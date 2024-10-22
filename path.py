#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv

from config import LEFT_HAND, RIGHT_HAND, OBSTACLE_PLACEMENT
import time

from tools import collision, setcubeplacement
from pinocchio.utils import rotate
from math import ceil
#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def robot_constraints():
    Rid = robot.model.getFrameId(RIGHT_HAND)
    oMframeR = robot.data.oMf[Rid]
    Lid = robot.model.getFrameId(LEFT_HAND)
    oMframeL = robot.data.oMf[Lid]
    oMframeChest = robot.data.oMf[1]
    rd = np.linalg.norm(oMframeL.translation - oMframeChest.translation)
    ld = np.linalg.norm(oMframeR.translation - oMframeChest.translation)
    return ld,rd

def random_cube_q(cube_q, step_size, cube_goal, goal_heuristic):
    while True:
        new_pos = np.random.uniform(-1, 1, 3)*step_size
        towards_goal = cube_goal.translation- cube_q.translation
        heuristic = (towards_goal/np.linalg.norm(towards_goal))*step_size
        direction = cube_q.translation + (new_pos * (1-goal_heuristic)) + (heuristic * goal_heuristic)
        oMf = pin.SE3(rotate('z', 0.),direction)
        setcubeplacement(robot, cube, oMf)
        cube_coll = pin.computeCollision(cube.collision_model, cube.collision_data, False)
        if direction[2] >= 0.93 and not cube_coll:
            return oMf

def lerp(q0,q1,t):    
    return q0.translation * (1 - t) + q1.translation * t

def lerp_config(q0,q1,t):    
    return q0 * (1 - t) + q1 * t

def distance(q1,q2):
    return np.linalg.norm(q1.translation-q2.translation)

def distance_config(q1, q2):
    return np.linalg.norm(q1-q2)

def nearest_vertex(G, q):
    nearest_vert = min([(i, distance(q, v)) for i, (p, v, r) in enumerate(G)], key= lambda x: x[1])[0]
    return nearest_vert

def new_conf(cubeq_from, cubeq_to, discretisationsteps, q_current, delta_q = None):
    cubeq_end = cubeq_to.copy()
    q, _ = computeqgrasppose(robot, q_current, cube, cubeq_to, viz)
    dist = distance(cubeq_from, cubeq_to)

    # if delta_q is not None and dist > delta_q:
    #     #compute the configuration that corresponds to a path of length delta_q
    #     q_end = lerp(q_near,q_rand,delta_q/dist)
    #     dist = delta_q

    dt = dist / discretisationsteps
    for i in range(1,discretisationsteps):
        oMf = pin.SE3(rotate('z', 0.), lerp(cubeq_from,cubeq_to,dt*i))
        q, success = computeqgrasppose(robot, q_current, cube, oMf, viz)  # can I form a valid grasp through the discretised distance?
        setcubeplacement(robot, cube, oMf)
        if pin.computeCollision(cube.collision_model, cube.collision_data, False) or not success:
            cubeq_prev = pin.SE3(rotate('z', 0.), lerp(cubeq_from,cubeq_to,dt*i))
            setcubeplacement(robot, cube, cubeq_prev)
            q_prev, success = computeqgrasppose(robot, q_current, cube, oMf, viz)
            return cubeq_prev, q_prev
    return cubeq_end, q

def valid_edge(q_new,q_goal,discretisationsteps, goal_dst_tolerance, q_current):
    return np.linalg.norm(q_goal.translation -new_conf(q_new, q_goal,discretisationsteps, q_current)[0].translation) < goal_dst_tolerance

def getpath(G):
    path = []
    node = G[-1]
    while node[0] is not None:
        path = [(node[1], node[2])] + path
        node = G[node[0]]
    path = [(G[0][1], G[0][2])] + path
    return path

def computepath(qinit,qgoal,cubeplacementq0, cubeplacementqgoal):# -> List[q]:
    step_size = 0.2
    goal_heuristic = 0.5
    discretisationsteps = 5
    goal_dst_tolerance = 1e-1
    rrt_k = 100
    cubeq_current = cubeplacementq0
    q_current = qinit
    
    #3d rrt loop
    G = [(None,cubeplacementq0, qinit)]
    for i in range(rrt_k):
        print("node:",i)
        viz.display(q_current)
        while True:
            # if np.linalg.norm(cubeq_current-cubeplacementqgoal) < goal_dst_tolerance:
            #     cube_curent = cubeplacementqgoal
            #     break
            # else:
            cubeq_rand = random_cube_q(cubeq_current, step_size, cubeplacementqgoal, goal_heuristic)
            q_rand, success = computeqgrasppose(robot, q_current, cube, cubeq_rand, viz)
            if success:
                cubeq_current = cubeq_rand
                break
        print("found cube rand")
        cubeq_near_index = nearest_vertex(G, cubeq_rand)
        print("found nearest")
        cubeq_near = G[cubeq_near_index][1]
        cubeq_new, q_new = new_conf(cubeq_near, cubeq_rand, discretisationsteps, q_rand)
        print("found new config")
        cubeq_current = cubeq_new
        q_current = q_new
        G.append([cubeq_near_index, cubeq_new, q_new])
        if valid_edge(cubeq_current, cubeplacementqgoal, discretisationsteps, goal_dst_tolerance, q_current):
            print("path found")
            break
    print("path wasn't found")
       
    path = getpath(G)
    path.append((cubeplacementqgoal, qgoal))
    return path

def displayedge(q0,q1, viz, vel=2.): #vel in sec.    
    '''Display the path obtained by linear interpolation of q0 to q1 at constant velocity vel'''
    fps = 40
    framerate = 1/fps
    t = ceil(distance_config(q0, q1)/vel)
    for f in range(fps):
        viz.display(lerp_config(q0, q1, t*f))
        time.sleep(framerate)
    
def displaypath(robot,path,dt,viz):
    # print(path)
    for q0, q1 in zip(path[:-1],path[1:]):
        cq0, rq0 = q0
        cq1, rq1 = q1
        setcubeplacement(robot, cube, cq0)
        displayedge(rq0, rq1, viz)
        time.sleep(dt)
    # for q in path:
    #     cq0, rq0 = q
    #     setcubeplacement(robot, cube, cq0)
    #     print("step")
    #     displayedge()
    #     time.sleep(dt)



if __name__ == "__main__":
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    
    robot, cube, viz = setupwithmeshcat()
    
    
    q = robot.q0.copy()
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    if not(successinit and successend):
        print ("error: invalid initial or end configuration")
    
    path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    input("display?")
    displaypath(robot,path,dt=.5,viz=viz) #you ll probably want to lower dt
    
