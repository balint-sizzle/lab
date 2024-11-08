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
from tools import setupwithmeshcat, distanceToObstacle
from inverse_geometry import computeqgrasppose
robot, cube, viz = setupwithmeshcat()
#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations

def cube_collision(oMf):
    setcubeplacement(robot, cube, oMf)
    dist = pin.computeDistance(cube.collision_model, cube.collision_data, 1).min_distance
    return dist<0.02

def pickup_cube_q(cube_q, q_current, viz):
    qt = cube_q.translation
    oMf = pin.SE3(rotate('z', 0.),  np.array([qt[0], qt[1], qt[2]+0.20]))
    q_pick, s = computeqgrasppose(robot, q_current, cube, oMf, viz)
    return oMf, q_pick

def random_cube_q(cube_q, cube_goal, q_current, viz):
    q_t, goal_t = cube_q.translation, cube_goal.translation
    x_bounds = (q_t[0], goal_t[0])
    y_bounds = (q_t[1], goal_t[1])
    z_steps = 0.05
    z_k = 10

    while True:
        x = np.random.uniform(x_bounds[0]*0.8, x_bounds[1]*1.2)
        y = np.random.uniform(y_bounds[0]*0.8, y_bounds[1]*1.2)
        for i in range(z_k, -1, -1):
            z = 0.93 + (z_steps * i)

            q_rand = np.array([x,y,z])
            oMf = pin.SE3(rotate('z', 0.),q_rand)
            q_rand, success = computeqgrasppose(robot, q_current, cube, oMf, viz)
            if not cube_collision(oMf) and success:
                return oMf, q_rand

def lerp(q0,q1,t):
    return q0.translation * (1 - t) + q1.translation * t

def lerp_config(q0,q1,t):
    return q0 * (1 - t) + q1 * t

def distance(q1,q2):
    return np.linalg.norm(q2.translation-q1.translation)

def distance_config(q1, q2):
    return np.linalg.norm(q1-q2)

def nearest_vertex(G, q):
    nearest_vert = min([(i, distance(q, v)) for i, (p, v, r) in enumerate(G)], key= lambda x: x[1])[0]
    return nearest_vert

def new_conf(cubeq_from, cubeq_to, discretisationsteps, q_current, delta_q = None):
    q, _ = computeqgrasppose(robot, q_current, cube, cubeq_to, viz)
    dist = distance(cubeq_from, cubeq_to)
    dt = dist / discretisationsteps

    for i in range(1,discretisationsteps):
        oMf = pin.SE3(rotate('z', 0.), lerp(cubeq_from,cubeq_to,(dt*i)/dist))
        q_h, success = computeqgrasppose(robot, q_current, cube, oMf, viz)
        # viz.display(q_h)
        # print(f"{i+1}: obs: {round(distanceToObstacle(robot, q_h), 3)}  to goal: {round(np.linalg.norm(cubeq_to.translation - oMf.translation), 3)}    current: {round(dt*i,3)}/{round(dist,3)}")
        # input()
        if cube_collision(oMf) or distanceToObstacle(robot, q_h) < 0.001:
            cubeq_prev = pin.SE3(rotate('z', 0.), lerp(cubeq_from,cubeq_to,(dt*(i-1))/dist))
            q_prev, success = computeqgrasppose(robot, q_current, cube, cubeq_prev, viz)
            return cubeq_prev, q_prev
        q_current = q_h
    return cubeq_to, q

def valid_edge(q_new,q_goal,discretisationsteps, goal_dst_tolerance, q_current):
    return np.linalg.norm(q_goal.translation - new_conf(q_new, q_goal, discretisationsteps, q_current)[0].translation) < goal_dst_tolerance

def getpath(G):
    path = []
    node = G[-1]
    while node[0] is not None:
        path = [node[2]] + path
        node = G[node[0]]
    path = [G[0][2]] + path
    return path

def computepath(qinit,qgoal,cubeplacementq0, cubeplacementqgoal):# -> List[q]:
    step_size = 0.3
    goal_heuristic = 0.5
    discretisationsteps = 50
    goal_dst_tolerance = 1e-4
    rrt_k = 1000
    cubeq_current = cubeplacementq0
    q_current = qinit
    pickup = True
    #3d rrt loop
    G = [(None,cubeplacementq0, qinit)]
    for i in range(rrt_k):
        print("node:",i)
        if pickup:
            cubeq_rand, q_rand = pickup_cube_q(cubeq_current, q_current, viz)
            pickup = False
        else:
            cubeq_rand, q_rand = random_cube_q(cubeplacementq0,cubeplacementqgoal, q_current, viz)
        cubeq_near_index = nearest_vertex(G, cubeq_rand)
        cubeq_near = G[cubeq_near_index][1]
        cubeq_new, q_new = new_conf(cubeq_near, cubeq_rand, discretisationsteps, q_rand)
        print("found new config")
        cubeq_current = cubeq_new
        q_current = q_new
        G.append([cubeq_near_index, cubeq_new, q_new])
        if valid_edge(cubeq_current, cubeplacementqgoal, discretisationsteps, goal_dst_tolerance, q_current):
            print("path found")
            path = getpath(G)
            path.append(qgoal)
            return path
    print("path wasn't found")

def displayedge(robot, cq0,cq1, rq0, viz, vel=2.): 
    '''Display the path obtained by linear interpolation of q0 to q1 at constant velocity vel'''
    fps = 60
    dist = distance(cq0, cq1)
    duration = dist/vel
    framerate = 1/fps
    nframes = ceil(fps * duration)
    for i in range(nframes):
        cube_qi = pin.SE3(rotate('z', 0.), lerp(cq0, cq1, float(i)/nframes))
        q_i,_ = computeqgrasppose(robot, rq0, cube, cube_qi, viz)
        viz.display(q_i)
        time.sleep(framerate)
    
def displaypath(robot,path,dt,viz):
    for i in range(1, len(path)):
        q0 = path[i-1]
        q1 = path[i]
        cq0, rq0 = q0
        cq1, rq1 = q1
        displayedge(robot, cq0, cq1, rq0, viz, vel=dt)
        time.sleep(dt)

def display_bezier(robot, path, dt, viz):
    from bezier import Bezier
    t_max = 5
    t = 0
    r = []
    c = []
    for q in path:
        r.append(q)
        r.append(q)
        r.append(q)

        # c.append(q[0])
        # c.append(q[0])
        # c.append(q[0])

    print(r)
    robot_path = np.array(r)
    # cube_path = np.array(c)
    robot_curve = Bezier(robot_path, t_max=t_max)
    # cube_curve = Bezier(cube_path, t_max=t_max)
    while t < t_max:
        q_robot = robot_curve.eval_horner(t)
        # q_cube = pin.SE3(cube_curve.eval_horner(t))

        # setcubeplacement(robot, cube, q_cube)
        viz.display(q_robot)
        t += dt
        time.sleep(dt)


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
    # path = np.array([np.array([-4.93444349e-01,  3.25351293e-17,  2.76401662e-17, -7.75288035e-01,
    #     7.52638877e-02, -3.11069264e-01,  3.42501444e-18,  2.35805377e-01,
    #    -2.94781621e-01, -4.16443347e-02, -3.47177833e-01,  1.54505051e-01,
    #    -3.55667798e-18,  1.92672782e-01,  2.10018064e+00]), np.array([-4.93444349e-01,  3.25351293e-17,  2.76401662e-17, -7.75288035e-01,
    #     7.52638877e-02, -3.11069264e-01,  3.42501444e-18,  2.35805377e-01,
    #    -2.94781621e-01, -4.16443347e-02, -3.47177833e-01,  1.54505051e-01,
    #    -3.55667798e-18,  1.92672782e-01,  2.10018064e+00]), np.array([-4.93444349e-01,  3.25351293e-17,  2.76401662e-17, -7.75288035e-01,
    #     7.52638877e-02, -3.11069264e-01,  3.42501444e-18,  2.35805377e-01,
    #    -2.94781621e-01, -4.16443347e-02, -3.47177833e-01,  1.54505051e-01,
    #    -3.55667798e-18,  1.92672782e-01,  2.10018064e+00]), np.array([-5.12183433e-01, -4.83464454e-04,  1.82840581e-05, -7.68048930e-01,
    #    -2.99416856e-01, -5.82784123e-01, -3.86318598e-03,  8.86017787e-01,
    #    -2.94334311e-01, -2.20289205e-02, -6.51315170e-01, -6.89792991e-02,
    #     3.83492414e-03,  7.24121462e-01,  2.10274246e+00]), np.array([-5.12183433e-01, -4.83464454e-04,  1.82840581e-05, -7.68048930e-01,
    #    -2.99416856e-01, -5.82784123e-01, -3.86318598e-03,  8.86017787e-01,
    #    -2.94334311e-01, -2.20289205e-02, -6.51315170e-01, -6.89792991e-02,
    #     3.83492414e-03,  7.24121462e-01,  2.10274246e+00]), np.array([-5.12183433e-01, -4.83464454e-04,  1.82840581e-05, -7.68048930e-01,
    #    -2.99416856e-01, -5.82784123e-01, -3.86318598e-03,  8.86017787e-01,
    #    -2.94334311e-01, -2.20289205e-02, -6.51315170e-01, -6.89792991e-02,
    #     3.83492414e-03,  7.24121462e-01,  2.10274246e+00]), np.array([-4.60013992e-01, -4.83464454e-04,  1.82840581e-05, -9.18801980e-01,
    #     6.70012570e-02, -1.00534714e+00, -5.24129055e-05,  9.38383815e-01,
    #    -1.92014390e-01,  8.01670962e-02, -3.14281370e-01, -5.72213475e-01,
    #     6.01721359e-05,  8.86530248e-01,  1.95217082e+00]), np.array([-4.60013992e-01, -4.83464454e-04,  1.82840581e-05, -9.18801980e-01,
    #     6.70012570e-02, -1.00534714e+00, -5.24129055e-05,  9.38383815e-01,
    #    -1.92014390e-01,  8.01670962e-02, -3.14281370e-01, -5.72213475e-01,
    #     6.01721359e-05,  8.86530248e-01,  1.95217082e+00]), np.array([-4.60013992e-01, -4.83464454e-04,  1.82840581e-05, -9.18801980e-01,
    #     6.70012570e-02, -1.00534714e+00, -5.24129055e-05,  9.38383815e-01,
    #    -1.92014390e-01,  8.01670962e-02, -3.14281370e-01, -5.72213475e-01,
    #     6.01721359e-05,  8.86530248e-01,  1.95217082e+00]), np.array([ 3.30087074e-02, -1.73208796e-03, -2.06247413e-03, -1.25272269e-01,
    #    -5.75999711e-02, -8.82182399e-01,  1.73041707e-06,  9.39785644e-01,
    #    -1.47854324e+00,  6.91904378e-01, -3.03086214e-02, -9.10889356e-01,
    #     6.17941462e-06,  9.41197629e-01,  8.47472323e-01]), np.array([ 3.30087074e-02, -1.73208796e-03, -2.06247413e-03, -1.25272269e-01,
    #    -5.75999711e-02, -8.82182399e-01,  1.73041707e-06,  9.39785644e-01,
    #    -1.47854324e+00,  6.91904378e-01, -3.03086214e-02, -9.10889356e-01,
    #     6.17941462e-06,  9.41197629e-01,  8.47472323e-01]), np.array([ 3.30087074e-02, -1.73208796e-03, -2.06247413e-03, -1.25272269e-01,
    #    -5.75999711e-02, -8.82182399e-01,  1.73041707e-06,  9.39785644e-01,
    #    -1.47854324e+00,  6.91904378e-01, -3.03086214e-02, -9.10889356e-01,
    #     6.17941462e-06,  9.41197629e-01,  8.47472323e-01]), np.array([ 0.1693032 ,  0.00230325,  0.00272233, -0.14798906, -0.02920587,
    #    -0.20259757, -0.00342804,  0.23524053, -1.58862521,  0.48690972,
    #     0.14708352, -0.37753085, -0.00343758,  0.2270218 ,  0.90584644]), np.array([ 0.1693032 ,  0.00230325,  0.00272233, -0.14798906, -0.02920587,
    #    -0.20259757, -0.00342804,  0.23524053, -1.58862521,  0.48690972,
    #     0.14708352, -0.37753085, -0.00343758,  0.2270218 ,  0.90584644]), np.array([ 0.1693032 ,  0.00230325,  0.00272233, -0.14798906, -0.02920587,
    #    -0.20259757, -0.00342804,  0.23524053, -1.58862521,  0.48690972,
    #     0.14708352, -0.37753085, -0.00343758,  0.2270218 ,  0.90584644])])
    input("display?")
    # displaypath(robot,path,dt=.2,viz=viz) #you ll probably want to lower dt
    display_bezier(robot, path, dt=.01, viz=viz)
