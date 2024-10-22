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

from tools import collision

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

def random_cube_q(cubeq0, cubeqgoal, step_size, checkcollision=True):
    oMframeChest = robot.data.oMf[1]
    from pinocchio.utils import rotate
    max_reach_d = 1
    distance_tolerance = 0.3
    ld, rd = robot_constraints()
    
    x_range, y_range, _ = abs(cubeqgoal.translation-cubeq0.translation)
    for _ in range(1000):
        # x = np.random.uniform(low=cubeq0.translation[0]-x_range, high=cubeq0.translation[0]+x_range)
        # y = np.random.uniform(low=cubeq0.translation[1]-y_range, high=cubeq0.translation[1]+y_range)
        
        new_pos = np.random.uniform(-1, 1, 3)*step_size
        # new_pos[:2] = new_pos[:2]
        direction = cubeq0.translation + new_pos
        q = pin.SE3(rotate('z', 0.),direction)
        
        d_obs = np.linalg.norm(q.translation-OBSTACLE_PLACEMENT.translation)
        if d_obs > distance_tolerance and direction[2] >= 0.93:
            return q
    return q

def computepath(qinit,qgoal,cubeplacementq0, cubeplacementqgoal):# -> List[q]:
    step_size = 2
    while True:
        q_rand = random_cube_q(cubeplacementq0, cubeplacementqgoal, step_size)

        pose, success = computeqgrasppose(robot, q, cube, q_rand, viz)
        if success:
            break
    #ensure that cube placements that are generated only occur in positions and orientations i like
    #robot must be holding cube, duh
    #
    return [qinit, qgoal]
    pass


def displaypath(robot,path,dt,viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    
    robot, cube, viz = setupwithmeshcat()
    
    
    q = robot.q0.copy()
    q_rand = random_cube_q(CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, 0.1)
    for i in range(5):
        q0,successinit = computeqgrasppose(robot, q, cube, q_rand, viz)
        q_rand = random_cube_q(q_rand, CUBE_PLACEMENT_TARGET, 0.1)
        
    # qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    # if not(successinit and successend):
    #     print ("error: invalid initial or end configuration")

    # path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    
    # displaypath(robot,path,dt=0.5,viz=viz) #you ll probably want to lower dt
    
