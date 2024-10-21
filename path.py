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
def random_cube_q(cubeq0, cubeqgoal, checkcollision=True):
    oMframeChest = robot.data.oMf[1]
    from pinocchio.utils import rotate
    max_reach_d = 1.3
    distance_tolerance = 0.2
    
    x_range, y_range, _ = abs(cubeqgoal.translation-cubeq0.translation)
    while True:
        # x = np.random.uniform(low=cubeq0.translation[0]-x_range, high=cubeq0.translation[0]+x_range)
        # y = np.random.uniform(low=cubeq0.translation[1]-y_range, high=cubeq0.translation[1]+y_range)
        Rid = robot.model.getFrameId(RIGHT_HAND)
        oMframeR = robot.data.oMf[Rid]
        Lid = robot.model.getFrameId(LEFT_HAND)
        oMframeL = robot.data.oMf[Lid]
        oMframeChest = robot.data.oMf[1]
        rd = np.linalg.norm(oMframeL.translation - oMframeChest.translation)
        ld = np.linalg.norm(oMframeR.translation - oMframeChest.translation)

        x = np.random.uniform(low=cubeq0.translation[0], high=cubeqgoal.translation[0])
        y = np.random.uniform(low=cubeq0.translation[1], high=cubeqgoal.translation[1])
        z = np.random.uniform(low=0.93, high=1.3)
        q = pin.SE3(rotate('z', 0.),np.array([x,y,z]))
        d_obs = np.linalg.norm(q.translation-OBSTACLE_PLACEMENT.translation)
        

        if d_obs > distance_tolerance and rd < max_reach_d and ld < max_reach_d:
            return q
    
def computepath(qinit,qgoal,cubeplacementq0, cubeplacementqgoal):# -> List[q]:
    #TODO
    q_rand = random_cube_q(cubeplacementq0, cubeplacementqgoal)
    return q_rand
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
    for i in range(1):
        q0,successinit = computeqgrasppose(robot, q, cube, random_cube_q(CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET), viz)
        
    # qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    # if not(successinit and successend):
    #     print ("error: invalid initial or end configuration")

    # path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    
    # displaypath(robot,path,dt=0.5,viz=viz) #you ll probably want to lower dt
    
