#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv

from config import LEFT_HAND, RIGHT_HAND
import time

from tools import collision
#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def random_cube_q(checkcollision=True):
    while True:
        q = pin.randomConfiguration(robot.model)  # sample between -3.2 and +3.2.
        if not (checkcollision and collision(robot, q)): return q
    
def computepath(qinit,qgoal,cubeplacementq0, cubeplacementqgoal):# -> List[q]:
    #TODO
    q_rand = random_cube_q()
    return [q_rand]
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
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)

    if not(successinit and successend):
        print ("error: invalid initial or end configuration")
    
    path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    
    displaypath(robot,path,dt=0.5,viz=viz) #you ll probably want to lower dt
    
