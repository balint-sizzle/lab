#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement

#CUSTOM IMPORTS
import time
from scipy.optimize import fmin_bfgs

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    
    
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)
    pin.framesForwardKinematics(robot.model,robot.data,q)
    pin.computeJointJacobians(robot.model,robot.data,q)

    
    oMcubeL = getcubeplacement(cube, LEFT_HOOK)
    oMcubeR = getcubeplacement(cube, RIGHT_HOOK)

    #align LEFT_HAND with LEFT_HOOK
    #3d first
    def cost(q):
        #now let's print the placement attached to the right hand
        pin.framesForwardKinematics(robot.model,robot.data,q)
        pin.computeJointJacobians(robot.model,robot.data,q)

        Rid = robot.model.getFrameId(RIGHT_HAND)
        oMframeR = robot.data.oMf[Rid]
        Lid = robot.model.getFrameId(LEFT_HAND)
        oMframeL = robot.data.oMf[Lid]

        return norm(pin.log(oMframeL.inverse() * oMcubeL).vector) + norm(pin.log(oMframeR.inverse() * oMcubeR).vector)

    qdes = fmin_bfgs(cost, qcurrent)
    return qdes, True
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    time.sleep(1)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    updatevisuals(viz, robot, cube, q0)
    
    
    
