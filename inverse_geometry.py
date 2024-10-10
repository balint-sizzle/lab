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
from tools import jointlimitsviolated

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    
    
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)
    pin.framesForwardKinematics(robot.model,robot.data,qcurrent)
    pin.computeJointJacobians(robot.model,robot.data,qcurrent)
    
    oMcubeL = getcubeplacement(cube, LEFT_HOOK)
    oMcubeR = getcubeplacement(cube, RIGHT_HOOK)

    def cost(q):
        #now let's print the placement attached to the right hand
        pin.framesForwardKinematics(robot.model,robot.data,q)

        Rid = robot.model.getFrameId(RIGHT_HAND)
        oMframeR = robot.data.oMf[Rid]
        Lid = robot.model.getFrameId(LEFT_HAND)
        oMframeL = robot.data.oMf[Lid]
        
        collision_cost = 1 if collision(robot, q) else 0
        jointlimit_cost = 1 if jointlimitsviolated(robot, q) else 0
        left_grasp = norm(pin.log(oMframeL.inverse() * oMcubeL).vector)
        right_grasp = norm(pin.log(oMframeR.inverse() * oMcubeR).vector)

        return (left_grasp+right_grasp)**2+collision_cost+jointlimit_cost
    
    def callback(q):
        updatevisuals(viz, robot, cube, q)
        time.sleep(0.02)

    qdes = fmin_bfgs(cost, qcurrent, epsilon=EPSILON) #callback=callback
    success = cost(qdes) < 0.1

    return qdes, success
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    updatevisuals(viz, robot, cube, q0)
    
    
