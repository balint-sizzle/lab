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
from tools import jointlimitscost, jointlimitsviolated, distanceToObstacle

def apply_joint_limits(robot, q):
    q_clamped = np.clip(q, robot.model.lowerPositionLimit, robot.model.upperPositionLimit)
    return q_clamped

def distanceToCubeOrTable(robot, q):
      '''Return the shortest distance between robot and the obstacle. '''
      geomidcube = robot.collision_model.getGeometryId('cubebase_0')
      geomidtable = robot.collision_model.getGeometryId('baseLink_0')
      pairs = [i for i, pair in enumerate(robot.collision_model.collisionPairs) if pair.second == geomidcube or pair.second == geomidtable]
      pin.framesForwardKinematics(robot.model,robot.data,q)
      pin.updateGeometryPlacements(robot.model,robot.data,robot.collision_model,robot.collision_data,q)
      dists = [pin.computeDistance(robot.collision_model, robot.collision_data, idx).min_distance for idx in pairs]      
      
      # pairsId = [pair.first for i, pair in enumerate(robot.collision_model.collisionPairs) if pair.second == geomidobs or pair.second == geomidtable]
      # names = [robot.collision_model.geometryObjects[idx].name for idx in pairsId ]
      # for name, dist in zip(names,dists):
      #     print ("name / distance ", name, " / ", dist)
      # print(min (dists))
      return min(dists)

def calculate_collision_cost(robot, q):
    tol = 0.0001
    scaling = 1
    dto = distanceToCubeOrTable(robot, q)
    if dto < tol:
        return scaling*dto
    else:
        return 0

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)
    DT = 1e-1
    convergence_tolerance = 0.004
    q = qcurrent
    for i in range(200):

        pin.framesForwardKinematics(robot.model,robot.data,q)
        pin.computeJointJacobians(robot.model,robot.data,q)

        
        oMcubeL = getcubeplacement(cube, LEFT_HOOK)
        oMcubeR = getcubeplacement(cube, RIGHT_HOOK)
        Rid = robot.model.getFrameId(RIGHT_HAND)
        oMframeR = robot.data.oMf[Rid]
        Lid = robot.model.getFrameId(LEFT_HAND)
        oMframeL = robot.data.oMf[Lid]

        v1 = pin.log(oMframeR.inverse() * oMcubeR).vector
        v2 = pin.log(oMframeL.inverse() * oMcubeL).vector
        # v1[0] = v1[0]+0.03
        # v2[0] = v2[0]-0.03
        combined_err = v1 + v2
        

        o_JRhand = pin.computeFrameJacobian(robot.model, robot.data, q, Rid)
        o_JLhand = pin.computeFrameJacobian(robot.model, robot.data, q, Lid)
        
        # o_Jcombined = np.vstack([o_JRhand, o_JLhand])
        if np.linalg.norm(v1) < convergence_tolerance and np.linalg.norm(v2) < convergence_tolerance:
            return q, True

        collision_cost = calculate_collision_cost(robot, q)
        vq1 = pinv(o_JRhand) @ v1
        P1 = np.eye(robot.nv)-pinv(o_JRhand) @ o_JRhand
        vq = vq1 + pinv(o_JLhand @ P1) @ (v2 - o_JLhand @ vq1)

        # if collision_cost > 0:
        #     repulsive_velocity = collision_cost * np.sign(vq)
        #     vq += repulsive_velocity

        q = pin.integrate(robot.model,q, vq * DT)
        q = apply_joint_limits(robot, q)

        # viz.display(q)
        # time.sleep(0.03)
    # def cost(q):
    #     #now let's print the placement attached to the right hand
    #     pin.framesForwardKinematics(robot.model,robot.data,q)


    #     collision_cost = 0.01 if collision(robot, q) else 0
    #     jointlimit_violated = 0.01 if jointlimitsviolated(robot, q) else 0
    #     jointlimit_cost = jointlimitscost(robot, q)
    #     left_grasp = norm(pin.log(oMframeL.inverse() * oMcubeL).vector)
    #     right_grasp = norm(pin.log(oMframeR.inverse() * oMcubeR).vector)

    #     return (left_grasp+right_grasp)**2+collision_cost+jointlimit_violated 
    
    # def callback(q):
    #     from setup_meshcat import updatevisuals
    #     updatevisuals(viz, robot, cube, q)
    #     time.sleep(0.02)

    # qdes = fmin_bfgs(cost, qcurrent, epsilon=EPSILON, disp=False) #callback=callback, 

    # success = cost(qdes) < 0.01
    return q, False
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    # print(successinit, successend)
    updatevisuals(viz, robot, cube, qe)
    
    
