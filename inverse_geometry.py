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
from tools import distanceToObstacle

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    """
    Returns a tuple of a robot configuration and a boolean on whether a valid pose was found.
    If a valid pose is found, the configuration will hold the cube.
    If a valid pose is not found, then the last calculated position
    in the direction of the cube will be returned.

    Args:
        robot: Pinocchio object of the robot
        qcurrent: Initial list of joint configuration
        cube: Pinocchio object of the cube
        cubetarget: Cube configuration to perform grasp around

    Returns:
        q: Robot configuration
        success: Result of grasping
    """
    setcubeplacement(robot, cube, cubetarget)
    DT = 1e-1
    convergence_tolerance = 0.004
    q = qcurrent
    for _ in range(200):
        pin.framesForwardKinematics(robot.model,robot.data,q)
        pin.computeJointJacobians(robot.model,robot.data,q)
        
        oMcubeL = getcubeplacement(cube, LEFT_HOOK)
        oMcubeR = getcubeplacement(cube, RIGHT_HOOK)
        right_hand_id = robot.model.getFrameId(RIGHT_HAND)
        oMframeR = robot.data.oMf[right_hand_id]
        left_hand_id = robot.model.getFrameId(LEFT_HAND)
        oMframeL = robot.data.oMf[left_hand_id]

        v1 = pin.log(oMframeR.inverse() * oMcubeR).vector
        v2 = pin.log(oMframeL.inverse() * oMcubeL).vector

        o_JRhand = pin.computeFrameJacobian(robot.model, robot.data, q, right_hand_id)
        o_JLhand = pin.computeFrameJacobian(robot.model, robot.data, q, left_hand_id)
        converged = np.linalg.norm(v1) < convergence_tolerance and np.linalg.norm(v2) < convergence_tolerance
        
        # Early exit condition
        if converged and distanceToObstacle(robot, q) > 0:
            return q, True

        # Performing two tasks at the same time using the null space
        # projector of task 1
        vq1 = pinv(o_JRhand) @ v1
        P1 = np.eye(robot.nv)-pinv(o_JRhand) @ o_JRhand
        vq = vq1 + pinv(o_JLhand @ P1) @ (v2 - o_JLhand @ vq1)

        # Making sure velocities are kept within the limits specified
        # by the manufacturer in the URDF file
        vq = np.clip(vq, -1, 1)
        q = pin.integrate(robot.model,q, vq * DT)
        q = projecttojointlimits(robot, q)

    # If IK didn't converge then it cannot be a valid grasp
    return q, False
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    from pinocchio.utils import rotate
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    updatevisuals(viz, robot, cube, qe)
