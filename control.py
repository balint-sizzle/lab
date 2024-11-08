#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np

from bezier import Bezier
    
import pinocchio as pin
from config import LEFT_HAND, RIGHT_HAND
from pinocchio.utils import rotate
import matplotlib.pyplot as plt
# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 5000               # proportional gain (P of PD)
Kd = 2*np.sqrt(Kp)   # derivative gain (D of PD)
contact_force = 100
error_e = []
error_ed = []
u_s = []
nle_r = []
times = []

def controllaw(sim, robot, trajs, tcurrent, cube):
    q, vq = sim.getpybulletstate()
    pin.computeAllTerms(robot.model, robot.data, q, vq)
    Rid = robot.model.getFrameId(RIGHT_HAND)
    Lid = robot.model.getFrameId(LEFT_HAND)
    pin.computeJointJacobians(robot.model,robot.data,q)
    J_r = pin.computeFrameJacobian(robot.model, robot.data, q, Rid)
    J_l = pin.computeFrameJacobian(robot.model, robot.data, q, Lid)
    
    q_ref, vq_ref, vvq_ref = trajs

    e = q_ref(tcurrent)-q
    e_d = vq_ref(tcurrent)-vq
    vvq_star = vvq_ref(tcurrent) + Kp*e + Kd * e_d

    M = pin.crba(robot.model, robot.data, q)
    f_nle = robot.data.nle
    f = np.array([0., -contact_force, 0., 0., 0., 0.,])
    J = np.vstack([J_r, J_l])
    f = np.hstack([f, f])
    u_star = M @ vvq_star + f_nle + (J.T @ f)

    sim.step(u_star)

if __name__ == "__main__":
        
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import DT
    
    robot, sim, cube = setupwithpybullet()
    
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    from pinocchio.utils import rotate
    from path import computepath
    
    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    sim.setqsim(q0)
    
    def maketraj(path ,T):
        r = []
        r.append(path[0])
        r.append(path[0])
        r.append(path[0])
        for q in path:
            r.append(q)
            r.append(q)
            r.append(q)
        r.append(path[-1])
        r.append(path[-1])
        r.append(path[-1])

        q_of_t = Bezier(r, t_max=T)
        vq_of_t = q_of_t.derivative(1)
        vvq_of_t = vq_of_t.derivative(1)
        return q_of_t, vq_of_t, vvq_of_t

    total_time=5.
    trajs = maketraj(path, total_time)
    tcur = 0.
    while tcur < total_time:
        rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        tcur += DT