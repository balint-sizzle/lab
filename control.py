#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np

from bezier import Bezier
    
import pinocchio as pin

# in my solution these gains were good enough for all joints but you might want to tune this.
Kp = 1e5               # proportional gain (P of PD)
Kv = 50   # derivative gain (D of PD)

def controllaw(sim, robot, trajs, tcurrent, cube):
    q, vq = sim.getpybulletstate()

    e = trajs[0](tcurrent)-q
    e_d = trajs[1](tcurrent)-vq
    pin.computeAllTerms(robot.model, robot.data, q, vq)
    # pin.computeMinverse(robot.model, robot.data, q)
    # pin.computeGeneralizedGravity(robot.model, robot.data, q)
    #pin.computeCentroidalMomentum(robot.model, robot.data, np.array([q, vq]))
    M = robot.data.M
    # tau = M *
    qt_star = trajs[2](tcurrent) + Kp*e + Kv * e_d
    torques = pin.rnea(robot.model, robot.data, e, e_d, qt_star)
    # print(torques)
    # u_star = M @ qt_star + G
    # torques = [0.0 for _ in sim.bulletCtrlJointsInPinOrder]
    sim.step(torques)

if __name__ == "__main__":
        
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import DT
    
    robot, sim, cube = setupwithpybullet()
    
    
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    from path import computepath
    
    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    # path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    
    #setting initial configuration
    sim.setqsim(q0)
    
    
    #TODO this is just an example, you are free to do as you please.
    #In any case this trajectory does not follow the path 
    #0 init and end velocities
    def maketraj(path ,T): #TODO compute a real trajectory !
        r = []
        for q in path:
            r.append(q)
            r.append(q)
            r.append(q)
        q_of_t = Bezier(r,t_max=T)
        vq_of_t = q_of_t.derivative(1)
        vvq_of_t = vq_of_t.derivative(1)
        return q_of_t, vq_of_t, vvq_of_t
        # cube_poses = [i[0].translation for i in path]
        # cube_of_t = Bezier(cube_poses, t_max = T)
        # cubev_of_t = cube_of_t.derivative(1)
        # cubevv_of_t = cubev_of_t.derivative(1)
        # return cube_of_t, cubev_of_t, cubev_of_t
    
    
    path = np.array([np.array([-4.93444349e-01,  3.25351293e-17,  2.76401662e-17, -7.75288035e-01,
        7.52638877e-02, -3.11069264e-01,  3.42501444e-18,  2.35805377e-01,
       -2.94781621e-01, -4.16443347e-02, -3.47177833e-01,  1.54505051e-01,
       -3.55667798e-18,  1.92672782e-01,  2.10018064e+00]), np.array([-4.93444349e-01,  3.25351293e-17,  2.76401662e-17, -7.75288035e-01,
        7.52638877e-02, -3.11069264e-01,  3.42501444e-18,  2.35805377e-01,
       -2.94781621e-01, -4.16443347e-02, -3.47177833e-01,  1.54505051e-01,
       -3.55667798e-18,  1.92672782e-01,  2.10018064e+00]), np.array([-4.93444349e-01,  3.25351293e-17,  2.76401662e-17, -7.75288035e-01,
        7.52638877e-02, -3.11069264e-01,  3.42501444e-18,  2.35805377e-01,
       -2.94781621e-01, -4.16443347e-02, -3.47177833e-01,  1.54505051e-01,
       -3.55667798e-18,  1.92672782e-01,  2.10018064e+00]), np.array([-5.12183433e-01, -4.83464454e-04,  1.82840581e-05, -7.68048930e-01,
       -2.99416856e-01, -5.82784123e-01, -3.86318598e-03,  8.86017787e-01,
       -2.94334311e-01, -2.20289205e-02, -6.51315170e-01, -6.89792991e-02,
        3.83492414e-03,  7.24121462e-01,  2.10274246e+00]), np.array([-5.12183433e-01, -4.83464454e-04,  1.82840581e-05, -7.68048930e-01,
       -2.99416856e-01, -5.82784123e-01, -3.86318598e-03,  8.86017787e-01,
       -2.94334311e-01, -2.20289205e-02, -6.51315170e-01, -6.89792991e-02,
        3.83492414e-03,  7.24121462e-01,  2.10274246e+00]), np.array([-5.12183433e-01, -4.83464454e-04,  1.82840581e-05, -7.68048930e-01,
       -2.99416856e-01, -5.82784123e-01, -3.86318598e-03,  8.86017787e-01,
       -2.94334311e-01, -2.20289205e-02, -6.51315170e-01, -6.89792991e-02,
        3.83492414e-03,  7.24121462e-01,  2.10274246e+00]), np.array([-4.60013992e-01, -4.83464454e-04,  1.82840581e-05, -9.18801980e-01,
        6.70012570e-02, -1.00534714e+00, -5.24129055e-05,  9.38383815e-01,
       -1.92014390e-01,  8.01670962e-02, -3.14281370e-01, -5.72213475e-01,
        6.01721359e-05,  8.86530248e-01,  1.95217082e+00]), np.array([-4.60013992e-01, -4.83464454e-04,  1.82840581e-05, -9.18801980e-01,
        6.70012570e-02, -1.00534714e+00, -5.24129055e-05,  9.38383815e-01,
       -1.92014390e-01,  8.01670962e-02, -3.14281370e-01, -5.72213475e-01,
        6.01721359e-05,  8.86530248e-01,  1.95217082e+00]), np.array([-4.60013992e-01, -4.83464454e-04,  1.82840581e-05, -9.18801980e-01,
        6.70012570e-02, -1.00534714e+00, -5.24129055e-05,  9.38383815e-01,
       -1.92014390e-01,  8.01670962e-02, -3.14281370e-01, -5.72213475e-01,
        6.01721359e-05,  8.86530248e-01,  1.95217082e+00]), np.array([ 3.30087074e-02, -1.73208796e-03, -2.06247413e-03, -1.25272269e-01,
       -5.75999711e-02, -8.82182399e-01,  1.73041707e-06,  9.39785644e-01,
       -1.47854324e+00,  6.91904378e-01, -3.03086214e-02, -9.10889356e-01,
        6.17941462e-06,  9.41197629e-01,  8.47472323e-01]), np.array([ 3.30087074e-02, -1.73208796e-03, -2.06247413e-03, -1.25272269e-01,
       -5.75999711e-02, -8.82182399e-01,  1.73041707e-06,  9.39785644e-01,
       -1.47854324e+00,  6.91904378e-01, -3.03086214e-02, -9.10889356e-01,
        6.17941462e-06,  9.41197629e-01,  8.47472323e-01]), np.array([ 3.30087074e-02, -1.73208796e-03, -2.06247413e-03, -1.25272269e-01,
       -5.75999711e-02, -8.82182399e-01,  1.73041707e-06,  9.39785644e-01,
       -1.47854324e+00,  6.91904378e-01, -3.03086214e-02, -9.10889356e-01,
        6.17941462e-06,  9.41197629e-01,  8.47472323e-01]), np.array([ 0.1693032 ,  0.00230325,  0.00272233, -0.14798906, -0.02920587,
       -0.20259757, -0.00342804,  0.23524053, -1.58862521,  0.48690972,
        0.14708352, -0.37753085, -0.00343758,  0.2270218 ,  0.90584644]), np.array([ 0.1693032 ,  0.00230325,  0.00272233, -0.14798906, -0.02920587,
       -0.20259757, -0.00342804,  0.23524053, -1.58862521,  0.48690972,
        0.14708352, -0.37753085, -0.00343758,  0.2270218 ,  0.90584644]), np.array([ 0.1693032 ,  0.00230325,  0.00272233, -0.14798906, -0.02920587,
       -0.20259757, -0.00342804,  0.23524053, -1.58862521,  0.48690972,
        0.14708352, -0.37753085, -0.00343758,  0.2270218 ,  0.90584644])])
    total_time=5.
    trajs = maketraj(path, total_time)
    # print("trajectory:", trajs[0](1))
    tcur = 0.
    
    
    while tcur < total_time:
        rununtil(controllaw, DT, sim, robot, trajs, tcur, cube)
        tcur += DT
    
    
    