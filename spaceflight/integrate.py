# A basic two-body astrodynamics library written in Python in
# support of teaching basic space flight mechanics for the
# "Introduction to Space Flight" Workshop.

# This file comprises functions that perform orbit propagation
# either numerically using Runge-Kutta or analytically using the
# two-body solution.

# Stanford Summer Academic Resource Center
# Written by: Samuel Y. W. Low, 2nd August 2022

import orbits
import numpy as np

###############################################################################
###      FUNCTION TO COMPUTE GRAVITATIONAL FORCE IN A TWO-BODY SCENARIO     ###
###############################################################################

def forces( pos, vel ):
    pos = np.array(pos)
    GM  = 398600.4418 # Gravitational constant of the Earth
    R = np.linalg.norm(pos) # Distance from Earth centre [km]
    return -1*GM*pos/(R**3)

###############################################################################
###      NUMERICAL INTEGRATOR THAT RETURNS A NxT MATRIX OF STATES GIVEN     ###
###       AN INITIAL POSITION, INITIAL VELOCITY, THE TOTAL DURATION OF      ###
###    INTEGRATION AND THE TIME STEP. USES RK4 WITH SIMPSON'S RULE VARIANT  ###
###############################################################################

def numerical( states, t, dt ):
    N = 6
    Ti = 1
    T = int(t/dt)+1
    c = 1.0/3.0
    stateMatrix = np.zeros((N,T))
    stateMatrix[0:3,0] = states[0:3]
    stateMatrix[3:6,0] = states[3:6]
    while Ti < T:
        pos = stateMatrix[0:3,Ti-1]
        vel = stateMatrix[3:6,Ti-1]
        k1p = vel
        k1v = forces( pos, vel )
        k2p = vel + dt * (c*k1v)
        k2v = forces( pos + dt*(c*k1p), vel + dt*(c*k1v) )
        k3p = vel + dt * (k2v-c*k1v)
        k3v = forces( pos + dt*(k2p-c*k1p), vel + dt*(k2v-c*k1v) )
        k4p = vel + dt * (k1v-k2v+k3v)
        k4v = forces( pos + dt*(k1p-k2p+k3p), vel + dt*(k1v-k2v+k3v) )
        posf = pos + (dt/8) * (k1p + 3*k2p + 3*k3p + k4p)
        velf = vel + (dt/8) * (k1v + 3*k2v + 3*k3v + k4v)
        stateMatrix[0:3,Ti] = posf
        stateMatrix[3:6,Ti] = velf
        Ti += 1
    return stateMatrix

###############################################################################
###     ANALYTICAL INTEGRATOR THAT RETURNS A NxT MATRIX OF STATES GIVEN     ###
###    A SET OF KEPLERIAN ELEMENTS AND THE TOTAL DURATION OF INTEGRATION    ###
###############################################################################

def analytical( a, e, i, w, R, M, t, dt ):
    N = 6
    Ti = 1
    T = int(t/dt)+1
    elements = [a,e,i,w,R,M]
    states = orbits.elements2states(elements)
    stateMatrix = np.zeros((N,T))
    stateMatrix[0:3,0] = states[0:3]
    stateMatrix[3:6,0] = states[3:6]
    PI = 3.141592653589793
    R2D = 180.0/PI # Degrees to radians
    GM = 398600.4418 # Gravitational constant of the Earth
    meanMotion = ( GM / (elements[0]**3) )**0.5
    a = elements[0]
    e = elements[1]
    i = elements[2]
    w = elements[3]
    R = elements[4]
    M = elements[5]
    while Ti < T:
        M = M + meanMotion * R2D * dt # Integrate using n
        M = (((M + 180.0) % (360.0))) - 180.0 # Wrap
        newElements = [a,e,i,w,R,M]
        stateMatrix[:,Ti] = orbits.elements2states(newElements)
        Ti += 1
    return stateMatrix

###############################################################################
###     ANALYTICAL INTEGRATOR THAT RETURNS A NxT MATRIX OF STATES GIVEN     ###
###    A SET OF KEPLERIAN ELEMENTS AND THE TOTAL DURATION OF INTEGRATION    ###
###############################################################################

def numerical_continuous_dv( states, t, dt, dv):
    N = 6
    Ti = 1
    T = int(t/dt)+1
    c = 1.0/3.0
    stateMatrix = np.zeros((N,T))
    stateMatrix[0:3,0] = states[0:3]
    stateMatrix[3:6,0] = states[3:6]
    while Ti < T:
        dv_inertial = orbits.hill_to_inertial(stateMatrix[:,Ti-1], dv)
        pos = stateMatrix[0:3,Ti-1]
        vel = stateMatrix[3:6,Ti-1] + dv_inertial
        k1p = vel
        k1v = forces( pos, vel )
        k2p = vel + dt * (c*k1v)
        k2v = forces( pos + dt*(c*k1p), vel + dt*(c*k1v) )
        k3p = vel + dt * (k2v-c*k1v)
        k3v = forces( pos + dt*(k2p-c*k1p), vel + dt*(k2v-c*k1v) )
        k4p = vel + dt * (k1v-k2v+k3v)
        k4v = forces( pos + dt*(k1p-k2p+k3p), vel + dt*(k1v-k2v+k3v) )
        posf = pos + (dt/8) * (k1p + 3*k2p + 3*k3p + k4p)
        velf = vel + (dt/8) * (k1v + 3*k2v + 3*k3v + k4v)
        stateMatrix[0:3,Ti] = posf
        stateMatrix[3:6,Ti] = velf
        Ti += 1
    return stateMatrix
    
###############################################################################
###                                END OF FILE                              ###
###############################################################################