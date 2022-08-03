# A basic two-body astrodynamics library written in Python in
# support of teaching basic space flight mechanics for the
# "Introduction to Space Flight" Workshop.

# This file comprises functions that perform conversions of state
# representations between orbital elements and cartesian form.

# Stanford Summer Academic Resource Center
# Written by: Samuel Y. W. Low, 2nd August 2022

import math
import numpy as np

###############################################################################
### FUNCTION TO CONVERT KEPLERIAN ORBIT ELEMENTS TO CARTESIAN STATES IN ECI ###
###############################################################################

def elements2states(elem):
    
    GM  = 398600.4418 # Gravitational constant of the Earth
    PI = 3.141592653589793
    D2R = PI/180.0 # Degrees to radian
    a, e, i, w, R, M = elem[0], elem[1], elem[2], elem[3], elem[4], elem[5]
    i, w, R, M = i*D2R, w*D2R, R*D2R, M*D2R
    
    # First, let us solve for the eccentric anomaly.
    eccAnom = mean2eccAnom(M,e)

    # With the eccentric anomaly, we can solve for position and velocity
    # in the local orbital frame, using the polar equation for an ellipse.
    # Note that the true anomaly would be included in the computation of
    # position and velocity in the perifocal frame below.
    pos_X = a * ( np.cos(eccAnom) - e)
    pos_Y = a * np.sqrt( 1 - e**2 ) * np.sin(eccAnom)
    pos_norm = np.sqrt( pos_X**2 + pos_Y**2 )
    vel_const = np.sqrt( GM * a ) / pos_norm
    vel_X = vel_const * ( -1 * np.sin(eccAnom) )
    vel_Y = vel_const * ( np.sqrt( 1 - e**2 ) * np.cos(eccAnom) )

    # The current coordinates are in the local hill frame, and thus 
    # conversion from hill to inertial would be the transpose of HN.
    DCM_HN = dcmZ(w) @ dcmX(i) @ dcmZ(R)
    DCM_NH = np.transpose(DCM_HN)

    # With the hill frame, we can now convert it to the ECI frame.
    pos = DCM_NH @ np.array([ pos_X, pos_Y, 0.0 ]).T
    vel = DCM_NH @ np.array([ vel_X, vel_Y, 0.0 ]).T
    
    return [pos[0],pos[1],pos[2],vel[0],vel[1],vel[2]] # Return state

###############################################################################
### FUNCTION TO CONVERT CARTESIAN STATES IN ECI TO KEPLERIAN ORBIT ELEMENTS ###
###############################################################################

def states2elements(states):
    
    GM  = 398600.4418 # Gravitational constant of the Earth
    pos, vel = np.array(states[0:3]), np.array(states[3:6])
    PI = 3.141592653589793
    R2D = 180.0/PI # Radians to degrees
    
    # First, compute the semi-major axis (assuming closed orbit).
    r = np.linalg.norm(pos)
    a = 1 / ( (2/r) - ( ( (np.linalg.norm(vel))**2 ) / GM ) )
    
    # Second, compute the angular momentum vector of the orbit.
    H      = np.cross(pos,vel)
    H_norm = np.linalg.norm(H)
    H_hat  = H / H_norm
    
    # Third, from the normalised angular momentum, derive the inclination.
    i = np.arctan2( math.sqrt( H_hat[0]**2 + H_hat[1]**2 ), H_hat[2] )
    
    # Fourth, from the normalised angular momentum, derive the RAAN.
    R = np.arctan2( H_hat[0], -1*H_hat[1] )
    
    # Fifth, compute the semi-latus rectum.
    p = ( H_norm**2 / GM )
    
    # Sixth, fetch the mean motion.
    n = (GM / (a**3))**0.5
    
    # Seventh, assuming an elliptical orbit, compute the eccentricity.
    if abs(p-a) < 1e-6:
        e = 0.0
    else:
        e = math.sqrt( 1 - (p/a) )
    
    # Eighth, compute the eccentric anomaly.
    E = np.arctan2( ( (np.dot(pos,vel) / ((a**2)*n)) ), (1 - r/a) )
    
    # Ninth, we can compute the mean anomaly using Kepler's equation.
    M = E - e*math.sin(E)
    
    # Tenth, the argument of latitude is computed.
    U = np.arctan2( pos[2], ( pos[1]*H_hat[0] - pos[0]*H_hat[1] ) )
    
    # Eleventh, the true anomaly is computed.
    nu = np.arctan2( math.sin(E)*math.sqrt(1-e**2), math.cos(E)-e )
    
    # Twelfth, the argument of perigee is computed.
    w = U - nu
    
    return [a,e,i*R2D,w*R2D,R*R2D,M*R2D] # Return the elements.

###############################################################################
###    FUNCTION TO SOLVE ECCENTRIC FROM MEAN ANOMALY VIA KEPLER'S EQUATION  ###
###############################################################################

def mean2eccAnom(M,e):
    E1 = M         # Initialise eccentric anomaly
    ei = e         # Initialise the float eccentricity
    residual = 1.0 # Initialise convergence residual
    while residual >= 0.000001:
        fn = E1 - (ei*np.sin(E1)) - M
        fd = 1 - (ei*np.cos(E1))
        E2 = E1 - (fn/fd)
        residual = abs(E2-E1) # Compute residual
        E1 = E2 # Update the eccentric anomaly
    return E2

################################################################################
###    FUNCTION TO GET A 3X3 ROTATION MATRIX FOR THE SPACECRAFT HILL FRAME   ###
################################################################################

def getHillDCM(states):
    pC = [states[0], states[1], states[2]]
    vC = [states[3], states[4], states[5]]
    hC = np.cross(pC, vC) # Angular momentum vector                      
    r_hat = pC / np.linalg.norm(pC) # Local X-axis
    h_hat = hC / np.linalg.norm(hC) # Local Z-axis
    y_hat = np.cross(h_hat, r_hat)  # Local Y-axis
    return np.array([r_hat, y_hat, h_hat])

################################################################################
###       FUNCTION TO CONVERT COORDINATES FROM HILL TO INERTIAL FRAME        ###
################################################################################

def hill_to_inertial(states, dv_hill):
    hillDCM = np.transpose(getHillDCM(states))
    return hillDCM @ np.array(dv_hill)

################################################################################
###    FUNCTION TO SOLVE ECCENTRIC FROM MEAN ANOMALY VIA KEPLER'S EQUATION   ###
################################################################################

def dcmX(t):
    dcm = np.array([[ 1.0,    0.0,         0.0         ],
                    [ 0.0,    math.cos(t), math.sin(t) ],
                    [ 0.0, -1*math.sin(t), math.cos(t) ]])
    return dcm

def dcmY(t):
    dcm = np.array([[ math.cos(t), 0.0, -1*math.sin(t) ],
                    [ 0.0,         1.0,    0.0         ],
                    [ math.sin(t), 0.0,    math.cos(t) ]])
    return dcm

def dcmZ(t):
    dcm = np.array([[    math.cos(t), math.sin(t), 0.0 ],
                    [ -1*math.sin(t), math.cos(t), 0.0 ],
                    [    0.0,         0.0,         1.0 ]])
    return dcm

################################################################################
###                                END OF FILE                               ###
################################################################################