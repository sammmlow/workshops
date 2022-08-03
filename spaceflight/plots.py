# A basic two-body astrodynamics library written in Python in
# support of teaching basic space flight mechanics for the
# "Introduction to Space Flight" Workshop.

# This file comprises functions plot orbits and trajectories.

# Stanford Summer Academic Resource Center
# Written by: Samuel Y. W. Low, 2nd August 2022

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import orbits
import integrate

###############################################################################
###     FUNCTION TO PLOT AN ORBIT IN THREE VIEWS GIVEN A 6xN STATE MATRIX   ###
###############################################################################

def plot_orbit( stateMatrix ):
    RE = 6378.140 # Earth's radius [km]
    lons = np.linspace(-180, 180, 100) * np.pi/180 
    lats = np.linspace(-90, 90, 100)[::-1] * np.pi/180 
    x = RE * np.outer(np.cos(lons), np.cos(lats)).T
    y = RE * np.outer(np.sin(lons), np.cos(lats)).T
    z = RE * np.outer(np.ones(np.size(lons)), np.sin(lats)).T
    plt.rcParams['figure.figsize'] = [12.8, 6.4]
    plt.rcParams['figure.dpi'] = 240
    fig = plt.figure()
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    axList = [ ax1,  ax2, ax3  ]
    elList = [ 30.0, 5.0, 90.0 ]
    azList = [ 60.0, 0.0, 0.0  ]
    titleList = ['Isometric View', 'Equatorial View', 'Polar View']
    for i in range(3):
        ax, el, az, title = axList[i], elList[i], azList[i], titleList[i]
        ax.plot_surface( x, y, z, cmap=cm.viridis, alpha = 0.25)
        ax.plot( stateMatrix[0,:], stateMatrix[1,:], stateMatrix[2,:] )
        ax.view_init( elev=el, azim=az )
        ax.set_xlabel('X [km]')
        ax.set_ylabel('Y [km]')
        ax.set_zlabel('Z [km]')
        ax.set_title(title)
    plt.show()


###############################################################################
###    FUNCTION TO PLOT AN ORBIT IN ONE VIEW, MEANT FOR INTERACTIVE PLOT    ###
###############################################################################

def plot_orbit_interactive(
    a=7000.0, e=0.0, i=45.0, w=0.0, R=0.0, M=0.0, t=3600.0, dt=60.0,
    azimuth=60.0, elevation=30.0):
    
    stateMatrix = integrate.analytical( a, e, i, w, R, M, t, dt )
    RE = 6378.140 # Earth's radius [km]
    lons = np.linspace(-180, 180, 100) * np.pi/180 
    lats = np.linspace(-90, 90, 100)[::-1] * np.pi/180 
    x = RE * np.outer(np.cos(lons), np.cos(lats)).T
    y = RE * np.outer(np.sin(lons), np.cos(lats)).T
    z = RE * np.outer(np.ones(np.size(lons)), np.sin(lats)).T
    plt.rcParams['figure.figsize'] = [12.8, 6.4]
    plt.rcParams['figure.dpi'] = 240
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface( x, y, z, cmap=cm.viridis, alpha = 0.25)
    ax.plot( stateMatrix[0,:], stateMatrix[1,:], stateMatrix[2,:] )
    ax.view_init( elev=elevation, azim=azimuth )
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.set_title('Plot of Orbit in Inertial Space')
    ax.axes.set_xlim3d(left=-1*a, right=a) 
    ax.axes.set_ylim3d(bottom=-1*a, top=a) 
    ax.axes.set_zlim3d(bottom=-1*a, top=a) 
    plt.show()


###############################################################################
###  FUNCTION TO PLOT HOHMANN TRANSFER FROM AN EQUATORIAL 7,000KM CIRCULAR  ###
###  ORBIT, GIVEN TWO IMPULSIVE THRUSTS AND A TIME OF FLIGHT BETWEEN THEM   ###
###############################################################################

def plot_hohmann(dV1, dV2, ToF):
    states = orbits.elements2states([7000,0,0,0,0,0])
    periodInitial = 2*np.pi*((orbits.states2elements( states )[0])**3 / 398600)**0.5
    statesIntegrated_Initial = integrate.numerical( states, periodInitial, 60 )
    statesFirstDV = statesIntegrated_Initial[:,-1]
    statesFirstDV[3:6] = statesFirstDV[3:6] + orbits.hill_to_inertial(statesFirstDV, dV1)
    statesIntegrated_Transfer = integrate.numerical( statesFirstDV, ToF, 60 )
    statesFinalDV = statesIntegrated_Transfer[:,-1]
    statesFinalDV[3:6] = statesFinalDV[3:6] + orbits.hill_to_inertial(statesFinalDV, dV2)
    periodFinal = 2*np.pi*((orbits.states2elements( statesFinalDV )[0])**3 / 398600)**0.5
    statesIntegrated_Final = integrate.numerical( statesFinalDV, periodFinal, 60 )
    statesTotal = np.hstack(( statesIntegrated_Initial, statesIntegrated_Transfer))
    statesTotal = np.hstack(( statesTotal, statesIntegrated_Final ))
    finalElem = orbits.states2elements( statesTotal[:,-1] )
    print("Your final orbit has a semi-major axis of:", finalElem[0], "km")
    print("Your final eccentricity is: ", finalElem[1])
    plot_orbit( statesTotal )