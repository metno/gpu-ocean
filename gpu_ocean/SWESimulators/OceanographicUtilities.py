# -*- coding: utf-8 -*-

"""
This python module implements misc helper functions

Copyright (C) 2019 SINTEF ICT

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from scipy import interpolate 

def intersectionsToMidpoints(a_i):
    """
    Converts values at cell intersections to values at midpoints. Use simple averaging
    """
    return 0.25*(a_i[:-1, :-1] + a_i[:-1, 1:] + a_i[1:, :-1] + a_i[1:, 1:])

    
def midpointsToIntersections(a_m, border_size, smoothing_factor=0.4):
    """
    Converts cell values at midpoints to cell values at midpoints using a cubic
    interpolating spline to generate first guess, followed by an iterative update. 
    """
    from scipy import interpolate 
    from scipy.ndimage.filters import gaussian_filter
    
    # Coordinates to midpoints
    x_m = np.mgrid[0:a_m.shape[1]] + 0.5
    y_m = np.mgrid[0:a_m.shape[0]] + 0.5
    
    # Coordinates to intersections
    x_i = np.mgrid[1:a_m.shape[1]]
    y_i = np.mgrid[1:a_m.shape[0]]
    
    # Create interpolating function and create first guess
    a_i1f = interpolate.interp2d(x_m, y_m, a_m, kind='cubic')
    a_i1 = a_i1f(x_i, y_i)

    # Iteratively refine intersections estimate
    for i in range(border_size):
        delta = a_m[1:-1,1:-1] - intersectionsToMidpoints(a_i1)
        smooth_delta = gaussian_filter(delta, max(1, border_size-i-1))
        delta = (1.0-smoothing_factor)*delta + smoothing_factor*smooth_delta
        
        a_i1[:-1, :-1] += delta*0.25
        a_i1[1:, :-1] += delta*0.25
        a_i1[1:, 1:] += delta*0.25
        a_i1[:-1, 1:] += delta*0.25 
        
    return a_i1[border_size-1:-border_size+1,border_size-1:-border_size+1]
    
    
def calcCoriolisParams(lat):
    #https://en.wikipedia.org/wiki/Coriolis_frequency
    #https://en.wikipedia.org/wiki/Beta_plane
    #Earths rotation rate: 7.2921 × 10−5 rad/s
    omega = 7.2921*1.0e-5
    
    #f is the angular velocity or frequency required to maintain a body at a fixed circle of latitude or zonal region
    f = 2.0*omega*np.sin(lat)
    
    #beta is df/dy where y points north
    #earth mean radius =  6,371.0088 km
    a = 6371008.8
    beta = 2.0*omega*np.cos(lat) / a
    
    return [f, beta]

def degToRad(deg):
    return deg*np.pi/180
    
    
    
def minmodY(eta, theta=1.3):
    backward = theta*(eta[1:-1,:] - eta[:-2,:])
    forward  = theta*(eta[2:,:] - eta[1:-1,:])
    central  = (eta[2:,:] - eta[:-2,:])/2
    
    positive = np.minimum(np.minimum(forward, backward), central)
    negative = np.maximum(np.maximum(forward, backward), central)
    
    positive = np.where((forward > 0) * (backward > 0) * (central > 0), positive, 0.0)
    negative = np.where((forward < 0) * (backward < 0) * (central < 0), negative, 0.0)
    
    retval = np.zeros_like(eta)
    retval[1:-1,:] = positive + negative
    retval[0,:] = theta*(eta[1,:] - eta[0,:])
    retval[-1,:] = theta*(eta[-2,:] - eta[-1,:])
    
    return retval

def minmodX(eta, theta=1.3):
    etaT = np.transpose(eta)
    D = minmodY(etaT)
    return np.transpose(D)

    
def rescaleMidpoints(data, nx1, ny1, **kwargs):
    ny0, nx0 = data.shape
    
    if (nx0 > nx1 and ny0 > ny1):
        # Subsample - non volume preserving
        dx0 = 1.0 / nx0
        dy0 = 1.0 / ny0
        
        x0 = np.linspace(0.5*dx0, 1.0-0.5*dx0, nx0)
        y0 = np.linspace(0.5*dy0, 1.0-0.5*dy0, ny0)
        
        data_int = interpolate.interp2d(x0, y0, data, kind='linear', **kwargs)
        
        dx1 = 1.0 / nx1
        dy1 = 1.0 / ny1
        
        x1 = np.linspace(0.5*dx1, 1.0-0.5*dx1, nx1)
        y1 = np.linspace(0.5*dy1, 1.0-0.5*dy1, ny1)
        
        return nx0/nx1, ny0/ny1, data_int(x1, y1)
    else:
        # Minmod reconstruction of slopes (volume preserving)
        dx = minmodX(data)
        dy = minmodY(data)
        
        x1 = nx0*np.linspace(0.5/nx1, 1-0.5/nx1, nx1)
        y1 = ny0*np.linspace(0.5/ny1, 1-0.5/ny1, ny1)
        
        x1, y1 = np.meshgrid(x1, y1)
        
        i = np.int32(x1)
        j = np.int32(y1)
        
        x1 = x1 - (i+0.5)
        y1 = y1 - (j+0.5)
        
        data_int = data[j, i] + x1 * dx[j, i] + y1 * dy[j, i]
        
        return nx0/nx1, ny0/ny1, data_int

def rescaleIntersections(data, nx1, ny1, **kwargs):
    ny0, nx0 = data.shape
        
    if (nx0 > nx1 and ny0 > ny1):
        # Subsample - using linear interpolation
        x0 = np.linspace(0, 1.0, nx0)
        y0 = np.linspace(0, 1.0, ny0)
        
        data_int = interpolate.interp2d(x0, y0, data, kind='linear', **kwargs)
        
        x1 = np.linspace(0, 1, nx1)
        y1 = np.linspace(0, 1, ny1)
        
        return (nx0-1)/(nx1-1), (ny0-1)/(ny1-1), data_int(x1, y1)
    else:
        # Rescales using bilinear interpolation        
        x1 = (nx0-1)*np.linspace(0, 1, nx1)
        y1 = (ny0-1)*np.linspace(0, 1, ny1)
        
        x1, y1 = np.meshgrid(x1, y1)
        
        #Get indices of four nearest neighbors
        i = np.int32(x1)
        j = np.int32(y1)
        k = np.minimum(i+1, nx0-1)
        l = np.minimum(j+1, ny0-1)
        
        #Get interpolation factors
        x1 = x1 - i
        y1 = y1 - j
        
        data_int = (1.0-y1) * ((1.0-x1)*data[j, i] + x1*data[j, k]) \
                       + y1 * ((1.0-x1)*data[l, i] + x1*data[l, k])
        
        return (nx0-1)/(nx1-1), (ny0-1)/(ny1-1), data_int
    
    
def calcGeostrophicBalance(eta, H_m, hu, hv, angle, f_beta, dx, dy, g=9.81, use_minmod=False, minmod_theta=1.3):
    
    #Calculate derivatives
    if (use_minmod):
        DetaDx = minmodX(eta, minmod_theta)/dx
        DetaDy = minmodY(eta, minmod_theta)/dy
    else:
        DetaDx, DetaDy = np.gradient(eta)
        DetaDx = DetaDx / dx
        DetaDy = DetaDy / dy
    
    #Get north and east vectors
    north = [np.sin(angle), np.cos(angle)]
    east = [np.cos(angle), -np.sin(angle)]
    
    #Calculate h
    h = H_m + eta
    
    #Get northward and eastward momentums
    hu_east = east[0]*hu + east[1]*hv
    hv_north = north[0]*hu + north[1]*hv
    
    #Calculat derivatives towards north and east for eta
    DetaDeast = east[0]*DetaDx + east[1]*DetaDy
    DetaDnorth = north[0]*DetaDx + north[1]*DetaDy
    
    #  f*hv - gh d\eta/dx = 0
    geos_x = f_beta*hv_north - g*h*DetaDeast

    #- f*hu - gh d\eta/dy = 0
    geos_y = -f_beta*hu_east - g*h*DetaDnorth
    
    return [geos_x, geos_y], [f_beta*hv_north, g*h*DetaDeast], [-f_beta*hu_east, g*h*DetaDnorth]