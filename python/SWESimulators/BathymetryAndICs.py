# -*- coding: utf-8 -*-

"""
This notebook sets up and runs a set of benchmarks to compare
different numerical discretizations of the SWEs

Copyright (C) 2017  SINTEF ICT

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

"""
This file contains functions for creating initial conditions
and bathymetry
"""

"""
make*Bump functions generate a bump on a selected part of the domain
and leave the rest of the water surface unchanged"
"""
def makeCornerBump(eta, nx, ny, dx, dy, halo):
    x_center = 4*dx
    y_center = 4*dy
    for j in range(-halo[2], ny + halo[0]):
        for i in range(-halo[3], nx + halo[1]):
            x = dx*i - x_center
            y = dy*j - y_center
            size = 500.0*min(dx, dy)
            if (np.sqrt(x**2 + y**2) < size):
                eta[j+halo[2], i+halo[3]] = np.exp(-(x**2/size+y**2/size))
                
def makeUpperCornerBump(eta, nx, ny, dx, dy, halo):
    x_center = (nx-4)*dx
    y_center = (ny-4)*dy
    for j in range(-halo[2], ny + halo[0]):
        for i in range(-halo[3], nx + halo[1]):
            x = dx*i - x_center
            y = dy*j - y_center
            size = 500.0*min(dx, dy)
            if (np.sqrt(x**2 + y**2) < size):
                eta[j+halo[2], i+halo[3]] = np.exp(-(x**2/size+y**2/size))

                
def makeCentralBump(eta, H0, nx, ny, dx, dy, halo):
    x_center = dx*nx/2.0
    y_center = dy*ny/2.0
    for j in range(-halo[2], ny + halo[0]):
        for i in range(-halo[3], nx + halo[1]):
            x = dx*i - x_center
            y = dy*j - y_center
            size = (0.015* min(nx, ny)*min(dx, dy))**2
            if (np.sqrt(x**2 + y**2) < size):
                eta[j+halo[2], i+halo[3]] = H0 + np.exp(-(x**2/size+y**2/size))
                
def makeLowerLeftBump(eta, nx, ny, dx, dy, halo):
    x_center = dx*nx*0.3
    y_center = dy*ny*0.2
    for j in range(-halo[2], ny + halo[0]):
        for i in range(-halo[3], nx + halo[1]):
            x = dx*i - x_center
            y = dy*j - y_center
            size = 500.0*min(dx, dy)
            if (np.sqrt(x**2 + y**2) < size):
                eta[j+halo[2], i+halo[3]] = np.exp(-(x**2/size+y**2/size))
                
                
## Adding initial conditions on top of an existing initialCondition:
"""
add*Bump functions add a bump to a selected part of the domain on top of 
the already existing input array
"""
def addCornerBump(eta, nx, ny, dx, dy, halo):
    x_center = 4*dx
    y_center = 4*dy
    for j in range(-halo[2], ny + halo[0]):
        for i in range(-halo[3], nx + halo[1]):
            x = dx*i - x_center
            y = dy*j - y_center
            size = 500.0*min(dx, dy)
            if (np.sqrt(x**2 + y**2) < size):
                eta[j+halo[2], i+halo[3]] += np.exp(-(x**2/size+y**2/size))
                
                
def addUpperCornerBump(eta, nx, ny, dx, dy, halo):
    x_center = (nx-4)*dx
    y_center = (ny-4)*dy
    for j in range(-halo[2], ny + halo[0]):
        for i in range(-halo[3], nx + halo[1]):
            x = dx*i - x_center
            y = dy*j - y_center
            size = 500.0*min(dx, dy)
            if (np.sqrt(x**2 + y**2) < size):
                eta[j+halo[2], i+halo[3]] += np.exp(-(x**2/size+y**2/size))

                
def addCentralBump(eta, nx, ny, dx, dy, halo):
    x_center = dx*nx/2.0
    y_center = dy*ny/2.0
    for j in range(-halo[2], ny + halo[0]):
        for i in range(-halo[3], nx + halo[1]):
            x = dx*i - x_center
            y = dy*j - y_center
            size = 500.0*min(dx, dy)
            #size = (0.015* min(nx, ny)*min(dx, dy))**2
            if (np.sqrt(x**2 + y**2) < size):
                eta[j+halo[2], i+halo[3]] += np.exp(-(x**2/size+y**2/size))

def addLowerLeftBump(eta, nx, ny, dx, dy, halo):
    print("addLowerLeftBump")
    print("halo", halo)
    print("nx", nx)
    print("ny", ny)
    x_center = dx*nx*0.3
    y_center = dy*ny*0.2
    for j in range(-halo[2], ny + halo[0]):
        for i in range(-halo[3], nx + halo[1]):
            x = dx*i - x_center
            y = dy*j - y_center
            size = 500.0*min(dx, dy)
            if (np.sqrt(x**2 + y**2) < size):
                eta[j+halo[2], i+halo[3]] += np.exp(-(x**2/size+y**2/size))
            
# This bump is for debug purposes and will be modified without mercy :)
def addDebugBump(eta, nx, ny, dx, dy, posx, posy, halo):
    x_center = dx*nx*posx
    y_center = dy*ny*posy
    for j in range(-halo[2], ny + halo[0]):
        for i in range(-halo[3], nx + halo[1]):
            x = dx*i - x_center
            y = dy*j - y_center
            size = 500.0*min(dx, dy)
            if (np.sqrt(x**2 + y**2) < size):
                eta[j+halo[2], i+halo[3]] += np.exp(-(x**2/size+y**2/size))


"""
Generate a radial dam break initial condition with a step function
"""                
def addCentralDamBreakStep(eta, nx, ny, dx, dy, step_size, halo):
    x_center = dx*nx/2.0
    y_center = dy*ny/2.0
    for j in range(-halo[2], ny + halo[0]):
        for i in range(-halo[3], nx + halo[1]):
            x = dx*i - x_center
            y = dy*j - y_center
            size = 10.0*min(dx, dy)
            #size = (0.015* min(nx, ny)*min(dx, dy))**2
            if (np.sqrt(x**2 + y**2) < size):
                eta[j+halo[2], i+halo[3]] += step_size

"""
Generate a radial dam break initial condition with a step function
"""                
def addCentralDamBreakSmooth(eta, nx, ny, dx, dy, step_size, halo):
    print "addCentralDamBreakSmooth not implemented"
    
"""
Generates a smooth jeté along the x-axis at y=0.25*ny
This is done by decreasing the water depth and is therefore best
suited for staggered schemes.
"""
def addTopographyBump(h, nx, ny, dx, dy, halo, bumpsize):
    # Creating a bump in y direction (uniform in x direction)
    yPos = np.floor(ny*0.25)
    print(yPos)
    print((-halo[2], ny))
    print((-halo[3], nx + halo[1]))
    for j in range(-halo[2], ny + halo[0]):
        for i in range(-halo[3], nx + halo[1]):
            r = 0.01*(yPos - j)**2
            h[j+halo[2], i+halo[3]] -= bumpsize*np.exp(-r) 

"""
Generates a crater in the bathymetry B centered in the middle of the domain.
Can also be described as a radial bottom bump.
"""            
def makeBathymetryCrater(B, nx, ny, dx, dy, halo):
    x_center = dx*nx/2.0
    y_center = dy*ny/2.0
    minReach = min(nx*dx, ny*dy)
    innerEdge = minReach*0.3/2.0
    outerEdge = minReach*0.7/2.0
    for j in range(-halo[2], ny + halo[0]):
        for i in range(-halo[3], nx + halo[1]):
            x = dx*i - x_center
            y = dy*j - y_center
            radius = np.sqrt(x**2 + y**2)
            if (radius > innerEdge) and (radius < outerEdge):
                B[j+halo[2], i+halo[3]] = 30.0*np.sin((radius - innerEdge)/(outerEdge - innerEdge)*np.pi )**2
            else:
                B[j+halo[2], i+halo[3]] = 0.0

"""
Adds a semi-crazy bottom consisting of a few periods of sines both in x and y direction. 
"""
def makeBathymetryCrazyness(B, nx, ny, dx, dy, halo):
    length = dx*nx*1.0
    height = dy*ny*1.0
    for j in range(-halo[2], ny + halo[0]):
        for i in range(-halo[3], nx + halo[1]):
            x = dx*i*1.0
            y = dy*j*1.0
            B[j+halo[2], i+halo[3]] = 25.0*(np.sin(np.pi*(x/length)*4)**2 + np.sin(np.pi*(y/height)*4)**2)

"""
Generates a bathymetry with a constant slope along the x-axis.
B(x,y) = low + x*(high-low)/(nx*dx)
"""
def linearBathymetryX(B, nx, ny, dx, dy, halo, low, high):
    length=dx*nx*1.0
    gradient = (high-low)/length
    for j in range(0, ny+1):
        for i in range(0, nx+1):
            B[j+halo[2], i+halo[3]] = low + i*dx*gradient
"""
Generates a bathymetry with a constant slope along the y-axis.
B(x,y) = low + y*(high-low)/(ny*dy)
"""
            
def linearBathymetryY(B, nx, ny, dx, dy, halo, low, high):
    length=dy*ny*1.0
    gradient = (high-low)/length
    for j in range(0, ny+1):
        for i in range(0, nx+1):
            B[j+halo[2], i+halo[3]] = low + j*dy*gradient
            
"""
Generates a smooth jeté diagonally across the domain
"""            
def diagonalWallBathymetry(B, nx, ny, dx, dy, halo, height):
    for j in range(0, ny+1):
        for i in range(0, nx+1):
            factor = 1.0
            if ( i-j > -30 and i-j < 10):
                factor = 1 - np.exp(-0.01*(abs(10 - j + i)**2))
            B[j+halo[2], i+halo[3]] = factor*height*np.exp(-0.006*(abs(100-j - i)**2))

"""
Generates initial conditions for a dam break, where the dam is diagonal in a 
corner of the domain
"""                
def addDiagonalDam(h, nx, ny, dx, dy, halo, height):
    for j in range(0, ny+1):
        for i in range(0, nx+1):
            if ( i+j < 50):
                h[j+halo[2], i+halo[3]] += height
            
"""
Generates a smooth jeté along the x-axis across the domain
"""                 
def straightWallBathymetry(B, nx, ny, dx, dy, halo, height):
    for j in range(0, ny+1):
        for i in range(0, nx+1):
            factor = 1.0
            if ( i > 40 and i < 60):
                factor = 1 - np.exp(-0.05*(abs(50 - i)**2))
            B[j+halo[2], i+halo[3]] = factor*height*np.exp(-0.01*(abs(80-j)**2))

"""
Generates initial conditions for a dam break, where the dam is across the
entire domain along the x-axis, located at y = dam_start_y
"""       
def addStraightDam(h, nx, ny, dx, dy, halo, height, dam_start_y=30):
    for j in range(0, ny+1):
        if ( j < dam_start_y):
            for i in range(0, nx+1):
                h[j+halo[2], i+halo[3]] += height

"""
Adds a continental shelf in the south of the domain.
The shelf is sharp and discontinuous
"""                
def addContinentalShelfBathymetry(B, nx, ny, halo, shallow, deep, where_in_y):
    where_in_ny = ny*where_in_y
    for j in range(0, ny+1):
        if ( j < where_in_ny):
            for i in range(0, nx+1):
                B[j+halo[2], i+halo[3]] = shallow;
        else:
            for i in range(0, nx+1):
                B[j+halo[2], i+halo[3]] = deep;

"""
Adds a smooth continental shelf in the south of the domain.
The shelf is smooth in order to avoid discontinuous bathymetry
"""
def addContinentalShelfBathymetrySmooth(B, nx, ny, halo, shallow, deep, where_in_y):
    print "addContinentalShelfBathymetrySmooth not implemented"
    

                
"""
Dual vortex initial conditions.
Qualitatively inspired by the figures in this paper:
http://journals.ametsoc.org/doi/pdf/10.1175/1520-0469(1992)049%3C2015:LMOASW%3E2.0.CO%3B2
"""
def addDualVortex(eta, u, v, nx, ny, dx, dy, halo):
    x_center = dx*nx*0.5
    y_center_pos = dy*ny*0.52
    y_center_neg = dy*ny*0.48
    y_center = dy*ny*0.5
    for j in range(-halo[2], ny + halo[0]-1):
        for i in range(-halo[3], nx + halo[1]-1):
            # The -1 in the for loop is to avoid out-of-range on staggered grids...
            x = dx*i - x_center
            y_pos = dy*j - y_center_pos
            y_neg = dy*j - y_center_neg
            y = dy*j - y_center
            size = 500.0*min(dx, dy)
            contribution = 0.0
            neg_cont = 0.0
            #size = (0.015* min(nx, ny)*min(dx, dy))**2
            if (np.sqrt(x**2 + y_pos**2) < size):
                contribution -= np.exp(-0.5*(x**2/size+y_pos**2/size))
                contribution += np.exp(-0.2*(x**2/size+y_pos**2/size))
                eta[j+halo[2], i+halo[3]] += 0.035*np.exp(-0.2*(x**2/size+y_pos**2/size))
            
            if (np.sqrt(x**2 + y_neg**2) < size):
                neg_cont -= np.exp(-0.5*(x**2/size+y_neg**2/size))
                neg_cont += np.exp(-0.2*(x**2/size+y_neg**2/size))
                eta[j+halo[2], i+halo[3]] -= 0.035*np.exp(-0.2*(x**2/size+y_neg**2/size))
            
            global_cont = 0.0
            global_cont -= np.exp(-0.02*(x**2/size + y**2/size))
            global_cont += np.exp(-0.005*(x**2/size + y**2/size))
            
            phi_pos = np.arctan2(x, y_pos)
            phi_neg = np.arctan2(x, y_neg)
            phi = np.arctan2(x, y)
            velocity_scale = 10.0
            u[j+halo[2], i+halo[3]] += velocity_scale*contribution*np.cos(phi_pos)
            v[j+halo[2], i+halo[3]] -= velocity_scale*contribution*np.sin(phi_pos)
            
            u[j+halo[2], i+halo[3]] -= velocity_scale*neg_cont*np.cos(phi_neg)
            v[j+halo[2], i+halo[3]] += velocity_scale*neg_cont*np.sin(phi_neg)

            ## Add these two lines in order to initate a rotation of water under the dual vortices
            #u[j+halo[2], i+halo[3]] += 0.2*velocity_scale*global_cont*np.cos(phi)
            #v[j+halo[2], i+halo[3]] -= 0.2*velocity_scale*global_cont*np.sin(phi)
            
def addDualVortexStaggered(eta, u, v, nx, ny, dx, dy, halo):
    x_center = dx*nx*0.5
    y_center_pos = dy*ny*0.52
    y_center_neg = dy*ny*0.48
    y_center = dy*ny*0.5
    for j in range(-halo[2], ny + halo[0]-1):
        for i in range(-halo[3], nx + halo[1]-1):
            # The -1 in the for loop is to avoid out-of-range on staggered grids...
            x = dx*(i+0.5) - x_center
            y_pos = dy*(j+0.5) - y_center_pos
            y_neg = dy*(j+0.5) - y_center_neg
            y = dy*(j+0.5) - y_center
            size = 500.0*min(dx, dy)
            #size = (0.015* min(nx, ny)*min(dx, dy))**2
            if (np.sqrt(x**2 + y_pos**2) < size):
                eta[j+halo[2], i+halo[3]] += 0.035*np.exp(-0.2*(x**2/size+y_pos**2/size))
            if (np.sqrt(x**2 + y_neg**2) < size):
                eta[j+halo[2], i+halo[3]] -= 0.035*np.exp(-0.2*(x**2/size+y_neg**2/size))

            x = x - 0.5*dx
            contribution = 0.0
            neg_cont = 0.0
            if (np.sqrt(x**2 + y_pos**2) < size):
                contribution -= np.exp(-0.5*(x**2/size+y_pos**2/size))
                contribution += np.exp(-0.2*(x**2/size+y_pos**2/size))
            if (np.sqrt(x**2 + y_neg**2) < size):
                neg_cont -= np.exp(-0.5*(x**2/size+y_neg**2/size))
                neg_cont += np.exp(-0.2*(x**2/size+y_neg**2/size))
            phi_pos = np.arctan2(x, y_pos)
            phi_neg = np.arctan2(x, y_neg)
            phi = np.arctan2(x, y)
            velocity_scale = 10.0
            u[j+halo[2], i+halo[3]] += velocity_scale*contribution*np.cos(phi_pos)                
            u[j+halo[2], i+halo[3]] -= velocity_scale*neg_cont*np.cos(phi_neg)

            x = x + 0.5*dx
            y = y - 0.5*dy
            y_pos = y_pos - 0.5*dy
            y_neg = y_neg - 0.5*dy
            contribution = 0.0
            neg_cont = 0.0
            if (np.sqrt(x**2 + y_pos**2) < size):
                contribution -= np.exp(-0.5*(x**2/size+y_pos**2/size))
                contribution += np.exp(-0.2*(x**2/size+y_pos**2/size))
            if (np.sqrt(x**2 + y_neg**2) < size):
                neg_cont -= np.exp(-0.5*(x**2/size+y_neg**2/size))
                neg_cont += np.exp(-0.2*(x**2/size+y_neg**2/size))
            phi_pos = np.arctan2(x, y_pos)
            phi_neg = np.arctan2(x, y_neg)
            phi = np.arctan2(x, y)
            v[j+halo[2], i+halo[3]] -= velocity_scale*contribution*np.sin(phi_pos)
            v[j+halo[2], i+halo[3]] += velocity_scale*neg_cont*np.sin(phi_neg)
            
            ## Add these two lines in order to initate a rotation of water under the dual vortices
            #u[j+halo[2], i+halo[3]] += 0.2*velocity_scale*global_cont*np.cos(phi)
            #v[j+halo[2], i+halo[3]] -= 0.2*velocity_scale*global_cont*np.sin(phi)
            
            ## Debug lines without the cos/sin terms
            #u[j+halo[2], i+halo[3]] += velocity_scale*contribution
            #u[j+halo[2], i+halo[3]] -= velocity_scale*neg_cont
            #u[j+halo[2], i+halo[3]] += 0.2*velocity_scale*global_cont
