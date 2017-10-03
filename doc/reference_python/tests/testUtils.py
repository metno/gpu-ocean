import os
import sys
import numpy as np

import pyopencl

testdir = 'timestep50'

def utils(a):
    return a+1

def make_cl_ctx():
    #Make sure we get compiler output from OpenCL
    os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

    #Set which CL device to use, and disable kernel caching
    if (str.lower(sys.platform).startswith("linux")):
        os.environ["PYOPENCL_CTX"] = "0"
    else:
        os.environ["PYOPENCL_CTX"] = "1"
    os.environ["CUDA_CACHE_DISABLE"] = "1"
    os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"
    os.environ["PYOPENCL_NO_CACHE"] = "1"
        
    #Create OpenCL context
    cl_ctx = pyopencl.create_some_context()
    #print "Using ", cl_ctx.devices[0].name
    return cl_ctx


## A common initial condition maker:
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

def makeCentralBump(eta, nx, ny, dx, dy, halo):
    x_center = dx*nx/2.0
    y_center = dy*ny/2.0
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

## Add initial conditions on top of existing ones:
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

def addCentralBump(eta, nx, ny, dx, dy, halo):
    x_center = dx*nx/2.0
    y_center = dy*ny/2.0
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
                
def makeBathymetryCrazyness(B, nx, ny, dx, dy, halo):
    length = dx*nx*1.0
    height = dy*ny*1.0
    for j in range(-halo[2], ny + halo[0]):
        for i in range(-halo[3], nx + halo[1]):
            x = dx*i*1.0
            y = dy*j*1.0
            B[j+halo[2], i+halo[3]] = 25.0*(np.sin(np.pi*(x/length)*4)**2 + np.sin(np.pi*(y/height)*4)**2)            

def saveResults(eta, u, v, method, BC, init):
    fileprefix = testdir + "/" + method + "_" + BC + "_" + init + "_"
    np.savetxt(fileprefix + "eta.dat", eta)
    np.savetxt(fileprefix + "u.dat", u)
    np.savetxt(fileprefix + "v.dat", v)

def loadResults(method, BC, init):
    fileprefix = testdir + "/" + method + "_" + BC + "_" + init + "_"
    eta = np.loadtxt(fileprefix + "eta.dat")
    u =   np.loadtxt(fileprefix + "u.dat")
    v =   np.loadtxt(fileprefix + "v.dat")
    return eta, u, v
