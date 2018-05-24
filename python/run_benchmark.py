import sys
sys.path.insert(0, "C:/Users/anbro/Documents/projects/ocean drift/gpu-ocean/python")


#Import packages we need
import numpy as np
from matplotlib import animation, rc
from matplotlib import pyplot as plt

import os
import pyopencl
import datetime
import sys

#Import our simulator
from SWESimulators import FBL, CTCS, LxF, KP07, CDKLM16, PlotHelper, Common
from SWESimulators.BathymetryAndICs import *


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
print "Using ", cl_ctx.devices[0].name


    
# Kurganov-Petrova 2007 paper
nx = 1000
ny = 1000

dx = 200.0
dy = 200.0

dt = 0.95/100
g = 9.81

f = 0.00
r = 0.0

# WIND
#wind = Common.WindStressParams(type=0, tau0=3.0, rho=1025, alpha=1.0/(100*dx))
wind = Common.WindStressParams(type=99)


ghosts = np.array([2,2,2,2]) # north, east, south, west
validDomain = np.array([2,2,2,2])
boundaryConditions = Common.BoundaryConditions()
dataShape = (ny + ghosts[0]+ghosts[2], 
             nx + ghosts[1]+ghosts[3])
             
waterHeight = 60
eta0 = np.zeros(dataShape, dtype=np.float32, order='C');
u0 = np.zeros(dataShape, dtype=np.float32, order='C');
v0 = np.zeros(dataShape, dtype=np.float32, order='C');

Hi = np.ones((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C') * waterHeight;

addCentralBump(eta0, nx, ny, dx, dy, validDomain)


#Initialize simulator
sim = KP07.KP07(cl_ctx, \
                eta0, Hi, u0, v0, \
                nx, ny, \
                dx, dy, dt, \
                g, f, r, \
                wind_stress=wind, \
                boundary_conditions=boundaryConditions,
                use_rk2=True)
T = 1
for i in range(T):
    t = sim.step(100.0)
    eta1, u1, v1 = sim.download()
    if (i%10 == 0):
        print "{:03.0f}".format(100*i / T) + " % => t=" + str(t) + "\tMax h: " + str(np.max(eta1))
