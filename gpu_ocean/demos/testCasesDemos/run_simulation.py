# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2018, 2019 SINTEF Digital
Copyright (C) 2018 Norwegian Meteorological Institute

This python program runs a 24 hour simulation for investigating the
performance of the different shallow water schemes.

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

import sys, os
current_dir = os.path.dirname(os.path.realpath(__file__))


if os.path.isdir(os.path.abspath(os.path.join(current_dir, '../../SWESimulators'))):
        sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../../')))

import argparse
parser = argparse.ArgumentParser(description='Benchmark a simulator.')
parser.add_argument('--nx', type=int, default=2048)
parser.add_argument('--ny', type=int, default=2048)
parser.add_argument('--block_width', type=int)
parser.add_argument('--block_height', type=int)
parser.add_argument('--simulator', type=str)
parser.add_argument('--output', type=str, default=None)


args = parser.parse_args()


# Import timing utilities
import time
tic = time.time();

# Import packages we need
import numpy as np
import json
from SWESimulators import FBL, CTCS, KP07, CDKLM16, PlotHelper, Common


toc = time.time()
print("{:02.4f} s: ".format(toc-tic) + "Imported packages")

# Create CUDA context
tic = time.time()
gpu_ctx = Common.CUDAContext()
device_name = gpu_ctx.cuda_device.name()
toc = time.time()
print("{:02.4f} s: ".format(toc-tic) + "Created context on " + device_name)

# Set benchmark sizes
dx = 250.0
dy = 250.0

courant_number = 0.2
g = 9.81

f = 0.0
r = 0.0

boundaryConditions = Common.BoundaryConditions()

# Generate initial conditions
waterHeight = 50

gravity_wave_speed = np.sqrt(g*(waterHeight + 1)) # 1 represent maximum of initial eta

nx = args.nx
ny = args.ny

               
def initEtaFV(eta0, ghosts):
    nx = eta0.shape[1] - ghosts[0] - ghosts[2]
    ny = eta0.shape[0] - ghosts[1] - ghosts[3]
    
    def my_cos(i, j):
        size = 0.6
        x = 2*(i + 0.5 - nx/2.0) / float(nx)
        y = 2*(j + 0.5 - ny/2.0) / float(ny)
        r = np.sqrt(x**2 + y**2)
        return 0.5*(1.0 + np.cos(np.pi*r/size)) * (r < size)
    
    #Generate disturbance 
    disturbance = np.fromfunction(lambda i, j: my_cos(i,j), (ny, nx))
    
    eta0.fill(0.0)
    x0, x1 = ghosts[0], nx+ghosts[0]
    y0, y1 = ghosts[1], ny+ghosts[1]
    eta0[y0:y1, x0:x1] += disturbance
    
    #Make sure solution is symmetric
    eta0 = 0.5*(eta0 +  eta0[::-1, ::-1])
    

def initEtaFD(eta0, ghosts):
    nx = eta0.shape[1] - ghosts[0] - ghosts[2]
    ny = eta0.shape[0] - ghosts[1] - ghosts[3]
    
    def my_cos(i, j):
        size = 0.6
        x = 2*(i - (nx-1)/2.0) / float(nx-1)
        y = 2*(j - (ny-1)/2.0) / float(ny-1)
        r = np.sqrt(x**2 + y**2)
        return 0.5*(1.0 + np.cos(np.pi*r/size)) * (r < size)
    
    #Generate disturbance 
    disturbance = np.fromfunction(lambda i, j: my_cos(i,j), (ny, nx))
    
    eta0.fill(0.0)
    x0, x1 = ghosts[0], nx+ghosts[0]
    y0, y1 = ghosts[1], ny+ghosts[1]
    eta0[y0:y1, x0:x1] += disturbance
    
    #Make sure solution is symmetric
    eta0 = 0.5*(eta0 +  eta0[::-1, ::-1])
    

    
def print_memory_req(sim, fbl=False, fvd=False):
    def get_buffer_size(buffer):
        return buffer.mem_size*buffer.itemsize
    
    mem = 0
    mem += get_buffer_size(sim.gpu_data.h0.data)
    mem += get_buffer_size(sim.gpu_data.hu0.data)
    mem += get_buffer_size(sim.gpu_data.hv0.data)
    if not fbl:
        mem += get_buffer_size(sim.gpu_data.h1.data)
        mem += get_buffer_size(sim.gpu_data.hu1.data)
        mem += get_buffer_size(sim.gpu_data.hv1.data)
    if fvd:
        mem += get_buffer_size(sim.bathymetry.Bi.data)
        mem += get_buffer_size(sim.bathymetry.Bm.data)
    else:
        mem += get_buffer_size(sim.H.data)

    memory_req = mem/(1024*1024)
    print(" -> Required memory: {:02.4f} MB".format(memory_req))
  
    
"""
Initializes the KP simulator
"""
def initKP():
    tic = time.time()

    ghosts = np.array([2,2,2,2]) # north, east, south, west
    dataShape = (args.ny + ghosts[0]+ghosts[2], 
                             args.nx + ghosts[1]+ghosts[3])

    eta0 = np.zeros(dataShape, dtype=np.float32, order='C');
    initEtaFV(eta0, ghosts)
    
    u0 = np.zeros(dataShape, dtype=np.float32, order='C');
    v0 = np.zeros(dataShape, dtype=np.float32, order='C');
    Hi = np.ones((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C') * waterHeight;
    
    dt = courant_number * dx/(4*gravity_wave_speed)
    toc = time.time()
    print("{:02.4f} s: ".format(toc-tic) + "Generated initial conditions")

    # Initialize simulator
    tic = time.time()

    kwargs = {'boundary_conditions': boundaryConditions, 'use_rk2': True}
    if (args.block_width != None):
            kwargs['block_width'] = args.block_width
    if (args.block_height != None):
            kwargs['block_height'] = args.block_height

    sim = KP07.KP07(gpu_ctx, \
                                    eta0, Hi, u0, v0, \
                                    args.nx, args.ny, \
                                    dx, dy, dt, \
                                    g, f, r, \
                                    **kwargs)
    toc = time.time()
    print("{:02.4f} s: ".format(toc-tic) + "Created KP simulator")

    print_memory_req(sim, fvd=True)
    
    return sim

"""
Initializes the CDKLM simulator
"""
def initCDKLM():
    tic = time.time()

    ghosts = np.array([2,2,2,2]) # north, east, south, west
    dataShape = (args.ny + ghosts[0]+ghosts[2], 
                             args.nx + ghosts[1]+ghosts[3])

    eta0 = np.zeros(dataShape, dtype=np.float32, order='C');
    initEtaFV(eta0, ghosts)
    
    u0 = np.zeros(dataShape, dtype=np.float32, order='C');
    v0 = np.zeros(dataShape, dtype=np.float32, order='C');
    Hi = np.ones((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C') * waterHeight;

    dt = courant_number * dx/(4*gravity_wave_speed)
    toc = time.time()
    print("{:02.4f} s: ".format(toc-tic) + "Generated initial conditions")

    # Initialize simulator
    tic = time.time()

    kwargs = {'boundary_conditions': boundaryConditions, 'rk_order': 2}
    if (args.block_width != None):
            kwargs['block_width'] = args.block_width
    if (args.block_height != None):
            kwargs['block_height'] = args.block_height

    sim = CDKLM16.CDKLM16(gpu_ctx, \
                                    eta0, u0, v0, Hi, \
                                    args.nx, args.ny, \
                                    dx, dy, dt, \
                                    g, f, r, \
                                    **kwargs)
    toc = time.time()
    print("{:02.4f} s: ".format(toc-tic) + "Created CDKLM simulator")

    print_memory_req(sim, fvd=True)
    
    return sim


"""
Initializes the FBL simulator
"""
def initFBL():
    tic = time.time()

    ghosts = [1, 1, 1, 1]
    dataShape = (args.ny+2, args.nx+2)

    eta0 = np.zeros(dataShape, dtype=np.float32, order='C');
    initEtaFD(eta0, ghosts)
    

    u0 = np.zeros((dataShape[0]+0, dataShape[1]-1), dtype=np.float32);
    v0 = np.zeros((dataShape[0]+1, dataShape[1]+0), dtype=np.float32);
    h0 = np.ones(dataShape, dtype=np.float32) * waterHeight;

    dt = courant_number * dx/(np.sqrt(2)*gravity_wave_speed)
    toc = time.time()
    print("{:02.4f} s: ".format(toc-tic) + "Generated initial conditions")

    # Initialize simulator
    tic = time.time()

    kwargs = {'boundary_conditions': boundaryConditions}
    if (args.block_width != None):
            kwargs['block_width'] = args.block_width
    if (args.block_height != None):
            kwargs['block_height'] = args.block_height

    sim = FBL.FBL(gpu_ctx, \
                                    h0, eta0, u0, v0, \
                                    args.nx, args.ny, \
                                    dx, dy, dt, \
                                    g, f, r, \
                                    **kwargs)
    toc = time.time()
    print("{:02.4f} s: ".format(toc-tic) + "Created FBL simulator")
    
    print_memory_req(sim, fbl=True)
  
    return sim

        
        
"""
Initializes the CTCS simulator
"""
def initCTCS():
    tic = time.time()

    ghosts = [1,1,1,1] # north, east, south, west
    dataShape = (args.ny + ghosts[0]+ghosts[2], 
                             args.nx + ghosts[1]+ghosts[3])

    eta0 = np.zeros(dataShape, dtype=np.float32, order='C');
    initEtaFD(eta0, ghosts)
    
    u0 = np.zeros((dataShape[0]+0, dataShape[1]+1), dtype=np.float32);
    v0 = np.zeros((dataShape[0]+1, dataShape[1]+0), dtype=np.float32);
    h0 = np.ones(dataShape, dtype=np.float32) * waterHeight;

    dt = courant_number * dx/(np.sqrt(2)*gravity_wave_speed)
    toc = time.time()
    print("{:02.4f} s: ".format(toc-tic) + "Generated initial conditions")

    # Initialize simulator
    tic = time.time()

    A = 1
    kwargs = {'boundary_conditions': boundaryConditions}
    if (args.block_width != None):
            kwargs['block_width'] = args.block_width
    if (args.block_height != None):
            kwargs['block_height'] = args.block_height

    sim = CTCS.CTCS(gpu_ctx, \
                                    h0, eta0, u0, v0, \
                                    args.nx, args.ny, \
                                    dx, dy, dt, \
                                    g, f, r, A, \
                                    **kwargs)
    toc = time.time()
    print("{:02.4f} s: ".format(toc-tic) + "Created CTCS simulator")
    
    print_memory_req(sim)
    
    return sim
        


sim = None

if (args.simulator == "KP"):
        sim = initKP()
elif (args.simulator == "CDKLM"): 
        sim = initCDKLM()
elif (args.simulator == "FBL"):
        sim = initFBL()
elif (args.simulator == "CTCS"):
        sim = initCTCS()
else:
        print("ERROR: Unknown simulator type '" + args.simulator + "', aborting")
        sys.exit(-1)

tic = time.time()
sim.step(5*sim.dt)
eta1, u1, v1 = sim.download()
toc = time.time()
print("{:02.4f} s: ".format(toc-tic) + "Spinup of simulator")

if (np.any(np.isnan(eta1))):
        print(" `-> ERROR: Not a number in spinup, aborting!")
        sys.exit(-1)

# Run simulator
print("=== Running 24 hours with domain size [{:02d} x {:02d}], block size [{:s} x {:s}] ===".format(args.nx, args.ny, str(args.block_width), str(args.block_height)))
        
num_iterations_pre = sim.num_iterations
    
gpu_ctx.synchronize()
tic = time.time()

t = sim.step(24*60*60)

gpu_ctx.synchronize()
toc = time.time()
        
mcells = args.nx*args.ny*(sim.num_iterations - num_iterations_pre)/(1e6*(toc-tic))
wall_time = toc - tic
print(" `-> {:02.4f} s: ".format(toc-tic) + "Step, " + "{:02.4f} mcells/sec".format(mcells))

tic = time.time()
eta1, u1, v1 = sim.download()
toc = time.time()
print(" `-> {:02.4f} s: ".format(toc-tic) + "Download")
print(" '->max(u): " + str(np.max(u1)))

if (np.any(np.isnan(eta1))):
        print(" `-> ERROR: Not a number in simulation, aborting!")
        sys.exit(-1)

print(" `-> t_sim={:02.4f}".format(t) + ", u_max={:02.4f}".format(np.max(u1)))

print(" === Wall time for 24 hour sim: {:02.8f} ===".format(wall_time))        
print(" === Maximum megacells: {:02.8f} ===".format(mcells))
print(" === Num iteration: {:02.8f} ===".format(sim.num_iterations - num_iterations_pre))

# Save benchmarking data to file 
# (if file exists, we append if data for scheme is not added and overwrite if already added)
if (args.output):
    if(os.path.isfile(args.output)):
        with np.load(args.output) as file_data:
            data = dict(file_data)
    else:
        data = {}
    data['megacells'] = max_mcells
    data['args'] = json.dumps(vars(args))
    np.savez(args.output, **data)
