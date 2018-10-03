# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2018 SINTEF Digital
Copyright (C) 2018 Norwegian Meteorological Institute

This python program runs a short program while reporting execution
metrics, and is intended to be used for meassuring computational 
performance and throughput.

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
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../../python/')))

import argparse
parser = argparse.ArgumentParser(description='Benchmark a simulator.')
parser.add_argument('--nx', type=int, default=1000)
parser.add_argument('--ny', type=int, default=1000)
parser.add_argument('--block_width', type=int)
parser.add_argument('--block_height', type=int)
parser.add_argument('--steps_per_download', type=int, default=2000)
parser.add_argument('--iterations', type=int, default=1)
parser.add_argument('--simulator', type=str)
parser.add_argument('--output', type=str, default=None)
args = parser.parse_args()


# Import timing utilities
import time
tic = time.time();

# Import packages we need
import os
import numpy as np
import pyopencl
from SWESimulators import FBL, CTCS, KP07, CDKLM16, PlotHelper, Common


toc = time.time()
print("{:02.4f} s: ".format(toc-tic) + "Imported packages")

# Create CUDA context
tic = time.time()
gpu_ctx = Common.GPUContext()
toc = time.time()
print("{:02.4f} s: ".format(toc-tic) + "Created context on " + gpu_ctx.devices[0].name)

# Set benchmark sizes
dx = 200.0
dy = 200.0

dt = 0.95/100
g = 9.81

f = 0.00
r = 0.0

boundaryConditions = Common.BoundaryConditions()

# Generate initial conditions
waterHeight = 60

x_center = dx*args.nx/2.0
y_center = dy*args.ny/2.0
size = 0.4*min(args.nx*dx, args.ny*dy)

def my_exp(i, j):
	x = dx*i - x_center
	y = dy*j - y_center
	return np.exp(-10*(x*x/(size*size)+y*y/(size*size))) * (np.sqrt(x**2 + y**2) < size)
	
"""
Initializes the KP simulator
"""
def initKP():
	tic = time.time()
	
	ghosts = np.array([2,2,2,2]) # north, east, south, west
	dataShape = (args.ny + ghosts[0]+ghosts[2], 
				 args.nx + ghosts[1]+ghosts[3])

	eta0 = np.fromfunction(lambda i, j: my_exp(i,j), dataShape, dtype=np.float32)
	u0 = np.zeros(dataShape, dtype=np.float32, order='C');
	v0 = np.zeros(dataShape, dtype=np.float32, order='C');
	Hi = np.ones((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C') * waterHeight;

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

	return sim

"""
Initializes the CDKLM simulator
"""
def initCDKLM():
	tic = time.time()
	
	ghosts = np.array([2,2,2,2]) # north, east, south, west
	dataShape = (args.ny + ghosts[0]+ghosts[2], 
				 args.nx + ghosts[1]+ghosts[3])

	eta0 = np.fromfunction(lambda i, j: my_exp(i,j), dataShape, dtype=np.float32)
	u0 = np.zeros(dataShape, dtype=np.float32, order='C');
	v0 = np.zeros(dataShape, dtype=np.float32, order='C');
	Hi = np.ones((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C') * waterHeight;

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
	
	return sim
	

"""
Initializes the FBL simulator
"""
def initFBL():
	tic = time.time()
	
	dataShape = (args.ny, args.nx)

	eta0 = np.fromfunction(lambda i, j: my_exp(i,j), dataShape, dtype=np.float32)
	u0 = np.zeros((dataShape[0]+0, dataShape[1]+1), dtype=np.float32);
	v0 = np.zeros((dataShape[0]+1, dataShape[1]+0), dtype=np.float32);
	h0 = np.ones(dataShape, dtype=np.float32) * waterHeight;

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
	
	return sim

	
	
"""
Initializes the CTCS simulator
"""
def initCTCS():
	tic = time.time()
	
	ghosts = [1,1,1,1] # north, east, south, west
	dataShape = (args.ny + ghosts[0]+ghosts[2], 
				 args.nx + ghosts[1]+ghosts[3])

	eta0 = np.fromfunction(lambda i, j: my_exp(i,j), dataShape, dtype=np.float32)
	u0 = np.zeros((dataShape[0]+0, dataShape[1]+1), dtype=np.float32);
	v0 = np.zeros((dataShape[0]+1, dataShape[1]+0), dtype=np.float32);
	h0 = np.ones(dataShape, dtype=np.float32) * waterHeight;

	toc = time.time()
	print("{:02.4f} s: ".format(toc-tic) + "Generated initial conditions")
			
	# Initialize simulator
	tic = time.time()
	
	A = 0.1*dx
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
sim.step(5*dt)
eta1, u1, v1 = sim.download()
toc = time.time()
print("{:02.4f} s: ".format(toc-tic) + "Spinup of simulator")

if (np.any(np.isnan(eta1))):
	print(" `-> ERROR: Not a number in spinup, aborting!")
	sys.exit(-1)

# Run simulator
print("=== Running with domain size [{:02d} x {:02d}], block size [{:s} x {:s}] ===".format(args.nx, args.ny, str(args.block_width), str(args.block_height)))
	
max_mcells = 0;
for i in range(args.iterations):
	print("{:03.0f} %".format(100*(i+1) / args.iterations))
	tic = time.time()
	t = sim.step(args.steps_per_download*dt)
	pyopencl.enqueue_barrier(sim.cl_queue)
	toc = time.time()
	mcells = args.nx*args.ny*args.steps_per_download/(1e6*(toc-tic))
	max_mcells = max(mcells, max_mcells);
	print(" `-> {:02.4f} s: ".format(toc-tic) + "Step, " + "{:02.4f} mcells/sec".format(mcells))
	tic = time.time()
	eta1, u1, v1 = sim.download()
	toc = time.time()
	print(" `-> {:02.4f} s: ".format(toc-tic) + "Download")
	
	if (np.any(np.isnan(eta1))):
		print(" `-> ERROR: Not a number in simulation, aborting!")
		sys.exit(-1)
		
	print(" `-> t_sim={:02.4f}".format(t) + ", h_max={:02.4f}".format(np.max(eta1)))

	
print(" === Maximum megacells: {:02.8f} ===".format(max_mcells))

# Save benchmarking data to file 
# (if file exists, we append if data for scheme is not added and overwrite if already added)
if (args.output):
    if(os.path.isfile(args.output)):
        with np.load(args.output) as file_data:
            data = dict(file_data)
    else:
        data = {}
    data[args.simulator] = max_mcells
    np.savez(args.output, **data)
