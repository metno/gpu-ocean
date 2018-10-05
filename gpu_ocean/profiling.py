#Import packages we need
import numpy as np
import os
import datetime
import sys
import argparse
import gc

import pycuda.driver as cuda
from pycuda.compiler import SourceModule

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../')))

#Import our simulator
from SWESimulators import FBL, CTCS, KP07, CDKLM16, Common

from SWESimulators.BathymetryAndICs import *

import importlib
importlib.reload(FBL)
importlib.reload(CTCS)
importlib.reload(KP07)
importlib.reload(CDKLM16)
importlib.reload(Common)


def runBenchmark(simulator, sim_args, sim_ic, iterations, steps_per_download):
    print("Creating context", flush=True)
    cuda_context = Common.CUDAContext()

    #Initialize simulator
    print("Creating simulator", flush=True)
    sim = simulator(gpu_ctx=cuda_context, **sim_args, **sim_ic)


    print("Simulating", flush=True)
    #Run a simulation and plot it
    for i in range(iterations):
        print(".", end='', flush=True)
        sim.step(steps_per_download*sim_args['dt'])
        eta1, u1, v1 = sim.download()
    print("", flush=True)

    print("eta: [{:f}, {:f}]".format(np.max(eta1), np.min(eta1)))
    print("u: [{:f}, {:f}]".format(np.max(u1), np.min(u1)))
    print("v: [{:f}, {:f}]".format(np.max(v1), np.min(v1)))
    

    
    
    
def genInitialConditions(simulator, sim_args):
    print("Creating initial conditions")
    
    ghost_cells = {}
    uv_staggering = None
    H_staggering = None
    
    if (simulator == FBL.FBL):
        ghost_cells = {'north': 0, 'east': 0, 'west': 0, 'south': 0}
        uv_staggering = 1
        H_staggering = 0
    elif (simulator == CTCS.CTCS):
        ghost_cells = {'north': 1, 'east': 1, 'west': 1, 'south': 1}
        uv_staggering = 1
        H_staggering = 0
    elif (simulator == KP07.KP07):
        ghost_cells = {'north': 2, 'east': 2, 'west': 2, 'south': 2}
        uv_staggering = 0
        H_staggering = 1
    elif (simulator == CDKLM16.CDKLM16):
        ghost_cells = {'north': 2, 'east': 2, 'west': 2, 'south': 2}
        uv_staggering = 0
        H_staggering = 1
        
        
    dataShape = (sim_args["ny"] + ghost_cells['south'] + ghost_cells['north'], 
                 sim_args["nx"] + ghost_cells['west']  + ghost_cells['east'])

    H0 = np.ones((dataShape[0] + H_staggering, dataShape[1]+H_staggering), dtype=np.float32) * 60
    eta0 = np.zeros(dataShape, dtype=np.float32)
    hu0 = np.zeros((dataShape[0], dataShape[1]+uv_staggering), dtype=np.float32)
    hv0 = np.zeros((dataShape[0]+uv_staggering, dataShape[1]), dtype=np.float32)
        
    #Create bump in to lower left of domain for testing
    print("Adding bump", flush=True)
    size = 0.75
    x = np.linspace(-1.0, 1.0, sim_args["nx"], dtype=np.float32)
    y = np.linspace(-1.0, 1.0, sim_args["ny"], dtype=np.float32)
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
    r = np.sqrt(xv**2 + yv**2)
    xv = None
    yv = None
    gc.collect()
    
    #Generate highres then downsample
    eta0[ghost_cells['south']:sim_args['ny']+ghost_cells['south'], ghost_cells['west']:sim_args['nx']+ghost_cells['west']] = 0.5*(1.0 + np.cos(np.pi*r/size)) * (r < size)

    #Initialize simulator
    sim_args = {"H": H0, "eta0": eta0, "hu0": hu0, "hv0": hv0}
    return sim_args
    
    
    
    
if __name__ == '__main__':

    sim_args_parser = argparse.ArgumentParser()
    sim_args_parser.add_argument('-nx', type=int, default=2048)
    sim_args_parser.add_argument('-ny', type=int, default=2048)
    sim_args_parser.add_argument('-dx', type=int, default=200)
    sim_args_parser.add_argument('-dy', type=int, default=200)
    sim_args_parser.add_argument('-dt', type=np.float32, default=1)
    sim_args_parser.add_argument('-g', type=np.float32, default=9.81)
    sim_args_parser.add_argument('-f', type=np.float32, default=0.001)
    sim_args_parser.add_argument('-r', type=np.float32, default=0.0)
    
    profiling_args_parser = argparse.ArgumentParser()
    profiling_args_parser.add_argument('-i', '--iterations', type=int, default=5)
    profiling_args_parser.add_argument('--steps_per_download', type=int, default=50)
    profiling_args_parser.add_argument('-fbl', '--fbl', dest='simulators', action='append_const', const=FBL.FBL)
    profiling_args_parser.add_argument('-ctcs', '--ctcs', dest='simulators', action='append_const', const=CTCS.CTCS)
    profiling_args_parser.add_argument('-kp', '--kp', dest='simulators', action='append_const', const=KP07.KP07)
    profiling_args_parser.add_argument('-cdklm', '--cdklm', dest='simulators', action='append_const', const=CDKLM16.CDKLM16)
    profiling_args_parser.add_argument('--block_width', type=int)
    profiling_args_parser.add_argument('--block_height', type=int)
    
    profiling_args, remaining = profiling_args_parser.parse_known_args()
    sim_vars, remaining = sim_args_parser.parse_known_args(args=remaining)
    sim_args = vars(sim_vars)

    
    if (profiling_args.block_width != None):
        sim_args['block_width'] = profiling_args.block_width
    if (profiling_args.block_height != None):
        sim_args['block_height'] = profiling_args.block_height
	

    
    
    if (remaining):
        sim_args_parser.print_help()
        profiling_args_parser.print_help()
        sys.exit(-1)
    
    for simulator in profiling_args.simulators:
        print("Running " + simulator.__name__ + " with " + str(sim_args) + ", " + str(vars(profiling_args)))
        
        sim_ic = genInitialConditions(simulator, sim_args)
        runBenchmark(simulator, sim_args, sim_ic, profiling_args.iterations, profiling_args.steps_per_download)
        
        
        
        
