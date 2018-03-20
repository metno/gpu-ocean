# -*- coding: utf-8 -*-

"""
This python class produces random perturbations that are to be added to 
the ocean state fields in order to generate model error.

Copyright (C) 2018  SINTEF ICT

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


from matplotlib import pyplot as plt
import numpy as np
import pyopencl
import gc

import Common
import FBL, CTCS

class OceanStateNoise(object):
    """
    Generating random perturbations for a ocean state.
   
    Perturbation for the surface field, dEta, is produced with a covariance structure according to a SOAR function,
    while dHu and dHv are found by the geostrophic balance to avoid shock solutions.
    """
    
    def __init__(self, cl_ctx, cl_queue,
                 nx, ny, 
                 boundaryConditions, staggered, cutoff=2,
                 block_width=16, block_height=16):
        
        self.cl_ctx = cl_ctx
        self.cl_queue = cl_queue
        
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.staggered = np.int(0)
        if staggered:
            self.staggered = np.int(1)
        
        self.periodicNorthSouth = np.int32(boundaryConditions.isPeriodicNorthSouth())
        self.periodicEastWest = np.int32(boundaryConditions.isPeriodicEastWest())
        
        # Size of random field and seed
        self.rand_nx = np.int32(nx + 2*(1+cutoff))
        self.rand_ny = np.int32(ny + 2*(1+cutoff))
        if self.periodicEastWest:
            self.rand_nx = np.int32(nx)
        if self.periodicNorthSouth:
            self.rand_ny = np.int32(ny)
        self.seed_ny = self.rand_ny
        self.seed_nx = np.int32(self.rand_nx/2) ### WHAT IF rand_nx IS ODD??
        
        # Generate seed:
        self.floatMax = 2147483648.0
        self.host_seed = np.random.rand(self.seed_ny, self.seed_nx)
        self.seed = Common.OpenCLArray2D(cl_ctx, self.seed_nx, self.seed_ny, 0, 0, self.host_seed)
        
        # Allocate memory for random numbers
        self.random_numbers_host = np.zeros((self.rand_ny, self.rand_nx), dtype=np.float32)
        self.random_numbers = Common.OpenCLArray2D(cl_ctx, self.rand_nx, self.rand_ny, 0, 0, self.random_numbers_host)
        
        # Generate kernels
        self.kernels = Common.get_kernel(self.cl_ctx, "ocean_noise.opencl", block_width, block_height)
 
        
        #Compute kernel launch parameters
        self.local_size = (block_width, block_height) 
        self.global_size_random_numbers = ( \
                       int(np.ceil(self.seed_nx / float(self.local_size[0])) * self.local_size[0]), \
                       int(np.ceil(self.seed_ny / float(self.local_size[1])) * self.local_size[1]) \
                                  ) 
        self.global_size_noise = ( \
                       int(np.ceil(self.rand_nx / float(self.local_size[0])) * self.local_size[0]), \
                       int(np.ceil(self.rand_ny / float(self.local_size[1])) * self.local_size[1]) \
                                  ) 
        
        
        
        
    def __del__(self):
        self.cleanUp()
     
    def cleanUp(self):
        self.seed.release()
        self.random_numbers.release()
        gc.collect()
        
    @classmethod
    def fromsim(cls, sim, cutoff=2, block_width=16, block_height=16):
        staggered = False
        if isinstance(cls, FBL.FBL) or isinstance(cls, CTCS.CTCS):
            staggered = True
        return cls(sim.cl_ctx, sim.cl_queue,
                   sim.nx, sim.ny,
                   sim.boundary_conditions, staggered, 
                   cutoff=cutoff,)
        
    def getSeed(self):
        return self.seed.download(self.cl_queue)
    
    def generateRandomNumbers(self):
        # Call kernel -> new random numbers 
        pass
    
    def getRandomNumbers(self):
        return self.random_numbers.download(self.cl_queue)
    
    # CPU versions of the above functions.
    def getSeedCPU(self):
        return self.host_seed
    
    def generateRandomNumbersCPU(self):
        pass
    
    def getRandomNumbersCPU(self):
        return self.random_numbers_host
    
    # CPU utility functions:
    
