# -*- coding: utf-8 -*-

"""
This python class represent an ensemble of particles stored on the GPU and using the GPU for drift calculations.

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
import matplotlib.gridspec as gridspec
import numpy as np
import time
import pyopencl

import Common
import CPUDrifter

class GPUDrifter(CPUDrifter.CPUDrifter):
    def __init__(self, cl_ctx, numParticles, \
                 observation_variance=0.1, \
                 boundaryConditions=Common.BoundaryConditions(), \
                 domain_size_x=1.0, domain_size_y=1.0, \
                 cl_queue=None, \
                 block_width = 64):
        # Define OpenCL environment:
        self.cl_ctx = cl_ctx
        self.block_width = block_width
        self.block_height = 1
        
        # TODO: Where should the cl_queue come from?
        # For sure, the drifter and the ocean simulator should use 
        # the same queue...
        self.cl_queue = cl_queue
        if self.cl_queue is None:
            self.cl_queue = pyopencl.CommandQueue(self.cl_ctx)
        
        self.numParticles = numParticles
        self.obs_index = self.numParticles
        self.observation_variance = observation_variance
        self.sensitivity = 1.0
        
        self.domain_size_x = domain_size_x
        self.domain_size_y = domain_size_y
        self.boundaryConditions = boundaryConditions
        
        self.particlesHost = np.zeros((self.numParticles + 1, 2)).astype(np.float32, order='C')
        self.particlesDevice = Common.OpenCLArray2D(self.cl_ctx, \
                                                    2, self.numParticles+1, 0, 0, \
                                                    self.particlesHost)
        
        self.driftKernels = Common.get_kernel(self.cl_ctx,\
            "driftKernels.opencl", self.block_width, self.block_height)
        self.local_size = (self.block_width, self.block_height)
        self.global_size = (int(np.ceil((self.numParticles + 2)/float(self.block_width))*self.block_width), 
                            self.block_height )
        
        #print "local_size: ", self.local_size
        #print "global_size: ", self.global_size
        #print "numParticles + obs: ", self.numParticles + 1
        # remember: shape = (y, x)
        
        
    def initializeParticles(self, domain_size_x = 1.0, domain_size_y = 1.0):
        
         # Initialize in unit square
        self.particlesHost = np.random.rand(self.numParticles + 1, 2)        
        # Ensure that the observation is in the middle 0.5x0.5 square:
        self.particlesHost[self.obs_index, :] = self.particlesHost[self.obs_index]*0.5 + 0.25
        
        # Map to given square
        self.particlesHost[:,0] = self.particlesHost[:,0]*domain_size_x
        self.particlesHost[:,1] = self.particlesHost[:,1]*domain_size_y
        
        self.domain_size_x = domain_size_x
        self.domain_size_y = domain_size_y
        self.particlesHost = self.particlesHost.astype(np.float32, order='C')
        
        self.particlesDevice.upload(self.cl_queue, self.particlesHost)
        
    def setParticlePositions(self, newParticlePositions):
        ### Need to attache the observation to the newParticlePositions, and then upload
        # to the GPU
        newPositionsAll = np.concatenate((newParticlePositions, np.array([self.getObservationPosition()])), \
                                         axis=0)
        #print newPositionsAll
        self.particlesDevice.upload(self.cl_queue, newPositionsAll)
    
    def setObservationPosition(self, newObservationPosition):
        newPositionsAll = np.concatenate((self.getParticlePositions(), np.array([newObservationPosition])))
        self.particlesDevice.upload(self.cl_queue, newPositionsAll)
        
    def setSensitivity(self, sensitivity):
        self.sensitivity = sensitivity
        
    def getParticlePositions(self):
        allParticles = self.particlesDevice.download(self.cl_queue)
        return allParticles[:-1, :]
    
    def getObservationPosition(self):
        allParticles = self.particlesDevice.download(self.cl_queue)
        return allParticles[self.obs_index, :]
    
    def drift(self, eta, hu, hv, H0, nx, ny, dx, dy, dt, \
              x_zero_ref, y_zero_ref):
        
        self.driftKernels.passiveDrifterKernel(self.cl_queue, self.global_size, self.local_size, \
                                               nx, ny, dx, dy, dt, x_zero_ref, y_zero_ref, \
                                               eta.data, eta.pitch, \
                                               hu.data, hu.pitch, \
                                               hv.data, hv.pitch, \
                                               H0, \
                                               np.int32(self.boundaryConditions.isPeriodicNorthSouth()), \
                                               np.int32(self.boundaryConditions.isPeriodicEastWest()), \
                                               np.int32(self.numParticles), \
                                               self.particlesDevice.data, \
                                               self.particlesDevice.pitch, \
                                               np.float32(self.sensitivity))

    def setCLQueue(self, cl_queue):
        self.cl_queue = cl_queue
        
    def cleanUp(self):
        if (self.particlesDevice is not None):
            self.particlesDevice.release()
            
    def enforceBoundaryConditions(self):
        if self.boundaryConditions.isPeriodicNorthSouth or self.boundaryConditions.isPeriodicEastWest:
            self.driftKernels.enforceBoundaryConditions(self.cl_queue, self.global_size, self.local_size, \
                                                        np.float32(self.domain_size_x), \
                                                        np.float32(self.domain_size_y), \
                                                        np.int32(self.boundaryConditions.isPeriodicNorthSouth()), \
                                                        np.int32(self.boundaryConditions.isPeriodicEastWest()), \
                                                        np.int32(self.numParticles), \
                                                        self.particlesDevice.data, \
                                                        self.particlesDevice.pitch)

