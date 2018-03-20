# -*- coding: utf-8 -*-

"""
This python class implements a DrifterCollection living on the GPU.

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
import BaseDrifterCollection

class GPUDrifterCollection(BaseDrifterCollection.BaseDrifterCollection):
    def __init__(self, cl_ctx, numDrifters, \
                 observation_variance=0.1, \
                 boundaryConditions=Common.BoundaryConditions(), \
                 domain_size_x=1.0, domain_size_y=1.0, \
                 cl_queue=None, \
                 block_width = 64):
        
        super(GPUDrifterCollection, self).__init__(numDrifters,
                                observation_variance=observation_variance,
                                boundaryConditions=boundaryConditions,
                                domain_size_x=domain_size_x, 
                                domain_size_y=domain_size_y)
        
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
        
        self.sensitivity = 1.0
         
        self.driftersHost = np.zeros((self.getNumDrifters() + 1, 2)).astype(np.float32, order='C')
        self.driftersDevice = Common.OpenCLArray2D(self.cl_ctx, \
                                                   2, self.getNumDrifters()+1, 0, 0, \
                                                    self.driftersHost)
        
        self.driftKernels = Common.get_kernel(self.cl_ctx,\
            "driftKernels.opencl", self.block_width, self.block_height)
        self.local_size = (self.block_width, self.block_height)
        self.global_size = (int(np.ceil((self.getNumDrifters() + 2)/float(self.block_width))*self.block_width), 
                            self.block_height )
        
        #print "local_size: ", self.local_size
        #print "global_size: ", self.global_size
        #print "numDrifters + obs: ", self.numDrifters + 1
        # remember: shape = (y, x)
        
    def copy(self):
        """
        Makes an independent indentical copy of the current object
        """
    
        copyOfSelf = GPUDrifterCollection(self.cl_ctx,
                                self.getNumDrifters(),
                                observation_variance = self.observation_variance,
                                boundaryConditions = self.boundaryConditions,
                                domain_size_x = self.domain_size_x, 
                                domain_size_y = self.domain_size_y,
                                cl_queue = self.cl_queue,
                                block_width = self.block_width)
        
        copyOfSelf.setDrifterPositions(self.getDrifterPositions())
        copyOfSelf.setObservationPosition(self.getObservationPosition())
        
        return copyOfSelf
    
    
    
    def setDrifterPositions(self, newDrifterPositions):
        ### Need to attache the observation to the newDrifterPositions, and then upload
        # to the GPU
        newPositionsAll = np.concatenate((newDrifterPositions, np.array([self.getObservationPosition()])), \
                                         axis=0)
        #print newPositionsAll
        self.driftersDevice.upload(self.cl_queue, newPositionsAll)
    
    def setObservationPosition(self, newObservationPosition):
        newPositionsAll = np.concatenate((self.getDrifterPositions(), np.array([newObservationPosition])))
        self.driftersDevice.upload(self.cl_queue, newPositionsAll)
        
    def setSensitivity(self, sensitivity):
        self.sensitivity = sensitivity
        
    def getDrifterPositions(self):
        allDrifters = self.driftersDevice.download(self.cl_queue)
        return allDrifters[:-1, :]
    
    def getObservationPosition(self):
        allDrifters = self.driftersDevice.download(self.cl_queue)
        return allDrifters[self.obs_index, :]
    
    def drift(self, eta, hu, hv, H0, nx, ny, dx, dy, dt, \
              x_zero_ref, y_zero_ref):
        
        #print "Calling drift with global_size " + str(self.global_size)
        self.driftKernels.passiveDrifterKernel(self.cl_queue, self.global_size, self.local_size, \
                                               nx, ny, dx, dy, dt, x_zero_ref, y_zero_ref, \
                                               eta.data, eta.pitch, \
                                               hu.data, hu.pitch, \
                                               hv.data, hv.pitch, \
                                               H0, \
                                               np.int32(self.boundaryConditions.isPeriodicNorthSouth()), \
                                               np.int32(self.boundaryConditions.isPeriodicEastWest()), \
                                               np.int32(self.getNumDrifters()), \
                                               self.driftersDevice.data, \
                                               self.driftersDevice.pitch, \
                                               np.float32(self.sensitivity))

    def setCLQueue(self, cl_queue):
        self.cl_queue = cl_queue
        
    def cleanUp(self):
        if (self.driftersDevice is not None):
            self.driftersDevice.release()
            
    def enforceBoundaryConditions(self):
        if self.boundaryConditions.isPeriodicNorthSouth or self.boundaryConditions.isPeriodicEastWest:
            self.driftKernels.enforceBoundaryConditions(self.cl_queue, self.global_size, self.local_size, \
                                                        np.float32(self.domain_size_x), \
                                                        np.float32(self.domain_size_y), \
                                                        np.int32(self.boundaryConditions.isPeriodicNorthSouth()), \
                                                        np.int32(self.boundaryConditions.isPeriodicEastWest()), \
                                                        np.int32(self.numDrifters), \
                                                        self.driftersDevice.data, \
                                                        self.driftersDevice.pitch)

