# -*- coding: utf-8 -*-

"""
Copyright (C) 2016  SINTEF ICT

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

#Import packages we need
import numpy as np
import pycuda.driver as cuda
from SWESimulators import Common, SimWriter
import gc
from abc import ABCMeta, abstractmethod

try:
    from importlib import reload
except:
    pass
    
reload(Common)

class Simulator(object):
    """
    Baseclass for different numerical schemes, all 'solving' the SW equations.
    """
    __metaclass__ = ABCMeta
    
    
    def __init__(self, \
                 gpu_ctx, \
                 nx, ny, \
                 ghost_cells_x, \
                 ghost_cells_y, \
                 dx, dy, dt, \
                 g, f, r, A, \
                 t, \
                 theta, rk_order, \
                 coriolis_beta, \
                 y_zero_reference_cell, \
                 wind_stress, \
                 write_netcdf, \
                 ignore_ghostcells, \
                 offset_x, offset_y, \
                 block_width, block_height):
        """
        Setting all parameters that are common for all simulators
        """
        self.gpu_stream = cuda.Stream()
        
        #Save input parameters
        #Notice that we need to specify them in the correct dataformat for the
        #CUDA kernel
        self.gpu_ctx = gpu_ctx
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.ghost_cells_x = np.int32(ghost_cells_x)
        self.ghost_cells_y = np.int32(ghost_cells_y)
        self.dx = np.float32(dx)
        self.dy = np.float32(dy)
        self.dt = np.float32(dt)
        self.g = np.float32(g)
        self.f = np.float32(f)
        self.r = np.float32(r)
        self.coriolis_beta = np.float32(coriolis_beta)
        self.wind_stress = wind_stress
        self.y_zero_reference_cell = np.float32(y_zero_reference_cell)
        
        self.offset_x = offset_x
        self.offset_y = offset_y
        
        # Upload wind stress params to device
        self.wind_stress_dev = cuda.mem_alloc(wind_stress.csize())
        cuda.memcpy_htod_async(int(self.wind_stress_dev), wind_stress.tostruct(), stream=self.gpu_stream)
        
        #Initialize time
        self.t = np.float32(t)
        
        if A is None:
            self.A = 'NA'  # Eddy viscocity coefficient
        else:
            self.A = np.float32(A)
        
        if theta is None:
            self.theta = 'NA'
        else:
            self.theta = np.float32(theta)
        if rk_order is None:
            self.rk_order = 'NA'
        else:
            self.rk_order = np.int32(rk_order)
            
        self.hasDrifters = False
        self.drifters = None
        
        # NetCDF related parameters
        self.write_netcdf = write_netcdf
        self.ignore_ghostcells = ignore_ghostcells
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.sim_writer = None
        
        # Compute kernel launch parameters
        self.local_size = (block_width, block_height, 1) 
        self.global_size = ( \
                       int(np.ceil(self.nx / float(self.local_size[0]))), \
                       int(np.ceil(self.ny / float(self.local_size[1]))) \
                      )
            
    @abstractmethod
    def step(self, t_end=0.0):
        """
        Function which steps n timesteps
        """
        pass
    
    @abstractmethod
    def fromfilename(cls, filename, cont_write_netcdf=True):
        """
        Initialize and hotstart simulation from nc-file.
        cont_write_netcdf: Continue to write the results after each superstep to a new netCDF file
        filename: Continue simulation based on parameters and last timestep in this file
        """
        pass
   
    def __del__(self):
        self.cleanUp()

    @abstractmethod
    def cleanUp(self):
        """
        Clean up function
        """
        pass
        
    def closeNetCDF(self):
        """
        Close the NetCDF file, if there is one
        """
        if self.write_netcdf:
            self.sim_writer.__exit__(0,0,0)
            self.write_netcdf = False
        
    def attachDrifters(self, drifters):
        ### Do the following type of checking here:
        #assert isinstance(drifters, GPUDrifters)
        #assert drifters.isInitialized()
        
        self.drifters = drifters
        self.hasDrifters = True
        self.drifters.setCLQueue(self.gpu_stream)
    
    def download(self, interior_domain_only=False):
        """
        Download the latest time step from the GPU
        """
        if interior_domain_only:
            eta, hu, hv = self.gpu_data.download(self.gpu_stream)
            return eta[self.interior_domain_indices[2]:self.interior_domain_indices[0],  \
                       self.interior_domain_indices[3]:self.interior_domain_indices[1]], \
                   hu[self.interior_domain_indices[2]:self.interior_domain_indices[0],   \
                      self.interior_domain_indices[3]:self.interior_domain_indices[1]],  \
                   hv[self.interior_domain_indices[2]:self.interior_domain_indices[0],   \
                      self.interior_domain_indices[3]:self.interior_domain_indices[1]]
        else:
            return self.gpu_data.download(self.gpu_stream)
    
    
    def downloadPrevTimestep(self):
        """
        Download the second-latest time step from the GPU
        """
        return self.gpu_data.downloadPrevTimestep(self.gpu_stream)
        
    def copyState(self, otherSim):
        """
        Copies the state ocean state (eta, hu, hv), the wind object and 
        drifters (if any) from the other simulator.
        
        This function is exposed to enable efficient re-initialization of
        resampled ocean states. This means that all parameters which can be 
        initialized/assigned a perturbation should be copied here as well.
        """
        
        assert type(otherSim) is type(self), "A simulator can only copy the state from another simulator of the same class. Here we try to copy a " + str(type(otherSim)) + " into a " + str(type(self))
        
        assert (self.ny, self.nx) == (otherSim.ny, otherSim.nx), "Simulators differ in computational domain. Self (ny, nx): " + str((self.ny, self.nx)) + ", vs other: " + ((otherSim.ny, otherSim.nx))
        
        self.gpu_data.h0.copyBuffer(self.gpu_stream, otherSim.gpu_data.h0)
        self.gpu_data.hu0.copyBuffer(self.gpu_stream, otherSim.gpu_data.hu0)
        self.gpu_data.hv0.copyBuffer(self.gpu_stream, otherSim.gpu_data.hv0)
        
        self.gpu_data.h1.copyBuffer(self.gpu_stream, otherSim.gpu_data.h1)
        self.gpu_data.hu1.copyBuffer(self.gpu_stream, otherSim.gpu_data.hu1)
        self.gpu_data.hv1.copyBuffer(self.gpu_stream, otherSim.gpu_data.hv1)
        
        # Question: Which parameters should we require equal, and which 
        # should become equal?
        self.wind_stress = otherSim.wind_stress
        
        if otherSim.hasDrifters and self.hasDrifters:
            self.drifters.setDrifterPositions(otherSim.drifters.getDrifterPositions())
            self.drifters.setObservationPosition(otherSim.drifters.getObservationPosition())
        
        
        
    def upload(self, eta0, hu0, hv0, eta1=None, hu1=None, hv1=None):
        """
        Reinitialize simulator with a new ocean state.
        """
        self.gpu_data.h0.upload(self.gpu_stream, eta0)
        self.gpu_data.hu0.upload(self.gpu_stream, hu0)
        self.gpu_data.hv0.upload(self.gpu_stream, hv0)
        
        if eta1 is None:
            self.gpu_data.h1.upload(self.gpu_stream, eta0)
            self.gpu_data.hu1.upload(self.gpu_stream, hu0)
            self.gpu_data.hv1.upload(self.gpu_stream, hv0)
        else:
            self.gpu_data.h1.upload(self.gpu_stream, eta1)
            self.gpu_data.hu1.upload(self.gpu_stream, hu1)
            self.gpu_data.hv1.upload(self.gpu_stream, hv1)
            
    def _set_interior_domain_from_sponge_cells(self):
        """
        Use possible existing sponge cells to correctly set the 
        variable self.interior_domain_incides
        """
        if (self.boundary_conditions.isSponge()):
            self.interior_domain_indices = self.boundary_conditions.spongeCells.copy()
            self.interior_domain_indices[0:2] = -self.interior_domain_indices[0:2]
            print("self.interior_domain_indices: " + str(self.interior_domain_indices))
    
    