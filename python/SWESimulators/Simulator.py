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
import pyopencl as cl #OpenCL in Python
import Common, SimWriter
import gc
from abc import ABCMeta, abstractmethod

reload(Common)

class Simulator(object):
    """
    Baseclass for different numerical schemes, all 'solving' the SW equations.
    """
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def step(self, t_end=0.0):
        """
        Function which steps n timesteps
        """
        pass
    
    @abstractmethod
    def fromfilename(cls, cl_ctx, filename, cont_write_netcdf=True):
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
            self.sim_writer = None
        
    def attachDrifters(self, drifters):
        ### Do the following type of checking here:
        #assert isinstance(drifters, SingleGPUPassiveDrifterEnsemble)
        #assert drifters.isInitialized()
        
        self.drifters = drifters
        self.hasDrifters = True
        self.drifters.setCLQueue(self.cl_queue)
    
    def download(self):
        """
        Download the latest time step from the GPU
        """
        return self.cl_data.download(self.cl_queue)
    
    
    def downloadPrevTimestep(self):
        """
        Download the second-latest time step from the GPU
        """
        return self.cl_data.downloadPrevTimestep(self.cl_queue)
        
    def upload(self, eta0, hu0, hv0, eta1=None, hu1=None, hv1=None):
        """
        Reinitialize simulator with a new ocean state.
        """
        self.cl_data.h0.upload(self.cl_queue, eta0)
        self.cl_data.hu0.upload(self.cl_queue, hu0)
        self.cl_data.hv0.upload(self.cl_queue, hv0)
        
        if eta1 is None:
            self.cl_data.h1.upload(self.cl_queue, eta0)
            self.cl_data.hu1.upload(self.cl_queue, hu0)
            self.cl_data.hv1.upload(self.cl_queue, hv0)
        else:
            self.cl_data.h1.upload(self.cl_queue, eta1)
            self.cl_data.hu1.upload(self.cl_queue, hu1)
            self.cl_data.hv1.upload(self.cl_queue, hv1)
        

    
    