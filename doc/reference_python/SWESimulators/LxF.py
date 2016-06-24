# -*- coding: utf-8 -*-

"""
This python module implements the classical Lax-Friedrichs numerical
scheme for the shallow water equations

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
import Common



        
        
        
        
        
        


"""
Class that solves the SW equations using the Forward-Backward linear scheme
"""
class LxF:

    """
    Initialization routine
    h0: Water depth incl ghost cells, (nx+1)*(ny+1) cells
    hu0: Initial momentum along x-axis incl ghost cells, (nx+1)*(ny+1) cells
    hv0: Initial momentum along y-axis incl ghost cells, (nx+1)*(ny+1) cells
    nx: Number of cells along x-axis
    ny: Number of cells along y-axis
    dx: Grid cell spacing along x-axis (20 000 m)
    dy: Grid cell spacing along y-axis (20 000 m)
    dt: Size of each timestep (90 s)
    g: Gravitational accelleration (9.81 m/s^2)
    """
    def __init__(self, \
                 cl_ctx, \
                 h0, hu0, hv0, \
                 nx, ny, \
                 dx, dy, dt, \
                 g, \
                 block_width=16, block_height=16):
        self.cl_ctx = cl_ctx

        #Create an OpenCL command queue
        self.cl_queue = cl.CommandQueue(self.cl_ctx)

        #Get kernels
        self.lxf_kernel = Common.get_kernel(self.cl_ctx, "LxF_kernel.opencl", block_width, block_height)
        
        #Create data by uploading to device
        ghost_cells_x = 1
        ghost_cells_y = 1
        self.cl_data = Common.SWEDataArkawaA(self.cl_ctx, nx, ny, ghost_cells_x, ghost_cells_y, h0, hu0, hv0)
        
        #Save input parameters
        #Notice that we need to specify them in the correct dataformat for the
        #OpenCL kernel
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.dx = np.float32(dx)
        self.dy = np.float32(dy)
        self.dt = np.float32(dt)
        self.g = np.float32(g)
        
        #Initialize time
        self.t = np.float32(0.0)
        
        #Compute kernel launch parameters
        self.local_size = (block_width, block_height) 
        self.global_size = ( \
                       int(np.ceil(self.nx / float(self.local_size[0])) * self.local_size[0]), \
                       int(np.ceil(self.ny / float(self.local_size[1])) * self.local_size[1]) \
                      ) 
    
    
    
    
    """
    Function which steps n timesteps
    """
    def step(self, t_end=0.0):
        n = int(t_end / self.dt + 1)
        
        for i in range(0, n):        
            local_dt = np.float32(min(self.dt, t_end-i*self.dt))
            
            if (local_dt <= 0.0):
                break
        
            self.lxf_kernel.swe_2D(self.cl_queue, self.global_size, self.local_size, \
                    self.nx, self.ny, \
                    self.dx, self.dy, local_dt, \
                    self.g, \
                    self.cl_data.h0.data, self.cl_data.h0.pitch, \
                    self.cl_data.hu0.data, self.cl_data.hu0.pitch, \
                    self.cl_data.hv0.data, self.cl_data.hv0.pitch, \
                    self.cl_data.h1.data, self.cl_data.h1.pitch, \
                    self.cl_data.hu1.data, self.cl_data.hu1.pitch, \
                    self.cl_data.hv1.data, self.cl_data.hv1.pitch)
                
            self.t += local_dt
            
            self.cl_data.swap()
        
        return self.t
        
        
    
    
    
    def download(self):
        return self.cl_data.download(self.cl_queue)

