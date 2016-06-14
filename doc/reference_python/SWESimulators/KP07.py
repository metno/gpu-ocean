# -*- coding: utf-8 -*-

"""
This python module implements the Kurganov-Petrova numerical scheme 
for the shallow water equations, described in 
A. Kurganov & Guergana Petrova
A Second-Order Well-Balanced Positivity Preserving Central-Upwind
Scheme for the Saint-Venant System Communications in Mathematical
Sciences, 5 (2007), 133-160. 

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
import os
import time
import numpy as np
import pyopencl as cl #OpenCL in Python







"""
Class that holds data for the SW equations in OpenCL
"""
class KP07DataCL:
    """
    Uploads initial data to the CL device
    """
    def __init__(self, cl_ctx, h0, u0, v0):
        if (not np.issubdtype(h0.dtype, np.float32) or np.isfortran(h0)):
            print "Converting H0"
            h0 = h0.astype(np.float32, order='C')
            
        if (not np.issubdtype(u0.dtype, np.float32) or np.isfortran(u0)):
            print "Converting U0"
            u0 = u0.astype(np.float32, order='C')
            
        if (not np.issubdtype(v0.dtype, np.float32) or np.isfortran(v0)):
            print "Converting V0"
            v0 = v0.astype(np.float32, order='C')
        
        self.ny, self.nx = h0.shape
        self.nx -= 4
        self.ny -= 4

        assert(h0.shape == (self.ny+4, self.nx+4))
        assert(u0.shape == (self.ny+4, self.nx+4))
        assert(v0.shape == (self.ny+4, self.nx+4))

        #Upload data to the device
        mf = cl.mem_flags
        self.h0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h0)
        self.u0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u0)
        self.v0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=v0)
        
        self.h0_pitch = np.int32((self.nx+4)*4)
        self.u0_pitch = np.int32((self.nx+4)*4)
        self.v0_pitch = np.int32((self.nx+4)*4)
        
        self.h1 = cl.Buffer(cl_ctx, mf.READ_WRITE, h0.nbytes)
        self.u1 = cl.Buffer(cl_ctx, mf.READ_WRITE, h0.nbytes)
        self.v1 = cl.Buffer(cl_ctx, mf.READ_WRITE, h0.nbytes)
        
        self.h1_pitch = np.int32((self.nx+4)*4)
        self.u1_pitch = np.int32((self.nx+4)*4)
        self.v1_pitch = np.int32((self.nx+4)*4)
        
    """
    Swaps the variables after a timestep has been completed
    """
    def swap(self):
        self.h1, self.h0 = self.h0, self.h1
        self.u1, self.u0 = self.u0, self.u1
        self.v1, self.v0 = self.v0, self.v1
        
    """
    Enables downloading data from CL device to Python
    """
    def download(self, cl_queue):
        #Allocate data on the host for result
        h1 = np.empty((self.ny+4, self.nx+4), dtype=np.float32, order='C')
        u1 = np.empty((self.ny+4, self.nx+4), dtype=np.float32, order='C')
        v1 = np.empty((self.ny+4, self.nx+4), dtype=np.float32, order='C')
        
        #Copy data from device to host
        cl.enqueue_copy(cl_queue, h1, self.h0)
        cl.enqueue_copy(cl_queue, u1, self.u0)
        cl.enqueue_copy(cl_queue, v1, self.v0)
        
        #Return
        return h1, u1, v1;
        
        
        
        
        
        


"""
Class that solves the SW equations using the Forward-Backward linear scheme
"""
class KP07:

    """
    Initialization routine
    h0: Water depth incl ghost cells, (nx+1)*(ny+1) cells
    u0: Initial momentum along x-axis incl ghost cells, (nx+1)*(ny+1) cells
    v0: Initial momentum along y-axis incl ghost cells, (nx+1)*(ny+1) cells
    nx: Number of cells along x-axis
    ny: Number of cells along y-axis
    dx: Grid cell spacing along x-axis (20 000 m)
    dy: Grid cell spacing along y-axis (20 000 m)
    dt: Size of each timestep (90 s)
    g: Gravitational accelleration (9.81 m/s^2)
    """
    def __init__(self, \
                 h0, u0, v0, \
                 nx, ny, \
                 dx, dy, dt, \
                 g, \
                 theta=1.3, use_rk2=True):
        #Make sure we get compiler output from OpenCL
        os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

        #Set which CL device to use
        os.environ["PYOPENCL_CTX"] = "1"

        #Create OpenCL context
        self.cl_ctx = cl.create_some_context()
        print "Using ", self.cl_ctx.devices[0].name

        #Create an OpenCL command queue
        self.cl_queue = cl.CommandQueue(self.cl_ctx)

        #Get kernels
        self.kp07_kernel = self.get_kernel("KP07_kernel.opencl")
        
        #Create data by uploading to device
        self.cl_data = KP07DataCL(self.cl_ctx, h0, u0, v0)
        
        #Save input parameters
        #Notice that we need to specify them in the correct dataformat for the
        #OpenCL kernel
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.dx = np.float32(dx)
        self.dy = np.float32(dy)
        self.dt = np.float32(dt)
        self.g = np.float32(g)
        self.theta = np.float32(theta)
        self.use_rk2 = use_rk2
        
        #Initialize time
        self.t = np.float32(0.0)
        
        #Compute kernel launch parameters
        self.local_size = (8, 8) 
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
        
            if (self.use_rk2):
                self.kp07_kernel.swe_2D(self.cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        self.dx, self.dy, local_dt, \
                        self.g, \
                        self.theta, \
                        np.int32(0), \
                        self.cl_data.h0, self.cl_data.h0_pitch, \
                        self.cl_data.u0, self.cl_data.u0_pitch, \
                        self.cl_data.v0, self.cl_data.v0_pitch, \
                        self.cl_data.h1, self.cl_data.h1_pitch, \
                        self.cl_data.u1, self.cl_data.u1_pitch, \
                        self.cl_data.v1, self.cl_data.v1_pitch)
                self.kp07_kernel.swe_2D(self.cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        self.dx, self.dy, local_dt, \
                        self.g, \
                        self.theta, \
                        np.int32(1), \
                        self.cl_data.h1, self.cl_data.h1_pitch, \
                        self.cl_data.u1, self.cl_data.u1_pitch, \
                        self.cl_data.v1, self.cl_data.v1_pitch, \
                        self.cl_data.h0, self.cl_data.h0_pitch, \
                        self.cl_data.u0, self.cl_data.u0_pitch, \
                        self.cl_data.v0, self.cl_data.v0_pitch)
            else:
                self.kp07_kernel.swe_2D(self.cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        self.dx, self.dy, local_dt, \
                        self.g, \
                        self.theta, \
                        np.int32(0), \
                        self.cl_data.h0, self.cl_data.h0_pitch, \
                        self.cl_data.u0, self.cl_data.u0_pitch, \
                        self.cl_data.v0, self.cl_data.v0_pitch, \
                        self.cl_data.h1, self.cl_data.h1_pitch, \
                        self.cl_data.u1, self.cl_data.u1_pitch, \
                        self.cl_data.v1, self.cl_data.v1_pitch)
                self.cl_data.swap()
                
            self.t += local_dt
            
        
        return self.t
    
    """
    Static function which reads a text file and creates an OpenCL kernel from that
    """
    def get_kernel(self, kernel_filename):
        #Read the proper program
        module_path = os.path.dirname(os.path.realpath(__file__))
        fullpath = os.path.join(module_path, kernel_filename)
        with open(fullpath, "r") as kernel_file:
            kernel_string = kernel_file.read()
            kernel = cl.Program(self.cl_ctx, kernel_string).build()
            
        return kernel
    
    
    
    def download(self):
        return self.cl_data.download(self.cl_queue)

