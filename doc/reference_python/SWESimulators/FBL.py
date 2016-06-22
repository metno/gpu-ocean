# -*- coding: utf-8 -*-

"""
This python module implements the Forward Backward Linear numerical 
scheme for the shallow water equations, described in 
L. P. Røed, "Documentation of simple ocean models for use in ensemble
predictions", Met no report 2012/3 and 2012/5 .

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
class FBLDataCL:
    """
    Uploads initial data to the CL device
    """
    def __init__(self, cl_ctx, h0, eta0, u0, v0):
        if (not np.issubdtype(h0.dtype, np.float32) or np.isfortran(h0)):
            print "Converting H0"
            h0 = h0.astype(np.float32, order='C')
            
        if (not np.issubdtype(eta0.dtype, np.float32) or np.isfortran(eta0)):
            print "Converting Eta0"
            eta0 = eta0.astype(np.float32, order='C')
            
        if (not np.issubdtype(u0.dtype, np.float32) or np.isfortran(u0)):
            print "Converting U0"
            u0 = u0.astype(np.float32, order='C')
            
        if (not np.issubdtype(v0.dtype, np.float32) or np.isfortran(v0)):
            print "Converting V0"
            v0 = v0.astype(np.float32, order='C')
        
        ny, nx = h0.shape
        self.nx = nx
        self.ny = ny

        assert(h0.shape == (ny, nx))
        assert(eta0.shape == (ny, nx))
        assert(u0.shape == (ny, nx+1))
        assert(v0.shape == (ny+1, nx))

        #Upload data to the device
        mf = cl.mem_flags
        self.h0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h0)
        self.eta0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=eta0)
        self.u0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u0)
        self.v0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=v0)
        
        self.h0_pitch = np.int32(nx*4)
        self.eta0_pitch = np.int32(nx*4)
        self.u0_pitch = np.int32((nx+1)*4)
        self.v0_pitch = np.int32(nx*4)
        
    """
    Enables downloading data from CL device to Python
    """
    def download(self, cl_queue):
        #Allocate data on the host for result
        eta1 = np.empty((self.ny, self.nx), dtype=np.float32, order='C')
        u1 = np.empty((self.ny, self.nx+1), dtype=np.float32, order='C')
        v1 = np.empty((self.ny+1, self.nx), dtype=np.float32, order='C')
        
        #Copy data from device to host
        cl.enqueue_copy(cl_queue, eta1, self.eta0)
        cl.enqueue_copy(cl_queue, u1, self.u0)
        cl.enqueue_copy(cl_queue, v1, self.v0)
        
        #Return
        return eta1, u1, v1;
        
        
        
        
        
        










"""
Class that solves the SW equations using the Forward-Backward linear scheme
"""
class FBL:

    """
    Initialization routine
    h0: Water depth incl ghost cells, (nx+2)*(ny+2) cells
    eta0: Initial deviation from mean sea level incl ghost cells, (nx+2)*(ny+2) cells
    u0: Initial momentum along x-axis incl ghost cells, (nx+1)*(ny+2) cells
    v0: Initial momentum along y-axis incl ghost cells, (nx+2)*(ny+1) cells
    nx: Number of cells along x-axis
    ny: Number of cells along y-axis
    dx: Grid cell spacing along x-axis (20 000 m)
    dy: Grid cell spacing along y-axis (20 000 m)
    dt: Size of each timestep (90 s)
    g: Gravitational accelleration (9.81 m/s^2)
    f: Coriolis parameter (1.2e-4 s^1)
    r: Bottom friction coefficient (2.4e-3 m/s)
    wind_type: TYpe of wind stress, 0=Uniform along shore, 1=bell shaped along shore, 2=moving cyclone
    wind_tau0: Amplitude of wind stress (Pa)
    wind_rho: Density of sea water (1025.0 kg / m^3)
    wind_alpha: Offshore e-folding length (1/(10*dx) = 5e-6 m^-1)
    wind_xm: Maximum wind stress for bell shaped wind stress
    wind_Rc: Distance to max wind stress from center of cyclone (10dx = 200 000 m)
    wind_x0: Initial x position of moving cyclone (dx*(nx/2) - u0*3600.0*48.0)
    wind_y0: Initial y position of moving cyclone (dy*(ny/2) - v0*3600.0*48.0)
    wind_u0: Translation speed along x for moving cyclone (30.0/sqrt(5.0))
    wind_v0: Translation speed along y for moving cyclone (-0.5*u0)
    """
    def __init__(self, \
                 h0, eta0, u0, v0, \
                 nx, ny, \
                 dx, dy, dt, \
                 g, f, r, \
                 wind_type=99, # "no wind" \
                 wind_tau0=0, wind_rho=0, wind_alpha=0, wind_xm=0, wind_Rc=0, \
                 wind_x0=0, wind_y0=0, \
                 wind_u0=0, wind_v0=0):
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
        self.u_kernel = self.get_kernel("FBL_U_kernel.opencl")
        self.v_kernel = self.get_kernel("FBL_V_kernel.opencl")
        self.eta_kernel = self.get_kernel("FBL_eta_kernel.opencl")
        
        #Create data by uploading to device
        self.cl_data = FBLDataCL(self.cl_ctx, h0, eta0, u0, v0)
        
        #Save input parameters
        #Notice that we need to specify them in the correct dataformat for the
        #OpenCL kernel
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.dx = np.float32(dx)
        self.dy = np.float32(dy)
        self.dt = np.float32(dt)
        self.g = np.float32(g)
        self.f = np.float32(f)
        self.r = np.float32(r)
        self.wind_type = np.int32(wind_type)
        self.wind_tau0 = np.float32(wind_tau0)
        self.wind_rho = np.float32(wind_rho)
        self.wind_alpha = np.float32(wind_alpha)
        self.wind_xm = np.float32(wind_xm)
        self.wind_Rc = np.float32(wind_Rc)
        self.wind_x0 = np.float32(wind_x0)
        self.wind_y0 = np.float32(wind_y0)
        self.wind_u0 = np.float32(wind_u0)
        self.wind_v0 = np.float32(wind_v0)
        
        #Initialize time
        self.t = np.float32(0.0)
        
        #Compute kernel launch parameters
        self.local_size = (8, 8) # WARNING::: MUST MATCH defines of block_width/height in kernels!
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
                
            self.u_kernel.computeUKernel(self.cl_queue, self.global_size, self.local_size, \
                    self.nx, self.ny, \
                    self.dx, self.dy, local_dt, \
                    self.g, self.f, self.r, \
                    self.cl_data.h0, self.cl_data.h0_pitch, \
                    self.cl_data.u0, self.cl_data.u0_pitch, \
                    self.cl_data.v0, self.cl_data.v0_pitch, \
                    self.cl_data.eta0, self.cl_data.eta0_pitch, \
                    self.wind_type, \
                    self.wind_tau0, self.wind_rho, self.wind_alpha, self.wind_xm, self.wind_Rc, \
                    self.wind_x0, self.wind_y0, \
                    self.wind_u0, self.wind_v0, \
                    self.t)
            
                
                
            self.v_kernel.computeVKernel(self.cl_queue, self.global_size, self.local_size, \
                    self.nx, self.ny, \
                    self.dx, self.dy, local_dt, \
                    self.g, self.f, self.r, \
                    self.cl_data.h0, self.cl_data.h0_pitch, \
                    self.cl_data.u0, self.cl_data.u0_pitch, \
                    self.cl_data.v0, self.cl_data.v0_pitch, \
                    self.cl_data.eta0, self.cl_data.eta0_pitch, \
                    self.wind_type, \
                    self.wind_tau0, self.wind_rho, self.wind_alpha, self.wind_xm, self.wind_Rc, \
                    self.wind_x0, self.wind_y0, \
                    self.wind_u0, self.wind_v0, \
                    self.t)
            
            self.eta_kernel.computeEtaKernel(self.cl_queue, self.global_size, self.local_size, \
                    self.nx, self.ny, \
                    self.dx, self.dy, local_dt, \
                    self.g, self.f, self.r, \
                    self.cl_data.h0, self.cl_data.h0_pitch, \
                    self.cl_data.u0, self.cl_data.u0_pitch, \
                    self.cl_data.v0, self.cl_data.v0_pitch, \
                    self.cl_data.eta0, self.cl_data.eta0_pitch)
                
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


        
        







