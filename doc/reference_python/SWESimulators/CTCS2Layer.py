# -*- coding: utf-8 -*-

"""
This python module implements the Centered in Time, Centered in Space
(leapfrog) numerical scheme for the shallow water equations, 
described in 
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
class CTCS2LayerDataCL:
    """
    Uploads initial data to the CL device
    """
    def __init__(self, cl_ctx, h_0, eta_0, u_0, v_0, \
                               h2_0, eta2_0, u2_0, v2_0):
        #Make sure that the data is single precision floating point
        if (not np.issubdtype(h_0.dtype, np.float32) or np.isfortran(h_0)):
            print "Converting H_0"
            h_0 = h_0.astype(np.float32, order='C')
        if (not np.issubdtype(eta_0.dtype, np.float32) or np.isfortran(eta_0)):
            print "Converting Eta_0"
            eta_0 = eta_0.astype(np.float32, order='C')
        if (not np.issubdtype(u_0.dtype, np.float32) or np.isfortran(u_0)):
            print "Converting U_0"
            u_0 = u_0.astype(np.float32, order='C')
        if (not np.issubdtype(v_0.dtype, np.float32) or np.isfortran(v_0)):
            print "Converting V_0"
            v_0 = v_0.astype(np.float32, order='C')
        
        #Same for second (deepest) layer
        if (not np.issubdtype(h2_0.dtype, np.float32) or np.isfortran(h2_0)):
            print "Converting H2_0"
            h2_0 = h2_0.astype(np.float32, order='C')
        if (not np.issubdtype(eta2_0.dtype, np.float32) or np.isfortran(eta2_0)):
            print "Converting Eta2_0"
            eta2_0 = eta2_0.astype(np.float32, order='C')
        if (not np.issubdtype(u2_0.dtype, np.float32) or np.isfortran(u2_0)):
            print "Converting U2_0"
            u2_0 = u2_0.astype(np.float32, order='C')
        if (not np.issubdtype(v2_0.dtype, np.float32) or np.isfortran(v2_0)):
            print "Converting V2_0"
            v2_0 = v2_0.astype(np.float32, order='C')
        
        self.ny, self.nx = h_0.shape
        self.nx = self.nx - 2 # Ghost cells
        self.ny = self.ny - 2

        assert(h_0.shape == (self.ny+2, self.nx+2))
        assert(eta_0.shape == (self.ny+2, self.nx+2))
        assert(u_0.shape == (self.ny+2, self.nx+1))
        assert(v_0.shape == (self.ny+1, self.nx+2))
        
        #Same for layer 2
        assert(h2_0.shape == (self.ny+2, self.nx+2))
        assert(eta2_0.shape == (self.ny+2, self.nx+2))
        assert(u2_0.shape == (self.ny+2, self.nx+1))
        assert(v2_0.shape == (self.ny+1, self.nx+2))

        #Upload data to the device
        mf = cl.mem_flags
        self.h_0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_0)
        
        self.eta_0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=eta_0)
        self.eta_1 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=eta_0)
        
        self.u_0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u_0)
        self.u_1 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u_0)
        
        self.v_0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=v_0)
        self.v_1 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=v_0)
        
        #Same for layer 2
        self.h2_0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_0)
        
        self.eta2_0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=eta2_0)
        self.eta2_1 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=eta2_0)
        
        self.u2_0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u2_0)
        self.u2_1 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u2_0)
        
        self.v2_0 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=v2_0)
        self.v2_1 = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=v2_0)
        
        
        
        
        
        #Compute pitches
        self.h_0_pitch = np.int32(h_0.shape[1]*4)
        
        self.eta_0_pitch = np.int32(eta_0.shape[1]*4)
        self.eta_1_pitch = np.int32(eta_0.shape[1]*4)
        
        self.u_0_pitch = np.int32(u_0.shape[1]*4)
        self.u_1_pitch = np.int32(u_0.shape[1]*4)
        
        self.v_0_pitch = np.int32(v_0.shape[1]*4)
        self.v_1_pitch = np.int32(v_0.shape[1]*4)
        
        #Same for layer 2
        self.h2_0_pitch = np.int32(h2_0.shape[1]*4)
        
        self.eta2_0_pitch = np.int32(eta2_0.shape[1]*4)
        self.eta2_1_pitch = np.int32(eta2_0.shape[1]*4)
        
        self.u2_0_pitch = np.int32(u2_0.shape[1]*4)
        self.u2_1_pitch = np.int32(u2_0.shape[1]*4)
        
        self.v2_0_pitch = np.int32(v2_0.shape[1]*4)
        self.v2_1_pitch = np.int32(v2_0.shape[1]*4)
        
       
    
    """
    Swaps the variables after a timestep has been completed
    """
    def swap(self):
        self.eta_1, self.eta_0 = self.eta_0, self.eta_1
        self.u_1, self.u_0 = self.u_0, self.u_1
        self.v_1, self.v_0 = self.v_0, self.v_1
        
        #Same for layer 2
        self.eta2_1, self.eta2_0 = self.eta2_0, self.eta2_1
        self.u2_1, self.u2_0 = self.u2_0, self.u2_1
        self.v2_1, self.v2_0 = self.v2_0, self.v2_1
        
        
    """
    Enables downloading data from CL device to Python
    """
    def download(self, cl_queue):
        #Allocate data on the host for result
        eta_1 = np.empty((self.ny+2, self.nx+2), dtype=np.float32, order='C')
        u_1 = np.empty((self.ny+2, self.nx+1), dtype=np.float32, order='C')
        v_1 = np.empty((self.ny+1, self.nx+2), dtype=np.float32, order='C')
        
        #Same for layer 2
        eta2_1 = np.empty((self.ny+2, self.nx+2), dtype=np.float32, order='C')
        u2_1 = np.empty((self.ny+2, self.nx+1), dtype=np.float32, order='C')
        v2_1 = np.empty((self.ny+1, self.nx+2), dtype=np.float32, order='C')
        
        #Copy data from device to host
        cl.enqueue_copy(cl_queue, eta_1, self.eta_1)
        cl.enqueue_copy(cl_queue, u_1, self.u_1)
        cl.enqueue_copy(cl_queue, v_1, self.v_1)
        
        #Same for layer 2
        cl.enqueue_copy(cl_queue, eta2_1, self.eta2_1)
        cl.enqueue_copy(cl_queue, u2_1, self.u2_1)
        cl.enqueue_copy(cl_queue, v2_1, self.v2_1)
        
        
        #Return
        return eta_1, u_1, v_1, eta2_1, u2_1, v2_1
        
        
        
        
        
        










"""
Class that solves the SW equations using the Centered in time centered in space scheme
"""
class CTCS2Layer:

    """
    Initialization routine
    h_0: Water depth incl ghost cells, (nx+2)*(ny+2) cells
    eta_0: Initial deviation from mean sea level incl ghost cells, (nx+2)*(ny+2) cells
    u_0: Initial momentum along x-axis incl ghost cells, (nx+1)*(ny+2) cells
    v_0: Initial momentum along y-axis incl ghost cells, (nx+2)*(ny+1) cells
    h2_0: Water depth (layer 2) incl ghost cells, (nx+2)*(ny+2) cells
    eta2_0: Initial deviation from mean sea level (layer 2) incl ghost cells, (nx+2)*(ny+2) cells
    u2_0: Initial momentum (layer 2) along x-axis incl ghost cells, (nx+1)*(ny+2) cells
    v2_0: Initial momentum (layer 2) along y-axis incl ghost cells, (nx+2)*(ny+1) cells
    nx: Number of cells along x-axis
    ny: Number of cells along y-axis
    dx: Grid cell spacing along x-axis (20 000 m)
    dy: Grid cell spacing along y-axis (20 000 m)
    dt: Size of each timestep (90 s)
    g: Gravitational accelleration (9.81 m/s^2)
    f: Coriolis parameter (1.2e-4 s^1)
    r: Bottom friction coefficient (2.4e-3 m/s)
    r2: Inter-layer friction coefficient (m/s)
    A: Eddy viscosity coefficient (O(dx))
    rho: Density of upper layer (1025.0 kg / m^3)
    rho2: Density of lower layer (1025.0 kg / m^3)
    wind_type: Type of wind stress, 0=Uniform along shore, 1=bell shaped along shore, 2=moving cyclone
    wind_tau0: Amplitude of wind stress (Pa)
    wind_alpha: Offshore e-folding length (1/(10*dx) = 5e-6 m^-1)
    wind_xm: Maximum wind stress for bell shaped wind stress
    wind_Rc: Distance to max wind stress from center of cyclone (10dx = 200 000 m)
    wind_x0: Initial x position of moving cyclone (dx*(nx/2) - u0*3600.0*48.0)
    wind_y0: Initial y position of moving cyclone (dy*(ny/2) - v0*3600.0*48.0)
    wind_u0: Translation speed along x for moving cyclone (30.0/sqrt(5.0))
    wind_v0: Translation speed along y for moving cyclone (-0.5*u0)
    """
    def __init__(self, \
                 h_0, eta_0, u_0, v_0, \
                 h2_0, eta2_0, u2_0, v2_0, \
                 nx, ny, \
                 dx, dy, dt, \
                 g, f, r, r2, A, \
                 rho1, rho2,
                 wind_type=99, # "no wind" \
                 wind_tau0=0, wind_alpha=0, wind_xm=0, wind_Rc=0, \
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
        self.u_kernel = self.get_kernel("CTCS_U_kernel.opencl")
        self.v_kernel = self.get_kernel("CTCS_V_kernel.opencl")
        self.eta_kernel = self.get_kernel("CTCS_eta_kernel.opencl")
        
        #Create data by uploading to device
        self.cl_data = CTCSDataCL(self.cl_ctx, h_0, eta_0, u_0, v_0, h2_0, eta2_0, u2_0, v2_0)
        
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
        self.r2 = np.float32(r2)
        self.A = np.float32(A)
        self.rho = np.float32(rho)
        self.rho2 = np.float32(rho2)
        self.wind_type = np.int32(wind_type)
        self.wind_tau0 = np.float32(wind_tau0)
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
            #Notation: 
            # cl_data.u0 => U^{n-1} before U kernel, U^{n+1} after U kernel
            # cl_data.u1 => U^{n}
            # When we call cl_data.swap(), we swap these, so that
            # cl_data.u0 => U^{n}
            # cl_data.u1 => U^{n+1} (U kernel has been executed)
            # Now we are ready for the next time step
            
            local_dt = np.float32(min(self.dt, t_end-i*self.dt))
            
            if (local_dt <= 0.0):
                break
            
            self.eta_kernel.computeEtaKernel(self.cl_queue, self.global_size, self.local_size, \
                    self.nx, self.ny, \
                    self.dx, self.dy, local_dt, \
                    self.g, self.f, self.r, \
                    self.cl_data.eta0, self.cl_data.eta0_pitch,   # eta^{n-1} => eta^{n+1} \
                    self.cl_data.u1, self.cl_data.u1_pitch,       # U^{n} \
                    self.cl_data.v1, self.cl_data.v1_pitch)       # V^{n}
            
            self.u_kernel.computeUKernel(self.cl_queue, self.global_size, self.local_size, \
                    self.nx, self.ny, \
                    self.dx, self.dy, local_dt, \
                    self.g, self.f, self.r, self.A,\
                    self.cl_data.h0, self.cl_data.h0_pitch, \
                    self.cl_data.eta1, self.cl_data.eta1_pitch,   # eta^{n} \
                    self.cl_data.u0, self.cl_data.u0_pitch,       # U^{n-1} => U^{n+1} \
                    self.cl_data.u1, self.cl_data.u1_pitch,       # U^{n} \
                    self.cl_data.v1, self.cl_data.v1_pitch,       # V^{n} \
                    self.wind_type, \
                    self.wind_tau0, self.wind_rho, self.wind_alpha, self.wind_xm, self.wind_Rc, \
                    self.wind_x0, self.wind_y0, \
                    self.wind_u0, self.wind_v0, \
                    self.t)
            
            self.v_kernel.computeVKernel(self.cl_queue, self.global_size, self.local_size, \
                    self.nx, self.ny, \
                    self.dx, self.dy, local_dt, \
                    self.g, self.f, self.r, self.A,\
                    self.cl_data.h0, self.cl_data.h0_pitch, \
                    self.cl_data.eta1, self.cl_data.eta1_pitch,   # eta^{n} \
                    self.cl_data.u1, self.cl_data.u1_pitch,       # U^{n} \
                    self.cl_data.v0, self.cl_data.v0_pitch,       # V^{n-1} => V^{n+1} \
                    self.cl_data.v1, self.cl_data.v1_pitch,       # V^{n} \
                    self.wind_type, \
                    self.wind_tau0, self.wind_rho, self.wind_alpha, self.wind_xm, self.wind_Rc, \
                    self.wind_x0, self.wind_y0, \
                    self.wind_u0, self.wind_v0, \
                    self.t)
            
            #After the kernels, swap the data pointers
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


        







