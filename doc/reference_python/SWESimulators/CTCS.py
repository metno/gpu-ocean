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
import numpy as np
import pyopencl as cl #OpenCL in Python
import Common









"""
Class that solves the SW equations using the Centered in time centered in space scheme
"""
class CTCS:

    """
    Initialization routine
    H: Water depth incl ghost cells, (nx+2)*(ny+2) cells
    eta0: Initial deviation from mean sea level incl ghost cells, (nx+2)*(ny+2) cells
    hu0: Initial momentum along x-axis incl ghost cells, (nx+1)*(ny+2) cells
    hv0: Initial momentum along y-axis incl ghost cells, (nx+2)*(ny+1) cells
    nx: Number of cells along x-axis
    ny: Number of cells along y-axis
    dx: Grid cell spacing along x-axis (20 000 m)
    dy: Grid cell spacing along y-axis (20 000 m)
    dt: Size of each timestep (90 s)
    g: Gravitational accelleration (9.81 m/s^2)
    f: Coriolis parameter (1.2e-4 s^1)
    r: Bottom friction coefficient (2.4e-3 m/s)
    A: Eddy viscosity coefficient (O(dx))
    wind_stress: Wind stress parameters
    """
    def __init__(self, \
                 cl_ctx, \
                 H, eta0, hu0, hv0, \
                 nx, ny, \
                 dx, dy, dt, \
                 g, f, r, A, \
                 wind_stress=Common.WindStressParams(), \
                 block_width=16, block_height=16):
        self.cl_ctx = cl_ctx

        #Create an OpenCL command queue
        self.cl_queue = cl.CommandQueue(self.cl_ctx)

        reload(Common)
        #Get kernels
        self.u_kernel = Common.get_kernel(self.cl_ctx, "CTCS_U_kernel.opencl", block_width, block_height)
        self.v_kernel = Common.get_kernel(self.cl_ctx, "CTCS_V_kernel.opencl", block_width, block_height)
        self.eta_kernel = Common.get_kernel(self.cl_ctx, "CTCS_eta_kernel.opencl", block_width, block_height)
        
        #Create data by uploading to device
        ghost_cells_x = 1
        ghost_cells_y = 1
        self.H = Common.OpenCLArray2D(self.cl_ctx, nx, ny, ghost_cells_x, ghost_cells_y, H)
        self.cl_data = Common.SWEDataArakawaC(self.cl_ctx, nx, ny, ghost_cells_x, ghost_cells_y, eta0, hu0, hv0)
        
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
        self.A = np.float32(A)
        self.wind_stress = wind_stress
        
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
                    self.cl_data.h0.data, self.cl_data.h0.pitch,     # eta^{n-1} => eta^{n+1} \
                    self.cl_data.hu1.data, self.cl_data.hu1.pitch,   # U^{n} \
                    self.cl_data.hv1.data, self.cl_data.hv1.pitch)   # V^{n}
            
            self.u_kernel.computeUKernel(self.cl_queue, self.global_size, self.local_size, \
                    self.nx, self.ny, \
                    self.dx, self.dy, local_dt, \
                    self.g, self.f, self.r, self.A,\
                    self.H.data, self.H.pitch, \
                    self.cl_data.h1.data, self.cl_data.h1.pitch,      # eta^{n} \
                    self.cl_data.hu0.data, self.cl_data.hu0.pitch,    # U^{n-1} => U^{n+1} \
                    self.cl_data.hu1.data, self.cl_data.hu1.pitch,    # U^{n} \
                    self.cl_data.hv1.data, self.cl_data.hv1.pitch,    # V^{n} \
                    self.wind_stress.type, \
                    self.wind_stress.tau0, self.wind_stress.rho, self.wind_stress.alpha, self.wind_stress.xm, self.wind_stress.Rc, \
                    self.wind_stress.x0, self.wind_stress.y0, \
                    self.wind_stress.u0, self.wind_stress.v0, \
                    self.t)
            
            self.v_kernel.computeVKernel(self.cl_queue, self.global_size, self.local_size, \
                    self.nx, self.ny, \
                    self.dx, self.dy, local_dt, \
                    self.g, self.f, self.r, self.A,\
                    self.H.data, self.H.pitch, \
                    self.cl_data.h1.data, self.cl_data.h1.pitch,     # eta^{n} \
                    self.cl_data.hu1.data, self.cl_data.hu1.pitch,   # U^{n} \
                    self.cl_data.hv0.data, self.cl_data.hv0.pitch,   # V^{n-1} => V^{n+1} \
                    self.cl_data.hv1.data, self.cl_data.hv1.pitch,   # V^{n} \
                    self.wind_stress.type, \
                    self.wind_stress.tau0, self.wind_stress.rho, self.wind_stress.alpha, self.wind_stress.xm, self.wind_stress.Rc, \
                    self.wind_stress.x0, self.wind_stress.y0, \
                    self.wind_stress.u0, self.wind_stress.v0, \
                    self.t)
            
            #After the kernels, swap the data pointers
            self.cl_data.swap()
            
            self.t += local_dt
        
        return self.t
    
    
    
    
    def download(self):
        return self.cl_data.download(self.cl_queue)


        







