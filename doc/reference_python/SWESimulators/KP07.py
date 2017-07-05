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
import numpy as np
import pyopencl as cl #OpenCL in Python
import Common





"""
Class that solves the SW equations using the Forward-Backward linear scheme
"""
class KP07:

    """
    Initialization routine
    w0: Water elevation incl ghost cells, (nx+4)*(ny+4) cells
    Bi: Bottom elevation defined on cell corners, (nx+5)*(ny+5) corners
    hu0: Initial momentum along x-axis incl ghost cells, (nx+4)*(ny+4) cells
    hv0: Initial momentum along y-axis incl ghost cells, (nx+4)*(ny+4) cells
    nx: Number of cells along x-axis
    ny: Number of cells along y-axis
    dx: Grid cell spacing along x-axis (20 000 m)
    dy: Grid cell spacing along y-axis (20 000 m)
    dt: Size of each timestep (90 s)
    g: Gravitational accelleration (9.81 m/s^2)
    f: Coriolis parameter (1.2e-4 s^1)
    r: Bottom friction coefficient (2.4e-3 m/s)
    theta: MINMOD theta used the reconstructions of the derivatives in the numerical scheme
    wind_type: Type of wind stress, 0=Uniform along shore, 1=bell shaped along shore, 2=moving cyclone
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
                 cl_ctx, \
                 w0, Bi, hu0, hv0, \
                 nx, ny, \
                 dx, dy, dt, \
                 g, f=0.0, r=0.0, \
                 theta=1.3, use_rk2=True,
                 wind_stress=Common.WindStressParams(), \
                 boundary_conditions=Common.BoundaryConditions(), \
                 block_width=16, block_height=16):
        self.cl_ctx = cl_ctx
            
        #Create an OpenCL command queue
        self.cl_queue = cl.CommandQueue(self.cl_ctx)

        #Get kernels
        self.kp07_kernel = Common.get_kernel(self.cl_ctx, "KP07_kernel.opencl", block_width, block_height)
        
        #Create data by uploading to device
        ghost_cells_x = 2
        ghost_cells_y = 2
        self.cl_data = Common.SWEDataArakawaA(self.cl_ctx, nx, ny, ghost_cells_x, ghost_cells_y, w0, hu0, hv0)
        
        self.bathymetry = Common.Bathymetry(self.cl_ctx, self.cl_queue, nx, ny, ghost_cells_x, ghost_cells_y, Bi)
        
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
        self.theta = np.float32(theta)
        self.use_rk2 = use_rk2
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
            local_dt = np.float32(min(self.dt, t_end-i*self.dt))
            
            if (local_dt <= 0.0):
                break
        
            if (self.use_rk2):
                self.kp07_kernel.swe_2D(self.cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        self.dx, self.dy, local_dt, \
                        self.g, \
                        self.theta, \
                        self.f, \
                        self.r, \
                        np.int32(0), \
                        self.cl_data.h0.data,  self.cl_data.h0.pitch,  \
                        self.cl_data.hu0.data, self.cl_data.hu0.pitch, \
                        self.cl_data.hv0.data, self.cl_data.hv0.pitch, \
                        self.cl_data.h1.data,  self.cl_data.h1.pitch,  \
                        self.cl_data.hu1.data, self.cl_data.hu1.pitch, \
                        self.cl_data.hv1.data, self.cl_data.hv1.pitch, \
                        self.bathymetry.Bi.data, self.bathymetry.Bi.pitch, \
                        self.bathymetry.Bm.data, self.bathymetry.Bm.pitch, \
                        self.wind_stress.type, \
                        self.wind_stress.tau0, self.wind_stress.rho, self.wind_stress.alpha, self.wind_stress.xm, self.wind_stress.Rc, \
                        self.wind_stress.x0, self.wind_stress.y0, \
                        self.wind_stress.u0, self.wind_stress.v0, \
                        self.t)
                self.kp07_kernel.swe_2D(self.cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        self.dx, self.dy, local_dt, \
                        self.g, \
                        self.theta, \
                        self.f, \
                        self.r, \
                        np.int32(1), \
                        self.cl_data.h1.data,  self.cl_data.h1.pitch,  \
                        self.cl_data.hu1.data, self.cl_data.hu1.pitch, \
                        self.cl_data.hv1.data, self.cl_data.hv1.pitch, \
                        self.cl_data.h0.data,  self.cl_data.h0.pitch,  \
                        self.cl_data.hu0.data, self.cl_data.hu0.pitch, \
                        self.cl_data.hv0.data, self.cl_data.hv0.pitch, \
                        self.bathymetry.Bi.data, self.bathymetry.Bi.pitch, \
                        self.bathymetry.Bm.data, self.bathymetry.Bm.pitch, \
                        self.wind_stress.type, \
                        self.wind_stress.tau0, self.wind_stress.rho, self.wind_stress.alpha, self.wind_stress.xm, self.wind_stress.Rc, \
                        self.wind_stress.x0, self.wind_stress.y0, \
                        self.wind_stress.u0, self.wind_stress.v0, \
                        self.t)
            else:
                self.kp07_kernel.swe_2D(self.cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        self.dx, self.dy, local_dt, \
                        self.g, \
                        self.theta, \
                        self.f, \
                        self.r, \
                        np.int32(0), \
                        self.cl_data.h0.data,  self.cl_data.h0.pitch,  \
                        self.cl_data.hu0.data, self.cl_data.hu0.pitch, \
                        self.cl_data.hv0.data, self.cl_data.hv0.pitch, \
                        self.cl_data.h1.data,  self.cl_data.h1.pitch,  \
                        self.cl_data.hu1.data, self.cl_data.hu1.pitch, \
                        self.cl_data.hv1.data, self.cl_data.hv1.pitch, \
                        self.bathymetry.Bi.data, self.bathymetry.Bi.pitch, \
                        self.bathymetry.Bm.data, self.bathymetry.Bm.pitch, \
                        self.wind_stress.type, \
                        self.wind_stress.tau0, self.wind_stress.rho, self.wind_stress.alpha, self.wind_stress.xm, self.wind_stress.Rc, \
                        self.wind_stress.x0, self.wind_stress.y0, \
                        self.wind_stress.u0, self.wind_stress.v0, \
                        self.t)
                self.cl_data.swap()
                
            self.t += local_dt
            
        
        return self.t
    
    
    
    
    def download(self):
        return self.cl_data.download(self.cl_queue)
    
    def downloadBathymetry(self):
        return self.bathymetry.download(self.cl_queue)

