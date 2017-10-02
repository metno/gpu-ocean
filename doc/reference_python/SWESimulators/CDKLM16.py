# -*- coding: utf-8 -*-

"""
This python module implements 
Alina Chertock, Michael Dudzinski, A. Kurganov & Maria Lukacova-Medvidova (2016)
Well-Balanced Schemes for the Shallow Water Equations with Coriolis Forces

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





        
        
        


"""
Class that solves the SW equations using the Coriolis well balanced reconstruction scheme, as given by the publication of Chertock, Dudzinski, Kurganov and Lukacova-Medvidova (CDFLM) in 2016.
"""
class CDKLM16:

    """
    Initialization routine
    h0: Water depth incl ghost cells, (nx+6)*(ny+6) cells
    u0: Initial momentum along x-axis incl ghost cells, (nx+6)*(ny+6) cells
    v0: Initial momentum along y-axis incl ghost cells, (nx+6)*(ny+6) cells
    Bi: Bottom topography defined on cell corners, (nx+7)*(ny+7) corners
    nx: Number of cells along x-axis
    ny: Number of cells along y-axis
    extra_ghosts_x: Number of extra ghost cells along x-axis
    extra_ghosts_y: Number of extra ghost cells along y-axis
    dx: Grid cell spacing along x-axis (20 000 m)
    dy: Grid cell spacing along y-axis (20 000 m)
    dt: Size of each timestep (90 s)
    g: Gravitational accelleration (9.81 m/s^2)
    f: Coriolis parameter (1.2e-4 s^1)
    r: Bottom friction coefficient (2.4e-3 m/s)
    """
    def __init__(self, \
                 cl_ctx, \
                 h0, hu0, hv0, \
                 Bi, \
                 nx, ny, \
                 dx, dy, dt, \
                 g, f, r, \
                 theta=1.3, rk_order=2, \
                 wind_stress=Common.WindStressParams(), \
                 boundary_conditions=Common.BoundaryConditions(), \
                 h0AsWaterElevation=True, \
                 reportGeostrophicEquilibrium=False, \
                 write_netcdf=False, \
                 block_width=16, block_height=16):
        self.cl_ctx = cl_ctx

        #Create an OpenCL command queue
        self.cl_queue = cl.CommandQueue(self.cl_ctx)

        #Get kernels
        self.kernel = Common.get_kernel(self.cl_ctx, "CDKLM16_kernel.opencl", block_width, block_height)
        
        #Create data by uploading to device
        self.ghost_cells_x = 3
        self.ghost_cells_y = 3
        ghost_cells_x = 3
        ghost_cells_y = 3
        self.cl_data = Common.SWEDataArakawaA(self.cl_ctx, nx, ny, ghost_cells_x, ghost_cells_y, h0, hu0, hv0)

        

        ## Allocating memory for geostrophical equilibrium variables
        self.reportGeostrophicEquilibrium = np.int32(reportGeostrophicEquilibrium)
        dummy_zero_array = np.zeros((ny+6, nx+6), dtype=np.float32, order='C') 
        self.geoEq_uxpvy = Common.OpenCLArray2D(cl_ctx, nx, ny, ghost_cells_x, ghost_cells_y, dummy_zero_array)
        self.geoEq_Kx = Common.OpenCLArray2D(cl_ctx, nx, ny, ghost_cells_x, ghost_cells_y, dummy_zero_array)
        self.geoEq_Ly = Common.OpenCLArray2D(cl_ctx, nx, ny, ghost_cells_x, ghost_cells_y, dummy_zero_array)

        #Bathymetry
        self.bathymetry = Common.Bathymetry(self.cl_ctx, self.cl_queue, nx, ny, ghost_cells_x, ghost_cells_y, Bi, boundary_conditions)

        assert( rk_order < 4 or rk_order > 0 ), "Only 1st, 2nd and 3rd order Runge Kutta supported"

        if (rk_order == 3):
            assert(r == 0.0), "3rd order Runge Kutta supported only without friction"
        
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
        self.rk_order = np.int32(rk_order)
        self.wind_stress = wind_stress
        self.h0AsWaterElevation = h0AsWaterElevation

        self.boundary_conditions = boundary_conditions
        self.boundaryType = np.int32(1)
        if (boundary_conditions.north == 2 and boundary_conditions.east == 2):
            self.boundaryType = np.int32(2)
        elif (boundary_conditions.north == 2):
            self.boundaryType = np.int32(3)
        elif (boundary_conditions.east == 2):
            self.boundaryType = np.int32(4)
        
        #Initialize time
        self.t = np.float32(0.0)
        
        #Compute kernel launch parameters
        self.local_size = (block_width, block_height) 
        self.global_size = ( \
                       int(np.ceil(self.nx / float(self.local_size[0])) * self.local_size[0]), \
                       int(np.ceil(self.ny / float(self.local_size[1])) * self.local_size[1]) \
                      ) 
    
        self.bc_kernel = Common.BoundaryConditionsArakawaA(self.cl_ctx, \
                                                           self.nx, \
                                                           self.ny, \
                                                           ghost_cells_x, \
                                                           ghost_cells_y, \
                                                           self.boundary_conditions, \
        )

        if self.h0AsWaterElevation:
            self.bathymetry.waterElevationToDepth(self.cl_data.h0)

        self.write_netcdf = write_netcdf
        self.sim_writer = None
        if self.write_netcdf:
            self.sim_writer = SimWriter.SimNetCDFWriter(self)
            

    """
    Clean up function
    """
    def cleanUp(self):
        if self.write_netcdf:
            self.sim_writer.__exit__(0,0,0)
            self.write_netcdf = False
        self.cl_data.release()
        self.geoEq_uxpvy.release()
        self.geoEq_Kx.release()
        self.geoEq_Ly.release()
        self.bathymetry.release()
        self.h0AsWaterElevation = False # Quick fix to stop waterDepthToElevation conversion
    
    """
    Function which steps n timesteps
    """
    def step(self, t_end=0.0):
        n = int(t_end / self.dt + 1)

        self.bc_kernel.boundaryCondition(self.cl_queue, \
                self.cl_data.h0, self.cl_data.hu0, self.cl_data.hv0)
        
        for i in range(0, n):        
            local_dt = np.float32(min(self.dt, t_end-i*self.dt))
            
            if (local_dt <= 0.0):
                break

            #self.bc_kernel.boundaryCondition(self.cl_queue, \
            #            self.cl_data.h1, self.cl_data.hu1, self.cl_data.hv1)
            
            # 2nd order Runge Kutta
            if (self.rk_order == 2):
                self.kernel.swe_2D(self.cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        self.dx, self.dy, local_dt, \
                        self.g, \
                        self.theta, \
                        self.f, \
                        self.r, \
                        self.rk_order, \
                        np.int32(0), \
                        self.cl_data.h0.data, self.cl_data.h0.pitch, \
                        self.cl_data.hu0.data, self.cl_data.hu0.pitch, \
                        self.cl_data.hv0.data, self.cl_data.hv0.pitch, \
                        self.cl_data.h1.data, self.cl_data.h1.pitch, \
                        self.cl_data.hu1.data, self.cl_data.hu1.pitch, \
                        self.cl_data.hv1.data, self.cl_data.hv1.pitch, \
                        self.bathymetry.Bi.data, self.bathymetry.Bi.pitch, \
                        self.bathymetry.Bm.data, self.bathymetry.Bm.pitch, \
                        self.wind_stress.type, \
                        self.wind_stress.tau0, self.wind_stress.rho, self.wind_stress.alpha, self.wind_stress.xm, self.wind_stress.Rc, \
                        self.wind_stress.x0, self.wind_stress.y0, \
                        self.wind_stress.u0, self.wind_stress.v0, \
                        self.t, \
                        self.boundaryType, \
                        np.int32(0), \
                        self.geoEq_uxpvy.data, self.geoEq_uxpvy.pitch, \
                        self.geoEq_Kx.data, self.geoEq_Kx.pitch, \
                        self.geoEq_Ly.data, self.geoEq_Ly.pitch )


                self.bc_kernel.boundaryCondition(self.cl_queue, \
                        self.cl_data.h1, self.cl_data.hu1, self.cl_data.hv1)
                
                self.kernel.swe_2D(self.cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        self.dx, self.dy, local_dt, \
                        self.g, \
                        self.theta, \
                        self.f, \
                        self.r, \
                        self.rk_order, \
                        np.int32(1), \
                        self.cl_data.h1.data, self.cl_data.h1.pitch, \
                        self.cl_data.hu1.data, self.cl_data.hu1.pitch, \
                        self.cl_data.hv1.data, self.cl_data.hv1.pitch, \
                        self.cl_data.h0.data, self.cl_data.h0.pitch, \
                        self.cl_data.hu0.data, self.cl_data.hu0.pitch, \
                        self.cl_data.hv0.data, self.cl_data.hv0.pitch, \
                        self.bathymetry.Bi.data, self.bathymetry.Bi.pitch, \
                        self.bathymetry.Bm.data, self.bathymetry.Bm.pitch, \
                        self.wind_stress.type, \
                        self.wind_stress.tau0, self.wind_stress.rho, self.wind_stress.alpha, self.wind_stress.xm, self.wind_stress.Rc, \
                        self.wind_stress.x0, self.wind_stress.y0, \
                        self.wind_stress.u0, self.wind_stress.v0, \
                        self.t, \
                        self.boundaryType, \
                        self.reportGeostrophicEquilibrium, \
                        self.geoEq_uxpvy.data, self.geoEq_uxpvy.pitch, \
                        self.geoEq_Kx.data, self.geoEq_Kx.pitch, \
                        self.geoEq_Ly.data, self.geoEq_Ly.pitch )

                self.bc_kernel.boundaryCondition(self.cl_queue, \
                        self.cl_data.h0, self.cl_data.hu0, self.cl_data.hv0)
                
            elif (self.rk_order == 1):
                self.kernel.swe_2D(self.cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        self.dx, self.dy, local_dt, \
                        self.g, \
                        self.theta, \
                        self.f, \
                        self.r, \
                        self.rk_order, \
                        np.int32(0), \
                        self.cl_data.h0.data, self.cl_data.h0.pitch, \
                        self.cl_data.hu0.data, self.cl_data.hu0.pitch, \
                        self.cl_data.hv0.data, self.cl_data.hv0.pitch, \
                        self.cl_data.h1.data, self.cl_data.h1.pitch, \
                        self.cl_data.hu1.data, self.cl_data.hu1.pitch, \
                        self.cl_data.hv1.data, self.cl_data.hv1.pitch, \
                        self.bathymetry.Bi.data, self.bathymetry.Bi.pitch, \
                        self.bathymetry.Bm.data, self.bathymetry.Bm.pitch, \
                        self.wind_stress.type, \
                        self.wind_stress.tau0, self.wind_stress.rho, self.wind_stress.alpha, self.wind_stress.xm, self.wind_stress.Rc, \
                        self.wind_stress.x0, self.wind_stress.y0, \
                        self.wind_stress.u0, self.wind_stress.v0, \
                        self.t, \
                        self.boundaryType, \
                        self.reportGeostrophicEquilibrium, \
                        self.geoEq_uxpvy.data, self.geoEq_uxpvy.pitch, \
                        self.geoEq_Kx.data, self.geoEq_Kx.pitch, \
                        self.geoEq_Ly.data, self.geoEq_Ly.pitch )
                
                self.cl_data.swap()

                self.bc_kernel.boundaryCondition(self.cl_queue, \
                        self.cl_data.h0, self.cl_data.hu0, self.cl_data.hv0)

            # 3rd order RK method:
            elif (self.rk_order == 3):
                self.kernel.swe_2D(self.cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        self.dx, self.dy, local_dt, \
                        self.g, \
                        self.theta, \
                        self.f, \
                        self.r, \
                        self.rk_order, \
                        np.int32(0), \
                        self.cl_data.h0.data, self.cl_data.h0.pitch, \
                        self.cl_data.hu0.data, self.cl_data.hu0.pitch, \
                        self.cl_data.hv0.data, self.cl_data.hv0.pitch, \
                        self.cl_data.h1.data, self.cl_data.h1.pitch, \
                        self.cl_data.hu1.data, self.cl_data.hu1.pitch, \
                        self.cl_data.hv1.data, self.cl_data.hv1.pitch, \
                        self.bathymetry.Bi.data, self.bathymetry.Bi.pitch, \
                        self.bathymetry.Bm.data, self.bathymetry.Bm.pitch, \
                        self.wind_stress.type, \
                        self.wind_stress.tau0, self.wind_stress.rho, self.wind_stress.alpha, self.wind_stress.xm, self.wind_stress.Rc, \
                        self.wind_stress.x0, self.wind_stress.y0, \
                        self.wind_stress.u0, self.wind_stress.v0, \
                        self.t, \
                        self.boundaryType, \
                        np.int32(0), \
                        self.geoEq_uxpvy.data, self.geoEq_uxpvy.pitch, \
                        self.geoEq_Kx.data, self.geoEq_Kx.pitch, \
                        self.geoEq_Ly.data, self.geoEq_Ly.pitch )


                self.bc_kernel.boundaryCondition(self.cl_queue, \
                        self.cl_data.h1, self.cl_data.hu1, self.cl_data.hv1)

                self.kernel.swe_2D(self.cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        self.dx, self.dy, local_dt, \
                        self.g, \
                        self.theta, \
                        self.f, \
                        self.r, \
                        self.rk_order, \
                        np.int32(1), \
                        self.cl_data.h1.data, self.cl_data.h1.pitch, \
                        self.cl_data.hu1.data, self.cl_data.hu1.pitch, \
                        self.cl_data.hv1.data, self.cl_data.hv1.pitch, \
                        self.cl_data.h0.data, self.cl_data.h0.pitch, \
                        self.cl_data.hu0.data, self.cl_data.hu0.pitch, \
                        self.cl_data.hv0.data, self.cl_data.hv0.pitch, \
                        self.bathymetry.Bi.data, self.bathymetry.Bi.pitch, \
                        self.bathymetry.Bm.data, self.bathymetry.Bm.pitch, \
                        self.wind_stress.type, \
                        self.wind_stress.tau0, self.wind_stress.rho, self.wind_stress.alpha, self.wind_stress.xm, self.wind_stress.Rc, \
                        self.wind_stress.x0, self.wind_stress.y0, \
                        self.wind_stress.u0, self.wind_stress.v0, \
                        self.t, \
                        self.boundaryType, \
                        self.reportGeostrophicEquilibrium, \
                        self.geoEq_uxpvy.data, self.geoEq_uxpvy.pitch, \
                        self.geoEq_Kx.data, self.geoEq_Kx.pitch, \
                        self.geoEq_Ly.data, self.geoEq_Ly.pitch )

                self.bc_kernel.boundaryCondition(self.cl_queue, \
                        self.cl_data.h1, self.cl_data.hu1, self.cl_data.hv1)

                self.kernel.swe_2D(self.cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        self.dx, self.dy, local_dt, \
                        self.g, \
                        self.theta, \
                        self.f, \
                        self.r, \
                        self.rk_order, \
                        np.int32(2), \
                        self.cl_data.h1.data, self.cl_data.h1.pitch, \
                        self.cl_data.hu1.data, self.cl_data.hu1.pitch, \
                        self.cl_data.hv1.data, self.cl_data.hv1.pitch, \
                        self.cl_data.h0.data, self.cl_data.h0.pitch, \
                        self.cl_data.hu0.data, self.cl_data.hu0.pitch, \
                        self.cl_data.hv0.data, self.cl_data.hv0.pitch, \
                        self.bathymetry.Bi.data, self.bathymetry.Bi.pitch, \
                        self.bathymetry.Bm.data, self.bathymetry.Bm.pitch, \
                        self.wind_stress.type, \
                        self.wind_stress.tau0, self.wind_stress.rho, self.wind_stress.alpha, self.wind_stress.xm, self.wind_stress.Rc, \
                        self.wind_stress.x0, self.wind_stress.y0, \
                        self.wind_stress.u0, self.wind_stress.v0, \
                        self.t, \
                        self.boundaryType, \
                        self.reportGeostrophicEquilibrium, \
                        self.geoEq_uxpvy.data, self.geoEq_uxpvy.pitch, \
                        self.geoEq_Kx.data, self.geoEq_Kx.pitch, \
                        self.geoEq_Ly.data, self.geoEq_Ly.pitch )

                self.bc_kernel.boundaryCondition(self.cl_queue, \
                        self.cl_data.h0, self.cl_data.hu0, self.cl_data.hv0)
                
            self.t += local_dt
            
        if self.write_netcdf:
            self.sim_writer.writeTimestep(self)
            
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
        if (self.h0AsWaterElevation):
            # Swap h0 with h1, fill h0 with w, download, swap back h0 with h1
            self.cl_data.h0, self.cl_data.h1 = self.cl_data.h1, self.cl_data.h0
            self.bathymetry.waterDepthToElevation(self.cl_data.h0, self.cl_data.h1)
            h1, hu1, hv1 = self.cl_data.download(self.cl_queue)
            self.cl_data.h0, self.cl_data.h1 = self.cl_data.h1, self.cl_data.h0
            return h1, hu1, hv1
        return self.cl_data.download(self.cl_queue)

    def downloadBathymetry(self):
        return self.bathymetry.download(self.cl_queue)

    def downloadGeoEqNorm(self):
        
        uxpvy_cpu = self.geoEq_uxpvy.download(self.cl_queue)
        Kx_cpu = self.geoEq_Kx.download(self.cl_queue)
        Ly_cpu = self.geoEq_Ly.download(self.cl_queue)

        uxpvy_norm = np.linalg.norm(uxpvy_cpu)
        Kx_norm = np.linalg.norm(Kx_cpu)
        Ly_norm = np.linalg.norm(Ly_cpu)

        return uxpvy_norm, Kx_norm, Ly_norm
