# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2016 SINTEF ICT, 
Copyright (C) 2017, 2018 SINTEF Digital
Copyright (C) 2017, 2018 Norwegian Meteorological Institute

This python module implements 
Alina Chertock, Michael Dudzinski, A. Kurganov & Maria Lukacova-Medvidova (2016)
Well-Balanced Schemes for the Shallow Water Equations with Coriolis Forces

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




        
        
        


"""
Class that solves the SW equations using the Coriolis well balanced reconstruction scheme, as given by the publication of Chertock, Dudzinski, Kurganov and Lukacova-Medvidova (CDFLM) in 2016.
"""
class CDKLM16:

    """
    Initialization routine
    eta0: Water level incl ghost cells, (nx+4)*(ny+4) cells
    u0: Initial momentum along x-axis incl ghost cells, (nx+4)*(ny+4) cells
    v0: Initial momentum along y-axis incl ghost cells, (nx+4)*(ny+4) cells
    Hi: Equilibrium water depth defined on cell corners, (nx+5)*(ny+5) corners
    nx: Number of cells along x-axis
    ny: Number of cells along y-axis
    dx: Grid cell spacing along x-axis (20 000 m)
    dy: Grid cell spacing along y-axis (20 000 m)
    dt: Size of each timestep (90 s)
    g: Gravitational accelleration (9.81 m/s^2)
    f: Coriolis parameter (1.2e-4 s^1), effectively as f = f + beta*y
    r: Bottom friction coefficient (2.4e-3 m/s)
    theta: minmod reconstruction parameter
    rk_order: Order of Runge Kutta method {1,2*,3}
    coriolis_beta: Coriolis linear factor -> f = f + beta*y
    y_zero_reference_cell: The cell representing y_0 in the above, defined as the lower face of the cell.
    wind_stress: Wind stress parameters
    boundary_conditions: Boundary conditions object
    h0AsWaterElevation: True if h0 is described by the surface elevation, and false if h0 is described by water depth
    reportGeostrophicEquilibrium: Calculate the Geostrophic Equilibrium variables for each superstep
    write_netcdf: Write the results after each superstep to a netCDF file
    """
    def __init__(self, \
                 cl_ctx, \
                 eta0, hu0, hv0, \
                 Hi, \
                 nx, ny, \
                 dx, dy, dt, \
                 g, f, r, \
                 theta=1.3, rk_order=2, \
                 coriolis_beta=0.0, \
                 y_zero_reference_cell = 0, \
                 wind_stress=Common.WindStressParams(), \
                 boundary_conditions=Common.BoundaryConditions(), \
                 h0AsWaterElevation=False, \
                 reportGeostrophicEquilibrium=False, \
                 write_netcdf=False, \
                 double_precision = False, \
                 block_width=16, block_height=16):
        
        self.cl_ctx = cl_ctx

        #Create an OpenCL command queue
        self.cl_queue = cl.CommandQueue(self.cl_ctx)
        self.A = "NA"  # Eddy viscocity coefficient

        ## After changing from (h, B) to (eta, H), several of the simulator settings used are wrong. This check will help detect that.
        if ( np.sum(eta0 - Hi[:-1, :-1] > 0) > nx):
            assert(False), "It seems you are using water depth/elevation h and bottom topography B, while you should use water level eta and equillibrium depth H."
        
        #Get kernels
        self.kernel = None
        if double_precision:
            self.kernel = Common.get_kernel(self.cl_ctx, "CDKLM16_double_kernel.opencl", block_width, block_height)
        else:
            self.kernel = Common.get_kernel(self.cl_ctx, "CDKLM16_kernel.opencl", block_width, block_height)

        self.ghost_cells_x = 2
        self.ghost_cells_y = 2
        ghost_cells_x = 2
        ghost_cells_y = 2
        self.y_zero_reference_cell = np.float32(2 + y_zero_reference_cell)
        
        # Boundary conditions
        self.boundary_conditions = boundary_conditions
        if (boundary_conditions.isSponge()):
            nx = nx + boundary_conditions.spongeCells[1] + boundary_conditions.spongeCells[3] - 2*self.ghost_cells_x
            ny = ny + boundary_conditions.spongeCells[0] + boundary_conditions.spongeCells[2] - 2*self.ghost_cells_y
            self.y_zero_reference_cell = np.float32(boundary_conditions.spongeCells[2] + y_zero_reference_cell)
        
        #Create data by uploading to device
        self.cl_data = Common.SWEDataArakawaA(self.cl_ctx, nx, ny, ghost_cells_x, ghost_cells_y, eta0, hu0, hv0)

        ## Allocating memory for geostrophical equilibrium variables
        self.reportGeostrophicEquilibrium = np.int32(reportGeostrophicEquilibrium)
        dummy_zero_array = np.zeros((ny+2*ghost_cells_y, nx+2*ghost_cells_x), dtype=np.float32, order='C') 
        self.geoEq_uxpvy = Common.OpenCLArray2D(cl_ctx, nx, ny, ghost_cells_x, ghost_cells_y, dummy_zero_array)
        self.geoEq_Kx = Common.OpenCLArray2D(cl_ctx, nx, ny, ghost_cells_x, ghost_cells_y, dummy_zero_array)
        self.geoEq_Ly = Common.OpenCLArray2D(cl_ctx, nx, ny, ghost_cells_x, ghost_cells_y, dummy_zero_array)

        #Bathymetry
        self.bathymetry = Common.Bathymetry(self.cl_ctx, self.cl_queue, nx, ny, ghost_cells_x, ghost_cells_y, Hi, boundary_conditions)

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
        self.coriolis_beta = np.float32(coriolis_beta)
        self.wind_stress = wind_stress
        self.h0AsWaterElevation = h0AsWaterElevation


        
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
        gc.collect() # Force run garbage collection to free up memory
        
    
    """
    Function which steps n timesteps
    """
    def step(self, t_end=0.0):
        n = int(t_end / self.dt + 1)

        if self.t == 0:
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

                self.callKernel(self.cl_data.h0, self.cl_data.hu0, self.cl_data.hv0, \
                                self.cl_data.h1, self.cl_data.hu1, self.cl_data.hv1, \
                                local_dt, 0)

                self.bc_kernel.boundaryCondition(self.cl_queue, \
                        self.cl_data.h1, self.cl_data.hu1, self.cl_data.hv1)

                self.callKernel(self.cl_data.h1, self.cl_data.hu1, self.cl_data.hv1, \
                                self.cl_data.h0, self.cl_data.hu0, self.cl_data.hv0, \
                                local_dt, 1)

                self.bc_kernel.boundaryCondition(self.cl_queue, \
                        self.cl_data.h0, self.cl_data.hu0, self.cl_data.hv0)
                
            elif (self.rk_order == 1):
                self.callKernel(self.cl_data.h0, self.cl_data.hu0, self.cl_data.hv0, \
                                self.cl_data.h1, self.cl_data.hu1, self.cl_data.hv1, \
                                local_dt, 0)
                                
                self.cl_data.swap()

                self.bc_kernel.boundaryCondition(self.cl_queue, \
                        self.cl_data.h0, self.cl_data.hu0, self.cl_data.hv0)

            # 3rd order RK method:
            elif (self.rk_order == 3):

                self.callKernel(self.cl_data.h0, self.cl_data.hu0, self.cl_data.hv0, \
                                self.cl_data.h1, self.cl_data.hu1, self.cl_data.hv1, \
                                local_dt, 0)
                
                self.bc_kernel.boundaryCondition(self.cl_queue, \
                        self.cl_data.h1, self.cl_data.hu1, self.cl_data.hv1)

                self.callKernel(self.cl_data.h1, self.cl_data.hu1, self.cl_data.hv1, \
                                self.cl_data.h0, self.cl_data.hu0, self.cl_data.hv0, \
                                local_dt, 1)

                self.bc_kernel.boundaryCondition(self.cl_queue, \
                        self.cl_data.h1, self.cl_data.hu1, self.cl_data.hv1)

                self.callKernel(self.cl_data.h1, self.cl_data.hu1, self.cl_data.hv1, \
                                self.cl_data.h0, self.cl_data.hu0, self.cl_data.hv0, \
                                local_dt, 2)
                
                self.bc_kernel.boundaryCondition(self.cl_queue, \
                        self.cl_data.h0, self.cl_data.hu0, self.cl_data.hv0)
                
            self.t += local_dt
            
        if self.write_netcdf:
            self.sim_writer.writeTimestep(self)
            
        return self.t


    def callKernel(self, \
                   h_in, hu_in, hv_in, \
                   h_out, hu_out, hv_out, \
                   local_dt, rk_step):
        self.kernel.swe_2D(self.cl_queue, self.global_size, self.local_size, \
                           self.nx, self.ny, \
                           self.dx, self.dy, local_dt, \
                           self.g, \
                           self.theta, \
                           self.f, \
                           self.coriolis_beta, \
                           self.y_zero_reference_cell, \
                           self.r, \
                           self.rk_order, \
                           np.int32(rk_step), \
                           h_in.data, h_in.pitch, \
                           hu_in.data, hu_in.pitch, \
                           hv_in.data, hv_in.pitch, \
                           h_out.data, h_out.pitch, \
                           hu_out.data, hu_out.pitch, \
                           hv_out.data, hv_out.pitch, \
                           self.bathymetry.Bi.data, self.bathymetry.Bi.pitch, \
                           self.bathymetry.Bm.data, self.bathymetry.Bm.pitch, \
                           self.wind_stress.type, \
                           self.wind_stress.tau0, self.wind_stress.rho, self.wind_stress.alpha, self.wind_stress.xm, self.wind_stress.Rc, \
                           self.wind_stress.x0, self.wind_stress.y0, \
                           self.wind_stress.u0, self.wind_stress.v0, \
                           self.t, \
                           self.boundary_conditions.north, self.boundary_conditions.east, self.boundary_conditions.south, self.boundary_conditions.west, \
                           self.reportGeostrophicEquilibrium, \
                           self.geoEq_uxpvy.data, self.geoEq_uxpvy.pitch, \
                           self.geoEq_Kx.data, self.geoEq_Kx.pitch, \
                           self.geoEq_Ly.data, self.geoEq_Ly.pitch )

    
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
