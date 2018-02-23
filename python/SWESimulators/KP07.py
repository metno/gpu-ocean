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
import gc

import Common, SimWriter, SimReader
import Simulator

class KP07(Simulator.Simulator):

    def __init__(self, \
                 cl_ctx, \
                 eta0, Hi, hu0, hv0, \
                 nx, ny, \
                 dx, dy, \
                 g, f=0.0, r=0.0, \
                 t=0.0, \
                 theta=1.3, use_rk2=True, coriolis_beta=0.0, \
                 y_zero_reference_cell = 0, \
                 dt_scale=1.0, \
                 wind_stress=Common.WindStressParams(), \
                 boundary_conditions=Common.BoundaryConditions(), \
                 write_netcdf=False, \
                 ignore_ghostcells=False, \
                 offset_x=0, offset_y=0, \
                 block_width=16, block_height=16, dt_block_size=128):
        """
        Initialization routine
        eta0: Initial deviation from mean sea level incl ghost cells, (nx+2)*(ny+2) cells
        hu0: Initial momentum along x-axis incl ghost cells, (nx+1)*(ny+2) cells
        hv0: Initial momentum along y-axis incl ghost cells, (nx+2)*(ny+1) cells
        Hi: Depth from equilibrium defined on cell corners, (nx+5)*(ny+5) corners
        nx: Number of cells along x-axis
        ny: Number of cells along y-axis
        dx: Grid cell spacing along x-axis (20 000 m)
        dy: Grid cell spacing along y-axis (20 000 m)
        dt: Size of each timestep (90 s)
        g: Gravitational accelleration (9.81 m/s^2)
        f: Coriolis parameter (1.2e-4 s^1), effectively as f = f + beta*y
        r: Bottom friction coefficient (2.4e-3 m/s)
        t: Start simulation at time t
        theta: MINMOD theta used the reconstructions of the derivatives in the numerical scheme
        use_rk2: Boolean if to use 2nd order Runge-Kutta (false -> 1st order forward Euler)
        coriolis_beta: Coriolis linear factor -> f = f + beta*(y-y_0)
        y_zero_reference_cell: The cell representing y_0 in the above, defined as the lower face of the cell .
        wind_stress: Wind stress parameters
        boundary_conditions: Boundary condition object
        write_netcdf: Write the results after each superstep to a netCDF file
        """
                 
        
        ## After changing from (h, B) to (eta, H), several of the simulator settings used are wrong. This check will help detect that.
        if ( np.sum(eta0 - Hi[:-1, :-1] > 0) > nx):
            assert(False), "It seems you are using water depth/elevation h and bottom topography B, while you should use water level eta and equillibrium depth H."
            
        self.cl_ctx = cl_ctx
        self.A = "NA"  # Eddy viscocity coefficient
            
        #Create an OpenCL command queue
        self.cl_queue = cl.CommandQueue(self.cl_ctx)

        #Get kernels
        self.kp07_kernel = Common.get_kernel(self.cl_ctx, "KP07_kernel.opencl", block_width, block_height)
        self.runge_kutta_kernel = Common.get_kernel(self.cl_ctx, "RungeKutta.opencl", block_width, block_height)
        self.dt_kernel = Common.get_kernel(self.cl_ctx, "max_dt.opencl", dt_block_size, 1)
        
        ghost_cells_x = 2
        ghost_cells_y = 2
        y_zero_reference_cell = 2.0 + y_zero_reference_cell
        
        # Boundary conditions
        self.boundary_conditions = boundary_conditions

        # Extend the computational domain if the boundary conditions
        # require it
        if (boundary_conditions.isSponge()):
            nx = nx + boundary_conditions.spongeCells[1] + boundary_conditions.spongeCells[3] - 2*ghost_cells_x
            ny = ny + boundary_conditions.spongeCells[0] + boundary_conditions.spongeCells[2] - 2*ghost_cells_y
            y_zero_reference_cell = boundary_conditions.spongeCells[2] + y_zero_reference_cell
            
        self.use_rk2 = use_rk2
        rk_order = np.int32(use_rk2 + 1)
        A = None
        dt = None
        super(KP07, self).__init__(cl_ctx, \
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
                                   block_width, block_height)
            
        #Create data by uploading to device    
        self.cl_data = Common.SWEDataArakawaA(self.cl_ctx, nx, ny, ghost_cells_x, ghost_cells_y, eta0, hu0, hv0)
        self.R1 = Common.OpenCLArray2D(self.cl_ctx, nx, ny, ghost_cells_x, ghost_cells_y)
        self.R2 = Common.OpenCLArray2D(self.cl_ctx, nx, ny, ghost_cells_x, ghost_cells_y)
        self.R3 = Common.OpenCLArray2D(self.cl_ctx, nx, ny, ghost_cells_x, ghost_cells_y)
        self.dt = Common.OpenCLArray2D(self.cl_ctx, nx, ny, ghost_cells_x, ghost_cells_y)
        
        #Bathymetry
        self.bathymetry = Common.Bathymetry(self.cl_ctx, self.cl_queue, nx, ny, ghost_cells_x, ghost_cells_y, Hi, boundary_conditions)
        
        #Save input parameters
        #Notice that we need to specify them in the correct dataformat for the
        #OpenCL kernel
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.dx = np.float32(dx)
        self.dy = np.float32(dy)
        self.g = np.float32(g)
        self.f = np.float32(f)
        self.r = np.float32(r)
        self.theta = np.float32(theta)
        self.use_rk2 = use_rk2
        self.coriolis_beta = np.float32(coriolis_beta)
        self.dt_scale = np.float32(dt_scale)
        self.y_zero_reference = np.int32(y_zero_reference_cell)
        self.rk_order = np.int32(use_rk2 + 1)
        self.wind_stress = wind_stress
        
        #Initialize time
        self.t = np.float32(0.0)
        
        #Compute kernel launch parameters
        self.local_size = (block_width, block_height) 
        self.global_size = ( \
                       int(np.ceil(self.nx / float(self.local_size[0])) * self.local_size[0]), \
                       int(np.ceil(self.ny / float(self.local_size[1])) * self.local_size[1]) \
                      ) 
        self.dt_block_size = dt_block_size
        
        self.bc_kernel = Common.BoundaryConditionsArakawaA(self.cl_ctx, \
                                                           self.nx, \
                                                           self.ny, \
                                                           ghost_cells_x, \
                                                           ghost_cells_y, \
                                                           self.boundary_conditions)
        
        if self.write_netcdf:
            self.sim_writer = SimWriter.SimNetCDFWriter(self, ignore_ghostcells=self.ignore_ghostcells, \
                                    offset_x=self.offset_x, offset_y=self.offset_y)

        self.bc_kernel.boundaryCondition(self.cl_queue, \
                self.cl_data.h0, self.cl_data.hu0, self.cl_data.hv0)

            
    @classmethod
    def fromfilename(cls, cl_ctx, filename, cont_write_netcdf=True):
        """
        Initialize and hotstart simulation from nc-file.
        cont_write_netcdf: Continue to write the results after each superstep to a new netCDF file
        filename: Continue simulation based on parameters and last timestep in this file
        """
        # open nc-file
        sim_reader = SimReader.SimNetCDFReader(filename, ignore_ghostcells=False)
        sim_name = str(sim_reader.get('simulator_short'))
        assert sim_name == cls.__name__, \
               "Trying to initialize a " + \
               cls.__name__ + " simulator with netCDF file based on " \
               + sim_name + " results."
        
        # read parameters
        nx = sim_reader.get("nx")
        ny = sim_reader.get("ny")

        dx = sim_reader.get("dx")
        dy = sim_reader.get("dy")

        width = nx * dx
        height = ny * dy

        dt = sim_reader.get("dt")
        g = sim_reader.get("g")
        r = sim_reader.get("bottom_friction_r")
        f = sim_reader.get("coriolis_force")
        beta = sim_reader.get("coriolis_beta")
        
        minmodTheta = sim_reader.get("minmod_theta")
        timeIntegrator = sim_reader.get("time_integrator")
        if (timeIntegrator == 2):
            using_rk2 = True
        else:
            using_rk2 = False 
        y_zero_reference_cell = sim_reader.get("y_zero_reference_cell")        
        
        wind_stress_type = sim_reader.get("wind_stress_type")
        wind = Common.WindStressParams(type=wind_stress_type)

        boundaryConditions = Common.BoundaryConditions( \
            sim_reader.getBC()[0], sim_reader.getBC()[1], \
            sim_reader.getBC()[2], sim_reader.getBC()[3], \
            sim_reader.getBCSpongeCells())

        Hi = sim_reader.getH();
        
        # get last timestep (including simulation time of last timestep)
        eta0, hu0, hv0, time0 = sim_reader.getLastTimeStep()
        
        return cls(cl_ctx, \
                 eta0, Hi, hu0, hv0, \
                 nx, ny, \
                 dx, dy, dt, \
                 g, f, r, \
                 t=time0, \
                 theta=minmodTheta, use_rk2=using_rk2, \
                 coriolis_beta=beta, \
                 y_zero_reference_cell = y_zero_reference_cell, \
                 wind_stress=wind, \
                 boundary_conditions=boundaryConditions, \
                 write_netcdf=cont_write_netcdf)

    def cleanUp(self):
        """
        Clean up function
        """
        self.closeNetCDF()
        
        self.cl_data.release()
        
        self.bathymetry.release()
        gc.collect()
                
    def fluxKernel(self, update_dt, U1, U2, U3):
        self.kp07_kernel.swe_2D(self.cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        self.dx, self.dy, \
                        self.g, \
                        self.theta, \
                        self.f, \
                        self.coriolis_beta, \
                        self.y_zero_reference, \
                        U1.data, U1.pitch,  \
                        U2.data, U2.pitch, \
                        U3.data, U3.pitch, \
                        self.R1.data, self.R1.pitch, \
                        self.R2.data, self.R2.pitch, \
                        self.R3.data, self.R3.pitch, \
                        self.dt.data, \
                        self.bathymetry.Bi.data, self.bathymetry.Bi.pitch, \
                        self.wind_stress.type, \
                        self.wind_stress.tau0, self.wind_stress.rho, self.wind_stress.alpha, self.wind_stress.xm, self.wind_stress.Rc, \
                        self.wind_stress.x0, self.wind_stress.y0, \
                        self.wind_stress.u0, self.wind_stress.v0, \
                        self.boundary_conditions.north, self.boundary_conditions.east, self.boundary_conditions.south, self.boundary_conditions.west, \
                        self.t, np.int32(update_dt))
                        
    def findDtKernel(self, max_dt):
        num_blocks_x = self.global_size[0]/self.local_size[0];
        num_blocks_y = self.global_size[1]/self.local_size[1];
        num_blocks = num_blocks_x*num_blocks_y
        self.dt_kernel.reduce_dt(self.cl_queue, (self.dt_block_size, 1), (self.dt_block_size, 1), \
                        self.dt.data, np.int32(num_blocks), 
                        self.dt_scale, np.float32(max_dt) )
                        
    
    def rungeKuttaKernel(self, substep, U1, U2, U3, Q1, Q2, Q3):
        self.runge_kutta_kernel.RungeKutta(self.cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        U1.data, U1.pitch,  \
                        U2.data, U2.pitch, \
                        U3.data, U3.pitch, \
                        self.R1.data, self.R1.pitch, \
                        self.R2.data, self.R2.pitch, \
                        self.R3.data, self.R3.pitch, \
                        Q1.data, Q1.pitch, \
                        Q2.data, Q2.pitch, \
                        Q3.data, Q3.pitch, \
                        self.bathymetry.Bm.data, self.bathymetry.Bm.pitch, \
                        self.r, \
                        self.dt.data, \
                        np.int32(substep) )
                        
    def boundaryConditionsKernel(self, U1, U2, U3):
        self.bc_kernel.boundaryCondition(self.cl_queue, U1, U2, U3)
        
    """
    Function which steps t_end in time
    """
    def step(self, t_end=0.0):
        n = 0
        t_end = np.float32(t_end);
        while (t_end > 0.0):
            if (self.use_rk2):
                # Substep one
                self.fluxKernel(1, self.cl_data.h0, self.cl_data.hu0, self.cl_data.hv0)
                self.findDtKernel(t_end)
                self.rungeKuttaKernel(0, self.cl_data.h0, self.cl_data.hu0, self.cl_data.hv0, \
                                         self.cl_data.h1, self.cl_data.hu1, self.cl_data.hv1)
                self.boundaryConditionsKernel(self.cl_data.h1, self.cl_data.hu1, self.cl_data.hv1)
                        
                # Substep two
                self.fluxKernel(0, self.cl_data.h1, self.cl_data.hu1, self.cl_data.hv1)
                self.rungeKuttaKernel(1, self.cl_data.h0, self.cl_data.hu0, self.cl_data.hv0, \
                                         self.cl_data.h1, self.cl_data.hu1, self.cl_data.hv1)
                self.boundaryConditionsKernel(self.cl_data.h1, self.cl_data.hu1, self.cl_data.hv1)
                
                # Swap h0 and h1
                self.cl_data.swap()
                
                        
            else:
                self.fluxKernel(1, self.cl_data.h0, self.cl_data.hu0, self.cl_data.hv0)
                self.findDtKernel(t_end)
                self.rungeKuttaKernel(0, self.cl_data.h0, self.cl_data.hu0, self.cl_data.hv0, \
                                         self.cl_data.h1, self.cl_data.hu1, self.cl_data.hv1)
                self.boundaryConditionsKernel(self.cl_data.h1, self.cl_data.hu1, self.cl_data.hv1)
                      
                # Swap h0 and h1  
                self.cl_data.swap()
            
            local_dt = np.zeros((128), dtype=np.float32)
            cl.enqueue_copy(self.cl_queue, local_dt, self.dt.data)
            t_end = t_end - local_dt[0];
            self.t += np.float32(local_dt[0]);
            print(local_dt[0]);
            n = n + 1;
            
            
        if self.write_netcdf:
            self.sim_writer.writeTimestep(self)
            
        print("Computed " + str(n) + " timesteps")
            
        return self.t
    
    
    

    def downloadBathymetry(self):
        return self.bathymetry.download(self.cl_queue)

