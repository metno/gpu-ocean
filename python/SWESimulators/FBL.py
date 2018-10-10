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
import numpy as np
import pyopencl as cl #OpenCL in Python
import gc

from SWESimulators import Common, SimWriter, SimReader
from SWESimulators import Simulator
from SWESimulators import WindStress
   

class FBL(Simulator.Simulator):
    """
    Class that solves the SW equations using the Forward-Backward linear scheme
    """

    def __init__(self, \
                 cl_ctx, \
                 H, eta0, hu0, hv0, \
                 nx, ny, \
                 dx, dy, dt, \
                 g, f, r, \
                 t=0.0, \
                 coriolis_beta=0.0, \
                 y_zero_reference_cell = 1, \
                 wind_stress=WindStress.NoWindStress(), \
                 boundary_conditions=Common.BoundaryConditions(), \
                 write_netcdf=False, \
                 ignore_ghostcells=False, \
                 offset_x=0, offset_y=0, \
                 block_width=16, block_height=16):
        """
        Initialization routine
        H: Water depth incl ghost cells, (nx+2)*(ny+2) cells
        eta0: Initial deviation from mean sea level incl ghost cells, (nx+2)*(ny+2) cells
        hu0: Initial momentum along x-axis incl ghost cells, (nx+1)*(ny+2) cells
        hv0: Initial momentum along y-axis incl ghost cells, (nx+2)*(ny+3) cells
        nx: Number of cells along x-axis
        ny: Number of cells along y-axis
        dx: Grid cell spacing along x-axis (20 000 m)
        dy: Grid cell spacing along y-axis (20 000 m)
        dt: Size of each timestep (90 s)
        g: Gravitational accelleration (9.81 m/s^2)
        f: Coriolis parameter (1.2e-4 s^1), effectively as f = f + beta*y
        r: Bottom friction coefficient (2.4e-3 m/s)
        coriolis_beta: Coriolis linear factor -> f = f + beta*y
        y_zero_reference_cell: The cell representing y_0 in the above, defined as the lower face of the cell .
        wind_stress: Wind stress parameters
        boundary_conditions: Boundary condition object
        write_netcdf: Write the results after each superstep to a netCDF file
        """
        
        #Create data by uploading to device
        ghost_cells_x = 1
        ghost_cells_y = 1
        y_zero_reference_cell = y_zero_reference_cell
        
        # Index range for interior domain (north, east, south, west)
        # so that interior domain of eta is
        # eta[self.interior_domain_indices[2]:self.interior_domain_indices[0], \
        #     self.interior_domain_indices[3]:self.interior_domain_indices[1] ]
        self.interior_domain_indices = np.array([-1, -1, 1, 1])
        
        self.boundary_conditions = boundary_conditions
        
        if boundary_conditions.isSponge():
            nx = nx - 2 + boundary_conditions.spongeCells[1] + boundary_conditions.spongeCells[3]
            ny = ny - 2 + boundary_conditions.spongeCells[0] + boundary_conditions.spongeCells[2]
            y_zero_reference_cell = y_zero_reference_cell + boundary_conditions.spongeCells[2]
        
        rk_order = None
        theta = None
        A = None
        super(FBL, self).__init__(cl_ctx, \
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
        
        
        self._set_interior_domain_from_sponge_cells()
        
        #--------------------------------------------!
        # Create compiler options here!
        compiler_options = ['-cl-fast-relaxed-math']
        #--------------------------------------------!
        
        #Get kernels
        self.step_kernel = Common.get_kernel(self.cl_ctx, "FBL_step_kernel.opencl", \
                                             block_width, block_height, \
                                             options = compiler_options)
                                            
        self.H = Common.OpenCLArray2D(self.cl_ctx, nx, ny, ghost_cells_x, ghost_cells_y, H)
        self.cl_data = Common.SWEDataArakawaC(self.cl_ctx, nx, ny, ghost_cells_x, ghost_cells_y, eta0, hu0, hv0, fbl=True)
        
        # Overwrite halo including ghost cells
        self.nx_halo = np.int32(nx + 2)
        self.ny_halo = np.int32(ny + 2)
        
        self.bc_kernel = FBL_periodic_boundary(self.cl_ctx, \
                                               self.nx, \
                                               self.ny, \
                                               self.boundary_conditions)

        self.totalNumIterations = 0
        # Bit-wise boolean for wall boundary conditions
        self.wall_bc = np.int32(0)
        if (self.boundary_conditions.north == 1):
            self.wall_bc = self.wall_bc | 0x01
        if (self.boundary_conditions.east == 1):
            self.wall_bc = self.wall_bc | 0x02
        if (self.boundary_conditions.south == 1):
            self.wall_bc = self.wall_bc | 0x04
        if (self.boundary_conditions.west == 1):
            self.wall_bc = self.wall_bc | 0x08

        if self.write_netcdf:
            self.sim_writer = SimWriter.SimNetCDFWriter(self, ignore_ghostcells=self.ignore_ghostcells, \
                                    staggered_grid=True, offset_x=self.offset_x, offset_y=self.offset_y)
            
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
        A = sim_reader.get("eddy_viscosity_coefficient")
        f = sim_reader.get("coriolis_force")
        beta = sim_reader.get("coriolis_beta")
        
        minmodTheta = sim_reader.get("minmod_theta")
        timeIntegrator = sim_reader.get("time_integrator")
        y_zero_reference_cell = sim_reader.get("y_zero_reference_cell")

        wind_stress_type = sim_reader.get("wind_stress_type")
        wind = Common.WindStressParams(type=wind_stress_type)

        boundaryConditions = Common.BoundaryConditions( \
            sim_reader.getBC()[0], sim_reader.getBC()[1], \
            sim_reader.getBC()[2], sim_reader.getBC()[3], \
            sim_reader.getBCSpongeCells())

        h0 = sim_reader.getH();

        # get last timestep (including simulation time of last timestep)
        eta0, hu0, hv0, time0 = sim_reader.getLastTimeStep()
        
        return cls(cl_ctx, \
                h0, eta0, hu0, hv0, \
                nx, ny, \
                dx, dy, dt, \
                g, f, r, \
                t=time0, \
                wind_stress=wind, \
                boundary_conditions=boundaryConditions, \
                write_netcdf=cont_write_netcdf)
    
    def cleanUp(self):
        """
        Clean up function
        """
        self.closeNetCDF()
        
        self.cl_data.release()
        
        self.H.release()
        self.cl_ctx = None
        gc.collect()
    
    # Over-riding Simulator's download.
    def download(self, interior_domain_only=False):
        """
        Download the latest time step from the GPU
        """
        return self.cl_data.download(self.cl_queue, \
                                     interior_domain_only=interior_domain_only)
       
    def step(self, t_end=0.0):
        """
        Function which steps n timesteps
        """
        n = int(t_end / self.dt + 1)
                
        ## Populate all ghost cells before we start
        if self.t == 0:
            self.bc_kernel.boundaryConditionU(self.cl_queue, self.cl_data.hu0)
            self.bc_kernel.boundaryConditionV(self.cl_queue, self.cl_data.hv0)
            self.bc_kernel.boundaryConditionEta(self.cl_queue, self.cl_data.h0)
            

        for i in range(0, n):        
            local_dt = np.float32(min(self.dt, t_end-i*self.dt))
            
            #if self.totalNumIterations > 240:
            #if self.totalNumIterations > 5:
            #    self.t += t_end
            #    return self.t


            if (local_dt <= 0.0):
                break
                
            wind_stress_t = np.float32(1.2)

            self.step_kernel.fblStepKernel(self.cl_queue, self.global_size, self.local_size, \
                    self.nx, self.ny, \
                    self.dx, self.dy, local_dt, \
                    self.g, self.f, self.coriolis_beta, self.y_zero_reference_cell, self.r, \
                    self.H.data, self.H.pitch, \
                    self.cl_data.hu0.data, self.cl_data.hu0.pitch, \
                    self.cl_data.hv0.data, self.cl_data.hv0.pitch, \
                    self.cl_data.h0.data, self.cl_data.h0.pitch, \
                    self.wall_bc, wind_stress_t)

            # Fix U boundary
            self.bc_kernel.boundaryConditionU(self.cl_queue, self.cl_data.hu0)
            
            # Fix V boundary
            self.bc_kernel.boundaryConditionV(self.cl_queue, self.cl_data.hv0)
            
            # Fix eta boundary
            self.bc_kernel.boundaryConditionEta(self.cl_queue, self.cl_data.h0)
   
            self.t += local_dt
            self.totalNumIterations += 1
            
        if self.write_netcdf:
            self.sim_writer.writeTimestep(self)
            
        return self.t
    

class FBL_periodic_boundary:
    def __init__(self, cl_ctx, nx, ny, \
                 boundary_conditions, \
                 block_width=16, block_height=16 ):

        self.boundary_conditions = boundary_conditions
        self.ghostsX = np.int32(2)
        self.ghostsY = np.int32(2)

        self.bc_north = np.int32(boundary_conditions.north)
        self.bc_east  = np.int32(boundary_conditions.east)
        self.bc_south = np.int32(boundary_conditions.south)
        self.bc_west  = np.int32(boundary_conditions.west)
        
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.nx_halo = np.int32(nx + self.ghostsX)
        self.ny_halo = np.int32(ny + self.ghostsY)

        # Debugging variables
        debug = False
        self.firstU = True
        self.firstV = True
        self.firstEta = True
        self.firstGhostU = debug
        self.firstGhostV = debug
        self.firstGhostEta = debug
        
        # Load kernel for periodic boundary.
        self.periodicBoundaryKernel \
            = Common.get_kernel(cl_ctx,\
            "FBL_periodic_boundary.opencl", block_width, block_height)

        # Reuse CTCS kernels for Flow Relaxation Scheme
        self.CTCSBoundaryKernels = Common.get_kernel(cl_ctx,\
                     "CTCS_boundary.opencl", block_width, block_height)
        
        #Compute kernel launch parameters
        self.local_size = (block_width, block_height) # WARNING::: MUST MATCH defines of block_width/height in kernels!
        self.global_size = ( \
                int(np.ceil((self.nx+2) / float(self.local_size[0])) * self.local_size[0]), \
                int(np.ceil((self.ny+3) / float(self.local_size[1])) * self.local_size[1]) )

    

    def boundaryConditionU(self, cl_queue, hu0):
        """
        Updates hu according periodic boundary conditions
        """

        # Start with fixing the potential sponge
        self.callSpongeNS(cl_queue, hu0, 1, 0)
        
        if self.firstU and (self.boundary_conditions.east == 1 or \
                            self.boundary_conditions.west == 1 or \
                            self.boundary_conditions.north == 1 or \
                            self.boundary_conditions.south == 1):
            
            self.periodicBoundaryKernel.closedBoundaryUKernel(cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        self.bc_north, self.bc_east, self.bc_south, self.bc_west, \
                        hu0.data, hu0.pitch)

        if (self.boundary_conditions.north == 2):
            self.periodicBoundaryKernel.periodicBoundaryUKernel_NS(cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        hu0.data, hu0.pitch)
        if (self.boundary_conditions.east == 2):
            self.periodicBoundaryKernel.periodicBoundaryUKernel_EW(cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        hu0.data, hu0.pitch)
        
        self.firstU = False    

    def boundaryConditionV(self, cl_queue, hv0):
        """
        Updates hv according to periodic boundary conditions
        """

        # Start with fixing the potential sponge
        self.callSpongeNS(cl_queue, hv0, 0, 1)
        
        if self.firstV and (self.boundary_conditions.east == 1 or \
                            self.boundary_conditions.west == 1 or \
                            self.boundary_conditions.north == 1 or \
                            self.boundary_conditions.south == 1):
          
            self.periodicBoundaryKernel.closedBoundaryVKernel(cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        self.bc_north, self.bc_east, self.bc_south, self.bc_west, \
                        hv0.data, hv0.pitch)

        if (self.boundary_conditions.north == 2):
            self.periodicBoundaryKernel.periodicBoundaryVKernel_NS(cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        hv0.data, hv0.pitch)
        if (self.boundary_conditions.east == 2):
            self.periodicBoundaryKernel.periodicBoundaryVKernel_EW(cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        hv0.data, hv0.pitch)
        
        self.firstV = False
        

    def boundaryConditionEta(self, cl_queue, eta0):
        """
        Updates eta boundary conditions (ghost cells)
        """
        # Start with fixing the potential sponge
        self.callSpongeNS(cl_queue, eta0, 0, 0)
        
        if self.firstEta and (self.boundary_conditions.east == 1 or \
                              self.boundary_conditions.west == 1 or \
                              self.boundary_conditions.north == 1 or \
                              self.boundary_conditions.south == 1):
            
            # self.closedBoundaryEtaKernel
            self.periodicBoundaryKernel.closedBoundaryEtaKernel(cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        self.bc_north, self.bc_east, self.bc_south, self.bc_west, \
                        eta0.data, eta0.pitch)
                              
        if (self.boundary_conditions.north == 2):
            self.periodicBoundaryKernel.periodicBoundaryEtaKernel_NS(cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        eta0.data, eta0.pitch)
        if (self.boundary_conditions.east == 2):
            self.periodicBoundaryKernel.periodicBoundaryEtaKernel_EW(cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        eta0.data, eta0.pitch)
                  
        self.firstEta = False
        
    def callSpongeNS(self, cl_queue, data, staggered_x, staggered_y):
        staggered_x_int32 = np.int32(staggered_x)
        staggered_y_int32 = np.int32(staggered_y)
        
        if (self.bc_north == 3) or (self.bc_south ==3):
            self.CTCSBoundaryKernels.boundary_flowRelaxationScheme_NS( \
                cl_queue, self.global_size, self.local_size, \
                self.nx, self.ny, \
                self.ghostsX, self.ghostsY, \
                staggered_x_int32, staggered_y_int32, \
                self.boundary_conditions.spongeCells[0], \
                self.boundary_conditions.spongeCells[2], \
                self.bc_north, self.bc_south, \
                data.data, data.pitch)

        if (self.bc_east == 3) or (self.bc_west == 3):
            self.CTCSBoundaryKernels.boundary_flowRelaxationScheme_EW( \
                cl_queue, self.global_size, self.local_size, \
                self.nx, self.ny, \
                self.ghostsX, self.ghostsY, \
                staggered_x_int32, staggered_y_int32, \
                self.boundary_conditions.spongeCells[1], \
                self.boundary_conditions.spongeCells[3], \
                self.bc_east, self.bc_west, \
                data.data, data.pitch)
