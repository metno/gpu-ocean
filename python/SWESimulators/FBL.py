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
import gc

from SWESimulators import Common, SimWriter, SimReader
from SWESimulators import Simulator
from SWESimulators import WindStress
   

class FBL(Simulator.Simulator):
    """
    Class that solves the SW equations using the Forward-Backward linear scheme
    """

    def __init__(self, \
                 gpu_ctx, \
                 H, eta0, hu0, hv0, \
                 nx, ny, \
                 dx, dy, dt, \
                 g, f, r, \
                 t=0.0, \
                 coriolis_beta=0.0, \
                 y_zero_reference_cell = 0, \
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
        hv0: Initial momentum along y-axis incl ghost cells, (nx+2)*(ny+1) cells
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
        ghost_cells_x = 0
        ghost_cells_y = 0
        y_zero_reference_cell = y_zero_reference_cell
        self.asym_ghost_cells = [0, 0, 0, 0] # [N, E, S, W]
        
        # Index range for interior domain (north, east, south, west)
        # so that interior domain of eta is
        # eta[self.interior_domain_indices[2]:self.interior_domain_indices[0], \
        #     self.interior_domain_indices[3]:self.interior_domain_indices[1] ]
        self.interior_domain_indices = np.array([None, None, 0, 0])
        
        self.boundary_conditions = boundary_conditions
        # Add asym ghost cell if periodic boundary condition:
        if (self.boundary_conditions.north == 2) or \
           (self.boundary_conditions.south == 2):
            self.asym_ghost_cells[0] = 1
            self.interior_domain_indices[0] = -1
        if (self.boundary_conditions.east == 2) or \
           (self.boundary_conditions.west == 2):
            self.asym_ghost_cells[1] = 1
            self.interior_domain_indices[1] = -1

        if boundary_conditions.isSponge():
            nx = nx + boundary_conditions.spongeCells[1] + boundary_conditions.spongeCells[3]# - self.asym_ghost_cells[1] - self.asym_ghost_cells[3]
            ny = ny + boundary_conditions.spongeCells[0] + boundary_conditions.spongeCells[2]# - self.asym_ghost_cells[0] - self.asym_ghost_cells[2]
            y_zero_reference_cell = y_zero_reference_cell + boundary_conditions.spongeCells[2]
          
        rk_order = None
        theta = None
        A = None
        super(FBL, self).__init__(gpu_ctx, \
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
        
        
        #Get kernels
        self.u_kernel = gpu_ctx.get_kernel("FBL_U_kernel.cu", block_width, block_height)
        self.v_kernel = gpu_ctx.get_kernel("FBL_V_kernel.cu", block_width, block_height)
        self.eta_kernel = gpu_ctx.get_kernel("FBL_eta_kernel.cu", block_width, block_height)

        # Get CUDA functions and define data types for prepared_{async_}call()
        self.computeUKernel = self.u_kernel.get_function("computeUKernel")
        self.computeUKernel.prepare("iiffffffffPiPiPiPiPf")
        self.computeVKernel = self.v_kernel.get_function("computeVKernel")
        self.computeVKernel.prepare("iiffffffffPiPiPiPiPf")
        self.computeEtaKernel = self.eta_kernel.get_function("computeEtaKernel")
        self.computeEtaKernel.prepare("iiffffffffPiPiPiPi")
        
        self.H = Common.CUDAArray2D(self.gpu_stream, nx, ny, ghost_cells_x, ghost_cells_y, H, self.asym_ghost_cells)
        self.gpu_data = Common.SWEDataArakawaC(self.gpu_stream, nx, ny, ghost_cells_x, ghost_cells_y, eta0, hu0, hv0, self.asym_ghost_cells)
        
        # Overwrite halo with asymetric ghost cells
        self.nx_halo = np.int32(nx + self.asym_ghost_cells[1] + self.asym_ghost_cells[3])
        self.ny_halo = np.int32(ny + self.asym_ghost_cells[0] + self.asym_ghost_cells[2])
       
        self.bc_kernel = FBL_periodic_boundary(self.gpu_ctx, \
                                               self.nx, \
                                               self.ny, \
                                               self.boundary_conditions, \
                                               self.asym_ghost_cells
        )

        self.totalNumIterations = 0
        if self.write_netcdf:
            self.sim_writer = SimWriter.SimNetCDFWriter(self, ignore_ghostcells=self.ignore_ghostcells, \
                                    staggered_grid=True, offset_x=self.offset_x, offset_y=self.offset_y)
            
    @classmethod
    def fromfilename(cls, gpu_ctx, filename, cont_write_netcdf=True):
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
        
        return cls(gpu_ctx, \
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
        
        self.gpu_data.release()
        
        self.H.release()
        gc.collect()
        
    def step(self, t_end=0.0):
        """
        Function which steps n timesteps
        """
        n = int(t_end / self.dt + 1)
                
        ## Populate all ghost cells before we start
        if self.t == 0:
            self.bc_kernel.boundaryConditionU(self.gpu_stream, self.gpu_data.hu0)
            self.bc_kernel.boundaryConditionV(self.gpu_stream, self.gpu_data.hv0)
            self.bc_kernel.boundaryConditionEta(self.gpu_stream, self.gpu_data.h0)
            

        for i in range(0, n):        
            local_dt = np.float32(min(self.dt, t_end-i*self.dt))
            
            #if self.totalNumIterations > 240:
            #if self.totalNumIterations > 5:
            #    self.t += t_end
            #    return self.t


            if (local_dt <= 0.0):
                break

            self.computeUKernel.prepared_async_call(self.global_size, self.local_size, self.gpu_stream, \
                    self.nx_halo, self.ny, \
                    self.dx, self.dy, local_dt, \
                    self.g, self.f, self.coriolis_beta, self.y_zero_reference_cell, self.r, \
                    self.H.data.gpudata, self.H.pitch, \
                    self.gpu_data.hu0.data.gpudata, self.gpu_data.hu0.pitch, \
                    self.gpu_data.hv0.data.gpudata, self.gpu_data.hv0.pitch, \
                    self.gpu_data.h0.data.gpudata, self.gpu_data.h0.pitch, \
                    self.wind_stress_dev, \
                    self.t)

            # Fix U boundary
            self.bc_kernel.boundaryConditionU(self.gpu_stream, self.gpu_data.hu0)
            
            self.computeVKernel.prepared_async_call(self.global_size, self.local_size, self.gpu_stream, \
                    self.nx, self.ny_halo, \
                    self.dx, self.dy, local_dt, \
                    self.g, self.f, self.coriolis_beta, self.y_zero_reference_cell, self.r, \
                    self.H.data.gpudata, self.H.pitch, \
                    self.gpu_data.hu0.data.gpudata, self.gpu_data.hu0.pitch, \
                    self.gpu_data.hv0.data.gpudata, self.gpu_data.hv0.pitch, \
                    self.gpu_data.h0.data.gpudata, self.gpu_data.h0.pitch, \
                    self.wind_stress_dev, \
                    self.t)

            # Fix V boundary
            self.bc_kernel.boundaryConditionV(self.gpu_stream, self.gpu_data.hv0)
            
            self.computeEtaKernel.prepared_async_call(self.global_size, self.local_size, self.gpu_stream, \
                    self.nx, self.ny, \
                    self.dx, self.dy, local_dt, \
                    self.g, self.f, self.coriolis_beta, self.y_zero_reference_cell, self.r, \
                    self.H.data.gpudata, self.H.pitch, \
                    self.gpu_data.hu0.data.gpudata, self.gpu_data.hu0.pitch, \
                    self.gpu_data.hv0.data.gpudata, self.gpu_data.hv0.pitch, \
                    self.gpu_data.h0.data.gpudata, self.gpu_data.h0.pitch)

            self.bc_kernel.boundaryConditionEta(self.gpu_stream, self.gpu_data.h0)
   
            self.t += local_dt
            self.totalNumIterations += 1
            
        if self.write_netcdf:
            self.sim_writer.writeTimestep(self)
            
        return self.t
    

class FBL_periodic_boundary:
    def __init__(self, \
                 gpu_ctx, \
                 nx, ny, \
                 boundary_conditions, asym_ghost_cells, \
                 block_width=16, block_height=16 ):

        self.boundary_conditions = boundary_conditions
        self.asym_ghost_cells = asym_ghost_cells
        self.ghostsX = np.int32(self.asym_ghost_cells[1] + self.asym_ghost_cells[3])
        self.ghostsY = np.int32(self.asym_ghost_cells[0] + self.asym_ghost_cells[2])

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
        self.firstU = debug
        self.firstV = debug
        self.firstGhostU = debug
        self.firstGhostV = debug
        self.firstGhostEta = debug
        
        # Load kernel for periodic boundary.
        self.periodicBoundaryKernel \
            = gpu_ctx.get_kernel("FBL_periodic_boundary.cu", block_width, block_height)
        # Get CUDA functions and define data types for prepared_{async_}call()
        self.closedBoundaryUKernel = self.periodicBoundaryKernel.get_function("closedBoundaryUKernel")
        self.closedBoundaryUKernel.prepare("iiiiPi")
        self.periodicBoundaryUKernel = self.periodicBoundaryKernel.get_function("periodicBoundaryUKernel")
        self.periodicBoundaryUKernel.prepare("iiiiPi")
        self.updateGhostCellsUKernel = self.periodicBoundaryKernel.get_function("updateGhostCellsUKernel")
        self.updateGhostCellsUKernel.prepare("iiiiPi")
        self.closedBoundaryVKernel = self.periodicBoundaryKernel.get_function("closedBoundaryVKernel")
        self.closedBoundaryVKernel.prepare("iiiiPi")
        self.periodicBoundaryVKernel = self.periodicBoundaryKernel.get_function("periodicBoundaryVKernel")
        self.periodicBoundaryVKernel.prepare("iiiiPi")
        self.updateGhostCellsVKernel = self.periodicBoundaryKernel.get_function("updateGhostCellsVKernel")
        self.updateGhostCellsVKernel.prepare("iiiiPi")
        self.periodicBoundaryEtaKernel = self.periodicBoundaryKernel.get_function("periodicBoundaryEtaKernel")
        self.periodicBoundaryEtaKernel.prepare("iiiiPi")

        # Reuse CTCS kernels for Flow Relaxation Scheme
        self.CTCSBoundaryKernels = gpu_ctx.get_kernel("CTCS_boundary.cu", block_width, block_height)
        # Get CUDA functions and define data types for prepared_{async_}call()
        self.boundary_flowRelaxationScheme_NS = self.CTCSBoundaryKernels.get_function("boundary_flowRelaxationScheme_NS")
        self.boundary_flowRelaxationScheme_NS.prepare("iiiiiiiiiiPi")
        self.boundary_flowRelaxationScheme_EW = self.CTCSBoundaryKernels.get_function("boundary_flowRelaxationScheme_EW")
        self.boundary_flowRelaxationScheme_EW.prepare("iiiiiiiiiiPi")
        
        #Compute kernel launch parameters
        self.local_size = (block_width, block_height, 1) # WARNING::: MUST MATCH defines of block_width/height in kernels!
        self.global_size = ( \
                int(np.ceil((self.nx_halo+1) / float(self.local_size[0]))), \
                int(np.ceil((self.ny_halo+1) / float(self.local_size[1]))) )

    

    def boundaryConditionU(self, gpu_stream, hu0):
        """
        Updates hu according periodic boundary conditions
        """

        # Start with fixing the potential sponge
        self.callSpongeNS(gpu_stream, hu0, 1, 0)
        
        if (self.boundary_conditions.east == 1 and \
            self.boundary_conditions.west == 1):
            if (self.nx_halo > self.nx):
                print("Closed east-west boundary, but nx_halo > nx")
                return
            
            self.closedBoundaryUKernel.prepared_async_call(self.global_size, self.local_size, gpu_stream, \
                        self.nx, self.ny, \
                        self.nx_halo, self.ny_halo, \
                        hu0.data.gpudata, hu0.pitch)

        elif (self.boundary_conditions.east == 2):
            ## Currently, this only works with 0 ghost cells:
            assert(hu0.nx == hu0.nx_halo), \
                "The current data does not have zero ghost cells"

            ## Call kernel that swaps the boundaries.
            if self.firstU:
                print("Periodic boundary conditions - U")
                self.firstU = False
                print("[nx, ny, nx_halo, ny_halo]")
                print([self.nx, self.ny, self.nx_halo, self.ny_halo])
                
            self.periodicBoundaryUKernel.prepared_async_call(self.global_size, self.local_size, gpu_stream, \
                        self.nx, self.ny, \
                        self.nx_halo, self.ny_halo, \
                        hu0.data.gpudata, hu0.pitch)
        
        
        # If there are ghost cells in north-south direction, update them!
        # TODO: Generalize to both ghost_north and ghost_south
        # Updating northern ghost cells
        if (self.ny_halo > self.ny):
            if self.firstGhostU:
                print("Updating U ghosts in north-south")
                self.firstGhostU = False
            
            self.updateGhostCellsUKernel.prepared_async_call(self.global_size, self.local_size, gpu_stream, \
                        self.nx, self.ny, \
                        self.nx_halo, self.ny_halo, \
                        hu0.data.gpudata, hu0.pitch)
            

    def boundaryConditionV(self, gpu_stream, hv0):
        """
        Updates hv according to periodic boundary conditions
        """

        # Start with fixing the potential sponge
        self.callSpongeNS(gpu_stream, hv0, 0, 1)
        
        if (self.boundary_conditions.north == 1 and \
            self.boundary_conditions.south == 1):
            if (self.ny_halo > self.ny):
                print("Closed north-south boundary, but ny_halo > ny")
                return
            self.closedBoundaryVKernel.prepared_async_call(self.global_size, self.local_size, gpu_stream, \
                        self.nx, self.ny, \
                        self.nx_halo, self.ny_halo, \
                        hv0.data.gpudata, hv0.pitch)

        elif (self.boundary_conditions.north == 2):
            # Periodic
            ## Currently, this only works with 0 ghost cells:
            assert(hv0.ny == hv0.ny_halo), \
                "The current data does not have zero ghost cells"
        
            ## Call kernel that swaps the boundaries.
            #print("Periodic boundary conditions")
            if self.firstV:
                print("Periodic boundary conditions - V")
                self.firstV = False
                print("[nx, ny, nx_halo, ny_halo]")
                print([self.nx, self.ny, self.nx_halo, self.ny_halo])
                
            self.periodicBoundaryVKernel.prepared_async_call(self.global_size, self.local_size, gpu_stream, \
                        self.nx, self.ny, \
                        self.nx_halo, self.ny_halo, \
                        hv0.data.gpudata, hv0.pitch)
        

        # Nonthereless: If there are ghost cells in east-west direction, update them!
        # TODO: Generalize to both ghost_east and ghost_west
        # Updating eastern ghost cells
        if (self.nx_halo > self.nx):
            if self.firstGhostV:
                print("Updating V ghosts in east-west")
                self.firstGhostV = False
            
            self.updateGhostCellsVKernel.prepared_async_call(self.global_size, self.local_size, gpu_stream, \
                        self.nx, self.ny, \
                        self.nx_halo, self.ny_halo, \
                        hv0.data.gpudata, hv0.pitch)
        

    def boundaryConditionEta(self, gpu_stream, eta0):
        """
        Updates eta boundary conditions (ghost cells)
        """
        # Start with fixing the potential sponge
        self.callSpongeNS(gpu_stream, eta0, 0, 0)
        
        if (self.boundary_conditions.north == 2 or
            self.boundary_conditions.east == 2):
            # Periodic

            if self.firstGhostEta:
                print("Updating eta ghosts")
                self.firstGhostEta = False
                print("[nx, ny, nx_halo, ny_halo]")
                print([self.nx, self.ny, self.nx_halo, self.ny_halo])
                    
            ## Call kernel that swaps the boundaries.
            #print("Periodic boundary conditions")
            self.periodicBoundaryEtaKernel.prepared_async_call(self.global_size, self.local_size, gpu_stream, \
                        self.nx, self.ny, \
                        self.nx_halo, self.ny_halo, \
                        eta0.data.gpudata, eta0.pitch)

        
    def callSpongeNS(self, gpu_stream, data, staggered_x, staggered_y):
        staggered_x_int32 = np.int32(staggered_x)
        staggered_y_int32 = np.int32(staggered_y)
        
        if (self.bc_north == 3) or (self.bc_south ==3):
            self.boundary_flowRelaxationScheme_NS.prepared_async_call( \
                self.global_size, self.local_size, self.gpu_stream, \
                self.nx, self.ny, \
                self.ghostsX, self.ghostsY, \
                staggered_x_int32, staggered_y_int32, \
                self.boundary_conditions.spongeCells[0], \
                self.boundary_conditions.spongeCells[2], \
                self.bc_north, self.bc_south, \
                data.data.gpudata, data.pitch)

        if (self.bc_east == 3) or (self.bc_west == 3):
            self.boundary_flowRelaxationScheme_EW.prepared_async_call( \
                self.global_size, self.local_size, self.gpu_stream, \
                self.nx, self.ny, \
                self.ghostsX, self.ghostsY, \
                staggered_x_int32, staggered_y_int32, \
                self.boundary_conditions.spongeCells[1], \
                self.boundary_conditions.spongeCells[3], \
                self.bc_east, self.bc_west, \
                data.data.gpudata, data.pitch)
