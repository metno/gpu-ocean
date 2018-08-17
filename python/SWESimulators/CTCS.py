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
import gc

import Common, SimWriter, SimReader
import Simulator
import WindStress

import time

class CTCS(Simulator.Simulator):
    """
    Class that solves the SW equations using the Centered in time centered in space scheme
    """

    def __init__(self, \
                 gpu_ctx, \
                 H, eta0, hu0, hv0, \
                 nx, ny, \
                 dx, dy, dt, \
                 g, f, r, A=0.0, \
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
        A: Eddy viscosity coefficient (O(dx))
        t: Start simulation at time t
        coriolis_beta: Coriolis linear factor -> f = f + beta*(y-y_0)
        y_zero_reference_cell: The cell representing y_0 in the above, defined as the lower face of the cell .
        wind_stress: Wind stress parameters
        boundary_conditions: Boundary condition object
        write_netcdf: Write the results after each superstep to a netCDF file
        """
        
       
        
        
        # Sort out internally represented ghost_cells in the presence of given
        # boundary conditions
        halo_x = 1
        halo_y = 1
        ghost_cells_x = 1
        ghost_cells_y = 1
        y_zero_reference_cell = y_zero_reference_cell + 1
        
        self.boundary_conditions = boundary_conditions
        if boundary_conditions.isSponge():
            nx = nx + boundary_conditions.spongeCells[1] + boundary_conditions.spongeCells[3] - 2*ghost_cells_x
            ny = ny + boundary_conditions.spongeCells[0] + boundary_conditions.spongeCells[2] - 2*ghost_cells_y
            y_zero_reference_cell = y_zero_reference_cell + boundary_conditions.spongeCells[2]

        # self.<parameters> are sat in parent constructor:
        rk_order = None
        theta = None
        super(CTCS, self).__init__(gpu_ctx, \
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
            
        # Index range for interior domain (north, east, south, west)
        # so that interior domain of eta is
        # eta[self.interior_domain_indices[2]:self.interior_domain_indices[0], \
        #     self.interior_domain_indices[3]:self.interior_domain_indices[1] ]
        self.interior_domain_indices = np.array([-1,-1,1,1])
        self._set_interior_domain_from_sponge_cells()

        #Get kernels
        self.u_kernel = gpu_ctx.get_kernel("CTCS_U_kernel.cu", block_width, block_height)
        self.v_kernel = gpu_ctx.get_kernel("CTCS_V_kernel.cu", block_width, block_height)
        self.eta_kernel = gpu_ctx.get_kernel("CTCS_eta_kernel.cu", block_width, block_height)
        
        # Get CUDA functions and define data types for prepared_{async_}call()
        self.computeUKernel = self.u_kernel.get_function("computeUKernel")
        self.computeUKernel.prepare("iiiifffffffffPiPiPiPiPiPf")
        self.computeVKernel = self.v_kernel.get_function("computeVKernel")
        self.computeVKernel.prepare("iiiifffffffffPiPiPiPiPiPf")
        self.computeEtaKernel = self.eta_kernel.get_function("computeEtaKernel")
        self.computeEtaKernel.prepare("iiffffffffPiPiPi")
        
        #Create data by uploading to device     
        self.H = Common.CUDAArray2D(self.gpu_stream, nx, ny, halo_x, halo_y, H)
        self.gpu_data = Common.SWEDataArakawaC(self.gpu_stream, nx, ny, halo_x, halo_y, eta0, hu0, hv0)
        
        # Global size needs to be larger than the default from parent.__init__
        self.global_size = ( \
                       int(np.ceil((self.nx+2*halo_x) / float(self.local_size[0]))), \
                       int(np.ceil((self.ny+2*halo_y) / float(self.local_size[1]))) \
                      ) 
    
        self.bc_kernel = CTCS_boundary_condition(gpu_ctx, \
                                                 self.nx, \
                                                 self.ny, \
                                                 self.boundary_conditions, \
                                                 halo_x, halo_y \
        )
        
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
        self.gpu_ctx = None
        gc.collect()
    
    def step(self, t_end=0.0):
        """
        Function which steps n timesteps
        """
        n = int(t_end / self.dt + 1)
        if self.t == 0:
            #print "N: ", n
            #print "np.float(min(self.dt, t_end-n*self.dt))", np.float32(min(self.dt, t_end-(n-1)*self.dt))
        
            # Ensure that the boundary conditions are satisfied before starting simulation
            self.bc_kernel.boundaryConditionEta(self.gpu_stream, self.gpu_data.h0)
            self.bc_kernel.boundaryConditionU(self.gpu_stream, self.gpu_data.hu0)
            self.bc_kernel.boundaryConditionV(self.gpu_stream, self.gpu_data.hv0)
            self.bc_kernel.boundaryConditionEta(self.gpu_stream, self.gpu_data.h1)
            self.bc_kernel.boundaryConditionU(self.gpu_stream, self.gpu_data.hu1)
            self.bc_kernel.boundaryConditionV(self.gpu_stream, self.gpu_data.hv1)
        
        for i in range(0, n):
            #Notation: 
            # gpu_data.u0 => U^{n-1} before U kernel, U^{n+1} after U kernel
            # gpu_data.u1 => U^{n}
            # When we call gpu_data.swap(), we swap these, so that
            # gpu_data.u0 => U^{n}
            # gpu_data.u1 => U^{n+1} (U kernel has been executed)
            # Now we are ready for the next time step
            
            local_dt = np.float32(min(self.dt, t_end-i*self.dt))
            
            if (local_dt <= 0.0):
                break
            
            self.computeEtaKernel.prepared_async_call(self.global_size, self.local_size, self.gpu_stream, \
                    self.nx, self.ny, \
                    self.dx, self.dy, local_dt, \
                    self.g, self.f, self.coriolis_beta, self.y_zero_reference_cell, self.r, \
                    self.gpu_data.h0.data.gpudata, self.gpu_data.h0.pitch,     # eta^{n-1} => eta^{n+1} \
                    self.gpu_data.hu1.data.gpudata, self.gpu_data.hu1.pitch,   # U^{n} \
                    self.gpu_data.hv1.data.gpudata, self.gpu_data.hv1.pitch)   # V^{n}

            self.bc_kernel.boundaryConditionEta(self.gpu_stream, self.gpu_data.h0)
            
            self.computeUKernel.prepared_async_call(self.global_size, self.local_size, self.gpu_stream, \
                    self.nx, self.ny, \
                    self.boundary_conditions.east, self.boundary_conditions.west, \
                    self.dx, self.dy, local_dt, \
                    self.g, self.f, self.coriolis_beta, self.y_zero_reference_cell, \
                    self.r, self.A,\
                    self.H.data.gpudata, self.H.pitch, \
                    self.gpu_data.h1.data.gpudata, self.gpu_data.h1.pitch,      # eta^{n} \
                    self.gpu_data.hu0.data.gpudata, self.gpu_data.hu0.pitch,    # U^{n-1} => U^{n+1} \
                    self.gpu_data.hu1.data.gpudata, self.gpu_data.hu1.pitch,    # U^{n} \
                    self.gpu_data.hv1.data.gpudata, self.gpu_data.hv1.pitch,    # V^{n} \
                    self.wind_stress_dev, \
                    self.t)

            self.bc_kernel.boundaryConditionU(self.gpu_stream, self.gpu_data.hu0)
            
            self.computeVKernel.prepared_async_call(self.global_size, self.local_size, self.gpu_stream, \
                    self.nx, self.ny, \
                    self.boundary_conditions.north, self.boundary_conditions.south, \
                    self.dx, self.dy, local_dt, \
                    self.g, self.f, self.coriolis_beta, self.y_zero_reference_cell, \
                    self.r, self.A,\
                    self.H.data.gpudata, self.H.pitch, \
                    self.gpu_data.h1.data.gpudata, self.gpu_data.h1.pitch,     # eta^{n} \
                    self.gpu_data.hu1.data.gpudata, self.gpu_data.hu1.pitch,   # U^{n} \
                    self.gpu_data.hv0.data.gpudata, self.gpu_data.hv0.pitch,   # V^{n-1} => V^{n+1} \
                    self.gpu_data.hv1.data.gpudata, self.gpu_data.hv1.pitch,   # V^{n} \
                    self.wind_stress_dev, \
                    self.t)

            self.bc_kernel.boundaryConditionV(self.gpu_stream, self.gpu_data.hv0)
            
            #After the kernels, swap the data pointers
            self.gpu_data.swap()
            
            self.t += local_dt
        
        if self.write_netcdf:
            self.sim_writer.writeTimestep(self)
            
        return self.t

        
class CTCS_boundary_condition:
    def __init__(self, gpu_ctx, nx, ny, \
                 boundary_conditions, halo_x, halo_y, \
                 block_width=16, block_height=16):

        self.boundary_conditions = boundary_conditions

        self.bc_north = np.int32(boundary_conditions.north)
        self.bc_east  = np.int32(boundary_conditions.east)
        self.bc_south = np.int32(boundary_conditions.south)
        self.bc_west  = np.int32(boundary_conditions.west)
        
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.halo_x = np.int32(halo_x)
        self.halo_y = np.int32(halo_y)
        self.nx_halo = np.int32(nx + 2*halo_x) 
        self.ny_halo = np.int32(ny + 2*halo_y)

        # Load kernel for periodic boundary
        self.boundaryKernels = gpu_ctx.get_kernel("CTCS_boundary.cu", block_width, block_height)
        
        # Get CUDA functions and define data types for prepared_{async_}call()
        self.boundaryUKernel_NS = self.boundaryKernels.get_function("boundaryUKernel_NS")
        self.boundaryUKernel_NS.prepare("iiiiiiPi")
        self.boundaryUKernel_EW = self.boundaryKernels.get_function("boundaryUKernel_EW")
        self.boundaryUKernel_EW.prepare("iiiiiiPi")
        self.boundaryVKernel_NS = self.boundaryKernels.get_function("boundaryVKernel_NS")
        self.boundaryVKernel_NS.prepare("iiiiiiPi")
        self.boundaryVKernel_EW = self.boundaryKernels.get_function("boundaryVKernel_EW")
        self.boundaryVKernel_EW.prepare("iiiiiiPi")
        self.boundaryEtaKernel_NS = self.boundaryKernels.get_function("boundaryEtaKernel_NS")
        self.boundaryEtaKernel_NS.prepare("iiiiiiPi")
        self.boundaryEtaKernel_EW = self.boundaryKernels.get_function("boundaryEtaKernel_EW")
        self.boundaryEtaKernel_EW.prepare("iiiiiiPi")
        self.boundary_linearInterpol_NS = self.boundaryKernels.get_function("boundary_linearInterpol_NS")
        self.boundary_linearInterpol_NS.prepare("iiiiiiiiiiPi")
        self.boundary_linearInterpol_EW = self.boundaryKernels.get_function("boundary_linearInterpol_EW")
        self.boundary_linearInterpol_EW.prepare("iiiiiiiiiiPi")
        self.boundary_flowRelaxationScheme_NS = self.boundaryKernels.get_function("boundary_flowRelaxationScheme_NS")
        self.boundary_flowRelaxationScheme_NS.prepare("iiiiiiiiiiPi")
        self.boundary_flowRelaxationScheme_EW = self.boundaryKernels.get_function("boundary_flowRelaxationScheme_EW")
        self.boundary_flowRelaxationScheme_EW.prepare("iiiiiiiiiiPi")

        # Set kernel launch parameters
        self.local_size = (block_width, block_height, 1)
        self.global_size = ( \
                             int(np.ceil((self.nx_halo + 1)/float(self.local_size[0]))), \
                             int(np.ceil((self.ny_halo + 1)/float(self.local_size[1]))) )

        
       
    def boundaryConditionU(self, gpu_stream, hu0):
        """
        Updates hu according periodic boundary conditions
        """
       
        if (self.bc_north < 3) or (self.bc_south < 3):
            self.boundaryUKernel_NS.prepared_async_call( \
                self.global_size, self.local_size, gpu_stream, \
                self.nx, self.ny, \
                self.halo_x, self.halo_y, \
                self.bc_north, self.bc_south, \
                hu0.data.gpudata, hu0.pitch)
        #self.callSpongeNS(gpu_stream, hu0, 0, 0)
        self.callSpongeNS(gpu_stream, hu0, 1, 0)
        
        if (self.bc_east < 3) or (self.bc_west < 3):
            self.boundaryUKernel_EW.prepared_async_call( \
                self.global_size, self.local_size, gpu_stream, \
                self.nx, self.ny, \
                self.halo_x, self.halo_y, \
                self.bc_east, self.bc_west, \
                hu0.data.gpudata, hu0.pitch)
        self.callSpongeEW(gpu_stream, hu0, 1, 0)
        #self.callSpongeEW(gpu_stream, hu0, 0, 0)
        
        
        
    def boundaryConditionV(self, gpu_stream, hv0):
        """
        Updates hv according to periodic boundary conditions
        """

        if (self.bc_north < 3) or (self.bc_south < 3):
            self.boundaryVKernel_NS.prepared_async_call( \
                self.global_size, self.local_size, gpu_stream, \
                self.nx, self.ny, \
                self.halo_x, self.halo_y, \
                self.bc_north, self.bc_south, \
                hv0.data.gpudata, hv0.pitch)
        self.callSpongeNS(gpu_stream, hv0, 0, 1)
        #self.callSpongeNS(gpu_stream, hv0, 0, 0)
        
        if (self.bc_east < 3) or (self.bc_west < 3):
            self.boundaryVKernel_EW.prepared_async_call( \
                self.global_size, self.local_size, gpu_stream, \
                self.nx, self.ny, \
                self.halo_x, self.halo_y, \
                self.bc_east, self.bc_west, \
                hv0.data.gpudata, hv0.pitch)
        self.callSpongeEW(gpu_stream, hv0, 0, 1)
        #self.callSpongeEW(gpu_stream, hv0, 0, 0)

    def boundaryConditionEta(self, gpu_stream, eta0):
        """
        Updates eta boundary conditions (ghost cells)
        """

        if (self.bc_north < 3) or (self.bc_south < 3):
            self.boundaryEtaKernel_NS.prepared_async_call( \
                self.global_size, self.local_size, gpu_stream, \
                self.nx, self.ny, \
                self.halo_x, self.halo_y, \
                self.bc_north, self.bc_south, \
                eta0.data.gpudata, eta0.pitch)
        self.callSpongeNS(gpu_stream, eta0, 0, 0)
            
        if (self.bc_east < 3) or (self.bc_west < 3):
            self.boundaryEtaKernel_EW.prepared_async_call( \
                self.global_size, self.local_size, gpu_stream, \
                self.nx, self.ny, \
                self.halo_x, self.halo_y, \
                self.bc_east, self.bc_west, \
                eta0.data.gpudata, eta0.pitch)
        self.callSpongeEW(gpu_stream, eta0, 0, 0)
            
              
    def callSpongeNS(self, gpu_stream, data, staggered_x, staggered_y):
        """
        Call othe approporary sponge-like boundary condition with the given data
        """
        staggered_x_int32 = np.int32(staggered_x)
        staggered_y_int32 = np.int32(staggered_y)

        #print "callSpongeNS"
        if (self.bc_north == 3) or (self.bc_south == 3):
            self.boundary_flowRelaxationScheme_NS.prepared_async_call( \
                self.global_size, self.local_size, gpu_stream, \
                self.nx, self.ny, \
                self.halo_x, self.halo_y, \
                staggered_x_int32, staggered_y_int32, \
                self.boundary_conditions.spongeCells[0], \
                self.boundary_conditions.spongeCells[2], \
                self.bc_north, self.bc_south, \
                data.data.gpudata, data.pitch) 
        if (self.bc_north == 4 ) or (self.bc_south == 4):
            self.boundary_linearInterpol_NS.prepared_async_call( \
                self.global_size, self.local_size, gpu_stream, \
                self.nx, self.ny, \
                self.halo_x, self.halo_y, \
                staggered_x_int32, staggered_y_int32, \
                self.boundary_conditions.spongeCells[0], \
                self.boundary_conditions.spongeCells[2], \
                self.bc_north, self.bc_south, \
                data.data.gpudata, data.pitch)                                

    def callSpongeEW(self, gpu_stream, data, staggered_x, staggered_y):
        staggered_x_int32 = np.int32(staggered_x)
        staggered_y_int32 = np.int32(staggered_y)

        #print "CallSpongeEW"
        if (self.bc_east == 3) or (self.bc_west == 3):
            self.boundary_flowRelaxationScheme_EW.prepared_async_call( \
                self.global_size, self.local_size, gpu_stream, \
                self.nx, self.ny, \
                self.halo_x, self.halo_y, \
                staggered_x_int32, staggered_y_int32, \
                self.boundary_conditions.spongeCells[1], \
                self.boundary_conditions.spongeCells[3], \
                self.bc_east, self.bc_west, \
                data.data.gpudata, data.pitch)   

        if (self.bc_east == 4 ) or (self.bc_west == 4):
            self.boundary_linearInterpol_EW.prepared_async_call( \
                self.global_size, self.local_size, gpu_stream, \
                self.nx, self.ny, \
                self.halo_x, self.halo_y, \
                staggered_x_int32, staggered_y_int32, \
                self.boundary_conditions.spongeCells[1], \
                self.boundary_conditions.spongeCells[3], \
                self.bc_east, self.bc_west, \
                data.data.gpudata, data.pitch)   
