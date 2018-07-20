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
import gc

import Common, SimWriter, SimReader
import Simulator
import WindStress
import OceanStateNoise


class CDKLM16(Simulator.Simulator):
    """
    Class that solves the SW equations using the Coriolis well balanced reconstruction scheme, as given by the publication of Chertock, Dudzinski, Kurganov and Lukacova-Medvidova (CDFLM) in 2016.
    """

    def __init__(self, \
                 cl_ctx, \
                 eta0, hu0, hv0, Hi, \
                 nx, ny, \
                 dx, dy, dt, \
                 g, f, r, \
                 t=0.0, \
                 theta=1.3, rk_order=2, \
                 coriolis_beta=0.0, \
                 y_zero_reference_cell = 0, \
                 max_wind_direction_perturbation = 0, \
                 wind_stress=WindStress.NoWindStress(), \
                 boundary_conditions=Common.BoundaryConditions(), \
                 small_scale_perturbation=False, \
                 small_scale_perturbation_amplitude=None, \
                 h0AsWaterElevation=False, \
                 reportGeostrophicEquilibrium=False, \
                 write_netcdf=False, \
                 ignore_ghostcells=False, \
                 offset_x=0, offset_y=0, \
                 block_width=32, block_height=4):
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
        rk_order: Order of Runge Kutta method {1,2*,3}
        coriolis_beta: Coriolis linear factor -> f = f + beta*(y-y_0)
        y_zero_reference_cell: The cell representing y_0 in the above, defined as the lower face of the cell .
        max_wind_direction_perturbation: Large-scale model error emulation by per-time-step perturbation of wind direction by +/- max_wind_direction_perturbation (degrees)
        wind_stress: Wind stress parameters
        boundary_conditions: Boundary condition object
        h0AsWaterElevation: True if h0 is described by the surface elevation, and false if h0 is described by water depth
        reportGeostrophicEquilibrium: Calculate the Geostrophic Equilibrium variables for each superstep
        write_netcdf: Write the results after each superstep to a netCDF file
        """
               
        

        ## After changing from (h, B) to (eta, H), several of the simulator settings used are wrong. This check will help detect that.
        if ( np.sum(eta0 - Hi[:-1, :-1] > 0) > nx):
            assert(False), "It seems you are using water depth/elevation h and bottom topography B, while you should use water level eta and equillibrium depth H."
        
        assert( rk_order < 4 or rk_order > 0 ), "Only 1st, 2nd and 3rd order Runge Kutta supported"

        if (rk_order == 3):
            assert(r == 0.0), "3rd order Runge Kutta supported only without friction"
        
        # Sort out internally represented ghost_cells in the presence of given
        # boundary conditions
        ghost_cells_x = 2
        ghost_cells_y = 2
        y_zero_reference_cell = 2 + y_zero_reference_cell
        
        # Boundary conditions
        self.boundary_conditions = boundary_conditions
        if (boundary_conditions.isSponge()):
            nx = nx + boundary_conditions.spongeCells[1] + boundary_conditions.spongeCells[3] - 2*ghost_cells_x
            ny = ny + boundary_conditions.spongeCells[0] + boundary_conditions.spongeCells[2] - 2*ghost_cells_y
            y_zero_reference_cell = boundary_conditions.spongeCells[2] + y_zero_reference_cell
        
        A = None
        self.max_wind_direction_perturbation = max_wind_direction_perturbation
        super(CDKLM16, self).__init__(cl_ctx, \
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
        self.interior_domain_indices = np.array([-2,-2,2,2])
        self._set_interior_domain_from_sponge_cells()
        
        #Get kernels
        self.kernel = Common.get_kernel(self.cl_ctx, "CDKLM16_kernel.opencl", block_width, block_height)
        
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
        self.h0AsWaterElevation = h0AsWaterElevation
        if self.h0AsWaterElevation:
            self.bathymetry.waterElevationToDepth(self.cl_data.h0)
        
        self.constant_equilibrium_depth = np.max(Hi)
        
        self.bc_kernel = Common.BoundaryConditionsArakawaA(self.cl_ctx, \
                                                           self.nx, \
                                                           self.ny, \
                                                           ghost_cells_x, \
                                                           ghost_cells_y, \
                                                           self.boundary_conditions, \
        )

        # Small scale perturbation:
        self.small_scale_perturbation = small_scale_perturbation
        self.small_scale_model_error = None
        if small_scale_perturbation:
            if small_scale_perturbation_amplitude is None:
                self.small_scale_model_error = OceanStateNoise.OceanStateNoise.fromsim(self)
            else:
                self.small_scale_model_error = OceanStateNoise.OceanStateNoise.fromsim(self, soar_q0=small_scale_perturbation_amplitude)
        
        if self.write_netcdf:
            self.sim_writer = SimWriter.SimNetCDFWriter(self, ignore_ghostcells=self.ignore_ghostcells, \
                                    offset_x=self.offset_x, offset_y=self.offset_y)

    
    def cleanUp(self):
        """
        Clean up function
        """
        self.closeNetCDF()
        
        self.cl_data.release()
        
        if self.small_scale_model_error is not None:
            self.small_scale_model_error.cleanUp()
        
        self.geoEq_uxpvy.release()
        self.geoEq_Kx.release()
        self.geoEq_Ly.release()
        self.bathymetry.release()
        self.h0AsWaterElevation = False # Quick fix to stop waterDepthToElevation conversion
        gc.collect()
           
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
                 eta0, hu0, hv0, \
                 Hi, \
                 nx, ny, \
                 dx, dy, dt, \
                 g, f, r, \
                 t=time0, \
                 theta=minmodTheta, rk_order=timeIntegrator, \
                 coriolis_beta=beta, \
                 y_zero_reference_cell = y_zero_reference_cell, \
                 wind_stress=wind, \
                 boundary_conditions=boundaryConditions, \
                 write_netcdf=cont_write_netcdf)
    
    
    
    def step(self, t_end=0.0, apply_stochastic_term=True):
        """
        Function which steps n timesteps.
        apply_stochastic_term: Boolean value for whether the stochastic
            perturbation (if any) should be applied.
        """
        n = int(t_end / self.dt + 1)

        if self.t == 0:
            self.bc_kernel.boundaryCondition(self.cl_queue, \
                self.cl_data.h0, self.cl_data.hu0, self.cl_data.hv0)
        
        for i in range(0, n):
            # Get new random wind direction (emulationg large-scale model error)
            if(self.max_wind_direction_perturbation > 0.0 and self.wind_stress.type() == 1):
                # max perturbation +/- max_wind_direction_perturbation deg within original wind direction (at t=0)
                perturbation = 2.0*(np.random.rand()-0.5) * self.max_wind_direction_perturbation;
                new_wind_stress = WindStress.GenericUniformWindStress( \
                    rho_air=self.wind_stress.rho_air, \
                    wind_speed=self.wind_stress.wind_speed, \
                    wind_direction=self.wind_stress.wind_direction + perturbation)
                # Upload new wind stress params to device
                cl.enqueue_copy(self.cl_queue, self.wind_stress_dev, new_wind_stress.tostruct())
                
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
            
            # Perturb ocean state with model error
            if self.small_scale_perturbation and apply_stochastic_term:
                self.small_scale_model_error.perturbSim(self)
            
            # Evolve drifters
            if self.hasDrifters:
                self.drifters.drift(self.cl_data.h0, self.cl_data.hu0, \
                                    self.cl_data.hv0, \
                                    np.float32(self.constant_equilibrium_depth), \
                                    self.nx, self.ny, self.dx, self.dy, \
                                    local_dt, \
                                    np.int32(2), np.int32(2))
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
                           self.wind_stress_dev, \
                           self.t, \
                           self.boundary_conditions.north, self.boundary_conditions.east, self.boundary_conditions.south, self.boundary_conditions.west, \
                           self.reportGeostrophicEquilibrium, \
                           self.geoEq_uxpvy.data, self.geoEq_uxpvy.pitch, \
                           self.geoEq_Kx.data, self.geoEq_Kx.pitch, \
                           self.geoEq_Ly.data, self.geoEq_Ly.pitch )
            
    
    def perturbState(self, q0_scale=None):
        self.small_scale_model_error.perturbSim(self, q0_scale=q0_scale)
    
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
