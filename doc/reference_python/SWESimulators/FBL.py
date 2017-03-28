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
import Common
        
        
        
        
        
        




reload(Common)






"""
Class that solves the SW equations using the Forward-Backward linear scheme
"""
class FBL:

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
    wind_stress: Wind stress parameters
    """
    def __init__(self, \
                 cl_ctx, \
                 H, eta0, hu0, hv0, \
                 nx, ny, \
                 dx, dy, dt, \
                 g, f, r, \
                 wind_stress=Common.WindStressParams(), \
                 boundary_conditions=Common.BoundaryConditions(), \
                 block_width=16, block_height=16):
        reload(Common)
        self.cl_ctx = cl_ctx
        self.boundary_conditions = boundary_conditions
        
        #Create an OpenCL command queue
        self.cl_queue = cl.CommandQueue(self.cl_ctx)

        #Get kernels
        self.u_kernel = Common.get_kernel(self.cl_ctx, "FBL_U_kernel.opencl", block_width, block_height)
        self.v_kernel = Common.get_kernel(self.cl_ctx, "FBL_V_kernel.opencl", block_width, block_height)
        self.eta_kernel = Common.get_kernel(self.cl_ctx, "FBL_eta_kernel.opencl", block_width, block_height)
                
        #Create data by uploading to device
        ghost_cells_x = 0
        ghost_cells_y = 0
        self.asym_ghost_cells = [0, 0, 0, 0] # [N, E, S, W]
        if not self.boundary_conditions.isDefault():
            if self.boundary_conditions.north == 2:
                self.asym_ghost_cells[0] = 1
            if self.boundary_conditions.east == 2:
                self.asym_ghost_cells[1] = 1
            
        self.H = Common.OpenCLArray2D(self.cl_ctx, nx, ny, ghost_cells_x, ghost_cells_y, H, self.asym_ghost_cells)
        self.cl_data = Common.SWEDataArakawaC(self.cl_ctx, nx, ny, ghost_cells_x, ghost_cells_y, eta0, hu0, hv0, self.asym_ghost_cells)
        
        #Save input parameters
        #Notice that we need to specify them in the correct dataformat for the
        #OpenCL kernel
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.nx_halo = np.int32(nx + self.asym_ghost_cells[1] + self.asym_ghost_cells[3])
        self.ny_halo = np.int32(ny + self.asym_ghost_cells[0] + self.asym_ghost_cells[2])
        self.dx = np.float32(dx)
        self.dy = np.float32(dy)
        self.dt = np.float32(dt)
        self.g = np.float32(g)
        self.f = np.float32(f)
        self.r = np.float32(r)
        self.wind_stress = wind_stress
        
        #Initialize time
        self.t = np.float32(0.0)
        
        #Compute kernel launch parameters
        self.local_size = (block_width, block_height) # WARNING::: MUST MATCH defines of block_width/height in kernels!
        self.global_size =  ( \
                       int(np.ceil(self.nx_halo / float(self.local_size[0])) * self.local_size[0]), \
                       int(np.ceil(self.ny_halo / float(self.local_size[1])) * self.local_size[1]) \
                      ) 
        #print("FBL.local_size: " + str(self.local_size))
        #print("FBL.global_size: " + str(self.global_size))

        self.bc_kernel = FBL_periodic_boundary(self.cl_ctx, \
                                               self.nx, \
                                               self.ny, \
                                               self.boundary_conditions, \
                                               self.asym_ghost_cells
        )

        self.totalNumIterations = 0
        
    
    """
    Function which steps n timesteps
    """
    def step(self, t_end=0.0):
        n = int(t_end / self.dt + 1)
        
        ## Populate all ghost cells before we start
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

            self.u_kernel.computeUKernel(self.cl_queue, self.global_size, self.local_size, \
                    self.nx_halo, self.ny, \
                    self.dx, self.dy, local_dt, \
                    self.g, self.f, self.r, \
                    self.H.data, self.H.pitch, \
                    self.cl_data.hu0.data, self.cl_data.hu0.pitch, \
                    self.cl_data.hv0.data, self.cl_data.hv0.pitch, \
                    self.cl_data.h0.data, self.cl_data.h0.pitch, \
                    self.wind_stress.type, \
                    self.wind_stress.tau0, self.wind_stress.rho, self.wind_stress.alpha, self.wind_stress.xm, self.wind_stress.Rc, \
                    self.wind_stress.x0, self.wind_stress.y0, \
                    self.wind_stress.u0, self.wind_stress.v0, \
                    self.t)

            # Fix U boundary
            self.bc_kernel.boundaryConditionU(self.cl_queue, self.cl_data.hu0)
            
            self.v_kernel.computeVKernel(self.cl_queue, self.global_size, self.local_size, \
                    self.nx, self.ny_halo, \
                    self.dx, self.dy, local_dt, \
                    self.g, self.f, self.r, \
                    self.H.data, self.H.pitch, \
                    self.cl_data.hu0.data, self.cl_data.hu0.pitch, \
                    self.cl_data.hv0.data, self.cl_data.hv0.pitch, \
                    self.cl_data.h0.data, self.cl_data.h0.pitch, \
                    self.wind_stress.type, \
                    self.wind_stress.tau0, self.wind_stress.rho, self.wind_stress.alpha, self.wind_stress.xm, self.wind_stress.Rc, \
                    self.wind_stress.x0, self.wind_stress.y0, \
                    self.wind_stress.u0, self.wind_stress.v0, \
                    self.t)

            # Fix V boundary
            self.bc_kernel.boundaryConditionV(self.cl_queue, self.cl_data.hv0)
            
            self.eta_kernel.computeEtaKernel(self.cl_queue, self.global_size, self.local_size, \
                    self.nx, self.ny, \
                    self.dx, self.dy, local_dt, \
                    self.g, self.f, self.r, \
                    self.H.data, self.H.pitch, \
                    self.cl_data.hu0.data, self.cl_data.hu0.pitch, \
                    self.cl_data.hv0.data, self.cl_data.hv0.pitch, \
                    self.cl_data.h0.data, self.cl_data.h0.pitch)

            self.bc_kernel.boundaryConditionEta(self.cl_queue, self.cl_data.h0)
   
            self.t += local_dt
            self.totalNumIterations += 1
            

        return self.t
    
    
    
    
    def download(self):
        return self.cl_data.download(self.cl_queue)


        
        





class FBL_periodic_boundary:
    def __init__(self, cl_ctx, nx, ny, \
                 boundary_conditions, asym_ghost_cells, \
                 block_width=16, block_height=16 ):

        self.cl_ctx = cl_ctx
        self.boundary_conditions = boundary_conditions
        self.asym_ghost_cells = asym_ghost_cells
        self.ghostsX = self.asym_ghost_cells[1] + self.asym_ghost_cells[3]
        self.ghostsY = self.asym_ghost_cells[0] + self.asym_ghost_cells[2]
        
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
            = Common.get_kernel(self.cl_ctx,\
            "FBL_periodic_boundary.opencl", block_width, block_height)

         #Compute kernel launch parameters
        self.local_size = (block_width, block_height) # WARNING::: MUST MATCH defines of block_width/height in kernels!
        self.global_size = ( \
                int(np.ceil(self.nx_halo+1 / float(self.local_size[0])) * self.local_size[0]), \
                int(np.ceil(self.ny_halo+1 / float(self.local_size[1])) * self.local_size[1]) )

    

    """
    Updates hu according periodic boundary conditions
    """
    def boundaryConditionU(self, cl_queue, hu0):
        if (self.boundary_conditions.east == 1 and \
            self.boundary_conditions.west == 1):
            if (self.nx_halo > self.nx):
                print("Closed east-west boundary, but nx_halo > nx")
                return
            self.periodicBoundaryKernel.closedBoundaryUKernel(cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        self.nx_halo, self.ny_halo, \
                        hu0.data, hu0.pitch)

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
                
            self.periodicBoundaryKernel.periodicBoundaryUKernel(cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        self.nx_halo, self.ny_halo, \
                        hu0.data, hu0.pitch)
        else:
            assert(False), 'Numerical sponge not yet supported'

        # Nonthereless: If there are ghost cells in north-south direction, update them!
        # TODO: Generalize to both ghost_north and ghost_south
        # Updating northern ghost cells
        if (self.ny_halo > self.ny):
            if self.firstGhostU:
                print("Updating U ghosts in north-south")
                self.firstGhostU = False
            
            self.periodicBoundaryKernel.updateGhostCellsUKernel(cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        self.nx_halo, self.ny_halo, \
                        hu0.data, hu0.pitch)
            

    """
    Updates hv according to periodic boundary conditions
    """
    def boundaryConditionV(self, cl_queue, hv0):
        if (self.boundary_conditions.north == 1 and \
            self.boundary_conditions.south == 1):
            if (self.ny_halo > self.ny):
                print("Closed north-south boundary, but ny_halo > ny")
                return
            self.periodicBoundaryKernel.closedBoundaryVKernel(cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        self.nx_halo, self.ny_halo, \
                        hv0.data, hv0.pitch)

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
                
            self.periodicBoundaryKernel.periodicBoundaryVKernel(cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        self.nx_halo, self.ny_halo, \
                        hv0.data, hv0.pitch)
        else:
            assert(False), 'Numerical sponge not yet supported'

        # Nonthereless: If there are ghost cells in east-west direction, update them!
        # TODO: Generalize to both ghost_east and ghost_west
        # Updating eastern ghost cells
        if (self.nx_halo > self.nx):
            if self.firstGhostV:
                print("Updating V ghosts in east-west")
                self.firstGhostV = False
            
            self.periodicBoundaryKernel.updateGhostCellsVKernel(cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        self.nx_halo, self.ny_halo, \
                        hv0.data, hv0.pitch)
        

    """
    Updates eta boundary conditions (ghost cells)
    """
    def boundaryConditionEta(self, cl_queue, eta0):
       
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
            self.periodicBoundaryKernel.periodicBoundaryEtaKernel(cl_queue, self.global_size, self.local_size, \
                        self.nx, self.ny, \
                        self.nx_halo, self.ny_halo, \
                        eta0.data, eta0.pitch)
        
