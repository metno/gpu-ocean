# -*- coding: utf-8 -*-

"""
This python class produces random perturbations that are to be added to 
the ocean state fields in order to generate model error.

Copyright (C) 2018  SINTEF ICT

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


from matplotlib import pyplot as plt
import numpy as np
import pyopencl
import gc

import Common
import FBL, CTCS, CDKLM16, KP07

class OceanStateNoise(object):
    """
    Generating random perturbations for a ocean state.
   
    Perturbation for the surface field, dEta, is produced with a covariance structure according to a SOAR function,
    while dHu and dHv are found by the geostrophic balance to avoid shock solutions.
    """
    
    def __init__(self, cl_ctx, cl_queue,
                 nx, ny, dx, dy,
                 boundaryConditions, staggered,
                 soar_q0=None, soar_L=None,
                 block_width=16, block_height=16):
        
        self.cl_ctx = cl_ctx
        self.cl_queue = cl_queue
        
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.dx = np.float32(dx)
        self.dy = np.float32(dy)
        self.staggered = np.int(0)
        if staggered:
            self.staggered = np.int(1)
            
        # The cutoff parameter is hard-coded.
        # The size of the cutoff determines the computational radius in the
        # SOAR function. Hence, the size of the local memory in the OpenCL 
        # kernels has to be hard-coded.
        self.cutoff = np.int32(2) 
        
        self.periodicNorthSouth = np.int32(boundaryConditions.isPeriodicNorthSouth())
        self.periodicEastWest = np.int32(boundaryConditions.isPeriodicEastWest())
        
        # Size of random field and seed
        self.rand_nx = np.int32(nx + 2*(1+self.cutoff))
        self.rand_ny = np.int32(ny + 2*(1+self.cutoff))
        if self.periodicEastWest:
            self.rand_nx = np.int32(nx)
        if self.periodicNorthSouth:
            self.rand_ny = np.int32(ny)
        self.seed_ny = np.int32(self.rand_ny)
        self.seed_nx = np.int32(self.rand_nx/2) ### WHAT IF rand_nx IS ODD??
        
        # Constants for the SOAR function:
        self.soar_q0 = np.float32(self.dx/100000)
        if soar_q0 is not None:
            self.soar_q0 = np.float32(soar_q0)
            
        self.soar_L = np.float32(0.75*self.dx)
        if soar_L is not None:
            self.soar_L = np.float32(soar_L)
        
        # Generate seed:
        self.floatMax = 2147483648.0
        self.host_seed = np.random.rand(self.seed_ny, self.seed_nx)*self.floatMax
        self.host_seed.astype(np.float32, order='C')
        self.seed = Common.OpenCLArray2D(cl_ctx, self.seed_nx, self.seed_ny, 0, 0, self.host_seed)
        
        # Allocate memory for random numbers
        self.random_numbers_host = np.zeros((self.rand_ny, self.rand_nx), dtype=np.float32, order='C')
        self.random_numbers = Common.OpenCLArray2D(cl_ctx, self.rand_nx, self.rand_ny, 0, 0, self.random_numbers_host)
        
        # Generate kernels
        self.kernels = Common.get_kernel(self.cl_ctx, "ocean_noise.opencl", block_width, block_height)
 
        
        #Compute kernel launch parameters
        self.local_size = (block_width, block_height) 
        self.global_size_random_numbers = ( \
                       int(np.ceil(self.seed_nx / float(self.local_size[0])) * self.local_size[0]), \
                       int(np.ceil(self.seed_ny / float(self.local_size[1])) * self.local_size[1]) \
                                  ) 
        self.global_size_noise = ( \
                       int(np.ceil(self.rand_nx / float(self.local_size[0])) * self.local_size[0]), \
                       int(np.ceil(self.rand_ny / float(self.local_size[1])) * self.local_size[1]) \
                                  ) 
        
        
        
        
    def __del__(self):
        self.cleanUp()
     
    def cleanUp(self):
        self.seed.release()
        self.random_numbers.release()
        gc.collect()
        
    @classmethod
    def fromsim(cls, sim, soar_q0=None, soar_L=None, 
                block_width=16, block_height=16):
        staggered = False
        if isinstance(sim, FBL.FBL) or isinstance(sim, CTCS.CTCS):
            staggered = True
        return cls(sim.cl_ctx, sim.cl_queue,
                   sim.nx, sim.ny, sim.dx, sim.dy,
                   sim.boundary_conditions, staggered,
                   soar_q0=soar_q0, soar_L=soar_L,
                   block_width=block_width, block_height=block_height)
        
        
        
    def getSeed(self):
        return self.seed.download(self.cl_queue)
    
    def getRandomNumbers(self):
        return self.random_numbers.download(self.cl_queue)
    
    def generateNormalDistribution(self):
        self.kernels.normalDistribution(self.cl_queue, 
                                        self.global_size_random_numbers, self.local_size,
                                        self.seed_nx, self.seed_ny,
                                        self.rand_nx,
                                        self.seed.data, self.seed.pitch,
                                        self.random_numbers.data, self.random_numbers.pitch)
        
    def generateUniformDistribution(self):
        # Call kernel -> new random numbers
        self.kernels.uniformDistribution(self.cl_queue, 
                                         self.global_size_random_numbers, self.local_size,
                                         self.seed_nx, self.seed_ny,
                                         self.rand_nx,
                                         self.seed.data, self.seed.pitch,
                                         self.random_numbers.data, self.random_numbers.pitch)
    
    def perturbSim(self, sim):
        assert(isinstance(sim, CDKLM16.CDKLM16))
        gcx = 0 # sim.ghost_cells_x
        gcy = 0 # sim.ghost_cells_y
        
        self.perturbOceanState(sim.cl_data.h0, sim.cl_data.hu0, sim.cl_data.hv0,
                               sim.bathymetry.Bi,
                               sim.f, beta=sim.coriolis_beta, g=sim.g, 
                               y0_reference_cell=sim.y_zero_reference_cell,
                               ghost_cells_x=gcx,
                               ghost_cells_y=gcy)
                               
    
    def perturbOceanState(self, eta, hu, hv, H, f, beta=0.0, g=9.81, 
                          y0_reference_cell=0, ghost_cells_x=0, ghost_cells_y=0):
        """
        Apply the SOAR Q covariance matrix on the random ocean field which is
        added to the provided buffers eta, hu and hv.
        eta: surface deviation - OpenCLArray2D object.
        hu: volume transport in x-direction - OpenCLArray2D object.
        hv: volume transport in y-dirextion - OpenCLArray2D object.
        """
        # Need to update the random field, requiering a global sync
        self.generateNormalDistribution()
        
        # Call applySOARQ_kernel and add to eta
        self.kernels.perturbOcean(self.cl_queue,
                                  self.global_size_noise, self.local_size,
                                  self.nx, self.ny,
                                  self.dx, self.dy,
                                  np.float32(ghost_cells_x), np.float32(ghost_cells_y),
                                  
                                  np.float32(g), np.float32(f),
                                  np.float32(beta), np.float32(y0_reference_cell),
                                  
                                  self.soar_q0, self.soar_L,
                                  self.periodicNorthSouth, self.periodicEastWest,
                                  
                                  self.random_numbers.data, self.random_numbers.pitch,
                                  eta.data, eta.pitch,
                                  hu.data, hu.pitch,
                                  hv.data, hv.pitch,
                                  H.data, H.pitch)
    
    
    
    ##### CPU versions of the above functions ####
    
    def getSeedCPU(self):
        return self.host_seed
    
    def generateNormalDistributionCPU(self):
        self._CPUUpdateRandom(True)
    
    def generateUniformDistributionCPU(self):
        self._CPUUpdateRandom(False)
    
    def getRandomNumbersCPU(self):
        return self.random_numbers_host
    
    def perturbEtaCPU(self, eta, use_existing_GPU_random_numbers=False,
                      ghost_cells_x=0, ghost_cells_y=0):
        """
        Apply the SOAR Q covariance matrix on the random field to add
        a perturbation to the incomming eta buffer.
        eta: numpy array
        """
        # Call CPU utility function
        if use_existing_GPU_random_numbers:
            self.random_numbers_host = self.getRandomNumbers()
        else:
            self.generateNormalDistributionCPU()
        d_eta = self._applyQ_CPU()
        
        interior = [-ghost_cells_y, -ghost_cells_x, ghost_cells_y, ghost_cells_x]
        for i in range(4):
            if interior[i] == 0:
                interior[i] = None
        
        eta[interior[2]:interior[0], interior[3]:interior[1]] = d_eta[1:-1, 1:-1]
    
    def perturbOceanStateCPU(self, eta, hu, hv, H, f,  beta=0.0, g=9.81,
                             ghost_cells_x=0, ghost_cells_y=0,
                             use_existing_GPU_random_numbers=False):
        # Call CPU utility function
        if use_existing_GPU_random_numbers:
            self.random_numbers_host = self.getRandomNumbers()
        else:
            self.generateNormalDistributionCPU()
        d_eta, d_hu, d_hv = self._obtainOceanPerturbations_CPU(H, f, beta, g)
        
        interior = [-ghost_cells_y, -ghost_cells_x, ghost_cells_y, ghost_cells_x]
        for i in range(4):
            if interior[i] == 0:
                interior[i] = None
        
        eta[interior[2]:interior[0], interior[3]:interior[1]] += d_eta[1:-1, 1:-1]
        hu[interior[2]:interior[0], interior[3]:interior[1]] += d_hu
        hv[interior[2]:interior[0], interior[3]:interior[1]] += d_hv
    
    # ------------------------------
    # CPU utility functions:
    # ------------------------------
    
    def _lcg(self, seed):
        seed = ((seed*1103515245) + 12345) % 0x7fffffff
        return seed / 2147483648.0, seed
    
    def _boxMuller(self, seed_in):
        seed = np.long(seed_in)
        u1, seed = self._lcg(seed)
        u2, seed = self._lcg(seed)
        r = np.sqrt(-2.0*np.log(u1))
        theta = 2*np.pi*u2
        n1 = r*np.cos(theta)
        n2 = r*np.sin(theta)
        return n1, n2, np.float32(seed)
    
    def _CPUUpdateRandom(self, normalDist):
        """
        Updating the random number buffer at the CPU.
        normalDist: Boolean parameter. 
            If True, the random numbers are from N(0,1)
            If False, the random numbers are from U[0,1]
        """
        #(ny, nx) = seed.shape
        #(domain_ny, domain_nx) = random.shape
        b_dim_x = self.local_size[0]
        b_dim_y = self.local_size[1]
        blocks_x = self.global_size_random_numbers[0]/b_dim_x
        blocks_y = self.global_size_random_numbers[1]/b_dim_y
        for by in range(blocks_y):
            for bx in range(blocks_x):
                for j in range(b_dim_y):
                    for i in range(b_dim_x):

                        ## Content of kernel:
                        y = b_dim_y*by + j # thread_id
                        x = b_dim_x*bx + i # thread_id
                        if (x < self.seed_nx and y < self.seed_ny):
                            n1, n2 = 0.0, 0.0
                            if normalDist:
                                n1, n2, self.host_seed[y,x]   = self._boxMuller(self.host_seed[y,x])
                            else:
                                n1, self.host_seed[y,x] = self._lcg(self.host_seed[y,x])
                                n2, self.host_seed[y,x] = self._lcg(self.host_seed[y,x])
                                
                            if x*2 + 1 < self.rand_nx:
                                self.random_numbers_host[y, x*2  ] = n1
                                self.random_numbers_host[y, x*2+1] = n2
                            elif x*2 == self.rand_nx:
                                self.random_numbers_host[y, x*2] = n1
    
    def _SOAR_Q_CPU(self, a_x, a_y, b_x, b_y):
        """
        CPU implementation of a SOAR covariance function between grid points
        (a_x, a_y) and (b_x, b_y)
        """
        dist = np.sqrt(  self.dx*self.dx*(a_x - b_x)**2  
                       + self.dy*self.dy*(a_y - b_y)**2 )
        return self.soar_q0*(1.0 + dist/self.soar_L)*np.exp(-dist/self.soar_L)
    
    def _applyQ_CPU(self):
        #xi, dx=1, dy=1, q0=0.1, L=1, cutoff=5):
        """
        Create the perturbation field for eta based on the SOAR covariance 
        structure
        """
                        
        # Assume in a GPU setting - we read xi into shared memory with ghostcells
        ny_halo = int(self.ny + (1 + self.cutoff)*2)
        nx_halo = int(self.nx + (1 + self.cutoff)*2)
        local_xi = np.zeros((ny_halo, nx_halo))
        for j in range(ny_halo):
            global_j = j
            if self.periodicNorthSouth:
                global_j = (j - self.cutoff - 1) % self.rand_ny
            for i in range(nx_halo):
                global_i = i
                if self.periodicEastWest:
                    global_i = (i - self.cutoff - 1) % self.rand_nx
                local_xi[j,i] = self.random_numbers_host[global_j, global_i]
                
        # Sync threads
        
        Qxi = np.zeros((self.ny+2, self.nx+2))
        for a_y in range(self.ny+2):
            for a_x in range(self.nx+2):
                # This is a OpenCL thread (a_x, a_y)
                local_a_x = a_x + self.cutoff
                local_a_y = a_y + self.cutoff

                start_b_y = local_a_y - self.cutoff
                end_b_y =  local_a_y + self.cutoff+1
                start_b_x = local_a_x - self.cutoff
                end_b_x =  local_a_x + self.cutoff+1

                Qx = 0
                for b_y in range(start_b_y, end_b_y):
                    for b_x in range(start_b_x, end_b_x):
                        Q = self._SOAR_Q_CPU(local_a_x, local_a_y, b_x, b_y)
                        Qx += Q*local_xi[b_y, b_x]
                Qxi[a_y, a_x] = Qx
        return Qxi
    
    
    def _obtainOceanPerturbations_CPU(self, H, f, beta, g):
        d_eta = self._applyQ_CPU()
        # d_eta.shape = (self.ny + 2, self.nx + 2)
        
        ####
        # Global sync (currently)
        #     Can be made into a local sync, as long as d_eta is given 
        #     periodic overlap (1 more global computated ghost cell)
        ####

        d_hu = np.zeros((self.ny, self.nx))
        d_hv = np.zeros((self.ny, self.nx))

        ### Find H_mid:
        # Read global H (def on intersections) to local, find H_mid
        # The local memory can then be reused to something else (perhaps use local_d_eta before computing local_d_eta?)
        #
        # Here, we just set it to 10
        H_mid = np.zeros((self.ny, self.nx))
        for j in range(self.ny):
            for i in range(self.nx):
                H_mid[j,i] = 0.25* (H[j,i] + H[j+1, i] + H[j, i+1] + H[j+1, i+1])
        
        ####
        # Local sync
        ####

        
        for j in range(0, self.ny):
            local_j = j + 1
            coriolis = f + beta*local_j*self.dy
            for i in range(0, self.nx):
                local_i = i + 1
                h_mid = d_eta[local_j,local_i] + H_mid[j, i]

                eta_diff_y = (d_eta[local_j+1, local_i] - d_eta[local_j-1, local_i])/(2.0*self.dy)
                d_hu[j,i] = -(g/coriolis)*h_mid*eta_diff_y

                eta_diff_x = (d_eta[local_j, local_i+1] - d_eta[local_j, local_i-1])/(2.0*self.dx)
                d_hv[j,i] = (g/coriolis)*h_mid*eta_diff_x   

        return d_eta, d_hu, d_hv
    