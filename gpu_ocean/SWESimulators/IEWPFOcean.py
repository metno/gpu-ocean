# -*- coding: utf-8 -*-

"""
This python class implements an the
Implicit Equal-Weight Particle Filter, for use on
simplified ocean models.

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
import matplotlib.gridspec as gridspec
import numpy as np
import time
import gc
import pycuda.driver as cuda

from SWESimulators import Common

class IEWPFOcean:
    
    def __init__(self, ensemble, debug=False):
        
        self.gpu_ctx = ensemble.gpu_ctx
        self.master_stream = cuda.Stream()
        self.debug = debug
        
        # Store information needed internally in the class
        self.dx = np.float32(ensemble.dx) 
        self.dy = np.float32(ensemble.dy)
        self.dt = np.float32(ensemble.dt)
        self.nx = np.int32(ensemble.nx)
        self.ny = np.int32(ensemble.ny)
        self.soar_q0 = np.float32(ensemble.small_scale_perturbation_amplitude)
        self.soar_L  = np.float32(ensemble.particles[0].small_scale_model_error.soar_L)
        self.f = np.float32(ensemble.f)
        self.g = np.float32(ensemble.g)
        self.const_H = np.float32(np.max(ensemble.base_H))

        self.boundaryConditions = ensemble.boundaryConditions
        
        self.geoBalanceConst = np.float32(self.g*self.const_H/(2.0*self.f))

        
        # The underlying assumptions are:
        # 1) that the equilibrium depth is constant:
        assert(np.max(ensemble.base_H) == np.min(ensemble.base_H))
        # 2) that both boundaries are periodic:
        assert(self.boundaryConditions.isPeriodicNorthSouth() and \
               self.boundaryConditions.isPeriodicEastWest())
        # 3) that the Coriolis force is constant for the entire domain:
        assert (ensemble.beta == 0)
        # 4) that dx and dy are the same
        assert (self.dx == self.dy)
        
        
        # Do not store the ensemble!!!
        
        # Create constant matrix S and copy to the GPU
        self.S_host, self.S_device = None, None
        self.S_host = self._createS(ensemble)
        self.S_device = Common.CUDAArray2D(self.master_stream, 2, 2, 0, 0, self.S_host)
        
        # Create constant localized SVD matrix and copy to the GPU
        self.localSVD_host, self.localSVD_device = None, None
        self.localSVD_host = self._generateLocaleSVDforP(ensemble)
        self.localSVD_device = Common.CUDAArray2D(self.master_stream, 49, 49, 0, 0, self.localSVD_host)
    
    def __del__(self):
        self.cleanUp()
        
        
    def cleanUp(self):
        # All allocated data needs to be freed from here
        if self.S_device is not None:
            self.S_device.release()
        if self.localSVD_device is not None:
            self.localSVD_device.release()
        
        self.gpu_ctx = None
        
        
    def download_S(self):
        return self.S_device.download(self.master_stream)
    
    def download_localSVD(self):
        return self.localSVD_device.download(self.master_stream)
    
        
    def showMatrices(self, x, y, title, z = None):
        num_cols = 2
        if z is not None:
            num_cols = 3
        fig = plt.figure(figsize=(num_cols*2,2))
        plt.subplot(1,num_cols,1)
        plt.imshow(x.copy(), origin="lower", interpolation="None")
        plt.xlabel('(%.2E, %.2E)' % (np.min(x), np.max(x)))
        plt.subplot(1,num_cols,2)
        plt.imshow(y.copy(), origin="lower", interpolation="None")
        plt.xlabel('(%.2E, %.2E)' % (np.min(y), np.max(y)))
        if z is not None:
            plt.subplot(1, num_cols, 3)
            plt.imshow(z.copy(), origin="lower", interpolation="None")
            plt.xlabel('(%.2E, %.2E)' % (np.min(z), np.max(z)))
        plt.suptitle(title)

        
        
    def _SOAR_Q_CPU(self, a_x, a_y, b_x, b_y):
        """
        CPU implementation of a SOAR covariance function between grid points
        (a_x, a_y) and (b_x, b_y) with periodic boundaries
        """
        dist_x = min((a_x - b_x)**2, (a_x - (b_x + self.nx))**2, (a_x - (b_x - self.nx))**2)
        dist_y = min((a_y - b_y)**2, (a_y - (b_y + self.ny))**2, (a_y - (b_y - self.ny))**2)
        
        dist = np.sqrt( self.dx*self.dx*dist_x  +  self.dy*self.dy*dist_y)

        return self.soar_q0*(1.0 + dist/self.soar_L)*np.exp(-dist/self.soar_L)

    def _createS(self, ensemble):
        """
        Create the 2x2 matrix S = (HQH^T + R)^-1

        Constant as long as
         - one drifter only,
         - H(x,y) = const, and
         - double periodic boundary conditions
        """

        # Local storage for x and y correlations:
        x_corr = np.zeros((7,7))
        y_corr = np.zeros((7,7))
        tmp_x = np.zeros((7,7))
        tmp_y = np.zeros((7,7))

        # Mid_coordinates:
        mid_i, mid_j = 3, 3

        # Fill the buffers with U_{GB}^T H^T
        x_corr[mid_j+1, mid_i] = -self.geoBalanceConst/self.dy
        x_corr[mid_j-1, mid_i] =  self.geoBalanceConst/self.dy
        y_corr[mid_j, mid_i+1] =  self.geoBalanceConst/self.dx
        y_corr[mid_j, mid_i-1] = -self.geoBalanceConst/self.dx
        if self.debug: self.showMatrices(x_corr, y_corr, "$U_{GB}^T  H^T$")
    
        # Apply the SOAR function to fill x and y with 7x5 and 5x7 respectively
        # First for x:
        for j,i in [mid_j+1, mid_i], [mid_j-1, mid_i]:
            for b in range(j-2, j+3):
                for a in range(i-2, i+3):
                    tmp_x[b, a] += x_corr[j,i]*self._SOAR_Q_CPU(a, b, i, j)
        # Then for y:
        for j,i in [mid_j, mid_i+1], [mid_j, mid_i-1]:
            for b in range(j-2, j+3):
                for a in range(i-2, i+3):
                    tmp_y[b, a] += y_corr[j,i]*self._SOAR_Q_CPU(a, b, i, j)
        if self.debug: self.showMatrices(tmp_x, tmp_y, "$Q_{SOAR} U_{GB}^T H^T$")   
        
        # Apply the SOARfunction again to fill the points needed to find drift in (mid_i, mid_j)
        # For both x and y:
        # This means that we only need to evaluate Q_{SOAR} Q_{SOAR} U_{GB}^T H^T at four points
        for j,i in [mid_j+1, mid_i], [mid_j-1, mid_i], [mid_j, mid_i-1], [mid_j, mid_i+1]:
            x_corr[j,i] = 0
            y_corr[j,i] = 0
            for b in range(j-2, j+3):
                for a in range(i-2, i+3):
                    SOAR_Q_res = self._SOAR_Q_CPU(a, b, i, j)
                    x_corr[j,i] += tmp_x[b, a]*SOAR_Q_res
                    y_corr[j,i] += tmp_y[b, a]*SOAR_Q_res
            if self.debug: print "(j, i ,x_corr[j,i], y_corr[j,i]): ", (j, i ,x_corr[j,i], y_corr[j,i])
        if self.debug: self.showMatrices(x_corr, y_corr, "$Q_{SOAR} Q_{SOAR} U_{GB}^T H^T$")

        # geostrophic balance:
        x_hu = -self.geoBalanceConst*(x_corr[mid_j+1, mid_i  ] - x_corr[mid_j-1, mid_i  ])/self.dy
        x_hv =  self.geoBalanceConst*(x_corr[mid_j  , mid_i+1] - x_corr[mid_j  , mid_i-1])/self.dx
        y_hu = -self.geoBalanceConst*(y_corr[mid_j+1, mid_i  ] - y_corr[mid_j-1, mid_i  ])/self.dy
        y_hv =  self.geoBalanceConst*(y_corr[mid_j  , mid_i+1] - y_corr[mid_j  , mid_i-1])/self.dx 

        # Structure the information as a  
        HQHT = np.matrix([[x_hu, y_hu],[x_hv, y_hv]])    
        if self.debug: print "HQHT\n", HQHT
        if self.debug: print "ensemble.observation_cov\n", ensemble.observation_cov
        S_inv = HQHT + ensemble.observation_cov
        if self.debug: print "S_inv\n", S_inv
        S = np.linalg.inv(S_inv)
        if self.debug: print "S\n", S
        return S.astype(np.float32, order='C')

    def _createCutoffSOARMatrixQ(self, ensemble, nx=None, ny=None, cutoff=2):
        
        if nx is None:
            nx = ensemble.nx
        if ny is None:
            ny = ensemble.ny
        
        Q = np.zeros((ny*nx, ny*nx))
        for a_y in range(ny):
            for a_x in range(nx):
                j = a_y*nx + a_x
                for b_y in range(a_y-cutoff, a_y+cutoff+1):
                    if b_y < 0:    
                         b_y = b_y + ny
                    if b_y > ny-1: 
                        b_y = b_y - ny
                    for b_x in range(a_x-cutoff, a_x+cutoff+1):
                        if b_x < 0:
                            b_x = b_x + nx
                        if b_x > nx-1: 
                            b_x = b_x - nx
                        i = b_y*nx + b_x
                        
                        # In the SOAR function, we use the ensemble nx and ny, to correctly 
                        # account for periodic boundaries.
                        Q[j, i] = self._SOAR_Q_CPU(a_x, a_y, b_x, b_y)
        return Q


    def _createUGBmatrix(self, ensemble, nx=None, ny=None):
    
        if nx is None:
            nx = ensemble.nx
        if ny is None:
            ny = ensemble.ny
        
        I = np.eye(nx*ny)
        A_hu = np.zeros((ny*nx, ny*nx))
        A_hv = np.zeros((ny*nx, ny*nx))
        for a_y in range(ny):
            for a_x in range(nx):
                j = a_y*nx + a_x
                
                # geo balance for hu:
                i = (a_y+1)*nx + a_x
                if a_y == ny-1:
                    i = 0*nx + a_x
                A_hu[j,i] = 1.0
                i = (a_y-1)*nx + a_x
                if a_y == 0:
                    i = (ny-1)*nx + a_x
                A_hu[j,i] = -1.0

                # geo balance for hv:
                i = a_y*nx + a_x + 1
                if a_x == nx-1:
                    i = a_y*nx + 0
                A_hv[j,i] = 1.0

                i = a_y*nx + a_x - 1
                if a_x == 0:
                    i = a_y*nx + nx - 1
                A_hv[j,i] = -1.0

        A_hu *= -self.geoBalanceConst/self.dy
        A_hv *=  self.geoBalanceConst/self.dx
            
        return np.bmat([[I], [A_hu], [A_hv]])

    def _createMatrixH(self, nx, ny, pos_x, pos_y):
        H = np.zeros((2, 3*nx*ny))
        index = pos_y*nx + pos_x
        H[0, 1*nx*ny + index] = 1
        H[1, 2*nx*ny + index] = 1
        return H

    def _generateLocaleSVDforP(self, ensemble):
        """
        Generates the local square root of the SVD-block needed for P^1/2.

        Finding:   U*Sigma*V^H = I - Q*U_GB^T*H^T*S*H*U_GB*Q
        Returning: U*sqrt(Sigma)
        """

        # Since the structure of the SVD-block is the same for any drifter position, we build the block
        # on a 7x7 domain with the observation in the middle cell
        local_nx = 7
        local_ny = 7
        pos_x = 3
        pos_y = 3

        # Create the matrices needed
        H      = self._createMatrixH(local_nx, local_ny, pos_x, pos_y)
        Q_soar = self._createCutoffSOARMatrixQ(ensemble, nx=local_nx, ny=local_ny)
        U_GB   = self._createUGBmatrix(ensemble, nx=local_nx, ny=local_ny)

        UQ = np.dot(U_GB, Q_soar)
        HUQ = np.dot(H, UQ)
        SHUQ = np.dot(self.S_host, HUQ)
        HTSHUQ = np.dot(H.transpose(), SHUQ)
        UTHTSHUQ = np.dot(U_GB.transpose(), HTSHUQ)
        QUTHTSHUQ = np.dot(Q_soar, UTHTSHUQ)

        svd_input = np.eye(local_nx*local_nx) - QUTHTSHUQ

        u, s, vh = np.linalg.svd(svd_input, full_matrices=True)

        if self.debug:
            SVD_prod = np.dot(u, np.dot(np.diag(s), vh))
            fig = plt.figure(figsize=(4,4))
            plt.imshow(SVD_prod, interpolation="None")
            plt.title("SVD_prod")
            plt.colorbar()

            fig = plt.figure(figsize=(4,4))
            plt.imshow(SVD_prod - np.eye(49), interpolation="None")
            plt.title("SVD_prod - I")
            plt.colorbar()

            fig = plt.figure(figsize=(4,4))
            plt.imshow(u, interpolation="None")
            plt.title("u")
            plt.colorbar()


        return np.dot(u, np.diag(np.sqrt(s))).astype(np.float32, order='C')
    
    
    # As we have S = (HQH^T + R)^-1, we can do step 1 of the IEWPF algorithm
    def obtainTargetWeight(self, ensemble, w_rest=None):
        if w_rest is None:
            w_rest = -np.log(1.0/ensemble.getNumParticles())*np.ones(ensemble.getNumParticles())
        
        d = ensemble.getInnovations() 
        Ne = ensemble.getNumParticles()
        c = np.zeros(Ne)
        for particle in range(Ne):
            # Obtain db = d^T S d
            db = 0.0
            for drifter in range(ensemble.driftersPerOceanModel):
                e = np.dot(self.S_host, d[particle,drifter,:])
                db += np.dot(e, d[particle, drifter, :])
            c[particle] = w_rest[particle] + 0.5*db
            if self.debug: print "c[" + str(particle) + "]: ", c[particle]
            if self.debug: print "exp(-c[" + str(particle) + "]: ", np.exp(-c[particle])
        return np.min(c)

    
    def _apply_periodic_boundary(self, index, dim_size):
        if index < 0:
            return index + dim_size
        elif index >= dim_size:
            return index - dim_size
        return index
    
    def _apply_local_SVD_to_global_xi(self, global_xi, pos_x, pos_y):
        """
        Despite the bad name, this is a good function!

        It takes as input:
         - the global_xi stored in a (ny, nx) buffer
         - the drifter cell position (pos_x, pos_y)
        
        It find the local sqrt (SVD) as U*sqrt(Sigma) in a (49, 49) buffer in self.
        

        The global_xi buffer is modified so that xi = U*sqrt(Sigma)*xi

        Note that we have to make a copy of xi so that we don't read already updated values.

        The function assumes periodic boundary conditions in both dimensions.
        """


        # Copy the result (representing the multiplication with I)
        read_global_xi = global_xi.copy()

        # Read the non-zero structure from tildeP to tildeP_block
        for loc_y_j in range(7):
            global_y_j = pos_y - 3 + loc_y_j
            global_y_j = self._apply_periodic_boundary(global_y_j, self.ny)
            for loc_x_j in range(7):
                global_x_j = pos_x - 3 + loc_x_j
                global_x_j = self._apply_periodic_boundary(global_x_j, self.nx)

                global_j = global_y_j*self.nx + global_x_j
                local_j = loc_y_j*7 + loc_x_j

                #loc_vec[local_j] = glob_vec[global_j]

                xi_j = 0.0
                for loc_y_i in range(7):
                    global_y_i = pos_y - 3 + loc_y_i
                    global_y_i = self._apply_periodic_boundary(global_y_i, self.ny)
                    for loc_x_i in range(7):
                        global_x_i = pos_x - 3 + loc_x_i
                        global_x_i = self._apply_periodic_boundary(global_x_i, self.nx)

                        global_i = global_y_i*self.nx + global_x_i
                        local_i = loc_y_i*7 + loc_x_i

                        xi_j += self.localSVD_host[local_j, local_i]*read_global_xi[global_y_i, global_x_i]

                global_xi[global_y_j, global_x_j] = xi_j

    