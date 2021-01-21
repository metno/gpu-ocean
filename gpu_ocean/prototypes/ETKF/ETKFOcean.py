# -*- coding: utf-8 -*-

"""
This software is a part of GPU Ocean.

Copyright (C) 2018  SINTEF Digital

This python class implements the Ensemble Transform Kalman Filter.

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
import logging

#from SWESimulators import Common, OceanStateNoise, config, EnsemblePlot

class ETKFOcean:
    """
    This class implements the Stochastic Ensemble Kalman Filter in square-root formulation
    for an ocean model with small scale ocean state perturbations as model errors.
    
    Input to constructor:
    ensemble: An object of super-type BaseOceanStateEnsemble.
            
    """

    def __init__(self, ensemble, inflation_factor=1.0):
        """
        Copying the ensemble to the member variables 
        and deducing frequently used ensemble quantities
        """

        self.ensemble = ensemble
        
        self.N_e = ensemble.getNumParticles()
        self.N_d = ensemble.getNumDrifters()

        # Size of state matrices (with ghost cells)
        self.n_i = self.ensemble.particles[0].ny + 2*self.ensemble.particles[-1].ghost_cells_y
        self.n_j = self.ensemble.particles[0].nx + 2*self.ensemble.particles[-1].ghost_cells_x

        self.inflation_factor = inflation_factor

        self.localObservationDistances = None
        self.localScale = 25.0
    

    def ETKF(self, ensemble=None):
        """
        Performing the analysis phase of the ETKF.
        Particles are observed and the analysis state is calculated and uploaded!

        ensemble: for better readability of the script when ETKF is called the ensemble can be passed again.
        Then it overwrites the initially defined member ensemble
        """

        if ensemble is not None:
            assert(self.N_e == ensemble.getNumParticles()), "ensemble changed size"
            assert(self.N_d == ensemble.getNumDrifters()), "ensemble changed number of drifter"
            assert(self.n_i == ensemble.particles[0].ny + 2*ensemble.particles[-1].ghost_cells_y), "ensemble changed size of physical domain"
            assert(self.n_j == ensemble.particles[0].nx + 2*ensemble.particles[-1].ghost_cells_x), "ensemble changed size of physical domain"
            
            self.ensemble = ensemble

            self.N_e_active = ensemble.getNumActiveParticles()

        X_f, X_f_mean, X_f_pert = self._giveX_f()
        HX_f_pert, HX_f_mean = self._giveHX_f()

        Rinv = self._constructRinv()

        D = self._giveD(HX_f_mean)

        P = self._giveP(HX_f_pert, Rinv)
        
        K = self._giveK(X_f_pert, P, HX_f_pert, Rinv)

        X_a = self._giveX_a(X_f_mean, X_f_pert, K, D, P)

        self.uploadAnalysisState(X_a)

    
    """
    The following methods stating with _ are simple matrix computations and reshaping operations
    which are separated for the seek of readability
    """

    def _deleteDeactivatedObservations(self, observation):
        """
        Delete inactive particles
        """
        idx = 0
        for p in range(self.N_e):
            if self.ensemble.particlesActive[p]:
                idx+=1
            elif not self.ensemble.particlesActive[p]:
                observation = np.delete(observation, idx, axis=0)
        return observation


    def _constructRinv(self):
        
        R_orig = self.ensemble.getObservationCov()

        R = np.zeros( (R_orig.shape[0]*self.N_d, R_orig.shape[1]*self.N_d) )

        for l in range(self.N_d):
            R[l,l] = R_orig[0,0]
            R[self.N_d+l, self.N_d+l] = R_orig[1,1]
            R[l,self.N_d+l] = R_orig[0,1]
            R[self.N_d+l,l] = R_orig[1,0]

        Rinv = np.linalg.inv(R)

        return Rinv

    def _giveX_f(self):

        """
        The download gives eta = 
        [
        [eta(x0,y0),...,eta(xN,y0)],
        ...,
        [eta(x0,yN),...,eta(xN,yN)]
        ]
        as an array of size Ny x Nx
        and analog for hu and hv.
        we use those as an 1D array eta = 
        [eta(x0,y0),...,eta(xN,y0),eta(x0,y1),...,eta(xN,y(N-1)),eta(x0,yN),...,eta(xN,yN)]
        and anlog for hu and hv 

        For further calculations the indivdual dimensions of the state variable are concatinated X = 
        [eta, hu, hv]

        Collecting the state perturbation for each ensemble member in a matrix Nx x Ne, where
        X_f_pert = 
        [ 
        [eta_pert(x0,y0) (particle 1),..., eta_pert],
        ...
        particle 2: [eta_pert,hu_pert,hv_pert]
        ]
        """

        X_f = np.zeros((3*self.n_i*self.n_j, self.N_e_active))

        idx = 0
        for e in range(self.N_e):
            if self.ensemble.particlesActive[e]:
                eta, hu, hv = self.ensemble.particles[e].download(interior_domain_only=False)
                eta = eta.reshape(self.n_i*self.n_j)
                hu  = hu.reshape(self.n_i*self.n_j)
                hv  = hv.reshape(self.n_i*self.n_j)
                X_f[:,idx] = np.append(eta, np.append(hu,hv))
                idx += 1

        X_f_mean = np.zeros( 3*self.n_i*self.n_j )
        for e in range(self.N_e_active):
            X_f_mean += 1/self.N_e_active * X_f[:,e]

        X_f_pert = np.zeros_like( X_f )
        for e in range(self.N_e_active):
            X_f_pert[:,e] = X_f[:,e] - X_f_mean

        return X_f, X_f_mean, X_f_pert


    def _giveHX_f(self):
        """
        Particles are observed in the following form:
        [
        particle 1:  [hu_1, hv_1], ... , [hu_D, hv_D],
        ...
        particle Ne: [hu_1, hv_1], ... , [hu_D, hv_D]
        ]

        In order to bring it in accordance with later data structure we use the following format for the storage of the perturbation of the observation:
        [
        [hu_1 (particle 1), ..., hu_1 (particle Ne)],
        ...
        [hu_D (particle 1), ..., hu_D (particle Ne)],
        [hv_1 (particle 1), ..., hv_1 (particle Ne)],
        ...
        [hv_D (particle 1), ..., hv_D (particle Ne)],
        ]

        """

        # Observation (nan for inactive particles)
        HX_f_orig = self._deleteDeactivatedObservations(self.ensemble.observeParticles())

        # Reshaping
        HX_f = np.zeros( (2*self.N_d, self.N_e_active) )
        for e in range(self.N_e_active):
            for l in range(self.N_d):
                HX_f[l,e]     = HX_f_orig[e,l,0]
            for l in range(self.N_d):
                HX_f[self.N_d+l,e] = HX_f_orig[e,l,1]

        HX_f_mean = 1/self.N_e_active * np.sum(HX_f, axis=1)

        HX_f_pert = HX_f - HX_f_mean.reshape((2*self.N_d,1))

        return HX_f_pert, HX_f_mean


    def _giveD(self, HX_f_mean):
        """
        Particles yield innovations in the following form:
        [x_1, y_1, hu_1, hv_1], ... , [x_D, y_D, hu_D, hv_D]

        In order to bring it in accordance with later data structure we use the following format for the storage of the perturbation of the observation:
        [hu_1, ..., hu_D, hv_1, ..., hv_D]
        """

        y_orig = self.ensemble.observeTrueState()

        y = np.zeros( (2*self.N_d) )
        for l in range(self.N_d):
            y[l]     = y_orig[l,2]
        for l in range(self.N_d):
            y[self.N_d+l] = y_orig[l,3]

        D = y - HX_f_mean

        return D


    def _giveP(self, HX_f_pert, Rinv):

        A1 = (self.N_e_active-1) * np.eye(self.N_e_active)
        A2 = np.dot(HX_f_pert.T, np.dot(Rinv, HX_f_pert))

        A = A1 + A2

        P = np.linalg.inv(A)

        return P

    def _giveK(self, X_f_pert, P, HX_f_pert, Rinv):

        K = np.dot(X_f_pert, np.dot(P, np.dot(HX_f_pert.T, Rinv)))

        return K


    def _giveX_a(self, X_f_mean, X_f_pert, K, D, P):

        X_a_mean = X_f_mean + np.dot(K, D)

        sigma, V = np.linalg.eigh( (self.N_e_active-1) * P )
        X_a_pert = np.dot( X_f_pert, np.dot( V, np.dot( np.diag( np.sqrt( np.real(sigma) ) ), V.T )))

        X_a = X_a_pert 
        for j in range(self.N_e_active):
            X_a[:,j] += X_a_mean
            
        return X_a


    def uploadAnalysisState(self, X_a):
        
        idx = 0
        for e in range(self.N_e):
            if self.ensemble.particlesActive[e]:
                eta = X_a[0:self.n_i*self.n_j, idx].reshape((self.n_i,self.n_j))
                hu  = X_a[self.n_i*self.n_j:2*self.n_i*self.n_j, idx].reshape((self.n_i,self.n_j))
                hv  = X_a[2*self.n_i*self.n_j:3*self.n_i*self.n_j, idx].reshape((self.n_i,self.n_j))
                self.ensemble.particles[e].upload(eta,hu,hv)
                idx += 1






    @staticmethod
    def distGC( obs, loc, r, lx, ly):
        """
        Calculating the Gasparin-Cohn value for the distance between obs 
        and loc for the localisation radius r.
        
        obs: drifter positions ([x,y])
        loc: current physical location to check (either [x,y] or [[x1,y1],...,[xd,yd]])
        r: localisation scale in the Gasparin Cohn function
        lx: domain extension in x-direction (necessary for periodic boundary conditions)
        ly: domain extension in y-direction (necessary for periodic boundary conditions)
        """
        if not obs.shape == loc.shape: 
            obs = np.tile(obs, (loc.shape[0],1))
        
        if len(loc.shape) == 1:
            dist = min(np.linalg.norm(np.abs(obs-loc)),
                    np.linalg.norm(np.abs(obs-loc) - np.array([lx,0 ])),
                    np.linalg.norm(np.abs(obs-loc) - np.array([0 ,ly])),
                    np.linalg.norm(np.abs(obs-loc) - np.array([lx,ly])) )
        else:
            dist = np.linalg.norm(obs-loc, axis=1)

        # scalar case
        if isinstance(dist, float):
            distGC = 0.0
            if dist/r < 1: 
                distGC = 1 - 5/3*(dist/r)**2 + 5/8*(dist/r)**3 + 1/2*(dist/r)**4 - 1/4*(dist/r)**5
            elif dist/r >= 1 and dist/r < 2:
                distGC = 4 - 5*(dist/r) + 5/3*(dist/r)**2 + 5/8*(dist/r)**3 -1/2*(dist/r)**4 + 1/12*(dist/r)**5 - 2/(3*(dist/r))
        # vector case
        else:
            distGC = np.zeros_like(dist)
            for i in range(len(dist)):
                if dist[i]/r < 1: 
                    distGC[i] = 1 - 5/3*(dist[i]/r)**2 + 5/8*(dist[i]/r)**3 + 1/2*(dist[i]/r)**4 - 1/4*(dist[i]/r)**5
                elif dist[i]/r >= 1 and dist[i]/r < 2:
                    distGC[i] = 4 - 5*(dist[i]/r) + 5/3*(dist[i]/r)**2 + 5/8*(dist[i]/r)**3 -1/2*(dist[i]/r)**4 + 1/12*(dist[i]/r)**5 - 2/(3*(dist[i]/r))

        return distGC


    @staticmethod
    def distGCField( obs, r, dx, dy, nx, ny):
        """
        Calculating the Gasparin-Cohn value for the distance between obs 
        and all (x,y)-grid points for the localisation radius r.
        
        obs: drifter positions (either [x,y] or [[x1,y1],...,[xd,yd]])
        loc: current physical location to check ([x,y])
        r: localisation scale in the Gasparin Cohn function
        lx: domain extension in x-direction (necessary for periodic boundary conditions)
        ly: domain extension in y-direction (necessary for periodic boundary conditions)

        distGCField: 
        """

        # compensation for ghost cells
        obs = obs + np.array([2*dx, 2*dy])

        lx = dx*nx
        ly = dy*ny
        dist_field = np.zeros((ny, nx))
        for j in range(ny):
            for i in range(nx):
                loc = np.array([(i+0.5)*dx, (j+0.5)*dy])
                dist_field[j, i] = ETKFOcean.distGC(obs, loc, r, lx, ly)
        return dist_field


    @staticmethod
    def boxDist(obs, loc, lx, ly):
        """
        Returning L1 norm between obs and loc for periodic boundary conditions
        """

        return min(np.max(np.abs(obs-loc)),
                np.max(np.abs(np.abs(obs-loc)-np.array([lx,0 ]))),
                np.max(np.abs(np.abs(obs-loc)-np.array([0 ,ly]))),
                np.max(np.abs(np.abs(obs-loc)-np.array([lx,ly]))))


    @staticmethod
    def localBox(obs, r, dx, dy, nx, ny):
        """
        Returning a (nx,ny)-sized array with 1 if (x,y) in boxDist of obs, 0 else
        """

        # compensation for ghost cells
        obs = obs + np.array([2*dx, 2*dy])

        lx = dx*nx
        ly = dy*ny
        box_field = np.zeros((ny, nx))
        for j in range(ny):
            for i in range(nx):
                loc = np.array([(i+0.5)*dx, (j+0.5)*dy])
                if ETKFOcean.boxDist(obs, loc, lx, ly) < r*2:
                    box_field[j, i] = 1
        return box_field


    @staticmethod
    def drifterDependencies(d, obs, r, lx, ly):
        """
        Returning a (N_d)-array with 1 if i is in boxDist of drifter d, 0 else
        """
        N_d = obs.shape[0]
        drifter_dependencies = np.zeros((N_d))
        for i in range(N_d):
            if ETKFOcean.boxDist(obs[d,:], obs[i,:], lx, ly) < r*2:
                drifter_dependencies[d] = 1

        print("Strange dependency scaling! Not consistent with local box!")

        return drifter_dependencies


    def initializeLocalPatches(self, localScale=0.0, x0=0.0, y0=0.0):
        """
        Preprocessing for the LETKF 
        which generates arrays storing the local observation indices for every grid cell (including 2 ghost cells)
        
        localScale: scale for the Gasparin-Cohn distance and the definition of local boxes
        x0: x-coordinate of physical position of the lower left corner in meter
        y0: y-coordinate of physical position of the lower left corner in meter
        
        self.localObservationDistances: 3D-array of shape (ny,nx,N_d) 
            where in the 3rd component of (j,i,k) the entry is Gasparin-Cohn distance of (x_i,y_j) to drifter k 
        self.localObservationBoxes: 3D-array of shape (ny,nx,N_d) 
            where in the 3rd component of (j,i,k) the entry is 1 if (x_i,y_j) in local box of drifter k, 0 else
        self.localObservationBoxesIndices: reshaping of self.localObservationBoxes to match shape of X_f (see above)

        TODO: x,y shifting if x0,y0 != 0.0
        TODO: adaptive ghost_cell handling (atm: hard coded 2 cells)
        """

        # Book keeping
        dy = self.ensemble.dy
        dx = self.ensemble.dx

        nx = self.n_j
        ny = self.n_i

        ly = nx*dy
        lx = ny*dx

        if localScale > 0.0:
            self.localScale = localScale

        # Get drifter position
        drifter_positions = self.ensemble.observeTrueDrifters()

        # Construct localPatches array
        self.localObservationDistances = np.zeros((ny, nx, self.N_d))
        self.localObservationDistancesIndices = np.zeros((3*ny*nx, self.N_d))

        self.localObservationBoxes = np.zeros((ny, nx, self.N_d))
        self.localObservationBoxesIndices = np.zeros((3*ny*nx, self.N_d))

        self.localObservationDependencies = np.zeros((self.N_d,self.N_d))

        for d in range(self.N_d):
            print("Construct localObservationDistances for observation_id ", d)
            self.localObservationDistances[:,:,d] = ETKFOcean.distGCField( drifter_positions[d,:], self.localScale*dx, dx, dy, nx, ny )
            dists = self.localObservationDistances[:,:,d].reshape(nx*ny)
            self.localObservationDistancesIndices[:,d] = np.append(dists, np.append(dists,dists))


            self.localObservationBoxes[:,:,d] = ETKFOcean.localBox( drifter_positions[d,:], self.localScale*dx, dx, dy, nx, ny )
            idxs = self.localObservationBoxes[:,:,d].reshape(nx*ny)
            self.localObservationBoxesIndices[:,d] = np.append(idxs, np.append(idxs,idxs))

            self.localObservationDependencies[d,:] = ETKFOcean.drifterDependencies(d, drifter_positions, self.localScale*dx, lx, ly)



    def LETKF(self, ensemble=None, localScale=0.0):
        """
        Performing the analysis phase of the ETKF.
        Particles are observed and the analysis state is calculated and uploaded!

        ensemble: for better readability of the script when ETKF is called the ensemble can be passed again.
        Then it overwrites the initially defined member ensemble
        """

        if ensemble is not None:
            assert(self.N_e == ensemble.getNumParticles()), "ensemble changed size"
            assert(self.N_d == ensemble.getNumDrifters()), "ensemble changed number of drifter"
            assert(self.n_i == ensemble.particles[0].ny + 2*ensemble.particles[-1].ghost_cells_y), "ensemble changed size of physical domain"
            assert(self.n_j == ensemble.particles[0].nx + 2*ensemble.particles[-1].ghost_cells_x), "ensemble changed size of physical domain"
            
            self.ensemble = ensemble

            self.N_e_active = ensemble.getNumActiveParticles()

        if localScale > 0.0:
            if localScale != self.localScale:
                self.localScale = localScale
                self.initializeLocalPatches( localScale=self.localScale, x0=0.0, y0=0.0)

        if self.localObservationDistances is None:
            self.initializeLocalPatches( localScale=self.localScale, x0=0.0, y0=0.0)

        local_X_as = X_f = np.zeros((3*self.n_i*self.n_j, self.N_e_active,self.N_d))

        X_f, X_f_mean, X_f_pert = self._giveX_f()
        HX_f_pert, HX_f_mean = self._giveHX_f()
        y_orig = self.ensemble.observeTrueState()

        for d in range(self.N_d):
            local_X_f = X_f[self.localObservationBoxesIndices[:,d]==1,:]
            local_X_f_mean = X_f_mean[self.localObservationBoxesIndices[:,d]==1,:]
            local_X_f_pert = X_f_pert[self.localObservationBoxesIndices[:,d]==1,:]

            local_HX_f_mean = HX_f_mean[self.localObservationDependencies[d,:]>0.0,:]
            local_HX_f_pert = HX_f_pert[self.localObservationDependencies[d,:]>0.0,:]

            local_Rinv = self._constructLocalRinv(d)

            local_D = self._giveLocalD(y_orig, HX_f_mean, d)

            local_P = self._giveP(local_HX_f_pert, local_Rinv)
            
            K = self._giveK(local_X_f_pert, local_P, local_HX_f_pert, local_Rinv)

            local_X_a = self._giveX_a(local_X_f_mean, local_X_f_pert, local_K, local_D, local_P)

            local_X_as[:,:,d] = self._reconstructXa(local_X_a, d)

        self.combineLocalAnalysis(X_f, local_X_as)
        self.uploadAnalysisState(X_a)


    def _constructLocalRinv(self, d):

        local_N_d = np.sum(self.localObservationDependencies[d,:])
        
        R_orig = self.ensemble.getObservationCov()

        local_R = np.zeros( (R_orig.shape[0]*local_N_d, R_orig.shape[1]*local_N_d) )

        for l in range(local_N_d):
            local_R[l,l] = R_orig[0,0]
            local_R[local_N_d+l, local_N_d+l] = R_orig[1,1]
            local_R[l,local_N_d+l] = R_orig[0,1]
            local_R[local_N_d+l,l] = R_orig[1,0]

        local_Rinv = np.linalg.inv(local_R)

        return local_Rinv

    def _giveLocalD(self, y_orig, HX_f_mean, d):

        local_N_d = np.sum(self.localObservationDependencies[d,:])

        local_y_orig = y_orig[self.localObservationDependencies[d,:]>0.0,:]

        local_y = np.zeros( (2*local_N_d) )

        for l in range(local_N_d):
            local_y[l]           = local_y_orig[l,2]
        for l in range(local_N_d):
            local_y[local_N_d+l] = local_y_orig[l,3]

        local_D = local_y - HX_f_mean[self.localObservationDependencies[d,:]>0.0]

        return local_D


    def _reconstructXa(self, local_X_a, d):
        
        reconstructed_X_a = np.zeros((3*self.n_i*self.n_j, self.N_e_active))

        for e in range(self.N_e_active):
            idx=0
            for i in range(3*self.n_i*self.n_j):
                if self.localObservationBoxesIndices[d,i] > 0.0:
                    reconstructed_X_a[i,e] = local_X_a[idx,e]
                    idx += 1

    def combineLocalAnalysis(self, X_f, local_X_as):

        X_a = np.zeros((3*self.n_i*self.n_j, self.N_e_active))

        for e in range(self.N_e_active):
            for i in range(3*self.n_i*self.n_j):
                observation_scaling = localObservationDistancesIndices[i,:][localObservationDistancesIndices[i,:]>0.0].shape[0]
                updates = 0.0
                update_weights = 0.0
                for d in range(self.N_d):
                    update_weight = 1/observation_scaling * self.localObservationDistancesIndices[i,d]
                    update_weights += update_weight
                    updates += update_weight * local_X_as[i,e,d]
                    
                X_a[i,e] = (1-update_weights) * X_f[i,e] + update_weights * updates

