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
    

    def ETKF(self, ensemble=None):
        """
        Performing the analysis phase of the EnKF.
        Particles are observed and the analysis state is calculated and uploaded!

        ensemble: for better readability of the script when EnKF is called the ensemble can be passed again.
        Then it overwrites the initially defined member ensemble
        """

        if ensemble is not None:
            assert(self.N_e == ensemble.getNumParticles()), "ensemble changed size"
            assert(self.N_d == ensemble.getNumDrifters()), "ensemble changed number of drifter"
            assert(self.n_i == ensemble.particles[0].ny + 2*ensemble.particles[-1].ghost_cells_y), "ensemble changed size of physical domain"
            assert(self.n_j == ensemble.particles[0].nx + 2*ensemble.particles[-1].ghost_cells_x), "ensemble changed size of physical domain"
            
            self.ensemble = ensemble

            self.N_e_active = ensemble.getNumActiveParticles()

        R = self._constructR()


        X_f, X_f_mean, X_f_pert = self._giveX_f()
        HX_f_pert, HX_f_mean = self._giveHX_f()
        D = self._giveD(HX_f_mean)

        P = self._giveP(HX_f_pert, R)
        K = self._giveK(X_f_pert, P, HX_f_pert, R)

        X_a = self._giveX_a(X_f_mean, K, D, P)

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


    def _constructR(self):
        
        R_orig = self.ensemble.getObservationCov()

        R = np.zeros( (R_orig.shape[0]*self.N_d, R_orig.shape[1]*self.N_d) )

        for l in range(self.N_d):
            R[l,l] = R_orig[0,0]
            R[self.N_d+l, self.N_d+l] = R_orig[1,1]
            R[l,self.N_d+l] = R_orig[0,1]
            R[self.N_d+l,l] = R_orig[1,0]

        return R

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
        [
        particle 1:  [hu_1, hv_1], ... , [hu_D, hv_D],
        ...
        particle Ne: [hu_1, hv_1], ... , [hu_D, hv_D]
        ]

        In order to bring it in accordance with later data structure we use the following format for the storage of the perturbation of the observation:
        [
        [d_hu_1 (particle 1), ..., d_hu_1 (particle Ne)],
        ...
        [d_hu_D (particle 1), ..., d_hu_D (particle Ne)],
        [d_hv_1 (particle 1), ..., d_hv_1 (particle Ne)],
        ...
        [d_hv_D (particle 1), ..., d_hv_D (particle Ne)],
        ]

        """

        y_orig = self.ensemble.observeTrueState()

        y = np.zeros( (2*self.N_d) )
        for l in range(self.N_d):
            y[l]     = y_orig[l,2]
        for l in range(self.N_d):
            y[N_d+l] = y_orig[l,3]

        D = y - HX_f_mean

        return D


    def _giveP(self, HX_f_pert, R):

        Rinv = np.linalg.inv(R)

        A = np.dot( (self.N_e_active-1)*np.eye(self.N_e_active), np.dot(HX_f_pert.T, np.dot(Rinv, HX_f_pert)))

        P = np.linalg.inv(A)

        return P

    def _giveK(self, X_f_pert, P, HX_f_pert, R):

        K = np.dot(X_f_pert, np.dot(P,np.dot(HX_f_pert.T,Rinv)))

        return K


    def _giveX_a(self, X_f_mean, K, D, P):

        X_a_mean = X_f_mean + np.dot(K, D)

        sigma, V = np.linalg.eigh( (self.N_e_active-1) * P )
        X_a_pert = np.dot( X_f_pert, np.dot( V, np.dot( np.diag( np.sqrt( sigma ) ), V.T )))

        X_a = X_a_pert 
        for j in range(N_e_active):
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