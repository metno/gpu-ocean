# -*- coding: utf-8 -*-

"""
This python class implements an abstract ensemble class, where each particle
will consist of an independent ocean state and one or more drifters.


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

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time
import abc

from SWESimulators import CDKLM16
from SWESimulators import GPUDrifterCollection
from SWESimulators import WindStress
from SWESimulators import Common
from SWESimulators import DataAssimilationUtils as dautils

class BaseOceanStateEnsemble(object):

    __metaclass__ = abc.ABCMeta
        
    def __init__(self, numParticles, cl_ctx, observation_type=dautils.ObservationType.DrifterPosition):
        
        self.cl_ctx = cl_ctx
        
        self.numParticles = numParticles
        self.particles = [None]*(self.numParticles + 1)
        
        self.obs_index = self.numParticles
        
        self.simType = 'CDKLM16'
        
        self.t = 0.0
        
        dautils.ObservationType._assert_valid(observation_type)
        self.observation_type = observation_type
        self.prev_observation = None
        
        
        # Observations are stored as [ [t^n, [[x_i^n, y_i^n]] ] ]
        # where n is time step and i is drifter
        self.observedDrifterPositions = []
        
        # Arrays to store statistical info for selected grid cells:
        self.varianceUnderDrifter_eta = []
        self.varianceUnderDrifter_hu = []
        self.varianceUnderDrifter_hv = []
        self.rmseUnderDrifter_eta = []
        self.rmseUnderDrifter_hu = []
        self.rmseUnderDrifter_hv = []
        self.rUnderDrifter_eta = []
        self.rUnderDrifter_hu = []
        self.rUnderDrifter_hv = []
        self.tArray = []

        
    def cleanUp(self):
        for oceanState in self.particles:
            if oceanState is not None:
                oceanState.cleanUp()
        
    # IMPROVED
    def setGridInfo(self, nx, ny, dx, dy, dt, 
                    boundaryConditions=Common.BoundaryConditions(), 
                    eta=None, hu=None, hv=None, H=None):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        
        self.boundaryConditions = boundaryConditions
        
        assert(self.simType == 'CDKLM16'), 'CDKLM16 is currently the only supported scheme'
        #if self.simType == 'CDKLM16':
        self.ghostCells = np.array([2,2,2,2])
        if self.boundaryConditions.isSponge():
            sponge = self.boundaryConditions.getSponge()
            for i in range(4):
                if sponge[i] > 0: 
                    self.ghostCells[i] = sponge[i]
        dataShape =  ( ny + self.ghostCells[0] + self.ghostCells[2], 
                       nx + self.ghostCells[1] + self.ghostCells[3]  )
            
        self.base_eta = eta
        self.base_hu = hu
        self.base_hv = hv
        self.base_H = H
            
        # Create base initial data: 
        if self.base_eta is None:
            self.base_eta = np.zeros(dataShape, dtype=np.float32, order='C')
        if self.base_hu is None:
            self.base_hu  = np.zeros(dataShape, dtype=np.float32, order='C');
        if self.base_hv is None:
            self.base_hv  = np.zeros(dataShape, dtype=np.float32, order='C');
        
        # Bathymetry:
        if self.base_H is None:
            waterDepth = 10
            self.base_H = np.ones((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C')*waterDepth
        
        # Ensure that parameters are initialized:
        self.setParameters()

    def setGridInfoFromSim(self, sim):
        eta, hu, hv = sim.download()
        Hi = sim.downloadBathymetry()[0]
        self.setGridInfo(sim.nx, sim.ny, sim.dx, sim.dy, sim.dt,
                         sim.boundary_conditions,
                         eta=eta, hu=hu, hv=hv, H=Hi)
        self.setParameters(f=sim.f, g=sim.g, beta=sim.coriolis_beta, r=sim.r, wind=sim.wind_stress)
    
    def setParameters(self, f=0, g=9.81, beta=0, r=0, wind=WindStress.NoWindStress()):
        self.g = g
        self.f = f
        self.beta = beta
        self.r = r
        self.wind = wind
    
    def setStochasticVariables(self, 
                               observation_variance = None, 
                               observation_variance_factor = 5.0,
                               small_scale_perturbation_amplitude = 0.0,
                               initialization_variance_factor_drifter_position = 0.0,
                               initialization_variance_factor_ocean_field = 0.0 
                              ):

        # Setting observation variance:
        self.observation_variance = observation_variance
        if self.observation_variance is None:
            self.observation_variance = (observation_variance_factor*self.dx)**2
        
        # Build observation covariance matrix:
        self.observation_cov = None
        self.observation_cov_inverse = None
        if np.isscalar(self.observation_variance):
            self.observation_cov = np.eye(2)*self.observation_variance
            self.observation_cov_inverse = np.eye(2)*(1.0/self.observation_variance)
        else:
            print("type(self.observation_variance): " + str(type(self.observation_variance)))
            # Assume that we have a correctly shaped matrix here
            self.observation_cov = self.observation_variance
            self.observation_cov_inverse = np.linalg.inv(self.observation_cov)
        
        
        # TODO: Check if this variable is used anywhere.
        # Should not be defined in the Base class.
        self.initialization_variance_drifters = initialization_variance_factor_drifter_position*self.dx
        self.initialization_cov_drifters = np.eye(2)*self.initialization_variance_drifters
        self.midPoint = 0.5*np.array([self.nx*self.dx, self.ny*self.dy])
        
        self.small_scale_perturbation_amplitude = small_scale_perturbation_amplitude
    
        # When initializing an ensemble, each member should be perturbed so that they 
        # have slightly different starting point.
        # This factor should be multiplied to the small_scale_perturbation_amplitude for that 
        # perturbation
        self.initialization_variance_factor_ocean_field = initialization_variance_factor_ocean_field
        
        
    @abc.abstractmethod
    def init(self, driftersPerOceanModel=1):
        # Initialize ocean models
        # add drifters
        # add noise
        # etc
        pass
    
    @abc.abstractmethod
    def resample(self, newSampleIndices, reinitialization_variance):
        # Resample and possibly perturb
        pass
        
        
    def _addObservation(self, observedDrifterPositions):
        # Observations are stored as [ [t^n, [[x_i^n, y_i^n]] ] ]
        # where n is time step and i is drifter
        
        print("Adding observation for time " + str(self.t))
        self.observedDrifterPositions.append([self.t, observedDrifterPositions])

        
    def observeDrifters(self):
        """
        Observing the drifters in all particles
        """
        drifterPositions = np.empty((self.getNumParticles(), self.driftersPerOceanModel, 2))
        for p in range(self.getNumParticles()):
            drifterPositions[p,:,:] = self.particles[p].drifters.getDrifterPositions()
        return drifterPositions
    
    def observeParticles(self):
        """
        Applying the observation operator on each particle.

        Structure on the output:
        [
        particle 1:  [u_1, v_1], ... , [u_D, v_D],
        particle 2:  [u_1, v_1], ... , [u_D, v_D],
        particle Ne: [u_1, v_1], ... , [u_D, v_D]
        ]
        numpy array with dimensions (particles, drifters, 2)

        The two values per particle drifter is either velocity or position, depending on 
        the observation type.
        """
        if self.observation_type == dautils.ObservationType.DrifterPosition:
            return self.observeDrifters()

        elif self.observation_type == dautils.ObservationType.UnderlyingFlow or \
             self.observation_type == dautils.ObservationType.DirectUnderlyingFlow:

            observedState = np.empty((self.getNumParticles(), \
                                      self.driftersPerOceanModel, 2))

            trueState = self.observeTrueState()
            # trueState = [[x1, y1, u1, v1], ..., [xD, yD, uD, vD]]

            for p in range(self.numParticles):
                # Downloading ocean state without ghost cells
                Hi = self.particles[0].downloadBathymetry()[1]
                eta, hu, hv = self.downloadParticleOceanState(p)

                for d in range(self.driftersPerOceanModel):
                    id_x = np.int(np.floor(trueState[d,0]/self.dx))
                    id_y = np.int(np.floor(trueState[d,1]/self.dy))

                    depth = Hi[id_y, id_x]
                    observedState[p,d,0] = hu[id_y, id_x]/(depth + eta[id_y, id_x])
                    observedState[p,d,1] = hv[id_y, id_x]/(depth + eta[id_y, id_x])
            return observedState
        
    def observeTrueDrifters(self):
        """
        Observing the drifters in the syntetic true state.
        """
        return self.particles[self.obs_index].drifters.getDrifterPositions()
        
        
    def observeTrueState(self):
        """
        Applying the observation operator on the syntetic true state.

        Returns a numpy array with D drifter positions and drifter velocities
        [[x_1, y_1, u_1, v_1], ... , [x_D, y_D, u_D, v_D]]
        If the observation operator is drifter positions, u and v are not included.
        """
        #print "(Remember to comment in this one again) CHECKIFALREADYOBSERVED"
        if self.observedDrifterPositions[-1][0] != self.t:
            self._addObservation(self.observeTrueDrifters())

        if self.observation_type == dautils.ObservationType.DrifterPosition:
            return self.observedDrifterPositions[-1][1]

        elif self.observation_type == dautils.ObservationType.UnderlyingFlow:
            dt = self.observedDrifterPositions[-1][0] - self.observedDrifterPositions[-2][0]
            trueState = np.empty((self.driftersPerOceanModel, 4))
            for d in range(self.driftersPerOceanModel):
                x = self.observedDrifterPositions[-1][1][d,0]
                y = self.observedDrifterPositions[-1][1][d,1]
                dx = x - self.observedDrifterPositions[-2][1][d, 0]
                dy = y - self.observedDrifterPositions[-2][1][d, 1]
                u = dx/dt
                v = dy/dt
                trueState[d,:] = np.array([x, y , u, v])
            return trueState

        elif self.observation_type == dautils.ObservationType.DirectUnderlyingFlow:
            trueState = np.empty((self.driftersPerOceanModel, 4))
            for d in range(self.driftersPerOceanModel):
                x = self.observedDrifterPositions[-1][1][d,0]
                y = self.observedDrifterPositions[-1][1][d,1]
                id_x = np.int(np.floor(x/self.dx))
                id_y = np.int(np.floor(y/self.dy))

                depth = self.particles[self.obs_index].downloadBathymetry()[1][id_y, id_x]

                # Downloading ocean state without ghost cells
                eta, hu, hv = self.downloadParticleOceanState(self.obs_index)
                u = hu[id_y, id_x]/(depth + eta[id_y, id_x])
                v = hv[id_y, id_x]/(depth + eta[id_y, id_x])

                trueState[d,:] = np.array([x, y, u, v])
            return trueState

    def step(self, t, stochastic_particles=True, stochastic_truth=True):
        """
        Function which makes all particles step until time t.
        apply_stochastic_term: Boolean value for whether the stochastic
            perturbation (if any) should be applied.
        """
        for p in range(self.getNumParticles()+1):
            #print "Starting sim " + str(p)
            if p == self.obs_index:
                self.t = self.particles[p].step(t, apply_stochastic_term=stochastic_truth)
            else:
                self.t = self.particles[p].step(t, apply_stochastic_term=stochastic_particles)
            #print "Finished sim " + str(p)      
        return self.t
    
    def step_truth(self, t, stochastic=True):
        self.t = self.particles[self.obs_index].step(t, apply_stochastic_term=stochastic)
        return self.t
    
    def step_particles(self, t, stochastic=True):
        for p in range(self.getNumParticles()):
            dummy_t = self.particles[p].step(t, apply_stochastic_term=stochastic)
        return self.t
    
    def getDistances(self, obs=None):
        """
        Shows the distance that each drifter is from the observation in the following structure:
        [
        particle 1: [dist_drifter_1, ..., dist_drifter_D], 
        ...
        particle Ne: [dist_drifter_1, ..., dist_drifter_D], 
        ]
        """
        return np.linalg.norm(self.observeTrueDrifters() - self.observeDrifters(),  axis=2)
    
    def getInnovations(self, obs=None):
        """
        Obtaining the innovation vectors, y^m - H(\psi_i^m)

        Returns a numpy array with dimensions (particles, drifters, 2)

        """
        if obs is None:
            trueState = self.observeTrueState()

        if self.observation_type == dautils.ObservationType.UnderlyingFlow or \
           self.observation_type == dautils.ObservationType.DirectUnderlyingFlow:
            # Change structure of trueState
            # from: [[x1, y1, u1, v1], ..., [xD, yD, uD, vD]]
            # to:   [[u1, v1], ..., [uD, vD]]
            trueState = trueState[:, 2:]

        #else, structure of trueState is already fine: [[x1, y1], ..., [xD, yD]]

        innovations = trueState - self.observeParticles()
        return innovations
            
    def getInnovationNorms(self, obs=None):
        
        # Innovations have the structure 
        # [ particle: [drifter: [x, y] ] ], or
        # [ particle: [drifter: [u, v] ] ]
        # We simply gather find the norm for each particle:
        innovations = self.getInnovations(obs=obs)
        return np.linalg.norm(np.linalg.norm(innovations, axis=2), axis=1)
            
    def printMaxOceanStates(self):
        simNo = 0
        for oceanState in self.particles:
            eta, hu, hv = oceanState.download()
            if simNo == self.obs_index:
                print("------- simNo: True state -------")
            else:
                print("------- simNo: " + str(simNo) + " -------")
            print("t = " + str(oceanState.t))
            print("Max eta: ", np.max(eta))
            print("Max hu:  ", np.max(hu))
            print("Max hv:  ", np.max(hv))
            simNo = simNo + 1
    

    def getGaussianWeight(self, innovations=None, normalize=True):
        """
        Calculates a weight associated to every particle, based on its innovation vector, using 
        Gaussian uncertainty for the observation.
        """

        if innovations is None:
            innovations = self.getInnovations()
        observationVariance = self.getObservationVariance()
        Rinv = None

        weights = np.zeros(innovations.shape[0])
        if len(innovations.shape) == 1:
            weights = (1.0/np.sqrt(2*np.pi*observationVariance))* \
                    np.exp(- (innovations**2/(2*observationVariance)))

        else:
            Ne = self.getNumParticles()
            Nd = innovations.shape[1] # number of drifters per particle
            Ny = innovations.shape[2]

            Rinv = self.observation_cov_inverse
            R = self.observation_cov

            for i in range(Ne):
                w = 0.0
                for d in range(Nd):
                    inn = innovations[i,d,:]
                    w += np.dot(inn, np.dot(Rinv, inn.transpose()))

                ## TODO: Restructure to do the normalization before applying
                # the exponential function. The current version is sensitive to overflows.
                weights[i] = (1.0/((2*np.pi)**Nd*np.linalg.det(R)**(Nd/2.0)))*np.exp(-0.5*w)
        if normalize:
            return weights/np.sum(weights)
        return weights
    
    def getCauchyWeight(self, distances=None, normalize=True):
        """
        Calculates a weight associated to every particle, based on its distance from the observation, 
        using Cauchy distribution based on the uncertainty of the position of the observation.
        This distribution should be used if wider tails of the distribution is beneficial.
        """
        
        if distances is None:
            distances = self.getDistances()
        observationVariance = self.getObservationVariance()
            
        weights = 1.0/(np.pi*np.sqrt(observationVariance)*(1 + (distances**2/observationVariance)))
        if normalize:
            return weights/np.sum(weights)
        return weights
    
    
            
    def getEnsembleMean(self):
        return None
    def getDomainSizeX(self):
        return self.nx*self.dx
    def getDomainSizeY(self):
        return self.ny*self.dy
    def getObservationVariance(self):
        return self.observation_variance
    def getNumParticles(self):
        return self.numParticles
    
    def findLargestPossibleTimeStep(self):
        """
        Un-optimized utility function to check largest allowed time-step for 
        the shallow water model.
        dt < 0.25*min[ dx/max(|u +- sqrt(g(H+eta))|), dy/max(|v +- sqrt(g(H+eta))|)]
        """
        max_u = 0
        max_v = 0
        for oceanState in self.particles:
            eta_tmp, hu, hv = oceanState.download()
            Hi = oceanState.downloadBathymetry()[0]
            w = 0.25*(Hi[1:,1:] + Hi[1:,:-1] + Hi[:-1,1:] + Hi[:-1,:-1]) + eta_tmp
            hu /= w
            hv /= w
            Hi = None
            w = np.sqrt(self.g*w)
            
            # using eta_tmp buffer for {u|v} +- sqrt(gw)
            max_u = max(max_u, np.max(np.abs(hu + w)))
            max_u = max(max_u, np.max(np.abs(hu - w)))
            max_v = max(max_v, np.max(np.abs(hv + w)))
            max_v = max(max_v, np.max(np.abs(hv - w)))
            
        return 0.25*min(self.dx/max_u, self.dy/max_v)
            
            
    def getEnsembleVarAndRMSEUnderDrifter(self, t, allDrifters=False):
        """
        Putting entries in the statistical arrays for single cells.
        """
        
        drifter_pos = self.observeTrueDrifters()[0,:]
        
        # downloadTrueOceanState and downloadParticleOceanState gives us interior domain only,
        # and no ghost cells.
        cell_id_x = int(np.floor(drifter_pos[0]/self.dx))
        cell_id_y = int(np.floor(drifter_pos[1]/self.dy))
        
        eta_true_array, hu_true_array, hv_true_array = self.downloadTrueOceanState()
        
        eta_true = eta_true_array[cell_id_y, cell_id_x]
        hu_true  =  hu_true_array[cell_id_y, cell_id_x]
        hv_true  =  hv_true_array[cell_id_y, cell_id_x]
        
        eta_mean = 0.0
        hu_mean = 0.0
        hv_mean = 0.0
        eta_rmse = 0.0
        hu_rmse = 0.0
        hv_rmse = 0.0
        eta_sigma = 0.0
        hu_sigma = 0.0
        hv_sigma = 0.0
        
        for p in range(self.getNumParticles()):
            tmp_eta, tmp_hu, tmp_hv = self.downloadParticleOceanState(p)
            eta_mean += tmp_eta[cell_id_y, cell_id_x]
            hu_mean += tmp_hu[cell_id_y, cell_id_x]
            hv_mean += tmp_hv[cell_id_y, cell_id_x]
            eta_rmse += (eta_true - tmp_eta[cell_id_y, cell_id_x])**2
            hu_rmse += (hu_true - tmp_hu[cell_id_y, cell_id_x])**2
            hv_rmse += (hv_true - tmp_hv[cell_id_y, cell_id_x])**2
        
        eta_rmse = np.sqrt(eta_rmse/(self.getNumParticles()+1))
        hu_rmse  = np.sqrt(hu_rmse /(self.getNumParticles()+1))
        hv_rmse  = np.sqrt(hv_rmse /(self.getNumParticles()+1))
        
        eta_mean = eta_mean/self.getNumParticles()
        hu_mean = hu_mean/self.getNumParticles()
        hv_mean = hv_mean/self.getNumParticles()
        
        # RMSE according to the paper draft
        eta_rmse = np.sqrt((eta_true - eta_mean)**2)
        hu_rmse  = np.sqrt((hu_true  - hu_mean )**2)
        hv_rmse  = np.sqrt((hv_true  - hv_mean )**2)
        
        for p in range(self.getNumParticles()):
            tmp_eta, tmp_hu, tmp_hv = self.downloadParticleOceanState(p)
            eta_sigma += (tmp_eta[cell_id_y, cell_id_x] - eta_mean)**2
            hu_sigma  += (tmp_hu[cell_id_y, cell_id_x]  - hu_mean )**2
            hv_sigma  += (tmp_hv[cell_id_y, cell_id_x]  - hv_mean )**2
        
        eta_sigma = np.sqrt(eta_sigma/(1.0 + self.getNumParticles()))
        hu_sigma  = np.sqrt( hu_sigma/(1.0 + self.getNumParticles()))
        hv_sigma  = np.sqrt( hv_sigma/(1.0 + self.getNumParticles()))
        
        eta_r = eta_sigma/eta_rmse
        hu_r  =  hu_sigma/hu_rmse
        hv_r  =  hv_sigma/hv_rmse
        
        self.varianceUnderDrifter_eta.append(eta_sigma)
        self.varianceUnderDrifter_hu.append(hu_sigma)
        self.varianceUnderDrifter_hv.append(hv_sigma)
        self.rmseUnderDrifter_eta.append(eta_rmse)
        self.rmseUnderDrifter_hu.append(hu_rmse)
        self.rmseUnderDrifter_hv.append(hv_rmse)
        self.rUnderDrifter_eta.append(eta_r)
        self.rUnderDrifter_hu.append(hu_r)
        self.rUnderDrifter_hv.append(hv_r)
        self.tArray.append(t)
        

        
    
    def downloadEnsembleStatisticalFields(self):
        """
        Find the ensemble mean, and the ensemble root mean-square error. 
        """
        eta_true, hu_true, hv_true = self.downloadTrueOceanState()
        
        eta_mean = np.zeros_like(eta_true)
        hu_mean = np.zeros_like(hu_true)
        hv_mean = np.zeros_like(hv_true)
        eta_rmse = np.zeros_like(eta_true)
        hu_rmse = np.zeros_like(hu_true)
        hv_rmse = np.zeros_like(hv_true)
        eta_r = np.zeros_like(eta_true)
        hu_r = np.zeros_like(hu_true)
        hv_r = np.zeros_like(hv_true)
        
        for p in range(self.getNumParticles()):
            tmp_eta, tmp_hu, tmp_hv = self.downloadParticleOceanState(p)
            eta_mean += tmp_eta
            hu_mean += tmp_hu
            hv_mean += tmp_hv
            eta_rmse += (eta_true - tmp_eta)**2
            hu_rmse += (hu_true - tmp_hu)**2
            hv_rmse += (hv_true - tmp_hv)**2
        
        eta_rmse = np.sqrt(eta_rmse/(self.getNumParticles()))
        hu_rmse  = np.sqrt(hu_rmse /(self.getNumParticles()))
        hv_rmse  = np.sqrt(hv_rmse /(self.getNumParticles()))
        
        eta_mean = eta_mean/self.getNumParticles()
        hu_mean = hu_mean/self.getNumParticles()
        hv_mean = hv_mean/self.getNumParticles()
        
        # RMSE according to the paper draft
        eta_rmse = np.sqrt((eta_true - eta_mean)**2)
        hu_rmse  = np.sqrt((hu_true  - hu_mean )**2)
        hv_rmse  = np.sqrt((hv_true  - hv_mean )**2)
        
        for p in range(self.getNumParticles()):
            tmp_eta, tmp_hu, tmp_hv = self.downloadParticleOceanState(p)
            eta_r += (tmp_eta - eta_mean)**2
            hu_r  += (tmp_hu  - hu_mean )**2
            hv_r  += (tmp_hv  - hv_mean )**2
            
        eta_r = np.sqrt(eta_r/(1.0 + self.getNumParticles()))/eta_rmse
        hu_r  = np.sqrt(hu_r /(1.0 + self.getNumParticles()))/hu_rmse
        hv_r  = np.sqrt(hv_r /(1.0 + self.getNumParticles()))/hv_rmse
        
        #print "min-max [eta, hu, hv]_r: ", [(np.min(eta_r), np.max(eta_r)), \
        #                                  (np.min(hu_r ), np.max(hu_r )), \
        #                                  (np.min(hv_r ), np.max(hv_r ))]
        
        return eta_mean, hu_mean, hv_mean, eta_rmse, hu_rmse, hv_rmse, eta_r, hu_r, hv_r
    
    def downloadParticleOceanState(self, particleNo):
        assert(particleNo < self.getNumParticles()+1), "particle out of range"
        return self.particles[particleNo].download(interior_domain_only=True)
        
    def downloadTrueOceanState(self):
        return self.particles[self.obs_index].download(interior_domain_only=True)
        
    def _updateMinMax(self, eta, hu, hv, fieldRanges):
        fieldRanges[0] = min(fieldRanges[0], np.min(eta))
        fieldRanges[1] = max(fieldRanges[1], np.max(eta))
        fieldRanges[2] = min(fieldRanges[2], np.min(hu ))
        fieldRanges[3] = max(fieldRanges[3], np.max(hu ))
        fieldRanges[4] = min(fieldRanges[4], np.min(hv ))
        fieldRanges[5] = max(fieldRanges[5], np.max(hv ))

    def plotEnsemble(self):
        """
        Utility function to plot:
            - the true state
            - the ensemble mean
            - the state of up to 5 first ensemble members
        """
        matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

        numParticlePlots = min(self.getNumParticles(), 5)
        numPlots = numParticlePlots + 3
        plotCols = 4
        fig = plt.figure(figsize=(7, 2*numPlots))

        eta_true, hu_true, hv_true = self.downloadTrueOceanState()
        fieldRanges = np.zeros(6) # = [eta_min, eta_max, hu_min, hu_max, hv_min, hv_max]
        
        self._updateMinMax(eta_true, hu_true, hv_true, fieldRanges)
        X,Y = np.meshgrid(np.arange(0, self.nx, 1.0), np.arange(0, self.ny, 1.0))


        eta_mean = np.zeros_like(eta_true)
        hu_mean = np.zeros_like(hu_true)
        hv_mean = np.zeros_like(hv_true)
        eta_mrse = np.zeros_like(eta_true)
        hu_mrse = np.zeros_like(hu_true)
        hv_mrse = np.zeros_like(hv_true)
        
        eta = [None]*numParticlePlots
        hu = [None]*numParticlePlots
        hv = [None]*numParticlePlots
        for p in range(self.getNumParticles()):
            if p < numParticlePlots:
                eta[p], hu[p], hv[p] = self.downloadParticleOceanState(p)
                eta_mean += eta[p]
                hu_mean += hu[p]
                hv_mean += hv[p]
                eta_mrse += (eta_true - eta[p])**2
                hu_mrse += (hu_true - hu[p])**2
                hv_mrse += (hv_true - hv[p])**2
                self._updateMinMax(eta[p], hu[p], hv[p], fieldRanges)
            else:
                tmp_eta, tmp_hu, tmp_hv = self.downloadParticleOceanState(p)
                eta_mean += tmp_eta
                hu_mean += tmp_hu
                hv_mean += tmp_hv
                eta_mrse += (eta_true - tmp_eta)**2
                hu_mrse += (hu_true - tmp_hu)**2
                hv_mrse += (hv_true - tmp_hv)**2
                self._updateMinMax(tmp_eta, tmp_hu, tmp_hv, fieldRanges)

        eta_mean = eta_mean/self.getNumParticles()
        hu_mean = hu_mean/self.getNumParticles()
        hv_mean = hv_mean/self.getNumParticles()
        eta_mrse = np.sqrt(eta_mrse/self.getNumParticles())
        hu_mrse = np.sqrt(hu_mrse/self.getNumParticles())
        hv_mrse = np.sqrt(hv_mrse/self.getNumParticles())

        eta_levels = np.linspace(fieldRanges[0], fieldRanges[1], 10)
        hu_levels = np.linspace(fieldRanges[2], fieldRanges[3], 10)
        hv_levels = np.linspace(fieldRanges[4], fieldRanges[5], 10)
        
        eta_lim = np.max(np.abs(fieldRanges[:2]))
        huv_lim = np.max(np.abs(fieldRanges[2:]))
        
        plt.subplot(numPlots, plotCols, 1)
        plt.imshow(eta_true, origin='lower', vmin=-eta_lim, vmax=eta_lim)
        plt.contour(eta_true, levels=eta_levels, colors='black', alpha=0.5)
        plt.title("true eta")
        plt.subplot(numPlots, plotCols, 2)
        plt.imshow(hu_true, origin='lower', vmin=-huv_lim, vmax=huv_lim)
        plt.contour(hu_true, levels=hu_levels, colors='black', alpha=0.5)
        plt.title("true hu")
        plt.subplot(numPlots, plotCols, 3)
        plt.imshow(hv_true, origin='lower', vmin=-huv_lim, vmax=huv_lim)
        plt.contour(hv_true, levels=hv_levels, colors='black', alpha=0.5)
        plt.title("true hv")
        plt.subplot(numPlots, plotCols, 4)
        plt.quiver(X, Y, hu_true, hv_true)
        plt.title("velocity field")
        
        plt.subplot(numPlots, plotCols, 5)
        plt.imshow(eta_mean, origin='lower', vmin=-eta_lim, vmax=eta_lim)
        plt.contour(eta_mean, levels=eta_levels, colors='black', alpha=0.5)
        plt.title("mean eta")
        plt.subplot(numPlots, plotCols, 6)
        plt.imshow(hu_mean, origin='lower', vmin=-huv_lim, vmax=huv_lim)
        plt.contour(hu_mean, levels=hu_levels, colors='black', alpha=0.5)
        plt.title("mean hu")
        plt.subplot(numPlots, plotCols, 7)
        plt.imshow(hv_mean, origin='lower', vmin=-huv_lim, vmax=huv_lim)
        plt.contour(hv_mean, levels=hv_levels, colors='black', alpha=0.5)
        plt.title("mean hv")
        plt.subplot(numPlots, plotCols, 8)
        plt.quiver(X, Y, hu_mean, hv_mean)
        plt.title("velocity field")
        
        mrse_max = max(np.max(eta_mrse), np.max(hu_mrse), np.max(hv_mrse))
        mrse_min = min(np.min(eta_mrse), np.min(hu_mrse), np.min(hv_mrse))
        mrse_levels = np.linspace(mrse_max, mrse_min, 10)
        
        plt.subplot(numPlots, plotCols, 9)
        plt.imshow(eta_mrse, origin='lower', vmin=-eta_lim, vmax=eta_lim)
        plt.contour(eta_mrse, levels=eta_levels, colors='black', alpha=0.5)
        plt.title("RMSE eta")
        plt.subplot(numPlots, plotCols, 10)
        plt.imshow(hu_mrse, origin='lower', vmin=-huv_lim, vmax=huv_lim)
        plt.contour(hu_mrse, levels=hu_levels, colors='black', alpha=0.5)
        plt.title("RMSE hu")
        plt.subplot(numPlots, plotCols, 11)
        plt.imshow(hv_mrse, origin='lower', vmin=-huv_lim, vmax=huv_lim)
        #plt.colorbar() # TODO: Find a nice way to include colorbar to this plot...
        plt.contour(hv_mrse, levels=hv_levels, colors='black', alpha=0.5)
        plt.title("RMSE hv")

        for p in range(numParticlePlots):
            plt.subplot(numPlots, plotCols, 13+p*plotCols)
            plt.imshow(eta[p], origin='lower', vmin=-eta_lim, vmax=eta_lim)
            plt.contour(eta[p], levels=eta_levels, colors='black', alpha=0.5)
            plt.title("particle eta")
            plt.subplot(numPlots, plotCols, 13+p*plotCols + 1)
            plt.imshow(hu[p], origin='lower', vmin=-huv_lim, vmax=huv_lim)
            plt.contour(hu[p], levels=hu_levels, colors='black', alpha=0.5)
            plt.title("particle hu")
            plt.subplot(numPlots, plotCols, 13+p*plotCols + 2)
            plt.imshow(hv[p], origin='lower', vmin=-huv_lim, vmax=huv_lim)
            plt.contour(hv[p], levels=hv_levels, colors='black', alpha=0.5)
            plt.title("particle hv")
            plt.subplot(numPlots, plotCols, 13+p*plotCols + 3)
            plt.quiver(X, Y, hu[p], hv[p])
            plt.title("velocity field")
            
        plt.axis('tight')
    
    def plotDistanceInfo(self, title=None, printInfo=False):
        """
        Utility function for generating informative plots of the ensemble relative to the observation
        """
        if self.observation_type == dautils.ObservationType.UnderlyingFlow or \
           self.observation_type == dautils.ObservationType.DirectUnderlyingFlow:
            return self.plotVelocityInfo(title=title, printInfo=printInfo)

        fig = plt.figure(figsize=(10,6))
        gridspec.GridSpec(2, 3)

        # PLOT POSITIONS OF PARTICLES AND OBSERVATIONS
        ax0 = plt.subplot2grid((2,3), (0,0))
        plt.plot(self.observeParticles()[:,:,0].flatten(), \
                 self.observeParticles()[:,:,1].flatten(), 'b.')
        plt.plot(self.observeTrueState()[:,0], \
                 self.observeTrueState()[:,1], 'r.')
        ensembleMean = self.getEnsembleMean()
        if ensembleMean is not None:
            plt.plot(ensembleMean[0], ensembleMean[1], 'r+')
        plt.xlim(0, self.getDomainSizeX())
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim(0, self.getDomainSizeY())
        plt.title("Particle positions")

        # PLOT DISCTRIBUTION OF PARTICLE DISTANCES AND THEORETIC OBSERVATION PDF
        ax0 = plt.subplot2grid((2,3), (0,1), colspan=2)
        innovations = self.getInnovationNorms()
        obs_var = self.getObservationVariance()
        plt.hist(innovations, bins=30, \
                 range=(0, max(min(self.getDomainSizeX(), self.getDomainSizeY()), np.max(innovations))),\
                 normed=True, label="particle innovations")

        # With observation 
        x = np.linspace(0, max(self.getDomainSizeX(), self.getDomainSizeY()), num=100)
        gauss_pdf = self.getGaussianWeight(x, normalize=False)
        plt.plot(x, gauss_pdf, 'g', label="pdf directly from innovations")
        plt.legend()
        plt.title("Distribution of particle innovations")

        # PLOT SORTED DISTANCES FROM OBSERVATION
        ax0 = plt.subplot2grid((2,3), (1,0), colspan=3)
        gaussWeights = self.getGaussianWeight()
        indices_sorted_by_observation = innovations.argsort()
        ax0.plot(gaussWeights[indices_sorted_by_observation]/np.max(gaussWeights), 'g', label="Gauss weight")
        ax0.set_ylabel('Relative weight')
        ax0.grid()
        ax0.set_ylim(0,1.4)
        plt.legend(loc=7)

        ax1 = ax0.twinx()
        ax1.plot(innovations[indices_sorted_by_observation], label="innovations")
        ax1.set_ylabel('Innovations from observation', color='b')

        plt.title("Sorted innovations from observation")

        if title is not None:
            plt.suptitle(title, fontsize=16)
        return fig
            
    def _fillPolarPlot(self, ax, drifter_id=0, printInfo=False):
        max_r = 0
        observedParticles = self.observeParticles()[:, drifter_id, :]
        if printInfo: print("observedParticles: \n" +str(observedParticles))
        for p in range(self.numParticles):
            u, v = observedParticles[p,0], observedParticles[p,1]
            r = np.sqrt(u**2 + v**2)
            max_r = max(max_r, r)
            theta = np.arctan(v/u)
            if (u < 0):
                theta += np.pi
            arr1 = plt.arrow(theta, 0, 0, r, alpha = 0.5, \
                             length_includes_head=True, \
                             edgecolor = 'green', facecolor = 'green', zorder = 5)

        obs_u = self.observeTrueState()[drifter_id, 2]
        obs_v = self.observeTrueState()[drifter_id, 3]
        if printInfo: print("observedTrueState: " + str((obs_u, obs_v)))
        obs_r = np.sqrt(obs_u**2 + obs_v**2)
        max_r = max(max_r, obs_r)
        obs_theta = np.arctan(obs_v/obs_u)
        if (obs_u < 0):
            obs_theta += np.pi
        arr1 = plt.arrow(obs_theta, 0, 0, obs_r, alpha = 0.5,\
                         length_includes_head=True, \
                         edgecolor = 'red', facecolor = 'red', zorder = 5)


        #ax.plot(theta, r, color='#ee8d18', lw=3)
        ax.set_rmax(max_r*1.2)
        plt.grid(True)
        plt.title("Observations from drifter " + str(drifter_id))


    def plotVelocityInfo(self, title=None, printInfo=False):
        """
        Utility function for generating informative plots of the ensemble relative to the observation
        """

        fig = None
        plotRows = 2
        if self.driftersPerOceanModel == 1:
            fig = plt.figure(figsize=(10,6))
        else:
            fig = plt.figure(figsize=(10,9))
            plotRows = 3
        gridspec.GridSpec(plotRows, 3)


        # PLOT POSITIONS OF PARTICLES AND OBSERVATIONS
        ax = plt.subplot2grid((plotRows,3), (0,0), polar=True, axisbg='#ffffff')
        self._fillPolarPlot(ax, drifter_id=0, printInfo=printInfo)

        # PLOT DISCTRIBUTION OF PARTICLE DISTANCES AND THEORETIC OBSERVATION PDF
        ax0 = plt.subplot2grid((plotRows,3), (0,1), colspan=2)
        innovations = self.getInnovationNorms()
        obs_var = self.getObservationVariance()
        range_x = np.sqrt(obs_var)*20

        # With observation 
        x = np.linspace(0, range_x, num=100)
        gauss_pdf = self.getGaussianWeight(x, normalize=False)
        plt.plot(x, gauss_pdf, 'g', label="pdf directly from innovations")
        plt.legend()
        plt.title("Distribution of particle innovations")

        #hisograms:
        ax1 = ax0.twinx()
        ax1.hist(innovations, bins=30, \
                 range=(0, range_x),\
                 normed=True, label="particle innovations (norm)")

        # PLOT SORTED DISTANCES FROM OBSERVATION
        ax0 = plt.subplot2grid((plotRows,3), (1,0), colspan=3)
        gaussWeights = self.getGaussianWeight()
        indices_sorted_by_observation = innovations.argsort()
        ax0.plot(gaussWeights[indices_sorted_by_observation]/np.max(gaussWeights),\
                 'g', label="Weight directly from innovations")
        ax0.set_ylabel('Weights directly from innovations', color='g')
        ax0.grid()
        ax0.set_ylim(0,1.4)
        #plt.legend(loc=7)
        ax0.set_xlabel('Particle ID')

        ax1 = ax0.twinx()
        ax1.plot(innovations[indices_sorted_by_observation], label="innovations")
        ax1.set_ylabel('Innovations', color='b')

        plt.title("Sorted distances from observation")

        if self.driftersPerOceanModel > 1:
            for drifter_id in range(1,min(4, self.driftersPerOceanModel)):
                ax = plt.subplot2grid((plotRows,3), (2,drifter_id-1), polar=True, axisbg='#ffffff')
                self._fillPolarPlot(ax, drifter_id=drifter_id, printInfo=printInfo)

        if title is not None:
            plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
            