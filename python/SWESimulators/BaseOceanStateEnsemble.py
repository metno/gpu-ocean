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

import CDKLM16
import GPUDrifterCollection
import WindStress
import Common
import DataAssimilationUtils as dautils


class BaseOceanStateEnsemble(object):

    __metaclass__ = abc.ABCMeta
        
    def __init__(self, numParticles, cl_ctx):
        
        self.cl_ctx = cl_ctx
        
        self.numParticles = numParticles
        self.particles = [None]*(self.numParticles + 1)
        
        self.obs_index = self.numParticles
        
        self.simType = 'CDKLM16'
        
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
        
        # Default values for now:
        self.initialization_variance = 10*dx
        self.observation_variance = 5*self.dx
        self.observation_cov = self.observation_variance*np.eye(2)
        
        self.midPoint = 0.5*np.array([self.nx*self.dx, self.ny*self.dy])
        self.initialization_cov = np.eye(2)*self.initialization_variance
        
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
                               observation_variance_factor=5,
                               initialization_variance_factor=10,
                               small_scale_perturbation_amplitude=0):

        # Observation_variance_per_drifter
        self.observation_variance = observation_variance_factor*self.dx
        self.observation_cov = self.observation_variance*np.eye(2)

        self.initialization_variance = initialization_variance_factor*self.dx
        
        # TODO: Check if this variable is used anywhere.
        # Should not be defined in the Base class.
        self.initialization_cov = np.eye(2)*self.initialization_variance
        
        self.small_scale_perturbation_amplitude = small_scale_perturbation_amplitude
    
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
        
    
    def observeParticles(self):
        """
        Applying the observation operator on each particle.
        """
        drifterPositions = np.empty((0,2))
        for oceanState in self.particles[:-1]:
            drifterPositions = np.append(drifterPositions, 
                                         oceanState.drifters.getDrifterPositions(),
                                         axis=0)
        return drifterPositions
    
    def observeTrueState(self):
        """
        Applying the observation operator on the syntetic true state,
        adding an observation error.
        """
        observation = self.particles[self.obs_index].drifters.getDrifterPositions()
        return observation[0,:]
        
    def step(self, t, apply_stochastic_term=True):
        """
        Function which makes all particles step until time t.
        apply_stochastic_term: Boolean value for whether the stochastic
            perturbation (if any) should be applied.
        """
        simNo = 0
        for oceanState in self.particles:
            #print "Starting sim " + str(simNo)
            output_t = oceanState.step(t, \
                           apply_stochastic_term=apply_stochastic_term)
            #print "Finished sim " + str(simNo)      
            simNo = simNo + 1
        return output_t
    
    def getDistances(self, obs=None):
        if obs is None:
            obs = self.observeTrueState()
        distances = np.empty(0)
        counter = 0
        for oceanState in self.particles[:-1]:
            distancesFromOceanState = oceanState.drifters.getDistances(obs)
            distances = np.append(distances,
                                  distancesFromOceanState,
                                  axis=0)
            counter += 1
        return distances
    
    def getInnovations(self, obs=None):
        if obs is None:
            obs = self.observeTrueState()
        innovations = np.empty((0,2))
        counter = 0
        for oceanState in self.particles[:-1]:
            innovationsFromOceanState = oceanState.drifters.getInnovations(obs)
            innovations = np.append(innovations,
                                    innovationsFromOceanState,
                                    axis=0)
            counter += 1
        return innovations
            
    def printMaxOceanStates(self):
        simNo = 0
        for oceanState in self.particles:
            eta, hu, hv = oceanState.download()
            print "------- simNo: " + str(simNo) + " -------"
            print "t = " + str(oceanState.t)
            print "Max eta: ", np.max(eta)
            print "Max hu:  ", np.max(hu)
            print "Max hv:  ", np.max(hv)
            simNo = simNo + 1
    

                    
            
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
            
            
    
    
    def downloadEnsembleStatisticalFields(self):
        eta_true, hu_true, hv_true = self.downloadTrueOceanState()
        
        eta_mean = np.zeros_like(eta_true)
        hu_mean = np.zeros_like(hu_true)
        hv_mean = np.zeros_like(hv_true)
        eta_mrse = np.zeros_like(eta_true)
        hu_mrse = np.zeros_like(hu_true)
        hv_mrse = np.zeros_like(hv_true)
        
        for p in range(self.getNumParticles()):
            tmp_eta, tmp_hu, tmp_hv = self.downloadParticleOceanState(p)
            eta_mean += tmp_eta
            hu_mean += tmp_hu
            hv_mean += tmp_hv
            eta_mrse += (eta_true - tmp_eta)**2
            hu_mrse += (hu_true - tmp_hu)**2
            hv_mrse += (hv_true - tmp_hv)**2
            
        eta_mean = eta_mean/self.getNumParticles()
        hu_mean = hu_mean/self.getNumParticles()
        hv_mean = hv_mean/self.getNumParticles()
        eta_mrse = np.sqrt(eta_mrse/self.getNumParticles())
        hu_mrse = np.sqrt(hu_mrse/self.getNumParticles())
        hv_mrse = np.sqrt(hv_mrse/self.getNumParticles())
        
        return eta_mean, hu_mean, hv_mean, eta_mrse, hu_mrse, hv_mrse
    
    def downloadParticleOceanState(self, particleNo):
        assert(particleNo < self.getNumParticles()+1), "particle out of range"
        eta, hu, hv = self.particles[particleNo].download()
        eta = eta[2:-2, 2:-2]
        hu = hu[2:-2, 2:-2]
        hv = hv[2:-2, 2:-2]
        return eta, hu, hv
    
    def downloadTrueOceanState(self):
        eta, hu, hv = self.particles[self.obs_index].download()
        eta = eta[2:-2, 2:-2]
        hu = hu[2:-2, 2:-2]
        hv = hv[2:-2, 2:-2]
        return eta, hu, hv
    
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
        plt.title("MRSE eta")
        plt.subplot(numPlots, plotCols, 10)
        plt.imshow(hu_mrse, origin='lower', vmin=-huv_lim, vmax=huv_lim)
        plt.contour(hu_mrse, levels=hu_levels, colors='black', alpha=0.5)
        plt.title("MRSE hu")
        plt.subplot(numPlots, plotCols, 11)
        plt.imshow(hv_mrse, origin='lower', vmin=-huv_lim, vmax=huv_lim)
        #plt.colorbar() # TODO: Find a nice way to include colorbar to this plot...
        plt.contour(hv_mrse, levels=hv_levels, colors='black', alpha=0.5)
        plt.title("MRSE hv")

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
    
    def plotDistanceInfo(self, title=None):
        """
        Utility function for generating informative plots of the ensemble relative to the observation
        """    
        fig = plt.figure(figsize=(10,6))
        gridspec.GridSpec(2, 3)
        
        # PLOT POSITIONS OF PARTICLES AND OBSERVATIONS
        ax0 = plt.subplot2grid((2,3), (0,0))
        plt.plot(self.observeParticles()[:,0], \
                 self.observeParticles()[:,1], 'b.')
        plt.plot(self.observeTrueState()[0], \
                 self.observeTrueState()[1], 'r.')
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
        distances = self.getDistances()
        obs_var = self.getObservationVariance()
        plt.hist(distances, bins=30, \
                 range=(0, max(min(self.getDomainSizeX(), self.getDomainSizeY()), np.max(distances))),\
                 normed=True, label="particle distances")
        
        # With observation 
        x = np.linspace(0, max(self.getDomainSizeX(), self.getDomainSizeY()), num=100)
        cauchy_pdf = dautils.getCauchyWeight(x, obs_var, normalize=False)
        gauss_pdf = dautils.getGaussianWeight(x, obs_var, normalize=False)
        plt.plot(x, cauchy_pdf, 'r', label="obs Cauchy pdf")
        plt.plot(x, gauss_pdf, 'g', label="obs Gauss pdf")
        plt.legend()
        plt.title("Distribution of particle distances from observation")
        
        # PLOT SORTED DISTANCES FROM OBSERVATION
        ax0 = plt.subplot2grid((2,3), (1,0), colspan=3)
        cauchyWeights = dautils.getCauchyWeight(distances, obs_var)
        gaussWeights = dautils.getGaussianWeight(distances, obs_var)
        indices_sorted_by_observation = distances.argsort()
        ax0.plot(cauchyWeights[indices_sorted_by_observation]/np.max(cauchyWeights), 'r', label="Cauchy weight")
        ax0.plot(gaussWeights[indices_sorted_by_observation]/np.max(gaussWeights), 'g', label="Gauss weight")
        ax0.set_ylabel('Relative weight')
        ax0.grid()
        ax0.set_ylim(0,1.4)
        plt.legend(loc=7)
        
        ax1 = ax0.twinx()
        ax1.plot(distances[indices_sorted_by_observation], label="distance")
        ax1.set_ylabel('Distance from observation', color='b')
        
        plt.title("Sorted distances from observation")

        if title is not None:
            plt.suptitle(title, fontsize=16)
            