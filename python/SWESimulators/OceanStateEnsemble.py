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


class OceanStateEnsemble(object):
        
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
        
    def setGridInfo(self, nx, ny, dx, dy, dt, boundaryConditions):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        
        self.observation_variance = 5*dx
        
        self.initialization_variance = 10*dx
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
            
        # Create base initial data:
        self.base_eta = np.zeros(dataShape, dtype=np.float32, order='C')
        self.base_hu  = np.zeros(dataShape, dtype=np.float32, order='C');
        self.base_hv  = np.zeros(dataShape, dtype=np.float32, order='C');
        
        # Bathymetry:
        waterDepth = 5
        self.base_Hi = np.ones((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C')*waterDepth
 
    
    def setParameters(self, f, g=9.81, beta=0, r=0, wind=WindStress.NoWindStress()):
        self.g = g
        self.f = f
        self.beta = beta
        self.r = r
        self.wind = wind
    
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
        drifterPositions = np.empty((0,2))
        for oceanState in self.particles[:-1]:
            drifterPositions = np.append(drifterPositions, 
                                         oceanState.drifters.getDrifterPositions(),
                                         axis=0)
        return drifterPositions
    
    def observeTrueState(self):
        observation = self.particles[self.obs_index].drifters.getDrifterPositions()
        return observation[0,:]
    
    def step(self, t):
        simNo = 0
        for oceanState in self.particles:
            #print "Starting sim " + str(simNo)
            oceanState.step(t)
            #print "Finished sim " + str(simNo)      
            simNo = simNo + 1
    
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
    
    
                    
            
    ### Duplication of code
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
            