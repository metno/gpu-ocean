# -*- coding: utf-8 -*-

"""
This python class takes care of the global ensemble of particles for EPS.

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

import Common
import DataAssimilationUtils as dautils

class Drifter(object):
    __metaclass__ = abc.ABCMeta
    
    """
    Class holding the global set of particles.
    """ 
    def __init__(self, numParticles, observation_variance=0.1, \
                 boundaryConditions=Common.BoundaryConditions(), \
                 domain_size_x=1.0, domain_size_y=1.0):
        """
        Creates a GlobalParticles object for drift trajectory ensemble.

        numParticles: number of particles in the ensemble, not included the observation
        observation_variance: uncertainty of observation position
        boundaryConditions: BoundaryConditions object, relevant during re-initialization of particles.    
        """
        
        self.numParticles = numParticles
        
        # Observation index is the last particle
        self.obs_index = self.numParticles 
        self.observation_variance = observation_variance
        
        self.positions = None # Needs to be allocated in the child class
        # Should represent all particles plus observation
        
        self.domain_size_x = domain_size_x
        self.domain_size_y = domain_size_y
        
        # Boundary conditions are read from a BoundaryConditions object
        self.boundaryConditions = boundaryConditions
    
    
    def copy(self):
        """
        Makes an independent indentical copy of the current object
        """
        pass
            
    
    ### Abstract GETs
    @abc.abstractmethod
    def getParticlePositions(self):
        pass
    
    @abc.abstractmethod
    def getObservationPosition(self):
        pass
    
    ### Abstract SETs
    @abc.abstractmethod
    def setParticlePositions(self, newParticlePositions):
        pass
    
    @abc.abstractmethod
    def setObservationPosition(self, newObservationPosition):
        pass
    
    
    
    ### GETs
    def getNumParticles(self):
        #print "\n\nUsing Drifter::getNumParticles()\n"
        return self.numParticles
    
    def getObservationVariance(self):
        return self.observation_variance
    
    def getBoundaryConditions(self):
        return self.boundaryConditions
    
    def getDomainSizeX(self):
        return self.domain_size_x
    
    def getDomainSizeY(self):
        return self.domain_size_y
        
        
    ### SETs
    def setBoundaryConditions(self, boundaryConditions):
        self.boundaryConditions = boundaryConditions
        
    def setDomainSize(self, size_x, size_y):
        self.domain_size_x = size_x
        self.domain_size_y = size_y
        
    
    ### Other abstract methods
    @abc.abstractmethod
    def enforceBoundaryConditions(self):
        """
        Enforces boundary conditions on all particles in the ensemble, and the observation.
        This function should be called whenever particles are moved, to enforce periodic boundary conditions for particles that have left the domain.
        """
        pass
        
        
    ### METHODS UNIQUELY DEFINED FOR ALL CHILD CLASSES
    
    def _getClosestPositions(self, obs=None):
        """
        Returns a set of coordinates corresponding to each particles closest position to the observation,
        considering possible periodic boundary conditions
        """
        if not (self.boundaryConditions.isPeriodicNorthSouth() or self.boundaryConditions.isPeriodicEastWest()):
            #return self.positions 
            return self.getParticlePositions()
        else:
            periodicPositions = self.getParticlePositions().copy()
            obs_x, obs_y = self.getObservationPosition()
            if obs is not None:
                obs_x = obs[0]
                obs_y = obs[1]
            if self.boundaryConditions.isPeriodicEastWest():
                for i in range(self.getNumParticles()):
                    x = periodicPositions[i,0]
                    
                    pos_x = np.array([x - self.getDomainSizeX(), x, \
                                      x + self.getDomainSizeX()])
                    dist_x = np.abs(pos_x - obs_x)
                    periodicPositions[i,0] = pos_x[np.argmin(dist_x)]

            if self.boundaryConditions.isPeriodicNorthSouth():
                for i in range(self.numParticles):
                    y = periodicPositions[i,1]
                    
                    pos_y = np.array([y - self.getDomainSizeY(), y, \
                                      y + self.getDomainSizeY()])
                    dist_y = np.abs(pos_y - obs_y)
                    periodicPositions[i,1] = pos_y[np.argmin(dist_y)]
        return periodicPositions
        
    
    def getDistances(self, obs=None):
        """
        Computes the distance between particles and observation. Possible periodic boundary conditions are taken care of.
        """
        distances = np.zeros(self.getNumParticles())
        if obs is None:
            obs = self.getObservationPosition()
            closestPositions = self._getClosestPositions()
        else:
            closestPositions = self._getClosestPositions(obs)
        for i in range(self.getNumParticles()):
            distances[i] = np.sqrt( (closestPositions[i,0]-obs[0])**2 +
                                    (closestPositions[i,1]-obs[1])**2)
        return distances
        
    
    def _enforceBoundaryConditionsOnPosition(self, x, y):
        """
        Maps the given coordinate to a coordinate within the domain. This function assumes that periodic boundary conditions are used, and should be considered as a private function.
        """
        ### TODO: SWAP the if's with while's?
        # Check what we assume is periodic boundary conditions
        if x < 0:
            x = self.domain_size_x + x
        if y < 0:
            y = self.domain_size_x + y
        if x > self.domain_size_x:
            x = x - self.domain_size_x
        if y > self.domain_size_y:
            y = y - self.domain_size_y
        return x, y
    
    
    def getEnsembleMean(self):
        """
        Calculates the mean position of all the particles in the ensemble.
        For cases with periodic boundary conditions, the every particle position is considered in the direction which minimize the distance to the observation.
        """
        closestPositions = self._getClosestPositions()
        mean_x, mean_y = np.mean(closestPositions, axis=0)
        mean_x, mean_y = self._enforceBoundaryConditionsOnPosition(mean_x, mean_y)
        return np.array([mean_x, mean_y])
    
    def initializeParticles(self, domain_size_x=1.0, domain_size_y=1.0):
        """
        Initialization of all particles (and observation) within a rectangle of given size.
        domain_size_x [default 1.0]: size of rectangle in x-direction 
        domain_size_y [default 1.0]: size of rectangle in y-direction
        """

        # Initialize in unit square
        np.copyto(self.positions, np.random.rand(self.numParticles + 1, 2))
        # Ensure that the observation is in the middle 0.5x0.5 square:
        self.positions[self.obs_index, :] = self.positions[self.obs_index]*0.5 + 0.25
        
        # Map to given square
        self.positions[:,0] = self.positions[:,0]*domain_size_x
        self.positions[:,1] = self.positions[:,1]*domain_size_y
        
        self.domain_size_x = domain_size_x
        self.domain_size_y = domain_size_y
        
    
    def plotDistanceInfo(self, title=None):
        """
        Utility function for generating informative plots of the ensemble relative to the observation
        """    
        fig = plt.figure(figsize=(10,6))
        gridspec.GridSpec(2, 3)
        
        # PLOT POSITIONS OF PARTICLES AND OBSERVATIONS
        ax0 = plt.subplot2grid((2,3), (0,0))
        plt.plot(self.getParticlePositions()[:,0], \
                 self.getParticlePositions()[:,1], 'b.')
        plt.plot(self.getObservationPosition()[0], \
                 self.getObservationPosition()[1], 'r.')
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
            
            