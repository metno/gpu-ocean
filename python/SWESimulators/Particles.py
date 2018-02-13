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

import Common

"""
Class holding the global set of particles.
"""
class GlobalParticles:
    
    """
    Creates a GlobalParticles object for drift trajectory ensemble.
    
    numParticles: number of particles in the ensemble, not included the observation
    observation_variance: uncertainty of observation position
    boundaryConditions: BoundaryConditions object, relevant during re-initialization of particles.    
    """
    def __init__(self, numParticles, observation_variance=0.1, boundaryConditions=Common.BoundaryConditions()):
        
        self.numParticles = numParticles
        
        # Observation index is the last particle
        self.obs_index = self.numParticles 
        self.observation_variance = observation_variance
        
        # One position for every particle plus observation
        self.positions = np.zeros((self.numParticles + 1, 2))
        
        self.domain_size_x = 1.0
        self.domain_size_y = 1.0
        
        # Boundary conditions are read from a BoundaryConditions object
        self.boundaryConditions = boundaryConditions
    
    """
    Makes an independent indentical copy of the current object
    """
    def copy(self):
        copyOfSelf = GlobalParticles(self.numParticles,
                                     observation_variance = self.observation_variance,
                                     boundaryConditions = self.boundaryConditions)
        
        copyOfSelf.positions = self.positions.copy()
        
        copyOfSelf.domain_size_x = self.domain_size_x
        copyOfSelf.domain_size_y = self.domain_size_y
        
        return copyOfSelf
    
    """
    Creates a GlobalParticle object where all parameters are similar to the current object, but without copying the particle positions.
    numParticles: The number of particles to be held by the new object, observation not included.
    """
    def copyEnv(self, numParticles):
        copyOfSelf = GlobalParticles(numParticles,
                                     observation_variance = self.observation_variance,
                                     boundaryConditions = self.boundaryConditions)
        
        # Copy observation
        copyOfSelf.positions[copyOfSelf.obs_index, :] = self.positions[self.obs_index,:]
        
        copyOfSelf.domain_size_x = self.domain_size_x
        copyOfSelf.domain_size_y = self.domain_size_y
        
        return copyOfSelf
        
    ### GETs
    def getNumParticles(self):
        return self.numParticles
    
    def getObservationVariance(self):
        return self.observation_variance
    
    def getParticlePositions(self):
        return self.positions[:-1,:]
    
    def getObservationPosition(self):
        return self.positions[-1, :]
    
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
        
    
    """
    Initialization of all particles (and observation) within a rectangle of given size.
    domain_size_x [default 1.0]: size of rectangle in x-direction 
    domain_size_y [default 1.0]: size of rectangle in y-direction
    """
    def initializeParticles(self, domain_size_x=1.0, domain_size_y=1.0):
        
        # Initialize in unit square
        self.positions = np.random.rand(self.numParticles + 1, 2)
        # Ensure that the observation is in the middle 0.5x0.5 square:
        self.positions[self.obs_index, :] = self.positions[self.obs_index]*0.5 + 0.25
        
        # Map to given square
        self.positions[:,0] = self.positions[:,0]*domain_size_x
        self.positions[:,1] = self.positions[:,1]*domain_size_y
        
        self.domain_size_x = domain_size_x
        self.domain_size_y = domain_size_y
        
    """
    Returns a set of coordinates corresponding to each particles closest position to the observation,
    considering possible periodic boundary conditions
    """
    def _getClosestPositions(self):
        if not (self.boundaryConditions.isPeriodicNorthSouth() or self.boundaryConditions.isPeriodicEastWest()):
            return self.positions
        else:
            periodicPositions = self.positions.copy()
            obs_x, obs_y = periodicPositions[self.obs_index, :]
            if self.boundaryConditions.isPeriodicEastWest():
                for i in range(self.numParticles):
                    x = periodicPositions[i,0]
                    
                    pos_x = np.array([x - self.domain_size_x, x, x + self.domain_size_x])
                    dist_x = np.abs(pos_x - obs_x)
                    periodicPositions[i,0] = pos_x[np.argmin(dist_x)]

            if self.boundaryConditions.isPeriodicNorthSouth():
                for i in range(self.numParticles):
                    y = periodicPositions[i,1]
                    
                    pos_y = np.array([y - self.domain_size_y, y, y + self.domain_size_y])
                    dist_y = np.abs(pos_y - obs_y)
                    periodicPositions[i,1] = pos_y[np.argmin(dist_y)]
        return periodicPositions
        
    """
    Computes the distance between particles and observation. Possible periodic boundary conditions are taken care of.
    """
    def getDistances(self):
        distances = np.zeros(self.numParticles)
        closestPositions = self._getClosestPositions()
        obs_x, obs_y = self.positions[self.obs_index, :]
        for i in range(self.numParticles):
            distances[i] = np.sqrt( (closestPositions[i,0]-obs_x)**2 +
                                    (closestPositions[i,1]-obs_y)**2)
        return distances
        
    """
    Calculates a weight associated to every particle, based on its distance from the observation, using Gaussian uncertainty of the position of the observation 
    """
    def getGaussianWeight(self, distance=None, normalize=True):
        if distance is None:
            distance = self.getDistances()
        weights = (1.0/np.sqrt(2*np.pi*self.observation_variance**2))* \
            np.exp(- (distance**2/(2*self.observation_variance**2)))
        if normalize:
            return weights/np.sum(weights)
        return weights
    
    
    
    """
    Calculates a weight associated to every particle, based on its distance from the observation, using Cauchy distribution based on the uncertainty of the position of the observation.
    This distribution should be used if wider tails of the distribution is beneficial.
    """
    def getCauchyWeight(self, distance=None, normalize=True):
        if distance is None:
            distance = self.getDistances()
        weights = 1.0/(np.pi*self.observation_variance*(1 + (distance/self.observation_variance)**2))
        if normalize:
            return weights/np.sum(weights)
        return weights
    
    """
    Calculates the mean position of all the particles in the ensemble.
    For cases with periodic boundary conditions, the every particle position is considered in the direction which minimize the distance to the observation.
    """
    def getEnsembleMean(self):
        closestPositions = self._getClosestPositions()
        mean_x, mean_y = np.mean(closestPositions[:-1, :], axis=0)
        mean_x, mean_y = self._enforceBoundaryConditionsOnPosition(mean_x, mean_y)
        return np.array([mean_x, mean_y])
    
    """
    Maps the given coordinate to a coordinate within the domain. This function assumes that periodic boundary conditions are used, and should be considered as a private function.
    """
    def _enforceBoundaryConditionsOnPosition(self, x, y):
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
    
    """
    Enforces boundary conditions on all particles in the ensemble, and the observation.
    This function should be called whenever particles are moved, to enforce periodic boundary conditions for particles that have left the domain.
    """
    def enforceBoundaryConditions(self):
        
        if (self.boundaryConditions.isPeriodicNorthSouth() and self.boundaryConditions.isPeriodicEastWest()):
            # Loop over particles
            for i in range(self.numParticles + 1):
                x, y = self.positions[i,0], self.positions[i,1]

                x, y = self._enforceBoundaryConditionsOnPosition(x,y)

                self.positions[i,0] = x
                self.positions[i,1] = y
        else:
            print "WARNING [GlobalParticle.enforceBoundaryConditions]: Functionality not defined for non-periodic boundary conditions"
            print "\t\tDoing nothing and continuing..."
    
    """
    Utility function for generating informative plots of the ensemble relative to the observation
    """    
    def plotDistanceInfo(self, title=None):
        fig = plt.figure(figsize=(10,6))
        gridspec.GridSpec(2, 3)
        
        # PLOT POSITIONS OF PARTICLES AND OBSERVATIONS
        ax0 = plt.subplot2grid((2,3), (0,0))
        plt.plot(self.getParticlePositions()[:,0], \
                 self.getParticlePositions()[:,1], 'b.')
        plt.plot(self.getObservationPosition()[0], \
                 self.getObservationPosition()[1], 'r.')
        ensembleMean = self.getEnsembleMean()
        plt.plot(ensembleMean[0], ensembleMean[1], 'r+')
        plt.xlim(0, self.domain_size_x)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim(0, self.domain_size_y)
        plt.title("Particle positions")
        
        # PLOT DISCTRIBUTION OF PARTICLE DISTANCES AND THEORETIC OBSERVATION PDF
        ax0 = plt.subplot2grid((2,3), (0,1), colspan=2)
        distances = self.getDistances()
        plt.hist(distances, bins=30, range=(0, max(min(self.domain_size_x, self.domain_size_y), np.max(distances))),\
                 normed=True, label="particle distances")
        
        # With observation 
        x = np.linspace(0, max(self.domain_size_x, self.domain_size_y), num=100)
        cauchy_pdf = self.getCauchyWeight(x, normalize=False)
        gauss_pdf = self.getGaussianWeight(x, normalize=False)
        plt.plot(x, cauchy_pdf, 'r', label="obs Cauchy pdf")
        plt.plot(x, gauss_pdf, 'g', label="obs Gauss pdf")
        plt.legend()
        plt.title("Distribution of particle distances from observation")
        
        # PLOT SORTED DISTANCES FROM OBSERVATION
        ax0 = plt.subplot2grid((2,3), (1,0), colspan=3)
        cauchyWeights = self.getCauchyWeight(distances)
        gaussWeights = self.getGaussianWeight(distances)
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
            
            