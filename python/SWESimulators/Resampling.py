# -*- coding: utf-8 -*-

"""
This module implements a selection of resampling schemes used for particle filters in EPS.

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
from Particles import *


"""
Create a new GlobalParticles instance based on the given indices, where duplicates are resampled from a gaussian distribution.
particles: The original ensemble before resampling
newSampleIndices: particle indices selected for resampling
reinitialization_variance: variance used when resampling duplicates
"""
def resampleParticles(particles, newSampleIndices, reinitialization_variance):
    newParticles = particles.copyEnv(len(newSampleIndices))
    #newParticles = GlobalParticles(len(newSampleIndices), particles.observation_variance, particles.boundaryConditions)
    
    if particles.numParticles != newParticles.numParticles:
        print "WARNING: The size of the new ensemble differs from the old size!"
        print "(old size, new size): ", (particles.numParticles, newParticles.numParticles)
    
    # We really do not the if. The random number with zero variance returns exactly the mean
    if reinitialization_variance == 0:
        # Simply copy the given positions
        newParticles.positions[:-1,:] = particles.positions[newSampleIndices,:].copy()
    else:
        # Make sure to make a clean copy of first resampled particle, and add a disturbance of the next ones.
        resampledOnce = np.full(particles.numParticles, False, dtype=bool)
        var = np.eye(2)*reinitialization_variance
        for i in range(len(newSampleIndices)):
            index = newSampleIndices[i]
            if resampledOnce[index]:
                newParticles.positions[i,:] = np.random.multivariate_normal(particles.positions[index,:], var)
            else:
                newParticles.positions[i,:] = particles.positions[index,:]
                resampledOnce[index] = True
                                                                            
        
        
    # Copy the observation:
    newParticles.positions[-1,:] = particles.positions[-1,:]
    
    # Enforce boundary conditions
    newParticles.enforceBoundaryConditions()
    
    return newParticles
    

"""
Probabilistic resampling of the particles based on the attached observation.
particles: A GlobalParticle object, which holds the ensemble of particles, the observation, and measures to compute the weight of particles based on this information.
reinitialization_variance: The variance used for resampling of particles that are already resampled. These duplicates are sampled around the original particle.
If reinitialization_variance is zero, exact duplications are generated.


"""
def probabilisticResampling(particles, reinitialization_variance=0):
    # Obtain weights:
    weights = particles.getGaussianWeight()
    #weights = particles.getCauchyWeight()
    
    # Create array of possible indices to resample:
    allIndices = range(particles.numParticles)
    
    # Draw new indices based from discrete distribution based on weights
    newSampleIndices = np.random.choice(allIndices, particles.numParticles, p=weights)
        
    # Return a new set of particles
    return resampleParticles(particles, newSampleIndices, reinitialization_variance)


def residualSampling(particles, reinitialization_variance=0, onlyDeterministic=False, onlyStochastic=False):
    # Obtain weights:
    #weights = particles.getCauchyWeight()
    weights = particles.getGaussianWeight()

    # Create array of possible indices to resample:
    allIndices = range(particles.numParticles)

    # Deterministic resampling based on the integer part of N*weights:
    weightsTimesN = weights*particles.numParticles
    weightsTimesNInteger = np.int64(np.floor(weightsTimesN))
    deterministicResampleIndices = np.repeat(allIndices, weightsTimesNInteger)
    
    # Stochastic resampling based on the decimal parts of N*weights:
    decimalWeights = np.mod(weightsTimesN, 1)
    decimalWeights = decimalWeights/np.sum(decimalWeights)
    stochasticResampleIndices = np.random.choice(allIndices, 
                                                 particles.numParticles - len(deterministicResampleIndices), 
                                                 p=decimalWeights)
    ### NOTE!
    # In numpy v >= 1.13, np.divmod can be used to get weightsTimesNInteger and decimalWeights from one function call.
    
    if onlyDeterministic:
        return resampleParticles(particles, deterministicResampleIndices, reinitialization_variance)
    if onlyStochastic:
        return resampleParticles(particles, stochasticResampleIndices, reinitialization_variance)
    
    return resampleParticles(particles, np.concatenate((deterministicResampleIndices, stochasticResampleIndices)), \
                             reinitialization_variance)
    
    
def stochasticUniversalSampling(particles, reinitialization_variance=0):
    # Obtain weights:
    #weights = particles.getCauchyWeight()
    weights = particles.getGaussianWeight()

    # Create array of possible indices to resample:
    allIndices = np.array(range(particles.numParticles))
    
    # Create histogram buckets based on the cumulative weights
    cumulativeWeights = np.concatenate(([0.0], np.cumsum(weights)))
    
    # Find first starting position:
    startPos = np.random.rand()/particles.numParticles
    lengths = 1.0/particles.numParticles
    #print startPos, lengths
    selectionValues = allIndices*lengths + startPos
    
    # Create a histogram of selectionValues within the cumulativeWeights buckets
    bucketValues, buckets = np.histogram(selectionValues, bins=cumulativeWeights)
    
    #newSampleIndices has now the number of times each index should be resampled
    # We need to go from [0, 0, 1, 4, 0] to [2,3,3,3,3]
    newSampleIndices = np.repeat(allIndices, bucketValues)
    
    # Return a new set of particles
    return resampleParticles(particles, newSampleIndices, reinitialization_variance)


def metropolisHastingSampling(particles,  reinitialization_variance=0):
    # Obtain weights:
    #weights = particles.getCauchyWeight()
    weights = particles.getGaussianWeight()
    
    # Create buffer for indices which should be in the new ensemble:
    newSampleIndices = np.zeros_like(weights, dtype=int)
    
    # The first member is automatically a member of the new ensemble
    newSampleIndices[0] = 0
    
    # Iterate through all weights, and apply the Metropolis-Hasting algorithm
    for i in range(1, particles.numParticles):
        # Draw random number U[0,1]
        p = np.random.rand()
        if p < weights[i]/weights[newSampleIndices[i-1]]:
            newSampleIndices[i] = i
        else:
            newSampleIndices[i] = newSampleIndices[i-1]
    
    # Return a new set of particles
    return resampleParticles(particles, newSampleIndices, reinitialization_variance)
