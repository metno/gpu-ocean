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


def resampleParticles(particles, newSampleIndices, reinitialization_variance):
    """
    Resamples the particle positions at the given indices. Duplicates are resampled from a gaussian distribution.
    
    particles: The ensemble to be resampled
    newSampleIndices: particle indices selected for resampling
    reinitialization_variance: variance used when resampling duplicates
    """
    
    oldParticlePositions = particles.getParticlePositions().copy()
    newNumberOfParticles = len(newSampleIndices)
    newParticlePositions = np.zeros((newNumberOfParticles, 2))
    
    if particles.getNumParticles() != newNumberOfParticles:
        raise RuntimeError("ERROR: The size of the new ensemble differs from the old size!\n" + \
                           "(old size, new size): " + str((particles.getNumParticles(), newNumberOfParticles)) + \
                           "\nWe can fix this in the future by requiring a function resizeEnsemble")
        
    # We really do not the if. The random number with zero variance returns exactly the mean
    if reinitialization_variance == 0:
        # Simply copy the given positions
        newParticlePositions[:,:] = oldParticlePositions[newSampleIndices, :]
    else:
        # Make sure to make a clean copy of first resampled particle, and add a disturbance of the next ones.
        resampledOnce = np.full(particles.numParticles, False, dtype=bool)
        var = np.eye(2)*reinitialization_variance
        for i in range(len(newSampleIndices)):
            index = newSampleIndices[i]
            if resampledOnce[index]:
                newParticlePositions[i,:] = np.random.multivariate_normal(oldParticlePositions[index,:], var)
            else:
                newParticlePositions[i,:] = oldParticlePositions[index,:]
                resampledOnce[index] = True
    
    # Set particle positions to the ensemble:            
    particles.setParticlePositions(newParticlePositions)
    
    # Enforce boundary conditions
    particles.enforceBoundaryConditions()
    
    
def probabilisticResampling(particles, reinitialization_variance=0):
    """
    Probabilistic resampling of the particles based on the attached observation.
    Particles are sampled directly from the discrete distribution given by their weights.

    particles: The ensemble to be resampled, holding the ensemble particles, the observation, and measures to compute the weight of particles based on this information.
    reinitialization_variance: The variance used for resampling of particles that are already resampled. These duplicates are sampled around the original particle.
    If reinitialization_variance is zero, exact duplications are generated.

    Implementation based on the description in van Leeuwen (2009) 'Particle Filtering in Geophysical Systems', Section 3a.1)
    """
    
    # Obtain weights:
    weights = particles.getGaussianWeight()
    #weights = particles.getCauchyWeight()
    
    # Create array of possible indices to resample:
    allIndices = range(particles.getNumParticles())
    
    # Draw new indices based from discrete distribution based on weights
    newSampleIndices = np.random.choice(allIndices, particles.getNumParticles(), p=weights)
        
    # Return a new set of particles
    resampleParticles(particles, newSampleIndices, reinitialization_variance)


def residualSampling(particles, reinitialization_variance=0, onlyDeterministic=False, onlyStochastic=False):
    """
    Residual resampling of particles based on the attached observation.
    Each particle is first resampled floor(N*w) times, which in total given M <= N particles. Afterwards, N-M particles are drawn from the discrete distribution given by weights N*w % 1.

   particles: The ensemble to be resampled, holding the ensemble particles, the observation, and measures to compute the weight of particles based on this information.
    reinitialization_variance: The variance used for resampling of particles that are already resampled. These duplicates are sampled around the original particle.
    If reinitialization_variance is zero, exact duplications are generated.

    Implementation based on the description in van Leeuwen (2009) 'Particle Filtering in Geophysical Systems', Section 3a.2)
    """
    
    # Obtain weights:
    #weights = particles.getCauchyWeight()
    weights = particles.getGaussianWeight()

    # Create array of possible indices to resample:
    allIndices = range(particles.getNumParticles())

    # Deterministic resampling based on the integer part of N*weights:
    weightsTimesN = weights*particles.getNumParticles()
    weightsTimesNInteger = np.int64(np.floor(weightsTimesN))
    deterministicResampleIndices = np.repeat(allIndices, weightsTimesNInteger)
    
    # Stochastic resampling based on the decimal parts of N*weights:
    decimalWeights = np.mod(weightsTimesN, 1)
    decimalWeights = decimalWeights/np.sum(decimalWeights)
    stochasticResampleIndices = np.random.choice(allIndices, 
                                                 particles.getNumParticles() - len(deterministicResampleIndices), 
                                                 p=decimalWeights)
    ### NOTE!
    # In numpy v >= 1.13, np.divmod can be used to get weightsTimesNInteger and decimalWeights from one function call.
    
    if onlyDeterministic:
        resampleParticles(particles, deterministicResampleIndices, reinitialization_variance)
    if onlyStochastic:
        resampleParticles(particles, stochasticResampleIndices, reinitialization_variance)
    
    resampleParticles(particles, np.concatenate((deterministicResampleIndices, stochasticResampleIndices)), \
                      reinitialization_variance)
    


def stochasticUniversalSampling(particles, reinitialization_variance=0):
    """
    Stochastic resampling of particles based on the attached observation.
    Consider all weights as line lengths, so that all particles represent segments completely covering the line [0, 1]. Draw u ~ U[0,1/N], and resample all particles representing points u + i/N, i = 0,...,N-1 on the line.

    particles: The ensemble to be resampled, holding the ensemble particles, the observation, and measures to compute the weight of particles based on this information.
    reinitialization_variance: The variance used for resampling of particles that are already resampled. These duplicates are sampled around the original particle.
    If reinitialization_variance is zero, exact duplications are generated.

    Implementation based on the description in van Leeuwen (2009) 'Particle Filtering in Geophysical Systems', Section 3a.3)
    """   
    
    # Obtain weights:
    #weights = particles.getCauchyWeight()
    weights = particles.getGaussianWeight()

    # Create array of possible indices to resample:
    allIndices = np.array(range(particles.getNumParticles()))
    
    # Create histogram buckets based on the cumulative weights
    cumulativeWeights = np.concatenate(([0.0], np.cumsum(weights)))
    
    # Find first starting position:
    startPos = np.random.rand()/particles.getNumParticles()
    lengths = 1.0/particles.getNumParticles()
    #print startPos, lengths
    selectionValues = allIndices*lengths + startPos
    
    # Create a histogram of selectionValues within the cumulativeWeights buckets
    bucketValues, buckets = np.histogram(selectionValues, bins=cumulativeWeights)
    
    #newSampleIndices has now the number of times each index should be resampled
    # We need to go from [0, 0, 1, 4, 0] to [2,3,3,3,3]
    newSampleIndices = np.repeat(allIndices, bucketValues)
    
    # Return a new set of particles
    resampleParticles(particles, newSampleIndices, reinitialization_variance)


def metropolisHastingSampling(particles,  reinitialization_variance=0):
    """
    Resampling based on the Monte Carlo Metropolis-Hasting algorithm.
    The first particle, having weight w_1, is automatically resampled. The next particle, with weight w_2, is then resampled with the probability p = w_2/w_1, otherwise the first particle is sampled again. The latest resampled particle make the comparement basis for the next particle. 

    particles: The ensemble to be resampled, holding the ensemble particles, the observation, and measures to compute the weight of particles based on this information.
    reinitialization_variance: The variance used for resampling of particles that are already resampled. These duplicates are sampled around the original particle.
    If reinitialization_variance is zero, exact duplications are generated.

    Implementation based on the description in van Leeuwen (2009) 'Particle Filtering in Geophysical Systems', Section 3a.4)
    """
    
    # Obtain weights:
    #weights = particles.getCauchyWeight()
    weights = particles.getGaussianWeight()
    
    # Create buffer for indices which should be in the new ensemble:
    newSampleIndices = np.zeros_like(weights, dtype=int)
    
    # The first member is automatically a member of the new ensemble
    newSampleIndices[0] = 0
    
    # Iterate through all weights, and apply the Metropolis-Hasting algorithm
    for i in range(1, particles.getNumParticles()):
        # Draw random number U[0,1]
        p = np.random.rand()
        if p < weights[i]/weights[newSampleIndices[i-1]]:
            newSampleIndices[i] = i
        else:
            newSampleIndices[i] = newSampleIndices[i-1]
    
    # Return a new set of particles
    resampleParticles(particles, newSampleIndices, reinitialization_variance)
