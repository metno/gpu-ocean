# -*- coding: utf-8 -*-

"""
This software is a part of GPU Ocean.

Copyright (C) 2018  SINTEF Digital

This python class implements an abstract ensemble class, where each particle
will consist of an independent ocean state and one or more drifters.

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
import warnings


import pycuda.driver as cuda

from SWESimulators import CDKLM16
from SWESimulators import GPUDrifterCollection
from SWESimulators import WindStress
from SWESimulators import Common
from SWESimulators import DataAssimilationUtils as dautils

class BaseOceanStateEnsemble(object):
    """
    Base class for ensembles of ocean states.
    """
    __metaclass__ = abc.ABCMeta
        
        
    def __init__(self):
        # Constructor. Needs to be specified for each child class
        # Not an abstract 
        pass
        
    def __del__(self):
        # 
        self.cleanUp()

        
    @abc.abstractmethod
    def cleanUp(self):
        # Clean up all allocated device memory
        pass
        
        
    @abc.abstractmethod
    def _init(self, driftersPerOceanModel=1):
        # Initialize ocean models
        # add drifters
        # add noise
        # etc
        pass
    
    @abc.abstractmethod
    def resample(self, newSampleIndices, reinitialization_variance):
        # Resample and possibly perturb
        pass

    @abc.abstractmethod
    def step_truth(self, t, stochastic=True):
        pass
        
    @abc.abstractmethod
    def step_particles(self, t, stochastic=True):
        pass
    
    
    def observeParticles(self, gpu=False):
        """
        Applying the observation operator on each particle.

        Structure on the output:
        [
        particle 1:  [hu_1, hv_1], ... , [hu_D, hv_D],
        particle 2:  [hu_1, hv_1], ... , [hu_D, hv_D],
        particle Ne: [hu_1, hv_1], ... , [hu_D, hv_D]
        ]
        numpy array with dimensions (particles, drifters, 2)

        The two values per particle drifter is either volume transport or position, depending on 
        the observation type.
        """
        if self.observation_type == dautils.ObservationType.DrifterPosition:
            raise ValueError('Observation type DataAssimilationUtils.ObservationType.DrifterPosition is not supported')

        elif self.observation_type == dautils.ObservationType.UnderlyingFlow or \
             self.observation_type == dautils.ObservationType.DirectUnderlyingFlow:

            observedState = np.empty((self.getNumParticles(), \
                                      self.driftersPerOceanModel, 2))

            trueState = self.observeTrueState()
            # trueState = [[x1, y1, hu1, hv1], ..., [xD, yD, huD, hvD]]

            for p in range(self.numParticles):
                if gpu:
                    sim = self.particles[p]
                    self.observeUnderlyingFlowKernel.prepared_async_call(self.global_size,
                                                                         self.local_size,
                                                                         self.gpu_stream,
                                                                         sim.nx, sim.ny, sim.dx, sim.dy,
                                                                         np.int32(2), np.int32(2),
                                                                         sim.gpu_data.h0.data.gpudata,
                                                                         sim.gpu_data.h0.pitch,
                                                                         sim.gpu_data.hu0.data.gpudata,
                                                                         sim.gpu_data.hu0.pitch,
                                                                         sim.gpu_data.hv0.data.gpudata,
                                                                         sim.gpu_data.hv0.pitch,
                                                                         np.max(self.base_H),
                                                                         self.driftersPerOceanModel,
                                                                         self.particles[self.obs_index].drifters.driftersDevice.data.gpudata,
                                                                         self.particles[self.obs_index].drifters.driftersDevice.pitch,
                                                                         self.observation_buffer.data.gpudata,
                                                                         self.observation_buffer.pitch)
                    
                    observedState[p,:,:] = self.observation_buffer.download(self.gpu_stream)
                                                                         
                
                else:
                    # Downloading ocean state without ghost cells
                    eta, hu, hv = self.downloadParticleOceanState(p)

                    for d in range(self.driftersPerOceanModel):
                        id_x = np.int(np.floor(trueState[d,0]/self.dx))
                        id_y = np.int(np.floor(trueState[d,1]/self.dy))

                        observedState[p,d,0] = hu[id_y, id_x]
                        observedState[p,d,1] = hv[id_y, id_x]
                        
                        
            #print "Particle positions obs index:"
            #print self.particles[self.obs_index].drifters.driftersDevice.download(self.gpu_stream)
            #print "true state used by the CPU:"
            #print trueState
            return observedState
        
    @abc.abstractmethod
    def observeTrueDrifters(self):
        """
        Observing the drifters in the syntetic true state.
        """
        pass
    
        
                             
    @abc.abstractmethod        
    def observeTrueState(self):
        """
        Applying the observation operator on the syntetic true state.
        The observation should be in state space, and therefore consists of 
        hu and hv, and not u and v.

        Returns a numpy array with D drifter positions and drifter velocities
        [[x_1, y_1, hu_1, hv_1], ... , [x_D, y_D, hu_D, hv_D]]
        If the observation operator is drifter positions, hu and hv are not included.
        """
        pass

    
    @abc.abstractmethod
    def getInnovations(self, obs=None):
        """
        Obtaining the innovation vectors, y^m - H(\psi_i^m)

        Returns a numpy array with dimensions (particles, drifters, 2)

        """
        pass
    
    
    @abc.abstractmethod
    def getInnovationNorms(self, obs=None):
        """
        Obtain the norm of the innovation vectors
        """
        # Innovations have the structure 
        # [ particle: [drifter: [x, y] ] ], or
        # [ particle: [drifter: [u, v] ] ]
        # We simply gather find the norm for each particle:
        pass
    
            
    @abc.abstractmethod
    def getGaussianWeight(self, innovations=None, normalize=True):
        pass


    
    # Some get functions that assume some private variables.
    # If assumptions are wrong, they should be overloaded.
    def getNx(self):
        return np.int32(self.nx)
    def getNy(self):
        return np.int32(self.ny)
    def getDx(self):
        return np.float32(self.dx)
    def getDy(self):
        return np.float32(self.dy)
    def getDt(self):
        return np.float32(self.dt)

    def getDomainSizeX(self):
        return self.getNx()*self.getDx()
    def getDomainSizeY(self):
        return self.getNy()*self.getDy()
    def getObservationVariance(self):
        return self.observation_variance
    def getNumParticles(self):
        return np.int32(self.numParticles)
    def getNumDrifters(self):
        return np.int32(self.driftersPerOceanModel)
    
    def getObservationCov(self):
        return self.observation_cov
    def getObservationCovInverse(self):
        return self.observation_cov_inverse

    
    def downloadParticleOceanState(self, particleNo):
        """
        Downloads ocean state without ghost cells.
        """
        assert(particleNo < self.getNumParticles() and particleNo > -1), "particle out of range"
        return self.particles[particleNo].download(interior_domain_only=True)
     
    @abc.abstractmethod
    def downloadTrueOceanState(self):
        pass
        

        