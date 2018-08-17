import unittest
import time
import numpy as np
import sys
import gc
import pyopencl

import abc

from testUtils import *

sys.path.insert(0, '../')
from SWESimulators import Common
from SWESimulators import DataAssimilationUtils as dautils

from SWESimulators import DrifterEnsemble
from dataAssimilation.BaseDrifterEnsembleTest import BaseDrifterEnsembleTest

class DrifterEnsembleTest(BaseDrifterEnsembleTest):
    #__metaclass__ = abc.ABCMeta
    
    def setUp(self):
        super(DrifterEnsembleTest, self).setUp()
        
        self.gpu_ctx = Common.CUDAContext()
                
    def tearDown(self):
        super(DrifterEnsembleTest, self).tearDown()

    ### Define required functions as abstract ###

    #@abc.abstractmethod
    def create_small_particle_set(self):
        self.smallParticleSet = DrifterEnsemble.DrifterEnsemble(self.gpu_ctx,
                                                                self.numParticles,
                                                                self.observationVariance)
        self.smallParticleSet.setGridInfo(self.nx, self.ny, self.dx, self.dy, self.dt,
                                          self.boundaryCondition)
        self.smallParticleSet.setParameters()
        self.smallParticleSet.init()
        

    #@abc.abstractmethod
    def create_resampling_particle_set(self):
        self.resamplingParticleSet = DrifterEnsemble.DrifterEnsemble(self.gpu_ctx,
                                                                     self.resampleNumParticles,
                                                                     observation_variance=self.resamplingObservationVariance)
        self.resamplingParticleSet.setGridInfo(self.nx, self.ny, self.dx, self.dy, self.dt)
        self.resamplingParticleSet.setParameters()
        self.resamplingParticleSet.init()
        

    #@abc.abstractmethod
    def create_large_particle_set(self, size, domain_x, domain_y):
        largeParticleSet = DrifterEnsemble.DrifterEnsemble(self.gpu_ctx,
                                                           size)
        largeParticleSet.setGridInfo(10, 10, domain_x/10.0, domain_y/10.0, self.dt,
                                               self.boundaryCondition)
        largeParticleSet.setParameters()
        largeParticleSet.init()
        return largeParticleSet

