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

from SWESimulators import CPUDrifterEnsemble
from dataAssimilation.BaseDrifterEnsembleTest import BaseDrifterEnsembleTest

class CPUDrifterEnsembleTest(BaseDrifterEnsembleTest):
    #__metaclass__ = abc.ABCMeta
    
    def setUp(self):
        super(CPUDrifterEnsembleTest, self).setUp()
                        
    def tearDown(self):
        super(CPUDrifterEnsembleTest, self).tearDown()

    ### Define required functions as abstract ###

    #@abc.abstractmethod
    def create_small_particle_set(self):
        self.smallParticleSet = CPUDrifterEnsemble.CPUDrifterEnsemble(self.numParticles,
                                                                      self.observationVariance)
        self.smallParticleSet.setGridInfo(self.nx, self.ny, self.dx, self.dy, self.dt,
                                          self.boundaryCondition)
        self.smallParticleSet.setParameters()
        self.smallParticleSet.init()
        

    #@abc.abstractmethod
    def create_resampling_particle_set(self):
        self.resamplingParticleSet = CPUDrifterEnsemble.CPUDrifterEnsemble(self.resampleNumParticles,
                                                                           observation_variance=self.resamplingObservationVariance)
        self.resamplingParticleSet.setGridInfo(self.nx, self.ny, self.dx, self.dy, self.dt)
        self.resamplingParticleSet.setParameters()
        self.resamplingParticleSet.init()
        

    #@abc.abstractmethod
    def create_large_particle_set(self, size, domain_x, domain_y):
        largeParticleSet = CPUDrifterEnsemble.CPUDrifterEnsemble(size)
        largeParticleSet.setGridInfo(10, 10, domain_x/10.0, domain_y/10.0, self.dt,
                                     self.boundaryCondition)
        largeParticleSet.setParameters()
        largeParticleSet.init()
        return largeParticleSet

