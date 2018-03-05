import unittest
import time
import numpy as np
import sys
import gc

from testUtils import *

sys.path.insert(0, '../')
from SWESimulators import Common
from SWESimulators.CPUDrifter import *
from dataAssimilation.DrifterTest import DrifterTest

class CPUDrifterTest(DrifterTest):

    def setUp(self):
        super(CPUDrifterTest, self).setUp()

    def tearDown(self):
        pass
    

    def create_small_particle_set(self):
        self.smallParticleSet = CPUDrifter(self.numParticles,
                                           self.observationVariance,
                                           self.boundaryCondition)

    def create_resampling_particle_set(self):
        self.resamplingParticleSet = CPUDrifter(self.resampleNumParticles)

    def create_large_particle_set(self, size):
        return CPUDrifter(size)

