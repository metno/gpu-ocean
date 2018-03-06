import unittest
import time
import numpy as np
import sys
import gc

from testUtils import *

sys.path.insert(0, '../')
from SWESimulators import Common
from SWESimulators.CPUDrifterCollection import *
from dataAssimilation.DrifterTest import DrifterTest

class CPUDrifterTest(DrifterTest):

    def setUp(self):
        super(CPUDrifterTest, self).setUp()

    def tearDown(self):
        pass
    

    def create_small_particle_set(self):
        self.smallParticleSet = CPUDrifterCollection(self.numParticles,
                                                     self.observationVariance,
                                                     self.boundaryCondition)

    def create_resampling_particle_set(self):
        self.resamplingParticleSet = CPUDrifterCollection(self.resampleNumParticles)

    def create_large_particle_set(self, size, domain_x, domain_y):
        return CPUDrifterCollection(size, domain_size_x=domain_x, domain_size_y=domain_y)

