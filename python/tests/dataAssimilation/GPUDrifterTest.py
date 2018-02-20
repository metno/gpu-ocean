import unittest
import time
import numpy as np
import sys
import gc
import pyopencl

from testUtils import *

sys.path.insert(0, '../')
from SWESimulators import Common
from SWESimulators.GPUDrifter import *
from SWESimulators import Resampling
from dataAssimilation.DrifterTest import DrifterTest



class GPUDrifterTest(DrifterTest):

    def setUp(self):
        super(GPUDrifterTest, self).setUp()
        self.cl_ctx = make_cl_ctx()
        
    def tearDown(self):
        self.cl_ctx = None
        if self.smallParticleSet is not None:
            self.smallParticleSet.cleanUp()
        if self.resamplingParticleSet is not None:
            self.resamplingParticleSet.cleanUp()


    def create_small_particle_set(self):
        self.smallParticleSet = GPUDrifter(self.cl_ctx,
                                           self.numParticles,
                                           self.observationVariance,
                                           self.boundaryCondition)

    def create_resampling_particle_set(self):
        self.resamplingParticleSet = GPUDrifter(self.cl_ctx,
                                                self.resampleNumParticles)
        
    def create_large_particle_set(self, size):
        return GPUDrifter(self.cl_ctx, size) 
        



