import unittest
import time
import numpy as np
import sys
import gc

from testUtils import *

sys.path.insert(0, '../')
from SWESimulators import Common
from SWESimulators.GPUDrifterCollection import *
from dataAssimilation.BaseDrifterTest import BaseDrifterTest



class GPUDrifterTest(BaseDrifterTest):

    def setUp(self):
        super(GPUDrifterTest, self).setUp()
        self.gpu_ctx = Common.CUDAContext(verbose=False)
        
    def tearDown(self):
        super(GPUDrifterTest, self).tearDown()
        gc.collect()
        
    def create_small_drifter_set(self):
        self.smallDrifterSet = GPUDrifterCollection(self.gpu_ctx,
                                                    self.numDrifters,
                                                    self.observationVariance,
                                                    self.boundaryCondition)

    def create_resampling_drifter_set(self):
        self.resamplingDrifterSet = GPUDrifterCollection(self.gpu_ctx,
                                                         self.resampleNumDrifters)
        
    def create_large_drifter_set(self, size, domain_x, domain_y):
        self.largeDrifterSet = GPUDrifterCollection(self.gpu_ctx, size, domain_size_x=domain_x, domain_size_y=domain_y) 
        



