import unittest
import time
import numpy as np
import sys
import gc
import pyopencl

from testUtils import *

sys.path.insert(0, '../')
from SWESimulators import Common
from SWESimulators.GPUDrifterCollection import *
from dataAssimilation.DrifterTest import DrifterTest



class GPUDrifterTest(DrifterTest):

    def setUp(self):
        super(GPUDrifterTest, self).setUp()
        self.cl_ctx = make_cl_ctx()
        
    def tearDown(self):
        self.cl_ctx = None
        if self.smallDrifterSet is not None:
            self.smallDrifterSet.cleanUp()
        if self.resamplingDrifterSet is not None:
            self.resamplingDrifterSet.cleanUp()


    def create_small_drifter_set(self):
        self.smallDrifterSet = GPUDrifterCollection(self.cl_ctx,
                                                    self.numDrifters,
                                                    self.observationVariance,
                                                    self.boundaryCondition)

    def create_resampling_drifter_set(self):
        self.resamplingDrifterSet = GPUDrifterCollection(self.cl_ctx,
                                                         self.resampleNumDrifters)
        
    def create_large_drifter_set(self, size, domain_x, domain_y):
        return GPUDrifterCollection(self.cl_ctx, size, domain_size_x=domain_x, domain_size_y=domain_y) 
        



