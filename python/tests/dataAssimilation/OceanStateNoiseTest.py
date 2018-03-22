import unittest
import time
import numpy as np
import sys
import gc
import pyopencl

from testUtils import *

sys.path.insert(0, '../')
from SWESimulators import Common
from SWESimulators.OceanStateNoise import *


class OceanStateNoiseTest(unittest.TestCase):

    def setUp(self):
        self.cl_ctx = make_cl_ctx()
        self.cl_queue = pyopencl.CommandQueue(self.cl_ctx)

        self.nx = 30
        self.ny = 40
        self.dx = 1.0
        self.dy = 1.0
        
        self.noise = None

        self.cutoff = 2
        self.nx_nonPeriodic = self.nx + 2*(1+self.cutoff)
        self.ny_nonPeriodic = self.ny + 2*(1+self.cutoff)
        
        # Standard setup is non-staggered, periodic
        self.staggered = False
        self.periodicNS = True
        self.periodicEW = True

        # Multiplies of block size: 16, 32, 48, 64
        self.glob_size_x = 16*2
        self.glob_size_y = 16*3
        self.glob_size_x_nonperiodic = 16*3
        self.glob_size_y_nonperiodic = 16*3
        self.glob_size_random_x = 16*1
        self.glob_size_random_x_nonperiodic = 16*2

        self.large_nx = 200
        self.large_ny = 200
        self.large_noise = None

        self.floatMax = 2147483648.0
        
    def tearDown(self):
        if self.noise is not None:
            self.noise.cleanUp()
        if self.large_noise is not None:
            self.large_noise.cleanUp()

    def create_noise(self):
        n,e,s,w = 1,1,1,1
        if self.periodicNS:
            n,s = 2,2
        if self.periodicEW:
            e,w = 2,2
        self.noise = OceanStateNoise(self.cl_ctx, self.cl_queue,
                                     self.nx, self.ny,
                                     self.dx, self.dy,
                                     Common.BoundaryConditions(n,e,s,w),
                                     staggered=self.staggered)
    def create_large_noise(self):
        n,e,s,w = 1,1,1,1
        if self.periodicNS:
            n,s = 2,2
        if self.periodicEW:
            e,w = 2,2
        self.large_noise = OceanStateNoise(self.cl_ctx, self.cl_queue,
                                           self.large_nx, self.large_ny,
                                           self.dx, self.dy,
                                           Common.BoundaryConditions(n,e,s,w),
                                           staggered = self.staggered)
        
    def compare_random(self, tol, msg):

        # The tolerance provided to testAlmostEqual makes the comparison wrt to the
        # number of decimal places, not the number of significant digits.
        # We therefore make sure that seed is in [0, 1]
        seed = self.noise.getSeed()/self.floatMax
        seedCPU = self.noise.getSeedCPU()/self.floatMax

        assert2DListAlmostEqual(self, seed.tolist(), seedCPU.tolist(), tol, msg+", seed")

        random = self.noise.getRandomNumbers()
        randomCPU = self.noise.getRandomNumbersCPU()

        assert2DListAlmostEqual(self, random.tolist(), randomCPU.tolist(), tol, msg+", random")

        
    def test_init_periodic_nonstaggered(self):
        self.create_noise()

        self.assertEqual(self.noise.rand_nx, self.nx)
        self.assertEqual(self.noise.rand_ny, self.ny)
        self.assertEqual(self.noise.seed_nx, self.nx/2)
        self.assertEqual(self.noise.seed_ny, self.ny)

        self.assertEqual(self.noise.global_size_noise, (self.glob_size_x, self.glob_size_y))
        self.assertEqual(self.noise.global_size_random_numbers, (self.glob_size_random_x, self.glob_size_y))

        self.compare_random(6, "test_init_periodic_nonstaggered")

        
    def test_init_periodic_staggered(self):
        self.staggered = True
        self.create_noise()

        self.assertEqual(self.noise.rand_nx, self.nx)
        self.assertEqual(self.noise.rand_ny, self.ny)
        self.assertEqual(self.noise.seed_nx, self.nx/2)
        self.assertEqual(self.noise.seed_ny, self.ny)

        self.assertEqual(self.noise.global_size_noise, (self.glob_size_x, self.glob_size_y))
        self.assertEqual(self.noise.global_size_random_numbers, (self.glob_size_random_x, self.glob_size_y))

        self.compare_random(6, "test_init_periodic_staggered")

    def test_init_non_periodic_nonstaggered(self):
        self.periodicEW = False
        self.periodicNS = False
        self.create_noise()

        self.assertEqual(self.noise.rand_nx, self.nx_nonPeriodic)
        self.assertEqual(self.noise.rand_ny, self.ny_nonPeriodic)
        self.assertEqual(self.noise.seed_nx, self.nx_nonPeriodic/2)
        self.assertEqual(self.noise.seed_ny, self.ny_nonPeriodic)        

        self.assertEqual(self.noise.global_size_noise, (self.glob_size_x_nonperiodic, self.glob_size_y_nonperiodic))
        self.assertEqual(self.noise.global_size_random_numbers, (self.glob_size_random_x_nonperiodic, self.glob_size_y_nonperiodic))

        self.compare_random(6, "test_init_non_periodi_nonstaggered")
        
    def test_init_perioidNS_nonstaggered(self):
        self.periodicEW = False
        self.create_noise()

        self.assertEqual(self.noise.rand_nx, self.nx_nonPeriodic)
        self.assertEqual(self.noise.rand_ny, self.ny)
        self.assertEqual(self.noise.seed_nx, self.nx_nonPeriodic/2)
        self.assertEqual(self.noise.seed_ny, self.ny)        

        self.assertEqual(self.noise.global_size_noise, (self.glob_size_x_nonperiodic, self.glob_size_y))
        self.assertEqual(self.noise.global_size_random_numbers, (self.glob_size_random_x_nonperiodic, self.glob_size_y))

        self.compare_random(6, "test_init_periodiNS_nonstaggered")
        
    def test_init_perioidEW_nonstaggered(self):
        self.periodicNS = False
        self.create_noise()

        self.assertEqual(self.noise.rand_nx, self.nx)
        self.assertEqual(self.noise.rand_ny, self.ny_nonPeriodic)
        self.assertEqual(self.noise.seed_nx, self.nx/2)
        self.assertEqual(self.noise.seed_ny, self.ny_nonPeriodic)        

        self.assertEqual(self.noise.global_size_noise, (self.glob_size_x, self.glob_size_y_nonperiodic))
        self.assertEqual(self.noise.global_size_random_numbers, (self.glob_size_random_x, self.glob_size_y_nonperiodic))

        self.compare_random(6, "test_init_periodiEW_nonstaggered")
        
    def test_random_uniform(self):
        self.create_large_noise()

        self.large_noise.generateUniformDistribution()

        U = self.large_noise.getRandomNumbers()

        mean = np.mean(U)
        var = np.var(U)

        # Check the mean and var with very low accuracy.
        # Gives error if the distribution is way off
        self.assertAlmostEqual(mean, 0.5, 1)
        self.assertAlmostEqual(var, 1.0/12.0, 2)

    def test_random_normal(self):
        self.create_large_noise()

        self.large_noise.generateNormalDistribution()

        U = self.large_noise.getRandomNumbers()

        mean = np.mean(U)
        var = np.var(U)

        # Check the mean and var with very low accuracy.
        # Gives error if the distribution is way off
        self.assertAlmostEqual(mean, 0.0, 1)
        self.assertAlmostEqual(var, 1.0, 1)


    def test_seed_diff(self):
        self.create_noise()
        tol = 6
        
        init_seed = self.noise.getSeed()/self.floatMax
        self.noise.generateNormalDistribution()
        normal_seed = self.noise.getSeed()/self.floatMax
        assert2DListNotAlmostEqual(self, normal_seed.tolist(), init_seed.tolist(), tol, "test_seed_diff, normal vs init_seed")
        
        self.noise.generateUniformDistribution()
        uniform_seed = self.noise.getSeed()/self.floatMax
        assert2DListNotAlmostEqual(self, uniform_seed.tolist(), init_seed.tolist(), tol, "test_seed_diff, uniform vs init_seed")
        assert2DListNotAlmostEqual(self, uniform_seed.tolist(), normal_seed.tolist(), tol, "test_seed_diff, uniform vs normal_seed")
