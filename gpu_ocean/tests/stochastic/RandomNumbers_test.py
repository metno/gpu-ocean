import unittest
import time
import numpy as np
import sys
import gc
import pycuda.driver as cuda

from testUtils import *

sys.path.insert(0, '../')
from SWESimulators import Common
from SWESimulators.OceanStateNoise import *
from tests.stochastic.OceanStateNoise_parent import OceanStateNoiseTestParent


class RandomNumbersTest(OceanStateNoiseTestParent):
        
    def test_random_uniform(self):
        self.create_large_noise()

        self.large_noise.generateUniformDistribution()

        U = self.large_noise.getRandomNumbers()

        mean = np.mean(U)
        var = np.var(U)

        # Check the mean and var with very low accuracy.
        # Gives error if the distribution is way off
        self.assertLess(np.abs(mean - 0.5), 0.005)
        self.assertLess(np.abs(var - 1/12), 0.001)


    def test_random_uniform_CPU(self):
        self.create_large_noise()

        self.large_noise.generateUniformDistributionCPU()

        U = self.large_noise.getRandomNumbersCPU()

        mean = np.mean(U)
        var = np.var(U)

        # Check the mean and var with very low accuracy.
        # Gives error if the distribution is way off
        self.assertLess(np.abs(mean - 0.5), 0.005)
        self.assertLess(np.abs(var - 1/12), 0.001)

    def test_random_normal(self):
        self.create_large_noise()

        self.large_noise.generateNormalDistribution()

        U = self.large_noise.getRandomNumbers()

        mean = np.mean(U)
        var = np.var(U)

        # Check the mean and var with very low accuracy.
        # Gives error if the distribution is way off
        self.assertLess(np.abs(mean), 0.01)
        self.assertLess(np.abs(var - 1.0), 0.01)
        
    def test_random_normal_CPU(self):
        self.create_large_noise()

        self.large_noise.generateNormalDistributionCPU()

        U = self.large_noise.getRandomNumbersCPU()

        mean = np.mean(U)
        var = np.var(U)

        # Check the mean and var with very low accuracy.
        # Gives error if the distribution is way off
        self.assertLess(np.abs(mean), 0.01)
        self.assertLess(np.abs(var - 1.0), 0.01)


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

        
    def test_empty_reduction_buffer(self):
        self.create_large_noise()
        
        buffer_host = self.large_noise.getReductionBuffer()
        self.assertEqual(buffer_host.shape, (1,3))
        self.assertEqual(buffer_host[0,0], 0.0)
        self.assertEqual(buffer_host[0,1], 0.0)
        self.assertEqual(buffer_host[0,2], 0.0)
        
    def test_reduction(self):
        self.create_large_noise()
        
        self.large_noise.generateNormalDistribution()
        obtained_random_numbers = self.large_noise.getRandomNumbers()
        gamma_from_numpy = np.linalg.norm(obtained_random_numbers)**2
        
        gamma = self.large_noise.getRandomNorm()

        # Checking relative difference 
        self.assertAlmostEqual((gamma_from_numpy-gamma)/gamma_from_numpy, 0.0, places=5)

        
