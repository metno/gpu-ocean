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


class OceanStateNoiseTest(unittest.TestCase):

    def setUp(self):
        self.gpu_ctx = Common.CUDAContext()
        self.gpu_stream = cuda.Stream()
        
        self.nx = 30
        self.ny = 40
        self.dx = 1.0
        self.dy = 1.0

        self.f = 0.02
        self.g = 9.81
        self.beta = 0.0
        
        self.noise = None

        self.cutoff = 2
        self.nx_nonPeriodic = self.nx + 2*(1+self.cutoff)
        self.ny_nonPeriodic = self.ny + 2*(1+self.cutoff)
        
        # Standard setup is non-staggered, periodic
        self.staggered = False
        self.periodicNS = True
        self.periodicEW = True

        # Total number of threads should be: 16, 32, 48, 64
        # Corresponding to the number of blocks: 1, 2, 3, 4
        self.glob_size_x = 2
        self.glob_size_y = 3
        self.glob_size_x_nonperiodic = 3
        self.glob_size_y_nonperiodic = 3
        self.glob_size_random_x = 1
        self.glob_size_random_x_nonperiodic = 2

        self.large_nx = 400
        self.large_ny = 400
        self.large_noise = None

        self.floatMax = 2147483648.0

        self.eta = None
        self.hu = None
        self.hv = None
        self.H = None

        
    def tearDown(self):
        if self.noise is not None:
            self.noise.cleanUp()
            del self.noise
        if self.large_noise is not None:
            self.large_noise.cleanUp()
            del self.large_noise
        if self.eta is not None:
            self.eta.release()
        if self.hu is not None:
            self.hu.release()
        if self.hv is not None:
            self.hv.release()
        if self.H is not None:
            self.H.release()
        if self.gpu_ctx is not None:
            self.assertEqual(sys.getrefcount(self.gpu_ctx), 2)
            self.gpu_ctx = None
   
        gc.collect()
            
    def create_noise(self):
        n,e,s,w = 1,1,1,1
        if self.periodicNS:
            n,s = 2,2
        if self.periodicEW:
            e,w = 2,2
        self.noise = OceanStateNoise(self.gpu_ctx, self.gpu_stream,
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
        self.large_noise = OceanStateNoise(self.gpu_ctx, self.gpu_stream,
                                           self.large_nx, self.large_ny,
                                           self.dx, self.dy,
                                           Common.BoundaryConditions(n,e,s,w),
                                           staggered = self.staggered)

    def allocateBuffers(self, HCPU):
        host_buffer = np.zeros((self.ny, self.nx))
        self.eta = Common.CUDAArray2D(self.gpu_stream, self.nx, self.ny, 0, 0, host_buffer)
        self.hu = Common.CUDAArray2D(self.gpu_stream, self.nx, self.ny, 0, 0, host_buffer)
        self.hv = Common.CUDAArray2D(self.gpu_stream, self.nx, self.ny, 0, 0, host_buffer)
        self.H = Common.CUDAArray2D(self.gpu_stream, self.nx+1, self.ny+1, 0, 0, HCPU)
        del host_buffer
        
    def compare_random(self, tol, msg):

        # The tolerance provided to testAlmostEqual makes the comparison wrt to the
        # number of decimal places, not the number of significant digits.
        # We therefore make sure that seed is in [0, 1]
        seed = self.noise.getSeed()
        seedCPU = self.noise.getSeedCPU()

        msg = msg+"\ntype(seed):    " + str(type(seed)) + ", " + str(type(seed[0,0]))\
              + "\ntype(seedCPU): " + str(type(seedCPU)) + ", " + str(type(seedCPU[0,0]))
        
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
        self.assertAlmostEqual(mean, 0.5, 2)
        self.assertAlmostEqual(var, 1.0/12.0, 2)


    def test_random_uniform_CPU(self):
        self.create_large_noise()

        self.large_noise.generateUniformDistributionCPU()

        U = self.large_noise.getRandomNumbersCPU()

        mean = np.mean(U)
        var = np.var(U)

        # Check the mean and var with very low accuracy.
        # Gives error if the distribution is way off
        self.assertAlmostEqual(mean, 0.5, 2)
        self.assertAlmostEqual(var, 1.0/12.0, 2)

    def test_random_normal(self):
        self.create_large_noise()

        self.large_noise.generateNormalDistribution()

        U = self.large_noise.getRandomNumbers()

        mean = np.mean(U)
        var = np.var(U)

        # Check the mean and var with very low accuracy.
        # Gives error if the distribution is way off
        self.assertAlmostEqual(mean, 0.0, 2)
        self.assertAlmostEqual(var, 1.0, 1)

        
    def test_random_normal_CPU(self):
        self.create_large_noise()

        self.large_noise.generateNormalDistributionCPU()

        U = self.large_noise.getRandomNumbersCPU()

        mean = np.mean(U)
        var = np.var(U)

        # Check the mean and var with very low accuracy.
        # Gives error if the distribution is way off
        self.assertAlmostEqual(mean, 0.0, 2)
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



    def perturb_eta(self, msg):
        etaCPU = np.zeros((self.ny, self.nx))
        HCPU = np.zeros((self.ny+1, self.nx+1))
        self.create_noise()
        self.allocateBuffers(HCPU)
        
        self.noise.perturbOceanState(self.eta, self.hu, self.hv, self.H,
                                     self.f, self.beta, self.g)
        self.noise.perturbEtaCPU(etaCPU, use_existing_GPU_random_numbers=True)
        etaFromGPU = self.eta.download(self.gpu_stream)

        # Scale so that largest value becomes ~ 1
        maxVal = np.max(etaCPU)
        etaFromGPU = etaFromGPU / maxVal
        etaCPU = etaCPU / maxVal
        
        assert2DListAlmostEqual(self, etaCPU.tolist(), etaFromGPU.tolist(), 6, msg)

        
    def test_perturb_eta_periodic(self):
        self.perturb_eta("test_perturb_eta_periodic")

    def test_perturb_eta_nonperiodic(self):
        self.periodicNS = False
        self.periodicEW = False
        self.perturb_eta("test_perturb_eta_nonperiodic")

    def test_perturb_eta_NS_periodic(self):
        self.periodicEW = False
        self.perturb_eta("test_perturb_eta_NS_periodic")

    def test_perturb_eta_EW_periodic(self):
        self.periodicNS = False
        self.perturb_eta("test_perturb_eta_EW_periodic")



    def perturb_ocean(self, msg):
        etaCPU = np.zeros((self.ny, self.nx))
        huCPU = np.zeros((self.ny, self.nx))
        hvCPU = np.zeros((self.ny, self.nx))
        HCPU = np.zeros((self.ny+1, self.nx+1))

        self.create_noise()
        self.allocateBuffers(HCPU)

        self.noise.perturbOceanState(self.eta, self.hu, self.hv, self.H,
                                     self.f, self.beta, self.g)
        self.noise.perturbOceanStateCPU(etaCPU, huCPU, hvCPU, HCPU,
                                        self.f, self.beta, self.g,
                                        use_existing_GPU_random_numbers=True)
        huFromGPU = self.hu.download(self.gpu_stream)
        hvFromGPU = self.hv.download(self.gpu_stream)

        # Scale so that largest value becomes ~ 1:
        maxVal = np.max(huCPU)
        huFromGPU = huFromGPU / maxVal
        hvFromGPU = hvFromGPU / maxVal
        huCPU = huCPU / maxVal
        hvCPU = hvCPU / maxVal
        
        assert2DListAlmostEqual(self, huCPU.tolist(), huFromGPU.tolist(), 5, msg+", hu")
        assert2DListAlmostEqual(self, hvCPU.tolist(), hvFromGPU.tolist(), 5, msg+", hv")
        

        
    def test_perturb_ocean_periodic(self):
        self.perturb_ocean("test_perturb_ocean_periodic")

        
    def test_perturb_ocean_nonperiodic(self):
        self.periodicNS = False
        self.periodicEW = False
        self.perturb_ocean("test_perturb_ocean_nonperiodic")

        
    def test_perturb_ocean_EW_periodic(self):
        self.periodicNS = False
        self.perturb_ocean("test_perturb_ocean_EW_periodic")

        
    def test_perturb_ocean_NS_periodic(self):
        self.periodicEW = False
        self.perturb_ocean("test_perturb_ocean_NS_periodic")
