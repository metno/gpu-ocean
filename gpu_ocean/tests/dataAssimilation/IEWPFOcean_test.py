import unittest
import numpy as np
import sys
import gc

from testUtils import *

sys.path.insert(0, '../')
from SWESimulators import Common
from SWESimulators import CDKLM16
from SWESimulators import OceanNoiseEnsemble
from SWESimulators import BathymetryAndICs as BC
from SWESimulators import DataAssimilationUtils as dautils
from SWESimulators import IEWPFOcean

class IEWPFOceanTest(unittest.TestCase):

    def setUp(self):
        self.sim = None
        self.ensemble = None
        self.iewpf = None
        
        self.gpu_ctx = Common.CUDAContext()

        self.setUpAndStartEnsemble()
        


    def tearDown(self):
        if self.sim is not None:
            self.sim.cleanUp()
            del self.sim
        if self.ensemble is not None:
            self.ensemble.cleanUp()
            del self.ensemble
        if self.iewpf is not None:
            self.iewpf.cleanUp()
            del self.iewpf
        if self.gpu_ctx is not None:
            self.assertEqual(sys.getrefcount(self.gpu_ctx), 2)
            self.gpu_ctx = None
        gc.collect()        
        

    def setUpAndStartEnsemble(self):
        self.nx = 40
        self.ny = 40
        
        self.dx = 4.0
        self.dy = 4.0

        self.dt = 0.05
        self.g = 9.81
        self.r = 0.0
        
        self.f = 0.05
        self.beta = 0.0

        self.waterDepth = 10.0
        
        self.ensembleSize = 3
        self.driftersPerOceanModel = 3
        
        ghosts = np.array([2,2,2,2]) # north, east, south, west
        validDomain = np.array([2,2,2,2])
        self.boundaryConditions = Common.BoundaryConditions(2,2,2,2)

        # Define which cell index which has lower left corner as position (0,0)
        x_zero_ref = 2
        y_zero_ref = 2

        dataShape = (self.ny + ghosts[0]+ghosts[2], 
                     self.nx + ghosts[1]+ghosts[3])
        dataShapeHi = (self.ny + ghosts[0]+ghosts[2]+1, 
                       self.nx + ghosts[1]+ghosts[3]+1)

        eta0 = np.zeros(dataShape, dtype=np.float32, order='C');
        eta0_extra = np.zeros(dataShape, dtype=np.float32, order='C')
        hv0 = np.zeros(dataShape, dtype=np.float32, order='C');
        hu0 = np.zeros(dataShape, dtype=np.float32, order='C');
        Hi = np.ones(dataShapeHi, dtype=np.float32, order='C')*self.waterDepth

        # Add disturbance:
        rel_grid_size = self.nx*1.0/self.dx
        BC.addBump(eta0, self.nx, self.ny, self.dx, self.dy, 0.3, 0.5, 0.05*rel_grid_size, validDomain)
        eta0 = eta0*0.3
        BC.addBump(eta0, self.nx, self.ny, self.dx, self.dy, 0.7, 0.3, 0.10*rel_grid_size, validDomain)
        eta0 = eta0*(-1.3)
        BC.addBump(eta0, self.nx, self.ny, self.dx, self.dy, 0.15, 0.8, 0.03*rel_grid_size, validDomain)
        eta0 = eta0*1.0
        BC.addBump(eta0, self.nx, self.ny, self.dx, self.dy, 0.6, 0.75, 0.06*rel_grid_size, validDomain)
        BC.addBump(eta0, self.nx, self.ny, self.dx, self.dy, 0.2, 0.2, 0.01*rel_grid_size, validDomain)
        eta0 = eta0*(-0.03)
        BC.addBump(eta0_extra, self.nx, self.ny, self.dx, self.dy, 0.5, 0.5, 0.4*rel_grid_size, validDomain)
        eta0 = eta0 + 0.02*eta0_extra
        BC.initializeBalancedVelocityField(eta0, Hi, hu0, hv0, \
                                           self.f, self.beta, self.g, \
                                           self.nx, self.ny, self.dx ,self.dy, ghosts)
        eta0 = eta0*0.5

        self.q0 = 0.5*self.dt*self.f/(self.g*self.waterDepth)
        
        self.sim = CDKLM16.CDKLM16(self.gpu_ctx, eta0, hu0, hv0, Hi, \
                                   self.nx, self.ny, self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r, \
                                   boundary_conditions=self.boundaryConditions, \
                                   write_netcdf=False, \
                                   small_scale_perturbation=True, \
                                   small_scale_perturbation_amplitude=self.q0)
        

        self.ensemble = OceanNoiseEnsemble.OceanNoiseEnsemble(self.gpu_ctx, self.ensembleSize, self.sim,
                                                              num_drifters =self.driftersPerOceanModel,
                                                              observation_type=dautils.ObservationType.DirectUnderlyingFlow,
                                                              observation_variance = 0.01**2)


        self.iewpf = IEWPFOcean.IEWPFOcean(self.ensemble)


    def run_ensemble(self):
        t = self.ensemble.step(1000*self.dt)

        



    def test_iewpf_attributes(self):

        self.assertEqual(self.iewpf.nx, self.nx)
        self.assertEqual(self.iewpf.ny, self.ny)
        self.assertAlmostEqual(self.iewpf.dx, self.dx)
        self.assertAlmostEqual(self.iewpf.dy, self.dy)
        self.assertAlmostEqual(self.iewpf.dt, self.dt)
        self.assertAlmostEqual(self.iewpf.soar_q0, self.q0)
        self.assertAlmostEqual(self.iewpf.soar_L, self.sim.small_scale_model_error.soar_L)
        self.assertAlmostEqual(self.iewpf.f, self.f)
        self.assertAlmostEqual(self.iewpf.g, self.g, places=6)
        self.assertAlmostEqual(self.iewpf.const_H, self.waterDepth)
        self.assertTrue(self.iewpf.boundaryConditions.isPeriodicNorthSouth())
        self.assertTrue(self.iewpf.boundaryConditions.isPeriodicEastWest())
        self.assertTrue(self.iewpf.Nx, self.nx*self.ny*3)
        self.assertEqual(self.iewpf.numParticles, self.ensembleSize)
        self.assertEqual(self.iewpf.numDrifters, self.driftersPerOceanModel)
        
    def test_S_matrix(self):
        S_from_GPU = self.iewpf.download_S()
        S_from_file = np.loadtxt("iewpfRefData/S.dat")

        self.assertEqual(self.iewpf.S_host.shape, (2,2))
        self.assertEqual(S_from_GPU.shape, (2,2))

        assert2DListAlmostEqual(self, S_from_GPU.tolist(), self.iewpf.S_host.tolist(), 7, "S matrix GPU vs CPU")
        assert2DListAlmostEqual(self, S_from_GPU.tolist(), S_from_file.tolist(), 7, "S matrix GPU vs file")
        

    def test_localSVD_matrix(self):
        localSVD_from_GPU = self.iewpf.download_localSVD()
        localSVD_from_file = np.loadtxt("iewpfRefData/localSVD.dat")

        self.assertEqual(self.iewpf.localSVD_host.shape, (49, 49))
        self.assertEqual(localSVD_from_GPU.shape, (49, 49))

        assert2DListAlmostEqual(self, localSVD_from_GPU.tolist(), self.iewpf.localSVD_host.tolist(), 7, "S matrix GPU vs CPU")
        assert2DListAlmostEqual(self, localSVD_from_GPU.tolist(), localSVD_from_file.tolist(), 7, "S matrix GPU vs file")

    def test_local_SVD_to_global_CPU_ref_data(self):
        test_data = np.loadtxt("iewpfRefData/preLocalSVDtoGlobal.dat")
        results_from_file = np.loadtxt("iewpfRefData/postLocalSVDtoGlobal.dat")

        self.assertEqual(test_data.shape, (self.ny, self.nx))
        
        self.iewpf._apply_local_SVD_to_global_xi_CPU(test_data, 30, 30)

        assert2DListAlmostEqual(self, test_data.tolist(), results_from_file.tolist(), 10, "test_local_SVD_to_global_CPU")
        
    def test_local_SVD_to_global_GPU_ref_data(self):
        test_data = np.loadtxt("iewpfRefData/preLocalSVDtoGlobal.dat")
        results_from_file = np.loadtxt("iewpfRefData/postLocalSVDtoGlobal.dat")

        self.assertEqual(test_data.shape, (self.ny, self.nx))

        sim = self.ensemble.particles[0]

        # Upload reference input data to the sim random numbers buffer:
        sim.small_scale_model_error.random_numbers.upload(self.iewpf.master_stream,
                                                          test_data.astype(np.float32))
        
        # Apply SVD centered in the chosen cell (30, 30):
        self.iewpf.applyLocalSVDOnGlobalXi(sim, 30, 30)
        
        # Download results:
        gpu_result = sim.small_scale_model_error.random_numbers.download(self.iewpf.master_stream)

        # Compare:
        rel_norm_results = np.linalg.norm(gpu_result - results_from_file)/np.max(results_from_file)
        self.assertAlmostEqual(rel_norm_results, 0.0, places=6)
        
    def test_empty_reduction_buffer(self):
        buffer_host = self.iewpf.download_reduction_buffer()
        self.assertEqual(buffer_host.shape, (1,1))
        self.assertEqual(buffer_host[0,0], 0.0)
        
    def test_reduction(self):
        self.ensemble.particles[0].small_scale_model_error.generateNormalDistribution()
        obtained_random_numbers = self.ensemble.particles[0].small_scale_model_error.getRandomNumbers()
        gamma_from_numpy = np.linalg.norm(obtained_random_numbers)**2
        
        gamma = self.iewpf.obtainGamma(self.ensemble.particles[0])

        # Checking relative difference 
        self.assertAlmostEqual((gamma_from_numpy-gamma)/gamma_from_numpy, 0.0, places=6)

    def test_set_buffer_to_zero(self):
        self.ensemble.particles[0].small_scale_model_error.generateNormalDistribution()
        self.iewpf.setNoiseBufferToZero(self.ensemble.particles[0])

        obtained_random_numbers = self.ensemble.particles[0].small_scale_model_error.getRandomNumbers()
        for j in range(self.ny):
            for i in range(self.nx):
                self.assertEqual(obtained_random_numbers[j,i], 0.0)

    def test_kalman_gain(self):
        self.run_ensemble()
        innovation = self.ensemble.getInnovations()[0]
        observed_drifter_positions = self.ensemble.observeTrueDrifters()

        # Set sim variables to zero, so that only the Kalman gain ends up in those fields
        zeros = np.zeros((self.iewpf.ny+4, self.iewpf.nx+4), dtype=np.float32)
        self.ensemble.particles[0].gpu_data.h0.upload(self.iewpf.master_stream, zeros)
        self.ensemble.particles[0].gpu_data.hu0.upload(self.iewpf.master_stream, zeros)
        self.ensemble.particles[0].gpu_data.hv0.upload(self.iewpf.master_stream, zeros)

        self.iewpf.addKalmanGain(self.ensemble.particles[0],
                                 observed_drifter_positions,
                                 innovation)

        eta, hu, hv = self.ensemble.particles[0].download(interior_domain_only=True)

        etaCPU, huCPU, hvCPU, gamma = self.iewpf.applyKalmanGain_CPU(self.ensemble.particles[0],
                                                                     observed_drifter_positions,
                                                                     innovation, returnKalmanGainTerm=True)
        rel_norm_eta = np.linalg.norm(eta - etaCPU)/np.max(eta)
        rel_norm_hu  = np.linalg.norm(hu - huCPU)/np.max(hu)
        rel_norm_hv  = np.linalg.norm(hv - hvCPU)/np.max(hv)
       
        self.assertAlmostEqual(rel_norm_eta, 0.0, places=5)
        self.assertAlmostEqual(rel_norm_hu,  0.0, places=5)
        self.assertAlmostEqual(rel_norm_hv,  0.0, places=5)
        
    def test_observation_operator_CPU_vs_GPU(self):
        self.run_ensemble()
                      
        cpu = self.ensemble.observeParticles(gpu=False)
        gpu = self.ensemble.observeParticles(gpu=True)
                
        assert2DListAlmostEqual(self, cpu.tolist(), gpu.tolist(), 6, "observation_operator_CPU_vs_GPU")
