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
        
        self.ensemble = OceanNoiseEnsemble.OceanNoiseEnsemble(self.ensembleSize, self.gpu_ctx,  
                                                              observation_type=dautils.ObservationType.DirectUnderlyingFlow)
        self.ensemble.setGridInfoFromSim(self.sim)
        self.ensemble.setStochasticVariables(observation_variance = 0.01**2,
                                             small_scale_perturbation_amplitude=self.q0)
        self.ensemble.init(driftersPerOceanModel=self.driftersPerOceanModel)

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

    def test_local_SVD_to_global_CPU(self):
        test_data = np.loadtxt("iewpfRefData/preLocalSVDtoGlobal.dat")
        results_from_file = np.loadtxt("iewpfRefData/postLocalSVDtoGlobal.dat")

        self.assertEqual(test_data.shape, (self.ny, self.nx))
        
        self.iewpf._apply_local_SVD_to_global_xi(test_data, 30, 30)

        assert2DListAlmostEqual(self, test_data.tolist(), results_from_file.tolist(), 10, "test_local_SVD_to_global_CPU")

        
        
