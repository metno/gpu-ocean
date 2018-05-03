import unittest
import time
import numpy as np
import sys
import gc

from testUtils import *

sys.path.insert(0, '../')
from SWESimulators import Common, CTCS

class CTCStest(unittest.TestCase):

    def setUp(self):
        self.cl_ctx = make_cl_ctx()

        self.nx = 50
        self.ny = 70
        
        self.dx = 200.0
        self.dy = 200.0
        
        self.dt = 1
        self.g = 9.81
        self.f = 0.0
        self.r = 0.0
        self.A = 1
        
        self.h0 = np.ones((self.ny+2, self.nx+2), dtype=np.float32) * 60;
        self.eta0 = np.zeros((self.ny+2, self.nx+2), dtype=np.float32);
        self.u0 = np.zeros((self.ny+2, self.nx+1+2), dtype=np.float32);
        self.v0 = np.zeros((self.ny+1+2, self.nx+2), dtype=np.float32);

        self.ghosts = [1,1,1,1] # north, east, south, west
        self.refEtaRange = [-1, -1, 1, 1]
        self.refURange = [-1, -1, 1, 1]
        self.refVRange = [-1, -1, 1, 1]
        self.etaRange = [-1, -1, 1, 1]
        self.uRange = [-1, -2, 1, 2]
        self.vRange = [-2, -1, 2, 1]
        self.boundaryConditions = None

        self.T = 50.0
        
        self.sim = None

    def tearDown(self):
        if self.sim != None:
            self.sim.cleanUp()
            self.sim = None
        self.h0 = None
        self.eta0 = None
        self.u0 = None
        self.v0 = None
        self.cl_ctx = None
        gc.collect() # Force run garbage collection to free up memory       

            
    def setBoundaryConditions(self, bcSettings=1):
        if (bcSettings == 1):
            self.boundaryConditions = Common.BoundaryConditions()
        elif (bcSettings == 2):
            self.boundaryConditions = Common.BoundaryConditions(2,2,2,2)
        elif bcSettings == 3:
            self.boundaryConditions = Common.BoundaryConditions(2,1,2,1)
        else:
            self.boundaryConditions = Common.BoundaryConditions(1,2,1,2)


        
    def checkResults(self, eta1, u1, v1, etaRef, uRef, vRef):
        diffEta = np.linalg.norm(eta1[self.etaRange[2]:self.etaRange[0], 
                                      self.etaRange[3]:self.etaRange[1]] - 
                                 etaRef[self.refEtaRange[2]:self.refEtaRange[0],
                                        self.refEtaRange[3]:self.refEtaRange[1]])
        diffU = np.linalg.norm(u1[self.uRange[2]:self.uRange[0],
                                  self.uRange[3]:self.uRange[1]] -
                               uRef[self.refURange[2]:self.refURange[0],
                                    self.refURange[3]:self.refURange[1]])
        diffV = np.linalg.norm(v1[self.vRange[2]:self.vRange[0],
                                  self.vRange[3]:self.vRange[1]] - 
                               vRef[ self.refVRange[2]:self.refVRange[0],
                                     self.refVRange[3]:self.refVRange[1]])
        maxDiffEta = np.max(eta1[self.etaRange[2]:self.etaRange[0], 
                                 self.etaRange[3]:self.etaRange[1]] - 
                            etaRef[self.refEtaRange[2]:self.refEtaRange[0],
                                   self.refEtaRange[3]:self.refEtaRange[1]])
        maxDiffU = np.max(u1[self.uRange[2]:self.uRange[0],
                             self.uRange[3]:self.uRange[1]] -
                          uRef[self.refURange[2]:self.refURange[0],
                               self.refURange[3]:self.refURange[1]])
        maxDiffV = np.max(v1[self.vRange[2]:self.vRange[0],
                             self.vRange[3]:self.vRange[1]] - 
                          vRef[ self.refVRange[2]:self.refVRange[0],
                                self.refVRange[3]:self.refVRange[1]])
        
        self.assertAlmostEqual(maxDiffEta, 0.0, places=0,
                               msg='Unexpected eta difference! Max diff: ' + str(maxDiffEta) + ', L2 diff: ' + str(diffEta))
        self.assertAlmostEqual(maxDiffU, 0.0, places=0,
                               msg='Unexpected U difference: ' + str(maxDiffU) + ', L2 diff: ' + str(diffU))
        self.assertAlmostEqual(maxDiffV, 0.0, places=0,
                               msg='Unexpected V difference: ' + str(maxDiffV) + ', L2 diff: ' + str(diffV))
    ## Wall boundary conditions
    
    def test_wall_central(self):
        self.setBoundaryConditions()
        makeCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.sim = CTCS.CTCS(self.cl_ctx, \
                             self.h0, self.eta0, self.u0, self.v0, \
                             self.nx, self.ny, \
                             self.dx, self.dy, self.dt, \
                             self.g, self.f, self.r, self.A, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CTCS", "wallBC", "central")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)


    def test_wall_corner(self):
        self.setBoundaryConditions()
        makeCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.sim = CTCS.CTCS(self.cl_ctx, \
                             self.h0, self.eta0, self.u0, self.v0, \
                             self.nx, self.ny, \
                             self.dx, self.dy, self.dt, \
                             self.g, self.f, self.r, self.A, boundary_conditions=self.boundaryConditions)
        
        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CTCS", "wallBC", "corner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)
       
    def test_wall_upperCorner(self):
        self.setBoundaryConditions()
        makeUpperCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.sim = CTCS.CTCS(self.cl_ctx, \
                             self.h0, self.eta0, self.u0, self.v0, \
                             self.nx, self.ny, \
                             self.dx, self.dy, self.dt, \
                             self.g, self.f, self.r, self.A, boundary_conditions=self.boundaryConditions)
        
        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CTCS", "wallBC", "upperCorner")
        
        self.checkResults(eta1, u1, v1, eta2, u2, v2)

        
    ## Periodic boundary conditions
        
    def test_periodic_central(self):
        self.setBoundaryConditions(2)
        makeCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.sim = CTCS.CTCS(self.cl_ctx, \
                             self.h0, self.eta0, self.u0, self.v0, \
                             self.nx, self.ny, \
                             self.dx, self.dy, self.dt, \
                             self.g, self.f, self.r, self.A, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CTCS", "wallBC", "central")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)

        
    
    def test_periodic_corner(self):
        self.setBoundaryConditions(2)
        makeCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.sim = CTCS.CTCS(self.cl_ctx, \
                             self.h0, self.eta0, self.u0, self.v0, \
                             self.nx, self.ny, \
                             self.dx, self.dy, self.dt, \
                             self.g, self.f, self.r, self.A, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CTCS", "periodic", "corner")

        self.refEtaRange = self.etaRange
        self.refURange = self.uRange
        self.refVRange = self.vRange
        self.checkResults(eta1, u1, v1, eta2, u2, v2) 

    def test_periodic_upperCorner(self):
        self.setBoundaryConditions(2)
        makeUpperCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.sim = CTCS.CTCS(self.cl_ctx, \
                             self.h0, self.eta0, self.u0, self.v0, \
                             self.nx, self.ny, \
                             self.dx, self.dy, self.dt, \
                             self.g, self.f, self.r, self.A, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CTCS", "periodic", "upperCorner")

        self.refEtaRange = self.etaRange
        self.refURange = self.uRange
        self.refVRange = self.vRange
        self.checkResults(eta1, u1, v1, eta2, u2, v2) 


    ### PERIODIC NS - closed EW
        
    def test_periodicNS_central(self):
        self.setBoundaryConditions(3)
        makeCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.sim = CTCS.CTCS(self.cl_ctx, \
                             self.h0, self.eta0, self.u0, self.v0, \
                             self.nx, self.ny, \
                             self.dx, self.dy, self.dt, \
                             self.g, self.f, self.r, self.A, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CTCS", "wallBC", "central")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)

    def test_periodicNS_corner(self):
        self.setBoundaryConditions(3)
        makeCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.sim = CTCS.CTCS(self.cl_ctx, \
                             self.h0, self.eta0, self.u0, self.v0, \
                             self.nx, self.ny, \
                             self.dx, self.dy, self.dt, \
                             self.g, self.f, self.r, self.A, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CTCS", "periodicNS", "corner")
        self.refEtaRange = self.etaRange
        self.refURange = self.uRange
        self.refVRange = self.vRange

        self.checkResults(eta1, u1, v1, eta2, u2, v2)

    def test_periodicNS_upperCorner(self):
        self.setBoundaryConditions(3)
        makeUpperCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.sim = CTCS.CTCS(self.cl_ctx, \
                             self.h0, self.eta0, self.u0, self.v0, \
                             self.nx, self.ny, \
                             self.dx, self.dy, self.dt, \
                             self.g, self.f, self.r, self.A, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CTCS", "periodicNS", "upperCorner")
        self.refEtaRange = self.etaRange
        self.refURange = self.uRange
        self.refVRange = self.vRange

        self.checkResults(eta1, u1, v1, eta2, u2, v2)


    ### PERIODIC EW - closed NS
    
    def test_periodicEW_central(self):
        self.setBoundaryConditions(4)
        makeCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.sim = CTCS.CTCS(self.cl_ctx, \
                             self.h0, self.eta0, self.u0, self.v0, \
                             self.nx, self.ny, \
                             self.dx, self.dy, self.dt, \
                             self.g, self.f, self.r, self.A, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CTCS", "wallBC", "central")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)

    def test_periodicEW_corner(self):
        self.setBoundaryConditions(4)
        makeCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.sim = CTCS.CTCS(self.cl_ctx, \
                             self.h0, self.eta0, self.u0, self.v0, \
                             self.nx, self.ny, \
                             self.dx, self.dy, self.dt, \
                             self.g, self.f, self.r, self.A, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CTCS", "periodicEW", "corner")
        self.refEtaRange = self.etaRange
        self.refURange = self.uRange
        self.refVRange = self.vRange

        self.checkResults(eta1, u1, v1, eta2, u2, v2)

    def test_periodicEW_upperCorner(self):
        self.setBoundaryConditions(4)
        makeUpperCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.sim = CTCS.CTCS(self.cl_ctx, \
                             self.h0, self.eta0, self.u0, self.v0, \
                             self.nx, self.ny, \
                             self.dx, self.dy, self.dt, \
                             self.g, self.f, self.r, self.A, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CTCS", "periodicEW", "upperCorner")
        self.refEtaRange = self.etaRange
        self.refURange = self.uRange
        self.refVRange = self.vRange

        self.checkResults(eta1, u1, v1, eta2, u2, v2)

        
    def test_coriolis_central(self):
        self.setBoundaryConditions()
        makeCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        self.f = 0.01
        self.sim = CTCS.CTCS(self.cl_ctx, \
                             self.h0, self.eta0, self.u0, self.v0, \
                             self.nx, self.ny, \
                             self.dx, self.dy, self.dt, \
                             self.g, self.f, self.r, self.A, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CTCS", "coriolis", "central")
        self.refEtaRange = self.etaRange
        self.refURange = self.uRange
        self.refVRange = self.vRange

        self.checkResults(eta1, u1, v1, eta2, u2, v2)
