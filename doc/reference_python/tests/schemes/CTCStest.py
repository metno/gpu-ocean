import unittest
import time
import numpy as np

from testUtils import *

import sys
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
        self.u0 = np.zeros((self.ny+2, self.nx+1), dtype=np.float32);
        self.v0 = np.zeros((self.ny+1, self.nx+2), dtype=np.float32);

        self.ghosts = None # [1,1,1,1]
        self.arrayRange = None
        self.boundaryConditions = None

        self.T = 50.0


    def setBoundaryConditions(self, bcSettings=1):
        if (bcSettings == 1):
            self.boundaryConditions = Common.BoundaryConditions()
            self.ghosts = [1,1,1,1] # north, east, south, west
            self.arrayRange = [-1, -1, 1, 1]

            # TODO: FIX below here
        elif (bcSettings == 2):
            self.boundaryConditions = Common.BoundaryConditions(2,2,2,2)
            self.ghosts = [1,1,0,0] # Both periodic
            self.arrayRange = [-1, -1, 0, 0]
        elif bcSettings == 3:
            self.boundaryConditions = Common.BoundaryConditions(2,1,2,1)
            self.ghosts = [1,0,0,0] # periodic north-south
            self.arrayRange = [-1, None, 0, 0]
        else:
            self.boundaryConditions = Common.BoundaryConditions(1,2,1,2)
            self.ghosts = [0,1,0,0] # periodic east-west
            self.arrayRange = [None, -1, 0, 0]


        
    def checkResults(self, eta1, u1, v1, etaRef, uRef, vRef, refRange=None):
        if refRange is None:
            diffEta = np.linalg.norm(eta1[self.arrayRange[2]:self.arrayRange[0], self.arrayRange[3]:self.arrayRange[1]] - etaRef)
            diffU = np.linalg.norm(u1[self.arrayRange[2]:self.arrayRange[0], :]-uRef)
            diffV = np.linalg.norm(v1[:, self.arrayRange[3]:self.arrayRange[1]]-vRef)
        else:
            diffEta = np.linalg.norm(eta1[self.arrayRange[2]:self.arrayRange[0], 
                                          self.arrayRange[3]:self.arrayRange[1]] - 
                                     etaRef[refRange[2]:refRange[0],
                                            refRange[3]:refRange[1]])
            diffU = np.linalg.norm(u1[self.arrayRange[2]:self.arrayRange[0], :] -
                                   uRef[refRange[2]:refRange[0], :])
            diffV = np.linalg.norm(v1[:, self.arrayRange[3]:self.arrayRange[1]] - 
                                   vRef[:, refRange[3]:refRange[1]])
            
        
        self.assertAlmostEqual(diffEta, 0.0,
                               msg='Unexpected eta - L2 difference: ' + str(diffEta))
        self.assertAlmostEqual(diffU, 0.0,
                               msg='Unexpected U - L2 difference: ' + str(diffU))
        self.assertAlmostEqual(diffV, 0.0,
                               msg='Unexpected V - L2 difference: ' + str(diffV))

        
    def test_wall_central(self):
        self.setBoundaryConditions()
        makeCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        sim = CTCS.CTCS(self.cl_ctx, \
                        self.h0, self.eta0, self.u0, self.v0, \
                        self.nx, self.ny, \
                        self.dx, self.dy, self.dt, \
                        self.g, self.f, self.r, self.A)

        t = sim.step(self.T)
        eta1, u1, v1 = sim.download()
        eta2, u2, v2 = loadResults("CTCS", "wallBC", "central")

        self.checkResults(eta1, u1, v1, eta2, u2, v2, self.arrayRange)


    def test_wall_corner(self):
        self.setBoundaryConditions()
        makeCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        sim = CTCS.CTCS(self.cl_ctx, \
                        self.h0, self.eta0, self.u0, self.v0, \
                        self.nx, self.ny, \
                        self.dx, self.dy, self.dt, \
                        self.g, self.f, self.r, self.A)
        
        t = sim.step(self.T)
        eta1, u1, v1 = sim.download()
        eta2, u2, v2 = loadResults("CTCS", "wallBC", "corner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2, self.arrayRange)
       
