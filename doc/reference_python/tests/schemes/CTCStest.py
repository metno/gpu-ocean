import unittest
import time
import numpy as np

from testUtils import *

import sys
sys.path.insert(0, '../')
from SWESimulators import CTCS

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

        self.ghostCells = [1,1,1,1]

        self.T = 50.0
        

    def test_wall_central(self):
        makeCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghostCells)
        sim = CTCS.CTCS(self.cl_ctx, \
                        self.h0, self.eta0, self.u0, self.v0, \
                        self.nx, self.ny, \
                        self.dx, self.dy, self.dt, \
                        self.g, self.f, self.r, self.A)
        
        t = sim.step(self.T)
        eta1, u1, v1 = sim.download()
        eta2, u2, v2 = loadResults("CTCS", "wallBC", "central")

        diffEta = np.linalg.norm(eta1 - eta2)
        diffU = np.linalg.norm(u1-u2)
        diffV = np.linalg.norm(v1-v2)

        self.assertAlmostEqual(diffEta, 0.0,
                               msg='Unexpected eta - L2 difference: ' + str(diffEta))

        self.assertAlmostEqual(diffU, 0.0,
                               msg='Unexpected U - L2 difference: ' + str(diffU))
        self.assertAlmostEqual(diffV, 0.0,
                               msg='Unexpected V - L2 difference: ' + str(diffV))
        


    def test_wall_corner(self):
        makeCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.ghostCells)
        sim = CTCS.CTCS(self.cl_ctx, \
                        self.h0, self.eta0, self.u0, self.v0, \
                        self.nx, self.ny, \
                        self.dx, self.dy, self.dt, \
                        self.g, self.f, self.r, self.A)
        
        t = sim.step(self.T)
        eta1, u1, v1 = sim.download()
        eta2, u2, v2 = loadResults("CTCS", "wallBC", "corner")

        diffEta = np.linalg.norm(eta1 - eta2)
        diffU = np.linalg.norm(u1-u2)
        diffV = np.linalg.norm(v1-v2)

        self.assertAlmostEqual(diffEta, 0.0,
                               msg='Unexpected eta - L2 difference: ' + str(diffEta))

        self.assertAlmostEqual(diffU, 0.0, places=6,
                               msg='Unexpected U - L2 difference: ' + str(diffU))
        self.assertAlmostEqual(diffV, 0.0, places=6,
                                msg='Unexpected V - L2 difference: ' + str(diffV))
       
