import unittest
import time
import numpy as np

from testUtils import *

import sys
sys.path.insert(0, '../')
from SWESimulators import FBL

class FBLtest(unittest.TestCase):

    def setUp(self):
        self.a = 123
        self.cl_ctx = make_cl_ctx()

        self.nx = 50
        self.ny = 70
        
        self.dx = 200.0
        self.dy = 200.0
        
        self.dt = 1
        self.g = 9.81
        self.f = 0.0
        self.r = 0.0

        self.h0 = np.ones((self.ny, self.nx), dtype=np.float32) * 60;
        self.eta0 = np.zeros((self.ny, self.nx), dtype=np.float32);
        self.u0 = np.zeros((self.ny, self.nx+1), dtype=np.float32);
        self.v0 = np.zeros((self.ny+1, self.nx), dtype=np.float32);

        self.T = 50.0
        

    def test_wall_central(self):
        makeCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, 0, 0)
        sim = FBL.FBL(self.cl_ctx, \
                      self.h0, self.eta0, self.u0, self.v0, \
                      self.nx, self.ny, \
                      self.dx, self.dy, self.dt, \
                      self.g, self.f, self.r)
        
        t = sim.step(self.T)
        eta1, u1, v1 = sim.download()
        eta2, u2, v2 = loadResults("FBL", "wallBC", "central")

        diffEta = np.linalg.norm(eta1 - eta2)
        diffU = np.linalg.norm(u1-u2)
        diffV = np.linalg.norm(v1-v2)

        self.assertEqual(diffEta, 0.0,
                         'Unexpected eta - L2 difference: ' + str(diffEta))

        self.assertEqual(diffU, 0.0,
                         'Unexpected U - L2 difference: ' + str(diffU))
        self.assertEqual(diffV, 0.0,
                         'Unexpected V - L2 difference: ' + str(diffV))
        


    def test_wall_corner(self):
        makeCornerBump(self.eta0, self.nx, self.ny, self.dx, self.dy, 0, 0)
        sim = FBL.FBL(self.cl_ctx, \
                      self.h0, self.eta0, self.u0, self.v0, \
                      self.nx, self.ny, \
                      self.dx, self.dy, self.dt, \
                      self.g, self.f, self.r)
        
        t = sim.step(self.T)
        eta1, u1, v1 = sim.download()
        eta2, u2, v2 = loadResults("FBL", "wallBC", "corner")

        diffEta = np.linalg.norm(eta1 - eta2)
        diffU = np.linalg.norm(u1-u2)
        diffV = np.linalg.norm(v1-v2)

        self.assertEqual(diffEta, 0.0,
                         'Unexpected eta - L2 difference: ' + str(diffEta))

        self.assertEqual(diffU, 0.0,
                         'Unexpected U - L2 difference: ' + str(diffU))
        self.assertEqual(diffV, 0.0,
                         'Unexpected V - L2 difference: ' + str(diffV))
       
