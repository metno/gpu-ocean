import unittest
import time
import numpy as np

from testUtils import *

import sys
sys.path.insert(0, '../')
from SWESimulators import Common, CDKLM16

class CDKLM16test(unittest.TestCase):

    def setUp(self):
        self.cl_ctx = make_cl_ctx()

        self.nx = 50
        self.ny = 70
        
        self.dx = 200.0
        self.dy = 200.0
        
        self.dt = 0.9
        self.g = 9.81
        self.f = 0.0
        self.r = 0.0
        self.A = 1
        
        #self.h0 = np.ones((self.ny+2, self.nx+2), dtype=np.float32) * 60;
        self.waterHeight = 60
        self.h0 = np.ones((self.ny+6, self.nx+6), dtype=np.float32) * self.waterHeight;
        self.u0 = np.zeros((self.ny+6, self.nx+6), dtype=np.float32);
        self.v0 = np.zeros((self.ny+6, self.nx+6), dtype=np.float32);

        self.ghosts = [3,3,3,3] # north, east, south, west
        self.refEtaRange = [-3, -3, 3, 3]
        self.refURange = [-3, -3, 3, 3] #[-1, -1, 1, 1]
        self.refVRange = [-3, -3, 3, 3] #[-1, -1, 1, 1]
        self.etaRange = [-3, -3, 3, 3] #[-1, -1, 1, 1]
        self.uRange = [-3, -3, 3, 3] #[-1, -2, 1, 2]
        self.vRange = [-3, -3, 3, 3] #[-2, -1, 2, 1]
        self.boundaryConditions = None

        self.T = 50.0


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
        
        self.assertAlmostEqual(diffEta, 0.0, places=6,
                               msg='Unexpected eta - L2 difference: ' + str(diffEta))
        self.assertAlmostEqual(diffU, 0.0, places=6,
                               msg='Unexpected U - L2 difference: ' + str(diffU))
        self.assertAlmostEqual(diffV, 0.0, places=6,
                               msg='Unexpected V - L2 difference: ' + str(diffV))

    ## Wall boundary conditions
    
    def test_wall_central(self):
        self.setBoundaryConditions()
        addCentralBump(self.h0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        sim = CDKLM16.CDKLM16(self.cl_ctx, \
                    self.h0, self.u0, self.v0, \
                    self.nx, self.ny, \
                    self.dx, self.dy, self.dt, \
                    self.g, self.f, self.r) #, boundary_conditions=self.boundaryConditions)

        t = sim.step(self.T)
        h1, u1, v1 = sim.download()
        eta1 = h1 - self.waterHeight
        eta2, u2, v2 = loadResults("CDKLM16", "wallBC", "central")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)


 
    def test_wall_corner(self):
        self.setBoundaryConditions()
        addCornerBump(self.h0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        sim = CDKLM16.CDKLM16(self.cl_ctx, \
                    self.h0, self.u0, self.v0, \
                    self.nx, self.ny, \
                    self.dx, self.dy, self.dt, \
                    self.g, self.f, self.r) #, boundary_conditions=self.boundaryConditions)

        t = sim.step(self.T)
        h1, u1, v1 = sim.download()
        eta1 = h1 - self.waterHeight
        eta2, u2, v2 = loadResults("CDKLM16", "wallBC", "corner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)

    def test_wall_upperCorner(self):
        self.setBoundaryConditions()
        addUpperCornerBump(self.h0, self.nx, self.ny, self.dx, self.dy, self.ghosts)
        sim = CDKLM16.CDKLM16(self.cl_ctx, \
                    self.h0, self.u0, self.v0, \
                    self.nx, self.ny, \
                    self.dx, self.dy, self.dt, \
                    self.g, self.f, self.r) #, boundary_conditions=self.boundaryConditions)

        t = sim.step(self.T)
        h1, u1, v1 = sim.download()
        eta1 = h1 - self.waterHeight
        eta2, u2, v2 = loadResults("CDKLM16", "wallBC", "upperCorner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)
