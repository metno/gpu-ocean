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
        self.h0 = None
        self.u0 = None
        self.v0 = None
        self.Bi = None
        
        self.ghosts = [3,3,3,3] # north, east, south, west
        self.validDomain = np.array([3,3,3,3])
        self.refRange = [-3, -3, 3, 3]
        self.dataRange = self.refRange
        self.boundaryConditions = None

        self.T = 50.0
        self.sim = None
        
    def tearDown(self):
        if self.sim != None:
            self.sim.cleanUp()
            self.sim = None

            
    def allocData(self):
        dataShape = (self.ny + self.ghosts[0]+self.ghosts[2], 
                     self.nx + self.ghosts[1]+self.ghosts[3])
        self.h0 = np.ones( dataShape, dtype=np.float32) * self.waterHeight
        self.u0 = np.zeros(dataShape, dtype=np.float32)
        self.v0 = np.zeros(dataShape, dtype=np.float32)
        self.Bi = np.zeros((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C')

        


    def setBoundaryConditions(self, bcSettings=1):

        if (bcSettings == 1):
            self.boundaryConditions = Common.BoundaryConditions()
        elif (bcSettings == 2):
            self.boundaryConditions = Common.BoundaryConditions(2,2,2,2)            
        elif bcSettings == 3:
            # Periodic NS
            self.boundaryConditions = Common.BoundaryConditions(2,1,2,1)
        else:
            # Periodic EW
            self.boundaryConditions = Common.BoundaryConditions(1,2,1,2)

        
    def checkResults(self, eta1, u1, v1, etaRef, uRef, vRef):
        diffEta = np.linalg.norm(eta1[self.dataRange[2]:self.dataRange[0], 
                                      self.dataRange[3]:self.dataRange[1]] - 
                                 etaRef[self.refRange[2]:self.refRange[0],
                                        self.refRange[3]:self.refRange[1]])
        diffU = np.linalg.norm(u1[self.dataRange[2]:self.dataRange[0],
                                  self.dataRange[3]:self.dataRange[1]] -
                               uRef[self.refRange[2]:self.refRange[0],
                                    self.refRange[3]:self.refRange[1]])
        diffV = np.linalg.norm(v1[self.dataRange[2]:self.dataRange[0],
                                  self.dataRange[3]:self.dataRange[1]] - 
                               vRef[ self.refRange[2]:self.refRange[0],
                                     self.refRange[3]:self.refRange[1]])
        
        self.assertAlmostEqual(diffEta, 0.0, places=6,
                               msg='Unexpected eta - L2 difference: ' + str(diffEta))
        self.assertAlmostEqual(diffU, 0.0, places=6,
                               msg='Unexpected U - L2 difference: ' + str(diffU))
        self.assertAlmostEqual(diffV, 0.0, places=6,
                               msg='Unexpected V - L2 difference: ' + str(diffV))

    ## Wall boundary conditions
    
    def test_wall_central(self):
        self.setBoundaryConditions()
        self.allocData()
        addCentralBump(self.h0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.cl_ctx, \
                                   self.h0, self.u0, self.v0, self.Bi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r) #, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        h1, u1, v1 = self.sim.download()
        eta1 = h1 - self.waterHeight
        eta2, u2, v2 = loadResults("CDKLM16", "wallBC", "central")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)


 
    def test_wall_corner(self):
        self.setBoundaryConditions()
        self.allocData()
        addCornerBump(self.h0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.cl_ctx, \
                                   self.h0, self.u0, self.v0, self.Bi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r) #, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        h1, u1, v1 = self.sim.download()
        eta1 = h1 - self.waterHeight
        eta2, u2, v2 = loadResults("CDKLM16", "wallBC", "corner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)

    def test_wall_upperCorner(self):
        self.setBoundaryConditions()
        self.allocData()
        addUpperCornerBump(self.h0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.cl_ctx, \
                                   self.h0, self.u0, self.v0, self.Bi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r) #, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        h1, u1, v1 = self.sim.download()
        eta1 = h1 - self.waterHeight
        eta2, u2, v2 = loadResults("CDKLM16", "wallBC", "upperCorner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)


## Full periodic boundary conditions

    def test_periodic_central(self):
        self.setBoundaryConditions(bcSettings=2)
        self.allocData()
        addCentralBump(self.h0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.cl_ctx, \
                                   self.h0, self.u0, self.v0, self.Bi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        h1, u1, v1 = self.sim.download()
        eta1 = h1 - self.waterHeight
        eta2, u2, v2 = loadResults("CDKLM16", "wallBC", "central")
        
        self.checkResults(eta1, u1, v1, eta2, u2, v2)


    def test_periodic_corner(self):
        self.setBoundaryConditions(bcSettings=2)
        self.allocData()
        addCornerBump(self.h0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.cl_ctx, \
                                   self.h0, self.u0, self.v0, self.Bi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        h1, u1, v1 = self.sim.download()
        eta1 = h1 - self.waterHeight
        eta2, u2, v2 = loadResults("CDKLM16", "periodic", "corner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)

    def test_periodic_upperCorner(self):
        self.setBoundaryConditions(bcSettings=2)
        self.allocData()
        addUpperCornerBump(self.h0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.cl_ctx, \
                                   self.h0, self.u0, self.v0, self.Bi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        h1, u1, v1 = self.sim.download()
        eta1 = h1 - self.waterHeight
        eta2, u2, v2 = loadResults("CDKLM16", "periodic", "upperCorner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)


## North-south periodic boundary conditions

    def test_periodicNS_central(self):
        self.setBoundaryConditions(bcSettings=3)
        self.allocData()
        addCentralBump(self.h0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.cl_ctx, \
                                   self.h0, self.u0, self.v0, self.Bi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        h1, u1, v1 = self.sim.download()
        eta1 = h1 - self.waterHeight
        eta2, u2, v2 = loadResults("CDKLM16", "wallBC", "central")
        
        self.checkResults(eta1, u1, v1, eta2, u2, v2)

        
    def test_periodicNS_corner(self):
        self.setBoundaryConditions(bcSettings=3)
        self.allocData()
        addCornerBump(self.h0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.cl_ctx, \
                                   self.h0, self.u0, self.v0, self.Bi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        h1, u1, v1 = self.sim.download()
        eta1 = h1 - self.waterHeight
        eta2, u2, v2 = loadResults("CDKLM16", "periodicNS", "corner")
        
        self.checkResults(eta1, u1, v1, eta2, u2, v2)


        
    def test_periodicNS_upperCorner(self):
        self.setBoundaryConditions(bcSettings=3)
        self.allocData()
        addUpperCornerBump(self.h0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.cl_ctx, \
                                   self.h0, self.u0, self.v0, self.Bi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        h1, u1, v1 = self.sim.download()
        eta1 = h1 - self.waterHeight
        eta2, u2, v2 = loadResults("CDKLM16", "periodicNS", "upperCorner")
        
        self.checkResults(eta1, u1, v1, eta2, u2, v2)

 ## East-west periodic boundary conditions

    def test_periodicEW_central(self):
        self.setBoundaryConditions(bcSettings=4)
        self.allocData()
        addCentralBump(self.h0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.cl_ctx, \
                                   self.h0, self.u0, self.v0, self.Bi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        h1, u1, v1 = self.sim.download()
        eta1 = h1 - self.waterHeight
        eta2, u2, v2 = loadResults("CDKLM16", "wallBC", "central")
        
        self.checkResults(eta1, u1, v1, eta2, u2, v2)       


    def test_periodicEW_corner(self):
        self.setBoundaryConditions(bcSettings=4)
        self.allocData()
        addCornerBump(self.h0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.cl_ctx, \
                                   self.h0, self.u0, self.v0, self.Bi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        h1, u1, v1 = self.sim.download()
        eta1 = h1 - self.waterHeight
        eta2, u2, v2 = loadResults("CDKLM16", "periodicEW", "corner")
        
        self.checkResults(eta1, u1, v1, eta2, u2, v2)       

    def test_periodicEW_upperCorner(self):
        self.setBoundaryConditions(bcSettings=4)
        self.allocData()
        addUpperCornerBump(self.h0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.cl_ctx, \
                                   self.h0, self.u0, self.v0, self.Bi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        h1, u1, v1 = self.sim.download()
        eta1 = h1 - self.waterHeight
        eta2, u2, v2 = loadResults("CDKLM16", "periodicEW", "upperCorner")
        
        self.checkResults(eta1, u1, v1, eta2, u2, v2)       

  
