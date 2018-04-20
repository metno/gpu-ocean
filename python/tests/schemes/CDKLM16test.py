import unittest
import time
import numpy as np
import sys
import gc

from testUtils import *

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
        self.eta0 = None
        self.u0 = None
        self.v0 = None
        self.Hi = None
        
        self.ghosts = [2,2,2,2] # north, east, south, west
        self.validDomain = np.array([2,2,2,2])
        self.refRange = [-3, -3, 3, 3]
        self.dataRange = [-2, -2, 2, 2]
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
        self.Hi = None
        self.cl_ctx = None
        gc.collect() # Force run garbage collection to free up memory
        


            
    def allocData(self):
        dataShape = (self.ny + self.ghosts[0]+self.ghosts[2], 
                     self.nx + self.ghosts[1]+self.ghosts[3])
        self.h0 = np.ones( dataShape, dtype=np.float32) * self.waterHeight
        self.eta0 = np.zeros(dataShape, dtype=np.float32);
        self.u0 = np.zeros(dataShape, dtype=np.float32)
        self.v0 = np.zeros(dataShape, dtype=np.float32)
        self.Hi = np.ones((dataShape[0]+1, dataShape[1]+1), dtype=np.float32, order='C') * self.waterHeight

        


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
        maxDiffEta = np.max(eta1[self.dataRange[2]:self.dataRange[0], 
                                 self.dataRange[3]:self.dataRange[1]] - 
                            etaRef[self.refRange[2]:self.refRange[0],
                                   self.refRange[3]:self.refRange[1]])
        maxDiffU = np.max(u1[self.dataRange[2]:self.dataRange[0],
                             self.dataRange[3]:self.dataRange[1]] -
                          uRef[self.refRange[2]:self.refRange[0],
                               self.refRange[3]:self.refRange[1]])
        maxDiffV = np.max(v1[self.dataRange[2]:self.dataRange[0],
                             self.dataRange[3]:self.dataRange[1]] - 
                          vRef[ self.refRange[2]:self.refRange[0],
                                self.refRange[3]:self.refRange[1]])
        
        self.assertAlmostEqual(maxDiffEta, 0.0, places=0,
                               msg='Unexpected eta difference! Max diff: ' + str(maxDiffEta) + ', L2 diff: ' + str(diffEta))
        #
        # W A R N I N G ! ! ! W A R N I N G ! ! ! W A R N I N G ! ! !
        #                             Disabled tests for u and v
        #
        #self.assertAlmostEqual(maxDiffU, 0.0, places=0,
        #                       msg='Unexpected U difference: ' + str(maxDiffU) + ', L2 diff: ' + str(diffU))
        #self.assertAlmostEqual(maxDiffV, 0.0, places=0,
        #                       msg='Unexpected V difference: ' + str(maxDiffV) + ', L2 diff: ' + str(diffV))
    ## Wall boundary conditions
    
    def test_wall_central(self):
        self.setBoundaryConditions()
        self.allocData()
        addCentralBump(self.h0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.cl_ctx, \
                                   self.eta0, self.u0, self.v0, self.Hi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r) #, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CDKLM16", "wallBC", "central")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)


 
    def test_wall_corner(self):
        self.setBoundaryConditions()
        self.allocData()
        addCornerBump(self.h0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.cl_ctx, \
                                   self.eta0, self.u0, self.v0, self.Hi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r) #, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CDKLM16", "wallBC", "corner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)

    def test_wall_upperCorner(self):
        self.setBoundaryConditions()
        self.allocData()
        addUpperCornerBump(self.h0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.cl_ctx, \
                                   self.eta0, self.u0, self.v0, self.Hi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r) #, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CDKLM16", "wallBC", "upperCorner")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)




  
    def test_coriolis_central(self):
        self.setBoundaryConditions()
        self.allocData()
        self.f = 0.01
        addCentralBump(self.eta0, self.nx, self.ny, self.dx, self.dy, self.validDomain)
        self.sim = CDKLM16.CDKLM16(self.cl_ctx, \
                                   self.eta0, self.u0, self.v0, self.Hi, \
                                   self.nx, self.ny, \
                                   self.dx, self.dy, self.dt, \
                                   self.g, self.f, self.r) #, boundary_conditions=self.boundaryConditions)

        t = self.sim.step(self.T)
        eta1, u1, v1 = self.sim.download()
        eta2, u2, v2 = loadResults("CDKLM16", "coriolis", "central")

        self.checkResults(eta1, u1, v1, eta2, u2, v2)
