import unittest
import time
import numpy as np
import sys
import gc

from testUtils import *

sys.path.insert(0, '../')
from SWESimulators import Common
from SWESimulators.Particles import *

#reload(GlobalParticles)

class GlobalParticlesTest(unittest.TestCase):

    def setUp(self):

        self.numParticles = 3
        self.observationVariance = 0.5
        self.boundaryCondition = Common.BoundaryConditions(2,2,2,2)
        self.smallParticleSet = GlobalParticles(self.numParticles,
                                                self.observationVariance,
                                                self.boundaryCondition)

        self.smallParticleSet.positions[0,:] = [0.9, 0.9]
        self.smallParticleSet.positions[1,:] = [0.9, 0.1]
        self.smallParticleSet.positions[2,:] = [0.1, 0.9]
        self.smallParticleSet.positions[3,:] = [0.1, 0.1]
                                                  

    #def tearDown(self):
    # Intentionally empty
        

    ### HANDY UTILS ###
    
    def assertListAlmostEqual(self, list1, list2, tol, testname):
        l = max(len(list1), len(list2))
        outro = ""
        if l < 6:
            outro = "\n\n- " + str(list1) + "\n+ " + str(list2)
        
        strList1 = str(list1)[:21]
        if (len(strList1) > 20):
            strList1 = strList1[:20] + "..."
        strList2 = str(list2)[:21]
        if (len(strList2) > 20):
            strList2 = strList2[:20] + "..."
            
        msg = "test case \'" + testname + "\' - lists differs: " + strList1 + " != " + strList2 + "\n\n"
        self.assertEqual(len(list1), len(list2),
                         msg=msg + "Not same lengths:\nlen(list1) = " + str(len(list1)) + "\nlen(list2) = " + str(len(list2)) + outro)

        l = len(list1)
        outro = ""
        if l < 6:
            outro = "\n\n- " + str(list1) + "\n+ " + str(list2)
        i = 0
        for a,b in zip(list1, list2):
            self.assertAlmostEqual(a, b, tol,
                                   msg = msg + "First dofferomg element " + str(i) + ":\n" + str(a) + "\n" + str(b) + outro)
            i = i + 1


            
    ### START TESTS ###
    
    def test_default_constructor(self):
        defaultParticleSet = GlobalParticles(self.numParticles)

        self.assertEqual(defaultParticleSet.getNumParticles(), self.numParticles)
        self.assertEqual(defaultParticleSet.getObservationVariance(), 0.1)

        positions = defaultParticleSet.getParticlePositions()
        defaultPosition = [0.0, 0.0]
        self.assertEqual(positions.shape, ((self.numParticles, 2)))
        self.assertEqual(positions[0,:].tolist(), defaultPosition)
        self.assertEqual(positions[1,:].tolist(), defaultPosition)
        self.assertEqual(positions[2,:].tolist(), defaultPosition)
                         
        observation = defaultParticleSet.getObservationPosition()
        self.assertEqual(observation.shape, ((2,)))
        self.assertEqual(observation.tolist(), defaultPosition)

        self.assertEqual(defaultParticleSet.getDomainSizeX(), 1.0)
        self.assertEqual(defaultParticleSet.getDomainSizeY(), 1.0)

        weight = 1.0/3
        weights = [weight, weight, weight]
        self.assertEqual(defaultParticleSet.getGaussianWeight().tolist(), weights)
        self.assertEqual(defaultParticleSet.getCauchyWeight().tolist(), weights)
        
        # Check boundary condition
        self.assertEqual(defaultParticleSet.getBoundaryConditions().get(), [1,1,1,1])
        
    def test_non_default_constructor(self):
        self.assertEqual(self.smallParticleSet.getNumParticles(), self.numParticles)
        self.assertEqual(self.smallParticleSet.getObservationVariance(), self.observationVariance)
        
        positions = self.smallParticleSet.getParticlePositions()
        self.assertEqual(positions.shape, ((self.numParticles, 2)))
        self.assertEqual(positions[0,:].tolist(), [0.9, 0.9])
        self.assertEqual(positions[1,:].tolist(), [0.9, 0.1])
        self.assertEqual(positions[2,:].tolist(), [0.1, 0.9])

        observation = self.smallParticleSet.getObservationPosition()
        self.assertEqual(observation.shape, ((2,)))
        self.assertEqual(observation.tolist(), [0.1, 0.1])
        
        self.assertEqual(self.smallParticleSet.getBoundaryConditions().get(), [2,2,2,2])


    def test_set_boundary_condition(self):
        self.smallParticleSet.setBoundaryConditions(Common.BoundaryConditions(2,1,2,1))
        self.assertEqual(self.smallParticleSet.getBoundaryConditions().get(), [2,1,2,1])


    def test_distances(self):
        longDiag = np.sqrt(2*0.8*0.8)
        longLine = 0.8
        shortDiag = np.sqrt(0.2*0.2 + 0.2*0.2)
        shortLine = 0.2
        semiDiag = np.sqrt(0.2*0.2 + 0.8*0.8)
                           
        
        # smallParticleSet is initially with periodic boundary conditions
        self.assertListAlmostEqual(self.smallParticleSet.getDistances().tolist(), \
                                   [shortDiag, shortLine, shortLine], 12,
                                   'distance with periodic boundaries')
        
        self.smallParticleSet.setBoundaryConditions(Common.BoundaryConditions(1,1,1,1))
        self.assertListAlmostEqual(self.smallParticleSet.getDistances().tolist(), \
                                   [longDiag, longLine, longLine], 12,
                                   'distances with non-periodic boundaries')
        
        self.smallParticleSet.setBoundaryConditions(Common.BoundaryConditions(1,2,1,2))
        self.assertListAlmostEqual(self.smallParticleSet.getDistances().tolist(), \
                                   [semiDiag, shortLine, longLine], 12,
                                   'distances with periodic boundaries in east-west')

        self.smallParticleSet.setBoundaryConditions(Common.BoundaryConditions(2,1,2,1))
        self.assertListAlmostEqual(self.smallParticleSet.getDistances().tolist(), \
                                   [semiDiag, longLine, shortLine], 12,
                                   'distances with periodic boundaries in north-south')


    #def test_init(self):
    #    largeParticleSet = GlobalParticles(1000)
    #    largeParticleSet.init
                                                

