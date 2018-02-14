import unittest
import time
import numpy as np
import sys
import gc

from testUtils import *

sys.path.insert(0, '../')
from SWESimulators import Common
from SWESimulators.Particles import *
from SWESimulators import Resampling

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

        self.resampleNumParticles = 6
        self.resamplingParticleSet = GlobalParticles(self.resampleNumParticles)
        for i in range(2):
            self.resamplingParticleSet.positions[3*i+0, :] = [0.25, 0.35+i*0.3]
            self.resamplingParticleSet.positions[3*i+1, :] = [0.4,  0.35+i*0.3]
            self.resamplingParticleSet.positions[3*i+2, :] = [0.65, 0.35+i*0.3]
        self.resamplingParticleSet.positions[6, :] = [0.25, 0.5]
        self.resamplingVar = 1e-8
        
    #def tearDown(self):
    # Intentionally empty
    

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

    def test_set_particle_positions(self):
        pos1 = [0.2, 0.5]
        pos2 = [0.8, 0.235]
        pos3 = [0.01, 0.01]
        newPositions = np.array([pos1, pos2, pos3])

        self.smallParticleSet.setParticlePositions(newPositions)

        positions = self.smallParticleSet.getParticlePositions()
        self.assertEqual(positions.shape, ((self.numParticles, 2)))
        self.assertEqual(positions[0,:].tolist(), pos1)
        self.assertEqual(positions[1,:].tolist(), pos2)
        self.assertEqual(positions[2,:].tolist(), pos3)

        observation = self.smallParticleSet.getObservationPosition()
        self.assertEqual(observation.shape, ((2,)))
        self.assertEqual(observation.tolist(), [0.1, 0.1])

    def test_set_observation_position(self):
        pos = np.array([0.523, 0.999])
        self.smallParticleSet.setObservationPosition(pos)
        
        positions = self.smallParticleSet.getParticlePositions()
        self.assertEqual(positions.shape, ((self.numParticles, 2)))
        self.assertEqual(positions[0,:].tolist(), [0.9, 0.9])
        self.assertEqual(positions[1,:].tolist(), [0.9, 0.1])
        self.assertEqual(positions[2,:].tolist(), [0.1, 0.9])

        observation = self.smallParticleSet.getObservationPosition()
        self.assertEqual(observation.shape, ((2,)))
        self.assertEqual(observation.tolist(), pos.tolist())
        
        self.assertEqual(self.smallParticleSet.getBoundaryConditions().get(), [2,2,2,2])
        
        
        
    def test_distances(self):
        longDiag = np.sqrt(2*0.8*0.8)
        longLine = 0.8
        shortDiag = np.sqrt(0.2*0.2 + 0.2*0.2)
        shortLine = 0.2
        semiDiag = np.sqrt(0.2*0.2 + 0.8*0.8)
                           
        
        # smallParticleSet is initially with periodic boundary conditions
        assertListAlmostEqual(self, self.smallParticleSet.getDistances().tolist(), \
                              [shortDiag, shortLine, shortLine], 12,
                              'distance with periodic boundaries')
        
        self.smallParticleSet.setBoundaryConditions(Common.BoundaryConditions(1,1,1,1))
        assertListAlmostEqual(self, self.smallParticleSet.getDistances().tolist(), \
                              [longDiag, longLine, longLine], 12,
                              'distances with non-periodic boundaries')
        
        self.smallParticleSet.setBoundaryConditions(Common.BoundaryConditions(1,2,1,2))
        assertListAlmostEqual(self, self.smallParticleSet.getDistances().tolist(), \
                              [semiDiag, shortLine, longLine], 12,
                              'distances with periodic boundaries in east-west')

        self.smallParticleSet.setBoundaryConditions(Common.BoundaryConditions(2,1,2,1))
        assertListAlmostEqual(self, self.smallParticleSet.getDistances().tolist(), \
                              [semiDiag, longLine, shortLine], 12,
                              'distances with periodic boundaries in north-south')

    def test_ensemble_mean(self):
        periodicMean = [1-0.1/3, 1-0.1/3]
        nonPeriodicMean = [(0.9 + 0.9 + 0.1)/3, (0.9 + 0.9 + 0.1)/3]
        semiPeriodicMean = [nonPeriodicMean[0], periodicMean[1]]
        
        assertListAlmostEqual(self, self.smallParticleSet.getEnsembleMean().tolist(),
                              periodicMean, 12,
                              'periodic mean')

        self.smallParticleSet.setBoundaryConditions(Common.BoundaryConditions(1,1,1,1))
        assertListAlmostEqual(self, self.smallParticleSet.getEnsembleMean().tolist(),
                              nonPeriodicMean, 12,
                              'non-periodic mean')

        self.smallParticleSet.setBoundaryConditions(Common.BoundaryConditions(2,1,2,1))
        assertListAlmostEqual(self, self.smallParticleSet.getEnsembleMean().tolist(),
                              semiPeriodicMean, 12,
                              'north-south-periodic mean')
        
        
    def test_init(self):
        largeParticleSet = GlobalParticles(1000)
        domain_x = 10.3
        domain_y = 5.4
        largeParticleSet.initializeParticles(domain_size_x = domain_x,
                                             domain_size_y = domain_y)

        self.assertEqual(largeParticleSet.getDomainSizeX(), domain_x)
        self.assertEqual(largeParticleSet.getDomainSizeY(), domain_y)

        p = largeParticleSet.getParticlePositions()
        self.assertGreaterEqual(np.min(p[:,0]), 0.0)
        self.assertLessEqual(np.max(p[:,0]), domain_x)
        self.assertGreaterEqual(np.min(p[:,1]) , 0.0)
        self.assertLessEqual(np.max(p[:,1]), domain_y)

    def test_set_domain_size(self):
        size_x = 10.3
        size_y = 5.4
        self.smallParticleSet.setDomainSize(size_x, size_y)

        self.assertEqual(self.smallParticleSet.getDomainSizeX(), size_x)
        self.assertEqual(self.smallParticleSet.getDomainSizeY(), size_y)

        longDiag = np.sqrt(2*0.8*0.8)
        longLine = 0.8
        # Distance should now be the above, even with periodic boundary conditions
        assertListAlmostEqual(self, self.smallParticleSet.getDistances().tolist(),
                              [longDiag, longLine, longLine], 12,
                              'getDistance() in big periodic domain')
        
        
    def test_copy(self):
        size_x = 10.3
        size_y = 5.4
        self.smallParticleSet.setDomainSize(size_x, size_y)
                        
        # Give non-standard domain_size before 
        copy = self.smallParticleSet.copy()
        
        self.assertEqual(copy.getNumParticles(), self.numParticles)
        self.assertEqual(copy.getObservationVariance(), self.observationVariance)
        
        positions = copy.getParticlePositions()
        self.assertEqual(positions.shape, ((self.numParticles, 2)))
        self.assertEqual(positions[0,:].tolist(), [0.9, 0.9])
        self.assertEqual(positions[1,:].tolist(), [0.9, 0.1])
        self.assertEqual(positions[2,:].tolist(), [0.1, 0.9])
        
        observation = copy.getObservationPosition()
        self.assertEqual(observation.shape, ((2,)))
        self.assertEqual(observation.tolist(), [0.1, 0.1])
        
        self.assertEqual(copy.getBoundaryConditions().get(), [2,2,2,2])

        self.assertEqual(copy.getDomainSizeX(), size_x)
        self.assertEqual(copy.getDomainSizeY(), size_y)
                                                           
        # Move a particle in the original dataset and check that it is still the same in
        # the copy

        self.smallParticleSet.positions[1,0] = 0.5
        self.smallParticleSet.positions[1,1] = 0.5

        positions = copy.getParticlePositions()
        self.assertEqual(positions[1,:].tolist(), [0.9, 0.1])


    def test_copy_env(self):
        size_x = 10.3
        size_y = 5.4
        self.smallParticleSet.setDomainSize(size_x, size_y)
                        
        # Give non-standard domain_size before
        newNumParticles = 4
        copy = self.smallParticleSet.copyEnv(newNumParticles)
        
        self.assertEqual(copy.getNumParticles(), newNumParticles)
        self.assertEqual(copy.getObservationVariance(), self.observationVariance)
        
        positions = copy.getParticlePositions()
        self.assertEqual(positions.shape, ((newNumParticles, 2)))
        self.assertEqual(positions[0,:].tolist(), [0.0, 0.0])
        self.assertEqual(positions[1,:].tolist(), [0.0, 0.0])
        self.assertEqual(positions[2,:].tolist(), [0.0, 0.0])
        self.assertEqual(positions[3,:].tolist(), [0.0, 0.0])
        
        observation = copy.getObservationPosition()
        self.assertEqual(observation.shape, ((2,)))
        self.assertEqual(observation.tolist(), [0.1, 0.1])
        
        self.assertEqual(copy.getBoundaryConditions().get(), [2,2,2,2])

        self.assertEqual(copy.getDomainSizeX(), size_x)
        self.assertEqual(copy.getDomainSizeY(), size_y)
        
    def test_gaussian_weights(self):
        obtainedWeights = self.resamplingParticleSet.getGaussianWeight()
        referenceWeights = [  3.77361928e-01,   1.22511481e-01,   1.26590824e-04,   3.77361928e-01, 1.22511481e-01,   1.26590824e-04]
        assertListAlmostEqual(self, obtainedWeights.tolist(),
                              referenceWeights, 9,
                              'gaussian weights')

    def test_cauchy_weights(self):
        obtainedWeights = self.resamplingParticleSet.getCauchyWeight()
        referenceWeights = [0.28413284,  0.16789668,  0.04797048,  0.28413284,  0.16789668,  0.04797048]
        assertListAlmostEqual(self, obtainedWeights.tolist(),
                              referenceWeights, 8,
                              'cauchy weights')

    def resample(self, indices_list):
        newParticlePositions = []
        for i in indices_list:
            newParticlePositions.append(self.resamplingParticleSet.getParticlePositions()[i,:].tolist())
        return newParticlePositions
        
    def test_resampling_predefined_indices(self):
        indices_list = [2,2,2,4,5,5]
        newParticlePositions = self.resample(indices_list)
        Resampling.resampleParticles(self.resamplingParticleSet, \
                                     indices_list, 0)
        self.assertEqual(self.resamplingParticleSet.getParticlePositions().tolist(), \
                         newParticlePositions)

    def test_probabilistic_resampling_with_duplicates(self):
        setNpRandomSeed()
        indices = [1,3,0,0,0,0]
        solutions = self.resample(indices)
        Resampling.probabilisticResampling(self.resamplingParticleSet)
        self.assertEqual(self.resamplingParticleSet.getParticlePositions().tolist(), \
                         solutions)

            
    def test_residual_sampling_with_duplicates(self):
        setNpRandomSeed()
        indices = [0,0,3,3,1,4]
        solutions = self.resample(indices)
        Resampling.residualSampling(self.resamplingParticleSet)
        self.assertEqual(self.resamplingParticleSet.getParticlePositions().tolist(), \
                         solutions)
        
    def test_stochastic_universal_sampling_with_duplicates(self):
        setNpRandomSeed()
        indices = [0,0,1,3,3,4]
        solutions = self.resample(indices)
        Resampling.stochasticUniversalSampling(self.resamplingParticleSet)
        self.assertEqual(self.resamplingParticleSet.getParticlePositions().tolist(), \
                         solutions)

    def test_monte_carlo_metropolis_hasting_sampling_with_duplicates(self):
        setNpRandomSeed()
        indices = [0,0,0,3,4,4]
        solutions = self.resample(indices)
        Resampling.metropolisHastingSampling(self.resamplingParticleSet)
        self.assertEqual(self.resamplingParticleSet.getParticlePositions().tolist(), solutions)


    def test_probabilistic_resampling(self):
        setNpRandomSeed()
        indices = [1,3,0,0,0,0]
        solutions = self.resample(indices)
        Resampling.probabilisticResampling(self.resamplingParticleSet, self.resamplingVar)
        assert2DListAlmostEqual(self, self.resamplingParticleSet.getParticlePositions().tolist(), solutions, 2, "probabilistic resampling, probabilistic duplicates")

            
    def test_residual_sampling(self):
        setNpRandomSeed()
        indices = [0,0,3,3,1,4]
        solutions = self.resample(indices)
        Resampling.residualSampling(self.resamplingParticleSet, self.resamplingVar)
        assert2DListAlmostEqual(self, self.resamplingParticleSet.getParticlePositions().tolist(), solutions, 2, "residual sampling, probabilistic duplicates")
                
    def test_stochastic_universal_sampling(self):
        setNpRandomSeed()
        indices = [0,0,1,3,3,4]
        solutions = self.resample(indices)
        Resampling.stochasticUniversalSampling(self.resamplingParticleSet, self.resamplingVar)
        assert2DListAlmostEqual(self, self.resamplingParticleSet.getParticlePositions().tolist(), solutions, 2, "stochastic universal sampling, probabilistic duplicates")

    def test_monte_carlo_metropolis_hasting_sampling(self):
        setNpRandomSeed()
        indices = [0,0,0,3,4,4]
        solutions = self.resample(indices)
        Resampling.metropolisHastingSampling(self.resamplingParticleSet, self.resamplingVar)
        print self.resamplingParticleSet.getParticlePositions().tolist()
        assert2DListAlmostEqual(self, self.resamplingParticleSet.getParticlePositions().tolist(), solutions, 2, "metropolis hasting sampling, probabilistic duplicates")
        

