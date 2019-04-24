# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2018  SINTEF Digital

This python class implements an ensemble of particles, each consisting
of a single drifter in its own ocean state. Each ocean model is 
perturbed during each timestep, using small scale perturbations.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time
import abc
import warnings 
import os, sys

import pycuda.driver as cuda

from SWESimulators import BaseOceanStateEnsemble
from SWESimulators import Common
from SWESimulators import CDKLM16
from SWESimulators import Observation
from SWESimulators import DataAssimilationUtils as dautils



try:
    from importlib import reload
except:
    pass


class EnsembleFromFiles(BaseOceanStateEnsemble.BaseOceanStateEnsemble):
    """
    Class that holds an ensemble of ocean states, which can be initialized
    from file, and where the true state can also be loaded from file.
    """

    
    def __init__(self, gpu_ctx, numParticles,
                 ensemble_directory,
                 true_state_directory,
                 observation_variance=None,
                 drifters="all",
                 cont_write_netcdf=False
                ):
        
        # We avoid calling the parent constructor by re-implementing the 
        # few parts of the parent constructor that still is relevant.
        # We therefore accept some double coding here, as the end justifies the means.
        
        print('Welcome to the EnsembleFromFile')
        print('Ensemble directory: ', ensemble_directory)
        print('True state directory: ', true_state_directory)
        
        self.gpu_ctx = gpu_ctx
        self.gpu_stream = cuda.Stream()
        
        self.simulate_true_state = False
        
        # Control that the ensemble and true state directories exist
        assert os.path.isdir(ensemble_directory), "Ensemble init folder does not exists: " + str(ensemble_directory)
        assert os.path.isdir(true_state_directory), "True state folder does not exists: " + str(true_state_directory)
        
        # List of files for NetCDF ensemble initialization, NetCDF truth, and pickle Observations  
        self.ensemble_init_nc_files = list(os.path.join(ensemble_directory, file) for file in os.listdir(ensemble_directory) if file.endswith('.nc'))
        self.true_state_nc_files = list(os.path.join(true_state_directory, file)  for file in os.listdir(true_state_directory) if file.endswith('.nc'))
        self.observation_files = list(os.path.join(true_state_directory, file)  for file in os.listdir(true_state_directory) if file.endswith('.pickle'))
        
        assert len(self.ensemble_init_nc_files) > 0, "There should be at least one NetCDF file in ensemble directory " + str(ensemble_directory)
        assert len(self.true_state_nc_files) == 1, "There should only be one single NetCDF file in the true state directory " + str(true_state_directory)
        assert len(self.observation_files) == 1,   "There should only be one single pickle file in the true state directory " + str(true_state_directory)
        
        self.numParticles = numParticles
        self.particles = [None]*(self.numParticles)
        
        # Declare variables for true state and observations
        self.true_state = None
        self.observations = None
        
        # Intentionally not declared to check that we don't use this variable.
        #self.obs_index = None
        
        
        self.simType = 'CDKLM16'
        self.cont_write_netcdf = cont_write_netcdf
        
        # We will not simulate the true state, but read it from file:
        self.simulate_true_state = False
        
        self.observation_type = dautils.ObservationType.DrifterPosition
        
        # Required GPU buffer:
        self.observation_buffer = None
        
        
        ### Then, call appropriate functions for initialization
        self._initializeEnsembleFromFile()
        self._readObservationsFromFile() 
        self._readTruthFromFile() 
        
        self._delcareStatisticalInfoArrays()
                
        self._setGridInfoFromSim(self.particles[0])
        self._setStochasticVariables(observation_variance = observation_variance)     
        #self._init(driftersPerOceanModel=num_drifters)
        
        
    def _initializeEnsembleFromFile(self):
        num_files = len(self.ensemble_init_nc_files)
        for particle_id in range(self.numParticles):
            file_id = particle_id % num_files
            self.particles[particle_id] = CDKLM16.CDKLM16.fromfilename(self.gpu_ctx, 
                                                                       self.ensemble_init_nc_files[file_id],
                                                                       cont_write_netcdf=self.cont_write_netcdf)
    
    def _readObservationsFromFile(self):
        self.true_state = CDKLM16.CDKLM16.fromfilename(self.gpu_ctx,
                                                       self.true_state_nc_files[0],
                                                       cont_write_netcdf = False)
    
    def _readTruthFromFile(self):
        self.observations = Observation.Observation()
        self.observations.read_pickle(self.observation_files[0])
    
        

    def cleanUp(self):
        if self.true_state is not None:
            self.true_state.cleanUp()
            self.true_state = None
        super(EnsembleFromFiles, self).cleanUp()
        

    def resample(self, newSampleIndices, reinitialization_variance):
        """
        Resampling the particles given by the newSampleIndicies input array.
        Here, the reinitialization_variance input is ignored, meaning that exact
        copies only are resampled.
        """
        obsTrueDrifter = self.observeTrueDrifters()
        positions = self.observeParticles()
        newPos = np.empty((self.driftersPerOceanModel, 2))
        newOceanStates = [None]*self.getNumParticles()
        for i in range(self.getNumParticles()):
            index = newSampleIndices[i]
            #print "(particle no, position, old direction, new direction): "
            if self.observation_type == dautils.ObservationType.UnderlyingFlow or \
               self.observation_type == dautils.ObservationType.DirectUnderlyingFlow:
                newPos[:,:] = obsTrueDrifter
            else:
                # Copy the drifter position from the particle that is resampled
                newPos[:,:] = positions[index,:]
            
            #print "\t", (index, positions[index,:], newPos)

            
            # Download index's ocean state:
            eta0, hu0, hv0 = self.particles[index].download()
            eta1, hu1, hv1 = self.particles[index].downloadPrevTimestep()
            newOceanStates[i] = (eta0, hu0, hv0, eta1, hu1, hv1)

            self.particles[i].drifters.setDrifterPositions(newPos)

        # New loop for transferring the correct ocean states back up to the GPU:
        for i in range(self.getNumParticles()):
            self.particles[i].upload(newOceanStates[i][0],
                                     newOceanStates[i][1],
                                     newOceanStates[i][2],
                                     newOceanStates[i][3],
                                     newOceanStates[i][4],
                                     newOceanStates[i][5])
                    
   