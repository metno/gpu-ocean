# -*- coding: utf-8 -*-

"""
This python class implements an ensemble of particles, each consisting
of a single drifter in its own ocean state. Each ocean model is 
perturbed during each timestep, using small scale perturbations.


Copyright (C) 2018  SINTEF ICT

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

import CDKLM16
import GPUDrifterCollection
import BaseOceanStateEnsemble
import Common
import DataAssimilationUtils as dautils

reload(BaseOceanStateEnsemble)
class OceanNoiseEnsemble(BaseOceanStateEnsemble.BaseOceanStateEnsemble):
        
    
    def init(self, driftersPerOceanModel=1):
        self.driftersPerOceanModel = driftersPerOceanModel
        
        for i in range(self.numParticles+1):
            self.particles[i] = CDKLM16.CDKLM16(self.cl_ctx, \
                                                self.base_eta, self.base_hu, self.base_hv, \
                                                self.base_H, \
                                                self.nx, self.ny, self.dx, self.dy, self.dt, \
                                                self.g, self.f, self.r, \
                                                boundary_conditions=self.boundaryConditions, \
                                                write_netcdf=False, \
                                                small_scale_perturbation=True, \
                                                small_scale_perturbation_amplitude=self.small_scale_perturbation_amplitude)
            
            if self.initialization_variance_factor_ocean_field != 0.0:
                self.particles[i].perturbState(q0_scale=self.initialization_variance_factor_ocean_field)
            
            drifters = GPUDrifterCollection.GPUDrifterCollection(self.cl_ctx, driftersPerOceanModel,
                                             observation_variance=self.observation_variance,
                                             boundaryConditions=self.boundaryConditions,
                                             domain_size_x=self.nx*self.dx, domain_size_y=self.ny*self.dy)
            
            initPos = np.random.multivariate_normal(self.midPoint, self.initialization_cov_drifters, driftersPerOceanModel)
            drifters.setDrifterPositions(initPos)
            #print "drifter particles: ", drifter.getParticlePositions()
            #print "drifter observations: ", drifter.getObservationPosition()
            self.particles[i].attachDrifters(drifters)
            
        # Put the initial positions into the observation array
        self._addObservation(self.observeTrueDrifters())

    

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
            if self.observation_type == dautils.ObservationType.UnderlyingFlow:
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
                    
   