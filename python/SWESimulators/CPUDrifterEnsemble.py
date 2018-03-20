# -*- coding: utf-8 -*-

"""
This python class implements a Ensemble of particles, each consisting of a single drifter in its own ocean state. The perturbation parameter is the wind direction.


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
import CPUDrifterCollection
import Common
import DataAssimilationUtils as dautils
import BaseDrifterEnsemble


class CPUDrifterEnsemble(BaseDrifterEnsemble.BaseDrifterEnsemble):
        
    def __init__(self, cl_ctx, numParticles, observation_variance=0.0):
         
        super(CPUDrifterEnsemble, self).__init__(cl_ctx, 
                                              numParticles, 
                                              observation_variance)
        
    
    # ---------------------------------------
    # Implementing abstract function
    # ---------------------------------------
    def init(self):

        self.sim = None

        self.drifters = CPUDrifterCollection.CPUDrifterCollection(self.numParticles,
                      observation_variance=self.observation_variance,
                      boundaryConditions=self.boundaryConditions,
                      domain_size_x=self.nx*self.dx, domain_size_y=self.ny*self.dy)
        
        self.drifters.initializeUniform()
    
    #--------------------
    ## Override
    #--------------------
    def step(self, t):
        # TODO: Steel particleDrift function from BasicParticleFilter.ipynb
        pass
    
    #-------------------
    ### NEW
    #-------------------
    def copy(self):
        copy = CPUDrifterEnsemble(None, self.numParticles, self.observation_variance)
        copy.setGridInfo(self.nx, self.ny, self.dx, self.dy, self.dt,
                         self.boundaryConditions,
                         self.base_eta, self.base_hu, self.base_hv, self.base_H)
        copy.setParameters(self.f, self.g, self.beta, self.r, self.wind)
        copy.init()
        copy.setParticleStates(self.observeParticles())
        copy.setObservationState(self.observeTrueState())
        return copy