# -*- coding: utf-8 -*-

"""
This software is a part of GPU Ocean.

Copyright (C) 2018  SINTEF Digital

This python class implements the Ensemble Transform Kalman Filter.

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
import logging

class SkillScore:
    """
    This class implements some skill scores
    
    Input to constructor:
    ensemble: An object of super-type BaseOceanStateEnsemble.
            
    """

    def __init__(self, ensemble):
        """
        Copying the ensemble to the member variables 
        and deducing frequently used ensemble quantities
        """
        
        self.count_DA_times = 0 
        self.running_skill_score = 0

        self.N_e = ensemble.getNumParticles()
        self.N_y = ensemble.getNumDrifters()
    

    def MSE(self, ensemble, perturb=False):
        """
        MSE as skill score.

        Taking the MSE error over all particle observations w.r.t. the true observation for all observation positions:
        1/N_e * (sum{j=1}^{N_y} sum_{i=1}^{N_e} (hu_i(x_j) - hu_true(x_j))^2 
         + sum{j=1}^{N_y} sum_{i=1}^{N_e} (hv_i(x_j) - hv_true(x_j))^2)
        """
        skill_ensemble = ensemble
        if perturb:
            for p in range(self.N_e):
                skill_ensemble.particles[p].perturbState()
        self.count_DA_times += 1
        self.running_skill_score += np.sum(1/(self.N_e*self.N_y)*(skill_ensemble.observeParticles()-skill_ensemble.observeTrueState()[:,2:4])**2)
        print("Running skill score = ", self.running_skill_score/self.count_DA_times)

    
    def evaluate(self):
        """
        Average skill score over all DA times 
        """
        assert(self.count_DA_times != 0), "Not a single DA step to evaluate"

        return self.running_skill_score/self.count_DA_times