# -*- coding: utf-8 -*-

"""
This software is a part of GPU Ocean.

Copyright (C) 2018  SINTEF Digital

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

import numpy as np
import os

class SkillScore:
    """
    This class implements some skill scores
    
    ensemble: An object of super-type BaseOceanStateEnsemble.
    scores: A list with the scores used for assessement,
    currently supported are 
        - "bias"
        - "MSE"
        - "CRPS".
    """

    def __init__(self, ensemble, scores):
        """
        Preparing dictionary to write skills.
        Copying frequently used ensemble quantities.
        """

        self.scores = scores

        allowed_scores = ["bias", "MSE", "CRPS"]
        assert (all(score in allowed_scores for score in scores)), "invalid scores"

        # index 0: bias, index 1: MSE, index 2: CRPS
        self.running_scoring = {}
        for score in scores:
            self.running_scoring[score] = np.zeros(0)

        self.N_e = ensemble.getNumParticles()
        self.N_y = ensemble.getNumDrifters()
    

    def assess(self, ensemble, perturb=False):
        """ Calculating different scores and saving in dict"""
        #TODO: Changing the logic that assessment takes ensemble and true observation 
        #(to avoid double observation in DA and skill classes)

        if "bias" in self.scores:
            bias = self.bias(ensemble, perturb)
            self.running_scoring["bias"] = np.append(self.running_scoring["bias"], bias)
        if "MSE" in self.scores:
            MSE = self.MSE(ensemble, perturb)
            self.running_scoring["MSE"] = np.append(self.running_scoring["MSE"], MSE)
        if "CRPS" in self.scores:
            CRPS = self.CRPS(ensemble, perturb)
            self.running_scoring["CRPS"] = np.append(self.running_scoring["CRPS"], CRPS)



    def MSE(self, ensemble, perturb=False):
        """
        MSE as skill score.

        Taking the MSE error over all particle observations w.r.t. the true observation for all observation positions:
        1/N_e * (sum{j=1}^{N_y} sum_{i=1}^{N_e} (hu_i(x_j) - hu_true(x_j))^2 
         + sum{j=1}^{N_y} sum_{i=1}^{N_e} (hv_i(x_j) - hv_true(x_j))^2)
        """
        # NOTE: The last time step of some PF is without noise 
        # to counteract this fact additional noise can be added
        skill_ensemble = ensemble
        if perturb:
            for p in range(self.N_e):
                if ensemble.particlesActive[p]:
                    skill_ensemble.particles[p].perturbState()

        MSE = np.nanmean(1/(ensemble.getNumActiveParticles()*self.N_y)*(skill_ensemble.observeParticles()-skill_ensemble.observeTrueState()[:,2:4])**2)
        
        print("Latest MSE = ", MSE)
        return  MSE


    def bias(self, ensemble, perturb=False):
        """
        bias as skill score.
        """
        # NOTE: The last time step of some PF is without noise 
        # to counteract this fact additional noise can be added
        skill_ensemble = ensemble
        if perturb:
            for p in range(self.N_e):
                if ensemble.particlesActive[p]:
                    skill_ensemble.particles[p].perturbState()

        bias =  np.nanmean((np.nanmean(skill_ensemble.observeParticles(), axis=0) - skill_ensemble.observeTrueState()[:,2:4]))
        
        print("Latest bias = ", bias)
        return bias


    def CPRS(self, ensemble, perturb=False):
        #TODO: Find suitable implementation. 
        """
        CRPS as skill score
        """
        print("Latest CRPS = ", "only dummy implementation")
        return 0 
    

    def evaluate(self, destination_dir=None):
        """
        Average skill score over all DA times 
        """

        scores = None
        avg_scores = {}
        if "bias" in self.scores:
            bias_scores = self.running_scoring["bias"]
            bias_scores = np.reshape(bias_scores, (len(bias_scores),1) )

            scores = bias_scores
            avg_scores["bias"] = np.average(bias_scores)
        
        if "MSE" in self.scores:
            MSE_scores = self.running_scoring["MSE"] 
            MSE_scores = np.reshape(MSE_scores, (len(MSE_scores),1) )

            if scores is None:
                scores = MSE_scores
            else:
                scores = np.hstack([scores,MSE_scores])
            avg_scores["MSE"] = np.average(MSE_scores)

        if "CRPS" in self.scores:
            CRPS_scores = self.running_scoring["CRPS"]
            CRPS_scores = np.reshape(CRPS_scores, (len(CRPS_scores),1) )

            if scores is None:
                scores = CRPS_scores
            else:
                scores = np.hstack([scores,CRPS_scores])
            avg_scores["CRPS"] = np.average(CRPS_scores)


        if destination_dir is not None:
            np.savetxt(os.path.join(destination_dir, 'scores.csv'), scores, header=" ".join(self.scores))

        return avg_scores 