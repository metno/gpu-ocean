# -*- coding: utf-8 -*-

"""
This python class represents an ensemble of ocean models with slightly
perturbed states. Runs on a single node.

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

import numpy as np
import logging

from SWESimulators import CDKLM16, Common, GPUDrifterCollection

class OceanModelEnsemble:
    """
    Class which holds a set of simulators on a single node, possibly with drifters attached
    """
    
    def __init__(self, gpu_ctx, sim_args, sim_ic, num_particles,
                 drifter_positions=[],
                 observation_variance = 0.01**2, 
                 initialization_variance_factor_ocean_field = 0.0):
        """
        Constructor which creates num_particles slighly different ocean models
        based on the same initial conditions
        """
        
        self.logger = logging.getLogger(__name__)
        self.gpu_ctx = gpu_ctx
        self.sim_args = sim_args
        self.observation_variance = observation_variance
        self.initialization_variance_factor_ocean_field = initialization_variance_factor_ocean_field
        
        
        
        
        # Build observation covariance matrix:
        if np.isscalar(self.observation_variance):
            self.observation_cov = np.eye(2)*self.observation_variance
            self.observation_cov_inverse = np.eye(2)*(1.0/self.observation_variance)
        else:
            # Assume that we have a correctly shaped matrix here
            self.observation_cov = self.observation_variance
            self.observation_cov_inverse = np.linalg.inv(self.observation_cov)
            
            
            
        # Generate ensemble members
        self.logger.debug("Creating %d particles (ocean models)", num_particles)
        self.particles = [None] * num_particles
        for i in range(num_particles):
            self.particles[i] = CDKLM16.CDKLM16(self.gpu_ctx, **sim_ic, **self.sim_args)
            
            if self.initialization_variance_factor_ocean_field != 0.0:
                self.particles[i].perturbState(q0_scale=self.initialization_variance_factor_ocean_field)
            
            # Attach drifters if requested
            self.logger.debug("Attaching %d drifters", len(drifter_positions))
            if (len(drifter_positions) > 0):
                drifters = GPUDrifterCollection.GPUDrifterCollection(self.gpu_ctx, len(drifter_positions),
                                                                     observation_variance=self.observation_variance,
                                                                     boundaryConditions=sim_ic['boundary_conditions'],
                                                                     domain_size_x=sim_args['nx']*sim_args['dx'], 
                                                                     domain_size_y=sim_args['ny']*sim_args['dy'])
                drifters.setDrifterPositions(drifter_positions)
                self.particles[i].attachDrifters(drifters)
            
    
    
    
    def cleanUp(self):
        for oceanState in self.particles:
            if oceanState is not None:
                oceanState.cleanUp()
    
    
    
    
    def modelStep(self, sub_t):
        self.logger.debug("Stepping all particles (ocean models) %f in time", sub_t)
        """
        Function which makes all particles step until time t.
        """
        for p in self.particles:
            self.t = p.step(sub_t)
        return self.t
    
    
    
    
    
    def getDrifterPositions(self, particle_index):
        self.logger.debug("Returning drifter positions")
        return self.particles[particle_index].drifters.getDrifterPositions()
    
    
    
    

    def getVelocity(self, drifter_positions):
        self.logger.debug("Computing velocities at given positions")
        """
        Applying the observation operator on each particle.

        Structure on the output:
        [
        particle 1:  [u_1, v_1], ... , [u_D, v_D],
        particle 2:  [u_1, v_1], ... , [u_D, v_D],
        particle Ne: [u_1, v_1], ... , [u_D, v_D]
        ]
        numpy array with dimensions (num_particles, num_drifters, 2)
        
        """
        num_particles = len(self.particles)
        num_drifters = len(drifter_positions)
        
        velocities = np.empty((num_particles, num_drifters, 2))

        # Assumes that all particles use the same bathymetry
        H = self.particles[0].downloadBathymetry()[1]
        
        for p in range(num_particles):
            # Downloading ocean state without ghost cells
            eta, hu, hv = self.particles[p].download(interior_domain_only=True)

            for d in range(num_drifters):
                id_x = np.int(np.floor(drifter_positions[d, 0]/self.sim_args['dx']))
                id_y = np.int(np.floor(drifter_positions[d, 1]/self.sim_args['dy']))

                depth = H[id_y, id_x]
                velocities[p,d,0] = hu[id_y, id_x]/(depth + eta[id_y, id_x])
                velocities[p,d,1] = hv[id_y, id_x]/(depth + eta[id_y, id_x])
                
        return velocities
        

