# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

This python class represents an ensemble of ocean models with slightly
perturbed states. Runs on multiple nodes using MPI

Copyright (C) 2018  SINTEF Digital
Copyright (C) 2018 Norwegian Meteorological Institute

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

import datetime
import logging
import numpy as np
from mpi4py import MPI
import gc

from SWESimulators import OceanModelEnsemble, Common


class MPIOceanModelEnsemble:
    """
    Class which holds a set of OceanModelEnsembles on different nodes. 
    Rank 0 holds the truth, and orchistrates the simulation
    Rank 1 to n-1 holds per_node_ensemble_size ocean models, so that the total
    number of ensemble members is (comm.size-1)*per_node_ensemble_size
    
    All ocean models are initialized using the same initial conditions
    """
    
    def __init__(self, comm, 
                 local_ensemble_size=None, drifter_positions=[], 
                 sim_args={}, data_args={},
                 ensemble_args={}):
        """
        Initialize the ensemble. Only rank 0 should receive the optional arguments.
        The constructor handles initialization across nodes
        """
        self.logger = logging.getLogger(__name__ + "_rank=" + str(comm.rank))
        self.logger.debug("Initializing")
        
        
        
        #Broadcast general information about ensemble
        ##########################
        self.comm = comm
        self.num_nodes = self.comm.size - 1 #Root does not participate
        assert self.comm.size >= 2, "You appear to not be using enough MPI nodes (at least two required)"
        
        self.local_ensemble_size = local_ensemble_size
        self.local_ensemble_size = self.comm.bcast(self.local_ensemble_size, root=0)
        
        self.num_drifters = len(drifter_positions)
        self.num_drifters = self.comm.bcast(self.num_drifters, root=0)
        
        # Ensure all particles in all processes use the same timestamp (common for each EPS run)
        if (self.comm.rank == 0):
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            timestamp_short = datetime.datetime.now().strftime("%Y_%m_%d")
            netcdf_filename = timestamp + ".nc"
        else:
            netcdf_filename = None
        netcdf_filename = self.comm.bcast(netcdf_filename, root=0)
        
        
        #Broadcast initial conditions for simulator
        ##########################
        self.sim_args = sim_args 
        self.sim_args = self.comm.bcast(self.sim_args, root=0)
        
        # FIXME: ! ! ! SLOW ! ! !
        self.data_args = data_args 
        self.data_args = self.comm.bcast(self.data_args, root=0)
        
        self.data_shape = (self.data_args['ny'], self.data_args['nx'])
        
        #if (self.comm.rank != 0):
        #    #FIXME: Hardcoded to sponge_cells=[80, 80, 80, 80]
        #    data_args['H'] = np.empty((self.data_shape[0]+161, self.data_shape[1]+161), dtype=np.float32)
        #    data_args['eta0'] = np.empty((self.data_shape[0]+160, self.data_shape[1]+160), dtype=np.float32)
        #    data_args['hu0'] = np.empty((self.data_shape[0]+160, self.data_shape[1]+160), dtype=np.float32)
        #    data_args['hv0'] = np.empty((self.data_shape[0]+160, self.data_shape[1]+160), dtype=np.float32)
        #else:
        #    data_args['H'] = np.float32(sim_ic['H'])
        #    data_args['eta0'] = np.float32(sim_ic['eta0'])
        #    data_args['hu0'] = np.float32(sim_ic['hu0'])
        #    data_args['hv0'] = np.float32(sim_ic['hv0'])
            
        #FIXME: Optimize this to one transfer by packing arrays?
        #self.comm.Bcast(data_args['H'], root=0)
        #self.comm.Bcast(data_args['eta0'], root=0)
        #self.comm.Bcast(data_args['hu0'], root=0)
        #self.comm.Bcast(data_args['hv0'], root=0)
        
        #self.logger.debug("eta0 is %s", str(data_args['eta0']))
        
        
        
        #Broadcast arguments that we do not store in self
        ##############################
        ensemble_args = self.comm.bcast(ensemble_args, root=0)
        
        
        #Create ensemble on local node
        ##############################
        self.logger.info("Creating ensemble with %d members", self.local_ensemble_size)
        self.gpu_ctx = Common.CUDAContext()
        
        # DEBUG
        self.sim_args["comm"] = self.comm
        
        if (self.comm.rank == 0):
            num_ensemble_members = 1
        else:
            num_ensemble_members = self.local_ensemble_size
        
        self.ensemble = OceanModelEnsemble.OceanModelEnsemble(
                            self.gpu_ctx, self.sim_args, self.data_args, 
                            num_ensemble_members, 
                            drifter_positions=drifter_positions, 
                            **ensemble_args,
                            netcdf_filename=netcdf_filename, rank=self.comm.rank)
        
        
        
        
    def modelStep(self, dt):
        return self.ensemble.modelStep(dt, self.comm.rank)
        
        
        
        
    
    
    
    
    
    def getNormalizedWeights(self):
        #Compute the innovations
        local_innovations = self._localGetInnovations()

        #Compute the gaussian pdf from the innovations
        local_gaussian_pdf = self._localGetGaussianPDF(local_innovations)
    
        #Gather the gaussian weights from all nodes to a global vector on rank 0
        global_gaussian_pdf = None
        if (self.comm.rank == 0):
            global_gaussian_pdf = np.empty(((self.num_nodes+1), self.local_ensemble_size))
        self.comm.Gather(local_gaussian_pdf, global_gaussian_pdf, root=0)

        #Compute the normalized weights on rank 0
        global_normalized_weights = None
        if (self.comm.rank == 0):
            #Remove ourselves (rank=0)
            global_gaussian_pdf = global_gaussian_pdf[1:].ravel()

            #Normalize
            global_sum = np.sum(global_gaussian_pdf)
            global_normalized_weights = global_gaussian_pdf/global_sum
        return global_normalized_weights
    
    
    
    
    

    def resampleParticles(self, global_normalized_weights):
        #Get the vector of "new" particles representing which particles should be copied etc
        resampling_indices = self._globalGetResamplingIndices(global_normalized_weights)

        #Compute which particles should be overwritten by copies of other particles
        resampling_pairs = self._globalGetResamplingPairs(resampling_indices)

        mpi_requests = []
        send_data = []
        receive_data = []
        num_resample = resampling_pairs.shape[0]
        for i in range(num_resample):
            #Get the source-destination pairs
            dst = resampling_pairs[i,0]
            dst_node = resampling_pairs[i,1]
            src = resampling_pairs[i,2]
            src_node = resampling_pairs[i,3]
            
            #Compute the local index of the src/dst particle
            local_src = src - (self.comm.rank-1)*self.local_ensemble_size
            local_dst = dst - (self.comm.rank-1)*self.local_ensemble_size

            if (self.comm.rank == src_node and src_node == dst_node):
                self.logger.debug("Node {:d} copying internally {:d} to {:d}".format(src_node, src, dst))
                eta0, hu0, hv0 = self.ensemble.particles[local_src].download()
                eta1, hu1, hv1 = self.ensemble.particles[local_src].downloadPrevTimestep()
                receive_data += [[local_dst, eta0, hu0, hv0, eta1, hu1, hv1]]

            elif (self.comm.rank == src_node):
                self.logger.debug("Node {:d} sending {:d} to node {:d}".format(src_node, src, dst_node))
                eta0, hu0, hv0 = self.ensemble.particles[local_src].download()
                eta1, hu1, hv1 = self.ensemble.particles[local_src].downloadPrevTimestep()
                send_data += [[eta0, hu0, hv0, eta1, hu1, hv1]]
                for j in range(6):
                    mpi_requests += [self.comm.Isend(send_data[-1][j], dest=dst_node, tag=6*i+j)]

            elif (self.comm.rank == dst_node):
                self.logger.debug("Node {:d} receiving {:d} from node {:d}".format(dst_node, src, src_node))
                #FIXME: hard coded ghost cells here
                data = [local_dst]
                for j in range(6):
                    buffer = np.empty((self.ensemble.sim_args['ny']+160, self.ensemble.sim_args['nx']+160), dtype=np.float32)
                    mpi_requests += [self.comm.Irecv(buffer, source=src_node, tag=6*i+j)]
                    data += [buffer]
                receive_data += [data]

        # Wait for communication to complete
        for request in mpi_requests:
            request.wait()

        # Clear sent data
        send_data = None
        gc.collect() 

        # Upload new data to the GPU for the right particle
        for local_dst, eta0, hu0, hv0, eta1, hu1, hv1 in receive_data:
            self.logger.debug("Resetting " + str(local_dst))
            stream = self.ensemble.particles[local_dst].gpu_stream
            self.ensemble.particles[local_dst].gpu_data.h0.upload(stream, eta0)
            self.ensemble.particles[local_dst].gpu_data.hu0.upload(stream, hu0)
            self.ensemble.particles[local_dst].gpu_data.hv0.upload(stream, hv0)

            self.ensemble.particles[local_dst].gpu_data.h1.upload(stream, eta1)
            self.ensemble.particles[local_dst].gpu_data.hu1.upload(stream, hu1)
            self.ensemble.particles[local_dst].gpu_data.hv1.upload(stream, hv1)
        receive_data = None
        
    
        
    def _localGetInnovations(self): 
        #First, broadcast the drifter positions from "truth"
        if (self.comm.rank == 0):
            truth_drifter_positions = self.ensemble.getDrifterPositions(0)
        else:
            truth_drifter_positions = np.empty((self.num_drifters, 2), dtype=np.float32)
        self.comm.Bcast(truth_drifter_positions, root=0)

        #Then get the velocity at the drifter positions at each local ensemble member
        local_velocities = self.ensemble.getVelocity(truth_drifter_positions)

        #Broadcast the truth to all nodes
        if (self.comm.rank == 0):
            truth = local_velocities[0]
        else:
            truth = np.empty((self.num_drifters, 2))
        self.comm.Bcast(truth, root=0)

        #Compute the innovations for each particle
        local_innovations = truth - local_velocities
        
        return local_innovations
        
        
        
        
    
    def _localGetGaussianPDF(self, local_innovations):
        """
        Computes the Gaussian probability density function based on the innovations
        """
        obs_var = self.ensemble.observation_variance
        obs_cov = self.ensemble.observation_cov
        obs_cov_inv = self.ensemble.observation_cov_inverse
        
        global_num_particles = self.num_nodes * self.local_ensemble_size
        local_num_particles = local_innovations.shape[0] #should be equal self.local_ensemble_size for all except root/master

        local_gaussian_pdf = np.zeros(local_num_particles)
        if global_num_particles == 1:
            #local_gaussian_pdf = (1.0/np.sqrt(2.0*np.pi*obs_var))*np.exp(-(innovationsF**2/(2.0*obs_var)))
            local_gaussian_pdf = np.exp(-(local_innovations**2/(2*obs_var))) / np.sqrt(2*np.pi*obs_var)
        else:
            for i in range(local_num_particles):
                w = 0.0
                for d in range(self.num_drifters):
                    innovation = local_innovations[i,d,:]
                    w += np.dot(innovation, np.dot(obs_cov_inv, innovation.transpose()))

                ## TODO: Restructure to do the normalization before applying
                # the exponential function. The current version is sensitive to overflows.
                #local_gaussian_pdf[i] = np.exp(-0.5*w) / ((2*np.pi) * np.sqrt(np.linalg.det(obs_cov)))**self.num_drifters
                #weights[i] = (1.0/((2*np.pi)**Nd*np.linalg.det(R)**(Nd/2.0)))*np.exp(-0.5*w)
                local_gaussian_pdf[i] = (1.0/((2*np.pi)**self.num_drifters * np.linalg.det(obs_cov)**(self.num_drifters/2.0))) * np.exp(-0.5*w)
                
        return local_gaussian_pdf
    
    
    
    def _globalGetResamplingIndices(self, global_gaussian_weights):
        """
        Generate list of indices to resample, e.g., 
        [0 0 0 1 1 5 6 6 7]
        will resample 0 three times, 1 two times, 5 once, 6 two times, and 7 one time. 



        Residual resampling of particles based on the attached observation.
        Each particle is first resampled floor(N*w) times, which in total given M <= N particles. Afterwards, N-M particles are drawn from the discrete distribution given by weights N*w % 1.

        ensemble: The ensemble to be resampled, holding the ensemble particles, the observation, and measures to compute the weight of particles based on this information.
        reinitialization_variance: The variance used for resampling of particles that are already resampled. These duplicates are sampled around the original particle.
        If reinitialization_variance is zero, exact duplications are generated.

        Implementation based on the description in van Leeuwen (2009) 'Particle Filtering in Geophysical Systems', Section 3a.2)
        """

        resampling_indices = None
        if (self.comm.rank == 0):
            num_particles = self.num_nodes * self.local_ensemble_size
            # Create array of possible indices to resample:
            allIndices = np.arange(num_particles)

            # Deterministic resampling based on the integer part of N*weights:
            weightsTimesN = global_gaussian_weights*num_particles
            weightsTimesNInteger = np.int64(np.floor(weightsTimesN))
            deterministic = np.repeat(allIndices, weightsTimesNInteger)
            
            # Stochastic resampling based on the decimal parts of N*weights:
            decimalWeights = np.mod(weightsTimesN, 1)
            decimalWeights = decimalWeights/np.sum(decimalWeights)

            #FIXME: This makes the stochastic process deterministic for debugging
            #np.random.seed(seed=(42 + comm.rank))

            stochastic = np.random.choice(allIndices, num_particles - len(deterministic), p=decimalWeights)

            resampling_indices = np.sort(np.concatenate((deterministic, stochastic)))
            
        return resampling_indices
    
    
    
    
    
    def _globalGetResamplingPairs(self, global_resampling_indices):
        if self.comm.rank == 0:
            num_particles = self.num_nodes*self.local_ensemble_size
            particle_ids = np.arange(num_particles, dtype=np.int32)
            particle_id_to_node_id = 1 + np.floor(particle_ids // self.local_ensemble_size).astype(np.int32)
            #print("Weights=", global_normalized_weights)
            #print("Remapping indices=", indices)

            # Find the list of indices that need to be copied (src), 
            # and those to be deleted (dst)
            src = np.sort(global_resampling_indices)
            src = src[:-1][src[1:] == src[:-1]]
            dst = np.setxor1d(global_resampling_indices, particle_ids)

            # FIXME: Then try to minimize MPI communication
            
            #Compute the node ids that the ids belong to
            src_node = particle_id_to_node_id[src]
            dst_node = particle_id_to_node_id[dst]
            
            self.logger.debug("Overwriting %s on %s with %s from %s", str(dst), str(dst_node), str(src), str(src_node))

            #Gather them into a remap array
            num_resample = len(src)
            resampling_pairs = np.empty((num_resample, 4), dtype=np.int32)
            resampling_pairs[:,0] = dst
            resampling_pairs[:,1] = dst_node
            resampling_pairs[:,2] = src
            resampling_pairs[:,3] = src_node
        else:
            num_resample = None

        #Broadcast with all nodes
        num_resample = self.comm.bcast(num_resample, root=0)
        if self.comm.rank > 0:
            resampling_pairs = np.empty((num_resample, 4), dtype=np.int32)
        self.comm.Bcast(resampling_pairs, root=0)
        
        return resampling_pairs
    
    
    