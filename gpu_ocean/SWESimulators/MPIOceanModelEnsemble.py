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
import gc, os, time

from SWESimulators import OceanModelEnsemble, Common, Observation
from SWESimulators import DataAssimilationUtils as dautils


class MPIOceanModelEnsemble:
    """
    Class which holds a set of OceanModelEnsembles on different nodes. 
    Rank 0 to n-1 holds per_node_ensemble_size ocean models, so that the total
    number of ensemble members is comm.size * per_node_ensemble_size
    Rank 0 also orchistrates the simulation
    
    All ocean models are initialized using the same initial conditions
    """
    
    def __init__(self, comm, 
                 observation_file, observation_type=dautils.ObservationType.UnderlyingFlow,
                 local_ensemble_size=None, 
                 sim_args={}, data_args={},
                 ensemble_args={}):
        """
        Initialize the ensemble. Only rank 0 should receive the optional arguments.
        The constructor handles initialization across nodes
        """
        self.logger = logging.getLogger(__name__ + "_rank=" + str(comm.rank))
        self.logger.debug("Initializing")
        
        self.t = 0
        
        assert(observation_file is not None)
        assert('observation_variance' in ensemble_args.keys())
        
        
        #Broadcast general information about ensemble
        ##########################
        self.comm = comm
        self.num_nodes = self.comm.size
        assert self.comm.size >= 1, "You appear to not be using enough MPI nodes (at least one required)"
        
        self.local_ensemble_size = local_ensemble_size
        self.local_ensemble_size = self.comm.bcast(self.local_ensemble_size, root=0)
        
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
        
        #Read observations from file
        self.observations = Observation.Observation(observation_type=observation_type,
                                                    domain_size_x=self.data_args["nx"]*self.data_args["dx"], 
                                                    domain_size_y=self.data_args["ny"]*self.data_args["dy"],
                                                    nx=self.data_args["nx"], ny=self.data_args["ny"],
                                                    observation_variance=ensemble_args["observation_variance"])
        self.observations.read_pickle(observation_file)
        
        self.num_drifters = self.observations.get_num_drifters()
        
        self.ensemble = OceanModelEnsemble.OceanModelEnsemble(
                            self.gpu_ctx, self.sim_args, self.data_args, 
                            self.local_ensemble_size,
                            **ensemble_args,
                            netcdf_filename=netcdf_filename, rank=self.comm.rank)
        
        
        
        
    def modelStep(self, sub_t):
        self.t = self.ensemble.modelStep(sub_t, self.comm.rank)
        return self.t
        
        
        
        
    
    
    
    
    
    def getNormalizedWeights(self):
        #Compute the innovations
        local_innovations = self._localGetInnovations()

        #Compute the gaussian pdf from the innovations
        local_gaussian_log_weights = self._localGetGaussianLogWeights(local_innovations)
        if(np.isnan(local_gaussian_log_weights).any()):
            print("local_gaussian_log_weights contains NaN")
    
        #Gather the gaussian weights from all nodes to a global vector on rank 0
        global_gaussian_log_weights = None
        if (self.comm.rank == 0):
            global_gaussian_log_weights = np.empty((self.num_nodes, self.local_ensemble_size))
        self.comm.Gather(local_gaussian_log_weights, global_gaussian_log_weights, root=0)

        #Compute the normalized weights on rank 0
        global_normalized_weights = None
        if (self.comm.rank == 0):
            #Remove ourselves (rank=0)
            global_gaussian_log_weights = global_gaussian_log_weights.ravel()

            #Normalize
            # see https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/'
            max_log_weight = global_gaussian_log_weights.max()
            global_normalized_weights = np.exp(global_gaussian_log_weights - max_log_weight)/np.exp(global_gaussian_log_weights-max_log_weight).sum()
        
        return global_normalized_weights
    
    
    
    
    

    def resampleParticles(self, global_normalized_weights):
        # FIXME: Do we need to exhange more data (than eta, hu, hv)
        
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
            local_src = src - self.comm.rank * self.local_ensemble_size
            local_dst = dst - self.comm.rank * self.local_ensemble_size

            if (self.comm.rank == src_node and src_node == dst_node):
                self.logger.info("Node {:d} copying internally {:d} to {:d}".format(src_node, src, dst))
                eta0, hu0, hv0 = self.ensemble.particles[local_src].download()
                eta1, hu1, hv1 = self.ensemble.particles[local_src].downloadPrevTimestep()
                receive_data += [[local_dst, eta0, hu0, hv0, eta1, hu1, hv1]] # FIXME: Don't send prev ts (for CDKLM)

            elif (self.comm.rank == src_node):
                self.logger.info("Node {:d} sending {:d} to node {:d}".format(src_node, src, dst_node))
                eta0, hu0, hv0 = self.ensemble.particles[local_src].download()
                eta1, hu1, hv1 = self.ensemble.particles[local_src].downloadPrevTimestep()
                send_data += [[eta0, hu0, hv0, eta1, hu1, hv1]]
                for j in range(6):
                    mpi_requests += [self.comm.Isend(send_data[-1][j], dest=dst_node, tag=6*i+j)]

            elif (self.comm.rank == dst_node):
                self.logger.info("Node {:d} receiving {:d} from node {:d}".format(dst_node, src, src_node))
                #FIXME: hard coded ghost cells here
                data = [local_dst]
                for j in range(6):
                    buffer = np.empty((self.ensemble.data_args['ny']+4, self.ensemble.data_args['nx']+4), dtype=np.float32)
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
            self.logger.info("Resetting " + str(local_dst))
            stream = self.ensemble.particles[local_dst].gpu_stream
            self.ensemble.particles[local_dst].gpu_data.h0.upload(stream, eta0)
            self.ensemble.particles[local_dst].gpu_data.hu0.upload(stream, hu0)
            self.ensemble.particles[local_dst].gpu_data.hv0.upload(stream, hv0)

            self.ensemble.particles[local_dst].gpu_data.h1.upload(stream, eta1)
            self.ensemble.particles[local_dst].gpu_data.hu1.upload(stream, hu1)
            self.ensemble.particles[local_dst].gpu_data.hv1.upload(stream, hv1)
        receive_data = None
        
    def observeTrueDrifters(self):
        return self.observations.get_observation(self.t, 12414) # FIXME: Use Hm as second arg
    
    def setBuoySet(self, buoy_set):
        self.observations.setBuoySet(buoy_set)
        self.num_drifters = self.observations.get_num_drifters()
        
    def _localGetInnovations(self): 
        #observations is a numpy array with D drifter positions and drifter velocities
        #[[x_1, y_1, hu_1, hv_1], ... , [x_D, y_D, hu_D, hv_D]]
        observations = self.observeTrueDrifters()
        
        #for particle in self.ensemble.particles:
        #    particle.writeState()
            
        truth_drifter_positions = observations[:,:2]
        observed_truth = observations[:,2:]
        
        #Then get the velocity at the drifter positions at each local ensemble member
        local_observed_particles = self.ensemble.observeParticles(truth_drifter_positions)

        #Compute the innovations for each particle
        local_innovations = observed_truth - local_observed_particles
        
        return local_innovations
        
    def _localGetGaussianLogWeights(self, local_innovations):
        """
        Computes the Gaussian probability density function based on the innovations
        """
        obs_var = self.ensemble.observation_variance
        obs_cov = self.ensemble.observation_cov
        obs_cov_inv = self.ensemble.observation_cov_inverse
        
        global_num_particles = self.num_nodes * self.local_ensemble_size
        local_num_particles = local_innovations.shape[0] #should be equal self.local_ensemble_size

        local_gaussian_log_weights = np.zeros(local_num_particles)
        for p in range(local_num_particles):
            w = 0.0
            for d in range(self.num_drifters):
                innovation = local_innovations[p,d,:]
                w += np.dot(innovation, np.dot(obs_cov_inv, innovation.transpose()))
            local_gaussian_log_weights[p] = -0.5*w
                
        return local_gaussian_log_weights
    
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

            stochastic = np.random.choice(allIndices, num_particles - len(deterministic), p=decimalWeights)

            resampling_indices = np.sort(np.concatenate((deterministic, stochastic)))
            
        return resampling_indices
    
    
    
    
    
    def _globalGetResamplingPairs(self, global_resampling_indices):
        if self.comm.rank == 0:
            num_particles = self.num_nodes*self.local_ensemble_size
            particle_ids = np.arange(num_particles, dtype=np.int32)
            particle_id_to_node_id = np.floor(particle_ids // self.local_ensemble_size).astype(np.int32)
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
    
    def dumpParticleSample(self, drifter_cells):
        self.ensemble.dumpParticleSample(drifter_cells)
        
    def dumpForecastParticleSample(self):
        self.ensemble.dumpForecastParticleSample()
        
    def getDrifterCells(self, t=None):
        if t is None:
            t = self.t
        drifter_positions = self.observations.get_drifter_position(t, applyDrifterSet=False)
        drifter_positions[:,0] = np.floor(drifter_positions[:,0]/self.data_args["dx"])
        drifter_positions[:,1] = np.floor(drifter_positions[:,1]/self.data_args["dy"])
        return drifter_positions.astype(np.int32)
    
    def dumpParticleInfosToFiles(self, prefix="particle_info"):
        """
        Default file name of dump will be particle_info_YYYY_mm_dd-HH_MM_SS_{rank}_{local_particle_id}.bz2
        """
        assert(self.ensemble.particleInfos[0] is not None), 'particleInfos[0] is None, and dumpParticleInfosToFile was called... This should not happend.'
        
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        timestamp_short = datetime.datetime.now().strftime("%Y_%m_%d")

        dir_name = prefix + "_" + timestamp_short
        
        if not os.path.isdir(dir_name) and self.comm.rank == 0:
            os.makedirs(dir_name)
        else:
            while True:
                if os.path.isdir(dir_name):
                    break
                time.sleep(1)
        
        filename_prefix = prefix + "_" + timestamp + "_" + str(self.comm.rank)
        
        self.ensemble.dumpParticleInfosToFiles(os.path.join(dir_name, filename_prefix))
        
    def dumpDrifterForecastToFiles(self, prefix="forecast_particle_info"):
        """
        Default file name of dump will be forecast_particle_info_YYYY_mm_dd-HH_MM_SS_{rank}_{local_particle_id}.bz2
        """
        assert(self.ensemble.drifterForecast[0] is not None), ' drifterForecast[0] is None, and dumpDrifterForecastToFiles was called... This should not happend.'
        
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        timestamp_short = datetime.datetime.now().strftime("%Y_%m_%d")

        dir_name = prefix + "_" + timestamp_short
        
        if not os.path.isdir(dir_name) and self.comm.rank == 0:
            os.makedirs(dir_name)
        else:
            while True:
                if os.path.isdir(dir_name):
                    break
                time.sleep(1)
        
        filename_prefix = prefix + "_" + timestamp + "_" + str(self.comm.rank)
        
        self.ensemble.dumpDrifterForecastToFiles(os.path.join(dir_name, filename_prefix))
        
    def initDriftersFromObservations(self):
        self.ensemble.attachDrifters(self.observations.get_drifter_position(self.t, applyDrifterSet=False, ignoreBuoys=True))
        