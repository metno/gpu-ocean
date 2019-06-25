# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2019 SINTEF Digital

This python module implements a class that read and writes drifter
observation to and from file.

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
import pandas as pd


from SWESimulators import CDKLM16
from SWESimulators import GPUDrifterCollection
from SWESimulators import DataAssimilationUtils as dautils




class Observation:
    """
    Class for creating and reading observations.
    """
    
    def __init__(self, observation_type=dautils.ObservationType.UnderlyingFlow,
                 domain_size_x=None, domain_size_y=None, nx=None, ny=None,
                 observation_variance=0.0):
        """
        Class for facilitating drifter observations in files.
        The pandas DataFrame contains drifter positions only for 
        each observation time. Values for hu and hv at provided static buoy
        positions can also be registered in the data frame.
        
        If the domain is considered to have periodic boundary conditions, the
        size of the domain should be provided to ensure correct estimated 
        velocities.
        
        All observations are based on velocities (u, v) and equilibrium ocean
        depth H. There are therefore no observations of eta.
        """
        
        self.observation_type = observation_type
        self._check_observation_type()
        
        self.time_key               = 'time'
        self.drifter_positions_key  = 'drifter_positions'
        self.drifter_obs_errors_key = 'drifter_obs_errors'
        self.buoy_observations_key  = 'buoy_observations'
        self.buoy_positions_key     = 'buoy_positions'
        self.buoy_obs_errors_key    = 'buoy_obs_errors'
        self.df_keys = [self.time_key, self.drifter_positions_key, self.drifter_obs_errors_key, 
                        self.buoy_observations_key, self.buoy_positions_key, self.buoy_obs_errors_key]
        
        #self.columns = ('time', 'drifter_positions', 'buoy_observations', 'buoy_positions', 'buoy_obs_errors', 'drifter_obs_errors')
        self.obs_df = pd.DataFrame(columns=self.df_keys) #columns)
        
        # For each time the data frame entry will look like this:
        # {'time' : t,
        #  'drifter_positions': [[x_1, y_1], [x_2, y_2], ..., [x_D, y_D]]
        #  'drifter_obs_error': [[eps_1,1, eps_1,2], [eps_2,1, eps_2,2], ..., [eps_D,1 eps_D,2]]
        #  'buoy_observations': [[hu_1, hv_1], [hu_2, hv_2], ..., [hu_B, hv_B]]
        #  'buoy_positions'   : [[x_1, y_1], [x_2, y_2], ..., [x_B, y_B]]
        #  'buoy_obs_error'   : [[eps_1,1, eps_1,2], [eps_2,1, eps_2,2], ..., [eps_B,1 eps_B,2]]
        # }
        # D = num drifters, B = num buoys
        #
        # buoy_positions is only provided for the first entry in the data frame, as it will be the same for all entries
        # The same values are also found in the variables self.buoy_positions and self.buoy_indices  
        
        self.register_buoys = False
        self.buoy_indices = None
        self.buoy_positions = None
        self.read_buoy = None
        
        # Configuration parameters:
        self.drifterSet = None
        self.observationInterval = 1
        self.obs_var = observation_variance
        self.obs_stddev = np.sqrt(observation_variance)
        
        if observation_type == dautils.ObservationType.StaticBuoys:
            assert(nx is not None), 'nx must be provided if observation_type is StaticBuoys'
            assert(ny is not None), 'ny must be provided if observation_type is StaticBuoys'
            assert(domain_size_x is not None), 'domain_size_x must be provided if observation_type is StaticBuoys'
            assert(domain_size_y is not None), 'domain_size_y must be provided if observation_type is StaticBuoys'
        
        self.domain_size_x = domain_size_x
        self.domain_size_y = domain_size_y
        self.nx = nx
        self.ny = ny
        
        
    def get_num_observations(self):
        """
        Returns the number of rows (drifter observations) stored in the DataFrame.
        """
        return self.obs_df[self.time_key].count()
    
    def get_num_drifters(self, applyDrifterSet=True, ignoreBuoys=False):
        """
        Returns the number of drifters in the observation set.
        """
         
        # Count buoys
        if (self.observation_type == dautils.ObservationType.StaticBuoys) and not ignoreBuoys:
            return np.sum(self.read_buoy)
           
        # Count drifters
        if (self.drifterSet is not None) and applyDrifterSet:
            return len(self.drifterSet)
        
        first_position = self.obs_df.iloc[0][self.drifter_positions_key]
        return first_position.shape[0]
    
    def add_observation_from_sim(self, sim):
        """
        Adds the current drifter positions to the observation DataFrame, or H*u and H*v 
        at the buoy positions.
        """
        
        # The timestamp is rounded to nearest integer, so that it is possible to compare to 
        # other simulation times.
        rounded_sim_t = round(sim.t)
        index = self.get_num_observations()
        
        buoy_positions = None
        buoy_observations = None
        buoy_obs_errors = None
        
        if not index == 0:
            assert(self.obs_df[self.obs_df[self.time_key]==rounded_sim_t].time.count() == 0), \
                "Observation for time " + str(rounded_sim_t) + " already exists in DataFrame"
        
        if self.register_buoys:
            if index == 0:
                buoy_positions = self.buoy_positions
            buoy_observations = np.zeros_like(self.buoy_positions)
            eta, hu, hv = sim.download(interior_domain_only=True)
            H = sim.downloadBathymetry()[1][2:-2, 2:-2] # H in cell centers
            
            for i in range(len(buoy_observations)):
                # buoy index
                x = self.buoy_indices[i, 0]
                y = self.buoy_indices[i, 1]
                
                # Observe current, and we know the depth, but not eta
                eta_ignorance_factor = H[y, x] / (H[y, x] + eta[y, x])
                buoy_observations[i, 0] = hu[y, x] * eta_ignorance_factor
                buoy_observations[i, 1] = hv[y, x] * eta_ignorance_factor
            
            buoy_obs_errors = np.random.normal(size=buoy_observations.shape)
            
        
        pos = sim.drifters.getDrifterPositions()
        drifter_obs_errors = np.random.normal(size=pos.shape)
        self.obs_df.loc[index] = {self.time_key: rounded_sim_t, self.drifter_positions_key: pos,
                                  self.buoy_observations_key: buoy_observations, self.buoy_positions_key: buoy_positions,
                                  self.buoy_obs_errors_key: buoy_obs_errors, self.drifter_obs_errors_key: drifter_obs_errors}
        
        
    #########################
    ### CONFIGURATIONS
    ########################
    def setDrifterSet(self, drifterSet):
        """
        Specify a subset of drifters that should be considered.
        The argument drifterSet should be a list of indices between 0 and the number of drifters - 1.
        
        The drifterSet is only considered while reading drifter positions that are already stored, not for
        adding new ones from simulators.
        """
        assert(type(drifterSet) is list), 'drifterSet is required to be a list, but is ' + str(type(drifterSet))
        assert(min(drifterSet) >= 0), 'drifterSet contains at least one negative drifter id'
        assert(max(drifterSet) < self.get_num_drifters()), 'drifterSet contains indices that are out-of-range'
        
        self.drifterSet = drifterSet
        
    def setObservationInterval(self, interval):
        self.observationInterval = interval
    
    def setBuoyCells(self, buoy_indices):
        self.buoy_indices = buoy_indices
        
        # Compute the absolute positions for the buoys in the middle of their cells
        self.buoy_positions = buoy_indices.copy().astype(np.float32)
        dx = self.domain_size_x/self.nx
        dy = self.domain_size_y/self.ny
        self.buoy_positions[:,0] = (self.buoy_positions[:, 0] + 0.5)*dx
        self.buoy_positions[:,1] = (self.buoy_positions[:, 1] + 0.5)*dy
        
        self.read_buoy = [True]*self.buoy_indices.shape[0]
        
        self.register_buoys = True
        
    def setBuoyCellsByFrequency(self, frequency_x, frequency_y):
        """
        Defines placements of buoys in the domain based on a given cell-frequency.
        
        E.g, if frequency_x = frequency_y = 25, and the domain is of size (500 x 300),
        (12 x 20) = 240 buoys are defined equally spaced throughout the domain.
        
        This cover slightly more than 0.1% of the state space.
        """
        static_y, static_x = int(self.ny/frequency_y), int(self.nx/frequency_x)
        num_buoys = static_y*static_x 

        buoy_indices = np.zeros((num_buoys, 2), dtype=np.int32)

        buoy_cell_id = 0
        for y in range(static_y):
            cell_y = y*frequency_y
            for x in range(static_x):
                cell_x = x*frequency_x

                buoy_indices[buoy_cell_id, :] = [cell_x, cell_y]

                buoy_cell_id += 1
                
        self.setBuoyCells(buoy_indices)
    
    def setBuoyReadingArea(self, area='all'):
        if self.observation_type == dautils.ObservationType.StaticBuoys:
            if area == "south":
                for i in range(len(self.buoy_indices)):
                    self.read_buoy[i] = self.buoy_indices[i,1] < self.ny/2
            elif area == "west":
                for i in range(len(self.buoy_indices)):
                    self.read_buoy[i] = self.buoy_indices[i,0] < self.nx/2
            elif area == 'all':
                self.read_buoy = [True]*self.buoy_indices.shape[0]
            else:
                assert(area == 'all'), 'Invalid area. Must be all, south or west'
            
            
    ############################
    ### FILE INTERFACE
    ############################        
    def to_pickle(self, path):
        """
        Write the observation DataFrame to file (pickle)
        """
        self.obs_df.to_pickle(path)
        
    def read_pickle(self, path):
        """
        Read observations from file
        """
        self.obs_df = pd.read_pickle(path)
        
        if self.observation_type == dautils.ObservationType.StaticBuoys:
            self.buoy_positions = self.obs_df.iloc[0][self.buoy_positions_key].copy()
            
            # Compute the cell indices for the buoys in the middle of their cells
            self.buoy_indices = self.buoy_positions.copy()
            dx = self.domain_size_x/self.nx
            dy = self.domain_size_y/self.ny
            self.buoy_indices[:,0] = np.floor(self.buoy_indices[:, 0]/dx)
            self.buoy_indices[:,1] = np.floor(self.buoy_indices[:, 1]/dy)
            self.buoy_indices = self.buoy_indices.astype(np.int32)
        
            self.read_buoy = [True]*self.buoy_indices.shape[0]
        
    def _check_observation_type(self):
        """
        Checking that we are not trying to use unsupported observation types
        """
        assert(self.observation_type == dautils.ObservationType.UnderlyingFlow) or \
              (self.observation_type == dautils.ObservationType.StaticBuoys), \
              "Only UnderlyingFlow and StaticBuoys ObservationType are supported at the moment."
        
    def _check_df_at_given_time(self, rounded_t):
        # Sanity check the DataFrame
        assert(self.obs_df[self.obs_df[self.time_key]==rounded_t].time.count() > 0), \
                "Observation for time " + str(rounded_t) + " does not exists in DataFrame"
        assert(self.obs_df[self.obs_df[self.time_key]==rounded_t].time.count() < 2), \
                "Observation for time " + str(rounded_t) + " has multiple entries in DataFrame"
        
        
    def get_observation_times(self):
        """
        Returns an array with the timestamps for which there exists observations of
        underlying current.
        """
        if self.get_num_observations() < 2:
            return np.array([])
                
        return self.obs_df.time.values[::self.observationInterval][1:].copy()
    
    def get_drifter_position(self, t, applyDrifterSet=True, ignoreBuoys=False):
        """
        Returns an array of drifter positions at time t, as
        [[x_1, y_1], ..., [x_D, y_D]]
        """
        
        # The timestamp is rounded to nearest integer, so that it is possible to compare to 
        # entries in the DataFrame.
        rounded_t = round(t)
        
        # Sanity check the DataFrame
        self._check_df_at_given_time(rounded_t)
        
        # Get index in data frame
        index = self.obs_df[self.obs_df[self.time_key]==rounded_t].index.values[0]
        
        if self.observation_type == dautils.ObservationType.StaticBuoys and not ignoreBuoys:
            return self.buoy_positions.copy()[self.read_buoy, :]
        
        current_pos = self.obs_df.iloc[index  ][self.drifter_positions_key]
        
        # Need to return a copy of the data frame data, elsewise we risk modifying the data frame!
        if applyDrifterSet and self.drifterSet is not None:
            return current_pos[self.drifterSet, :].copy()

        return current_pos.copy()
    
    

        
    def get_observation(self, t, waterDepth):
        """
        Makes an observation of the underlying current for the provided time.
        Transforms the drifter positions to an observation relative to the previous
        drifter observation.

        Returns a numpy array with D drifter positions and drifter velocities
        [[x_1, y_1, hu_1, hv_1], ... , [x_D, y_D, hu_D, hv_D]]
        """
        
        # The timestamp is rounded to nearest integer, so that it is possible to compare to 
        # entries in the DataFrame.
        rounded_t = round(t)
        
        # Sanity check the DataFrame
        self._check_df_at_given_time(rounded_t)
        
        # Check that we are not trying to use unsupported observation types
        self._check_observation_type()

        index = self.obs_df[self.obs_df[self.time_key]==rounded_t].index.values[0]
        
        assert(index > self.observationInterval-1), "Observation can not be made this early in the DataFrame."
        
        # If Buoys
        if self.observation_type == dautils.ObservationType.StaticBuoys:
            num_buoys = self.get_num_drifters()
            
            observation = np.zeros((num_buoys, 4))
            
            observation[:, :2] = self.buoy_positions.copy()[self.read_buoy, :]
            observation[:, 2:] = self.obs_df.iloc[index][self.buoy_observations_key][self.read_buoy, :]
            
            # Add observation error:
            if self.buoy_obs_errors_key in self.obs_df.columns:
                obs_error = self.obs_df.iloc[index][self.buoy_obs_errors_key]
                if obs_error is not None:
                    observation[:, 2:] += obs_error[self.read_buoy, :] * self.obs_stddev
            
            return observation
        
        # Else drifters:
        prev_index = index - self.observationInterval
        dt = self.obs_df.iloc[index     ][self.time_key] - \
             self.obs_df.iloc[prev_index][self.time_key]

        current_pos = self.obs_df.iloc[index     ][self.drifter_positions_key]
        prev_pos    = self.obs_df.iloc[prev_index][self.drifter_positions_key]
        if self.drifterSet is not None:
            current_pos = current_pos[self.drifterSet, :]
            prev_pos = prev_pos[self.drifterSet, :]
        
        num_drifters = prev_pos.shape[0]
        hu_hv = (current_pos - prev_pos)*waterDepth/dt        
        
        
        observation = np.zeros((num_drifters, 4))
        observation[:,:2] = current_pos
        observation[:,2:] = hu_hv 
        
        # Correct velocities for drifters that travel through the domain boundary
        if self.domain_size_x or self.domain_size_y:
            for d in range(observation.shape[0]):
                
                if self.domain_size_x:
                    velocity_x_p = (current_pos[d,0] - prev_pos[d,0] + self.domain_size_x)*waterDepth/dt
                    velocity_x_m = (current_pos[d,0] - prev_pos[d,0] - self.domain_size_x)*waterDepth/dt
                    if abs(velocity_x_p) < abs(observation[d,2]):
                        observation[d,2] = velocity_x_p
                    if abs(velocity_x_m) < abs(observation[d,2]):
                        observation[d,2] = velocity_x_m
                
                if self.domain_size_y:
                    velocity_y_p = (current_pos[d,1] - prev_pos[d,1] + self.domain_size_y)*waterDepth/dt
                    velocity_y_m = (current_pos[d,1] - prev_pos[d,1] - self.domain_size_y)*waterDepth/dt
                    if abs(velocity_y_p) < abs(observation[d,3]):
                        observation[d,3] = velocity_y_p
                    if abs(velocity_y_m) < abs(observation[d,3]):
                        observation[d,3] = velocity_y_m
        
        # Add observation error
        if self.drifter_obs_errors_key in self.obs_df.columns:
            obs_error = self.obs_df.iloc[index][self.drifter_obs_errors_key]
            
            if obs_error is not None:
                if self.drifterSet is not None:
                    obs_error = obs_error[self.drifterSet, :]
                observation[:,2:] += obs_error * self.obs_stddev
        
        return observation
        
        
    def _detect_jump(self, pos0, pos1, jump_limit=100000):
        ds = np.sqrt((pos1[0] - pos0[0])**2 + \
                     (pos1[1] - pos0[1])**2)
        if ds > jump_limit:
            return True
        return False

    def get_drifter_path(self, drifter_id, start_t, end_t): #, flexiframe=False):
        """
        Creates a list of paths for the given drifter in the given time interval,
        so that the drift trajectory can be plotted.
        We create a list of paths rather than a single path, as the path is disconnected 
        when the drifter passes through the boundary.
        """
        paths = []
        observation_times = self.get_observation_times()
        
        start_obs_index = 0
        end_obs_index = len(observation_times)
        try:
            start_obs_index = np.where(observation_times == start_t)[0][0]
            end_obs_index   = np.where(observation_times == end_t  )[0][0]+1
        except IndexError:
            pass
        
        total_num_observations = end_obs_index - start_obs_index
        
        # Filter the given drifter based from the data frame only once for efficiency
        all_drifter_positions_df = self.obs_df[self.drifter_positions_key].values[::self.observationInterval][1:].copy()
        all_drifter_positions = np.stack(all_drifter_positions_df, axis=0)[:, drifter_id,:]
        
        path = np.zeros((total_num_observations, 2))
        path_index = 0
        for i in range(start_obs_index, end_obs_index):
            obs_t = observation_times[i]
            if obs_t < start_t or obs_t > end_t:
                continue
            current_pos = all_drifter_positions[i, :]
            if path_index > 0:
                if self._detect_jump(path[path_index-1,:], current_pos):
                    paths.append(path[:path_index,:])
                    
                    path_index = 0
                    path = np.zeros((total_num_observations, 2))
            path[path_index,:] = current_pos
            path_index += 1
        paths.append(path[:path_index, :])
        return paths