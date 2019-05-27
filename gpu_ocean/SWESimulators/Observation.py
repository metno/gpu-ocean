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
                 domain_size_x=None, domain_size_y=None):
        """
        Class for facilitating drifter observations in files.
        The pandas DataFrame contains drifter positions only for 
        each observation time. 
        
        If the domain is considered to have periodic boundary conditions, the
        size of the domain should be provided to ensure correct estimated 
        velocities.
        """
        
        self.observation_type = observation_type
        self._check_observation_type()
        
        self.columns = ('time', 'drifter_positions')
        self.obs_df = pd.DataFrame(columns=self.columns)
        
        # Configuration parameters:
        self.drifterSet = None
        self.observationInterval = 1
        
        self.domain_size_x = domain_size_x
        self.domain_size_y = domain_size_y
        
        
    def get_num_observations(self):
        """
        Returns the number of rows (drifter observations) stored in the DataFrame.
        """
        return self.obs_df[self.columns[0]].count()
    
    def get_num_drifters(self):
        """
        Returns the number of drifters in the observation set.
        """
        if self.drifterSet is not None:
            return len(self.drifterSet)
        
        first_position = self.obs_df.iloc[0][self.columns[1]]
        return first_position.shape[0]
    
    def add_observation_from_sim(self, sim):
        """
        Adds the current drifter positions to the observation DataFrame.
        """
        
        # The timestamp is rounded to nearest integer, so that it is possible to compare to 
        # other simulation times.
        rounded_sim_t = round(sim.t)
        index = self.get_num_observations()
        
        if not index == 0:
            assert(self.obs_df[self.obs_df[self.columns[0]]==rounded_sim_t].time.count() == 0), \
                "Observation for time " + str(rounded_sim_t) + " already exists in DataFrame"

        pos = sim.drifters.getDrifterPositions()
        self.obs_df.loc[index] = {self.columns[0]: rounded_sim_t, self.columns[1]: pos}
        
        
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
        
    
        
        
    def _check_observation_type(self):
        """
        Checking that we are not trying to use unsupported observation types
        """
        assert(self.observation_type == dautils.ObservationType.UnderlyingFlow), \
            "UnderlyingFlow is the only supported ObservationType at the moment."
        
    def _check_df_at_given_time(self, rounded_t):
        # Sanity check the DataFrame
        assert(self.obs_df[self.obs_df[self.columns[0]]==rounded_t].time.count() > 0), \
                "Observation for time " + str(rounded_t) + " does not exists in DataFrame"
        assert(self.obs_df[self.obs_df[self.columns[0]]==rounded_t].time.count() < 2), \
                "Observation for time " + str(rounded_t) + " has multiple entries in DataFrame"
        
        
    def get_observation_times(self):
        """
        Returns an array with the timestamps for which there exists observations of
        underlying current.
        """
        if self.get_num_observations() < 2:
            return np.array([])
                
        return self.obs_df.time.values[::self.observationInterval][1:].copy()
    
    def get_drifter_position(self, t, applyDrifterSet=True):
        """
        Returns an array of drifter positions at time t.
        """
        # The timestamp is rounded to nearest integer, so that it is possible to compare to 
        # entries in the DataFrame.
        rounded_t = round(t)
        
        # Sanity check the DataFrame
        self._check_df_at_given_time(rounded_t)

        # Get index in data frame
        index = self.obs_df[self.obs_df[self.columns[0]]==rounded_t].index.values[0]
        
        current_pos = self.obs_df.iloc[index  ][self.columns[1]]
        
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

        index = self.obs_df[self.obs_df[self.columns[0]]==rounded_t].index.values[0]
        
        assert(index > self.observationInterval-1), "Observation can not be made this early in the DataFrame."
        
        prev_index = index - self.observationInterval
        dt = self.obs_df.iloc[index     ][self.columns[0]] - \
             self.obs_df.iloc[prev_index][self.columns[0]]

        current_pos = self.obs_df.iloc[index     ][self.columns[1]]
        prev_pos    = self.obs_df.iloc[prev_index][self.columns[1]]
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
                
        return observation
        
        
    def _detect_jump(self, pos0, pos1, jump_limit=100000):
        ds = np.sqrt((pos1[0] - pos0[0])**2 + \
                     (pos1[1] - pos0[1])**2)
        if ds > jump_limit:
            return True
        return False

    def get_drifter_path(self, drifter_id, start_t, end_t):
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
        all_drifter_positions_df = self.obs_df[self.columns[1]].values[::self.observationInterval][1:].copy()
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