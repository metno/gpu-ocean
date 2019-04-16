# -*- coding: utf-8 -*-

"""
 

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
    
    def __init__(self, observation_type=dautils.ObservationType.UnderlyingFlow):
        """
        Class for facilitating drifter observations in files.
        The pandas DataFrame contains drifter positions only for 
        each observation time. 
        """
        
        self.observation_type = observation_type
        self._check_observation_type()
        
        self.columns = ('time', 'drifter_positions')
        self.obs_df = pd.DataFrame(columns=self.columns)
        
        
    def get_num_observations(self):
        """
        Returns the number of rows (drifter observations) stored in the DataFrame.
        """
        return self.obs_df[self.columns[0]].count()
    
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
        
    def get_observation_times(self):
        """
        Returns an array with the timestamps for which there exists observations of
        underlying current.
        """
        if self.get_num_observations() < 2:
            return np.array([])
                
        return self.obs_df.time.values[1:]
    
    def get_drifter_position(self, t):
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
        
        return current_pos

        
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
        assert(self.obs_df[self.obs_df[self.columns[0]]==rounded_t].time.count() > 0), \
                "Observation for time " + str(rounded_t) + " does not exists in DataFrame"
        assert(self.obs_df[self.obs_df[self.columns[0]]==rounded_t].time.count() < 2), \
                "Observation for time " + str(rounded_t) + " has multiple entries in DataFrame"
        
        # Check that we are not trying to use unsupported observation types
        self._check_observation_type()

        index = self.obs_df[self.obs_df[self.columns[0]]==rounded_t].index.values[0]
        
        assert(index > 0), "Observation can not be made from the first entry in the DataFrame."
        
        dt = self.obs_df.iloc[index][self.columns[0]] - self.obs_df.iloc[index-1][self.columns[0]]
        print(dt)

        current_pos = self.obs_df.iloc[index  ][self.columns[1]]
        prev_pos    = self.obs_df.iloc[index-1][self.columns[1]]
        num_drifters = prev_pos.shape[0]
        
        hu_hv = (current_pos - prev_pos)*waterDepth/dt
        
        observation = np.zeros((num_drifters, 4))
        observation[:,:2] = current_pos
        observation[:,2:] = hu_hv
        
        return observation
        