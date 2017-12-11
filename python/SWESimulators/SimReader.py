# -*- coding: utf-8 -*-

"""
This python module implements saving shallow water simulations to a
netcdf file.

Copyright (C) 2017  SINTEF ICT

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
import datetime
from netCDF4 import Dataset
import matplotlib.pyplot as plt


class SimNetCDFReader:

    def __init__(self, filename, ignore_ghostcells=True):
        
        self.filename = filename
        self.ignore_ghostcells = ignore_ghostcells
        
        self.ncfile = Dataset(filename, 'r')
        
        self.ghostCells = [self.ncfile.getncattr('ghost_cells_north'), \
                           self.ncfile.getncattr('ghost_cells_east'), \
                           self.ncfile.getncattr('ghost_cells_south'), \
                           self.ncfile.getncattr('ghost_cells_west')]
        self.staggered_grid = str(self.ncfile.getncattr('staggered_grid')) == 'True'
       
        
    def printVariables(self):
        for var in self.ncfile.variables:
            print var
        
    def printAttributes(self):
        for attr in self.ncfile.ncattrs():
            print attr, "\t--> ", self.ncfile.getncattr(attr)
    
    def getNumTimeSteps(self):
        time = self.ncfile.variables['time']
        for t in time:
            print t

        return time.size

    def getLastTimeStep(self):
        return self.getTimeStep(-1)
        
    def getTimeStep(self, index):
        time = self.ncfile.variables['time']
        eta  = self.ncfile.variables['eta'][index, :, :]
        hu = self.ncfile.variables['hu'][index, :, :]
        hv = self.ncfile.variables['hv'][index, :, :]
        if self.ignore_ghostcells:
            eta = eta[self.ghostCells[2]:-self.ghostCells[0], \
                      self.ghostCells[3]:-self.ghostCells[1]]
            hu = hu[self.ghostCells[2]:-self.ghostCells[0], \
                  self.ghostCells[3]:-self.ghostCells[1]]
            hv = hv[self.ghostCells[2]:-self.ghostCells[0], \
                  self.ghostCells[3]:-self.ghostCells[1]]
        return eta, hu, hv, time[index]


    
