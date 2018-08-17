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
from matplotlib import animation, rc
from SWESimulators import PlotHelper


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

        self.text_font_size = 12
        
    def get(self, attr):
        try:
            return self.ncfile.getncattr(attr)
        except:
            return "not found"
        
    def printVariables(self):
        for var in self.ncfile.variables:
            print(var)
        
    def printAttributes(self):
        for attr in self.ncfile.ncattrs():
            print(attr + "\t--> " + str(self.ncfile.getncattr(attr)))
    
    def getNumTimeSteps(self):
        time = self.ncfile.variables['time']
        #for t in time:
            #print t
        return time.size
    
    def getBC(self):
        return np.fromstring(self.get("boundary_conditions_mr")[1:-1], dtype=int, sep=',')
    
    def getBCSpongeCells(self):
        return np.fromstring(self.get("boundary_conditions_sponge_mr")[1:-1], dtype=int, sep=' ')

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
    
    def getH(self):
        H = self.ncfile.variables['H'][0, :, :]
        return H

    def getEtaAtTimeStep(self, index):
        time = self.ncfile.variables['time']
        eta = self.ncfile.variables['eta'][index, :, :]
        if self.ignore_ghostcells:
            eta = eta[self.ghostCells[2]:-self.ghostCells[0], \
                      self.ghostCells[3]:-self.ghostCells[1]]
        return eta, time[index]

    def getAxis(self):
        x = self.ncfile.variables['x']
        y = self.ncfile.variables['y']
        if self.ignore_ghostcells:
            x = x[self.ghostCells[2]:-self.ghostCells[0]]
            y = y[self.ghostCells[3]:-self.ghostCells[1]]
        return x, y
    
    def getEtaXSlice(self, t, y):
        y_index = int(y) + int(self.get('ghost_cells_south'))
        return self.ncfile.variables['eta'][t, y_index, self.ghostCells[3]:-self.ghostCells[1] ]
        
    def _getWaterHeight(self):
        if self.staggered_grid:
            return 0.0
        return 60.0
    
    def _animate(self, i):
        eta1, u1, v1, t = self.getTimeStep(i)
        self.plotter.plot(eta1-self._getWaterHeight(), u1, v1)
                

    
    def makeAnimation(self):
        nx = self.ncfile.getncattr('nx')
        ny = self.ncfile.getncattr('ny')
        dx = self.ncfile.getncattr('dx')
        dy = self.ncfile.getncattr('dy')
        #Calculate radius from center for plotting
        x_center = dx*nx*0.5
        y_center = dy*ny*0.5
        y_coords, x_coords = np.mgrid[0:ny*dy:dy, 0:nx*dx:dx]
        x_coords = np.subtract(x_coords, x_center)
        y_coords = np.subtract(y_coords, y_center)
        radius = np.sqrt(np.multiply(x_coords, x_coords) + np.multiply(y_coords, y_coords))

        eta0, hu0, hv0, t0 = self.getTimeStep(0)
        waterHeight = self._getWaterHeight()
        fig = plt.figure()
        self.plotter = PlotHelper.PlotHelper(fig, x_coords, y_coords, radius, \
                                             eta0-waterHeight, hu0, hv0)

        anim = animation.FuncAnimation(fig, self._animate, range(self.getNumTimeSteps()), interval=100)
        plt.close(anim._fig)
        return anim


    def _addText(self, ax, msg):
        bp = 70 # breakpoint
        if len(msg) > bp:
            rest = '     ' + msg[bp:]
            ax.text(0.1, self.textPos, msg[0:bp], fontsize=self.text_font_size)
            self.textPos -= 0.2
            self._addText(ax, rest)
        else:
            ax.text(0.1, self.textPos, msg, fontsize=self.text_font_size)
            #print len(msg)
            self.textPos -= 0.2

    def makeInfoPlot(self, ax, text_font_size=8):
        self.text_font_size = text_font_size
        self.textPos = 2.3
        # Ax is the subplot object
        ax.text(1, 2.8, 'NetCDF INFO', fontsize=self.text_font_size)
        
        #self._addText(ax, 'working directory: ' + self.current_directory)
        self._addText(ax, 'filename: ' + self.filename)
        self._addText(ax, '')
        self._addText(ax, 'git hash: ' + self.get('git_hash'))
        self._addText(ax, '')
        self._addText(ax, 'Simulator: ' + self.get('simulator_short'))
        self._addText(ax, 'BC: ' + self.get('boundary_conditions'))
        self._addText(ax, 'f:  ' + str(self.get('coriolis_force')) + ', beta: ' + str(self.get('coriolis_beta')))
        self._addText(ax, 'dt: ' + str(self.get('dt')) + ", auto_dt: " + self.get('auto_dt') + ", dx: " + str(self.get('dx')) + ", dy: " + str(self.get('dy')))
        self._addText(ax, 'wind type: ' + str(self.get('wind_stress_type')))
        
        ax.axis([0, 6, 0, 3])
