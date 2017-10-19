# -*- coding: utf-8 -*-

"""
This python module implements saving shallow water simulations to a
netcdf file.

Copyright (C) 2016  SINTEF ICT

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
import subprocess
import matplotlib.pyplot as plt
import os as os

class SimNetCDFWriter:
    def __init__(self, sim, num_layers=1, staggered_grid=False, \
                 ignore_ghostcells=False, \
                 width=1, height=1):

        # OpenCL queue:
        self.cl_queue = sim.cl_queue

        # Write options:
        self.ignore_ghostcells = ignore_ghostcells
        self.num_layers = num_layers
        self.staggered_grid = staggered_grid

        # Identification of solution
        self.timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.timestamp_short = datetime.datetime.now().strftime("%Y_%m_%d")
        try:
            self.git_hash = str.strip(subprocess.check_output(['git', 'rev-parse', 'HEAD']))
        except:
            self.git_hash = "git info missing..."

        
        self.simulator_long = str(sim.__class__)
        self.simulator_short = (self.simulator_long.split("."))[-1]

        self.dir_name = "netcdf_" + self.timestamp_short + "/"
        self.output_file_name = self.dir_name + self.simulator_short + "_" + self.timestamp + ".nc"

        self.current_directory =os.getcwd()
        self.textPos = -1
        
        # Simulator info
        self.sim = sim
        self.boundary_conditions = str(sim.boundary_conditions)
        
        self.dt = sim.dt
        if self.staggered_grid:
            self.bathymetry = -sim.H.download(self.cl_queue)
        else:
            self.bathymetry = sim.bathymetry.download(self.cl_queue)[1] # Bm
        self.time_integrator = sim.rk_order
        self.minmod_theta = sim.theta
        self.coriolis_force = sim.f
        self.wind_stress = sim.wind_stress
        self.eddy_viscosity_coefficient = sim.A
        g = sim.g
        nx = sim.nx
        ny = sim.ny
        dx = sim.dx
        dy = sim.dy
        self.ghost_cells_x = sim.ghost_cells_x
        self.ghost_cells_y = sim.ghost_cells_y
        self.bottom_friction_r = sim.r

        # Organize directory and create file:
        if not os.path.isdir(self.dir_name):
            os.makedirs(self.dir_name)
        self.ncfile = Dataset(self.output_file_name,'w', clobber=True) 

        # Write global attributes
        self.ncfile.gitHash = self.git_hash
        self.ncfile.ignore_ghostcells = str(self.ignore_ghostcells)
        self.ncfile.num_layers = self.num_layers
        self.ncfile.staggered_grid = str(self.staggered_grid)
        self.ncfile.simulator_long = self.simulator_long
        self.ncfile.simulator_short = self.simulator_short
        self.ncfile.boundary_conditions = self.boundary_conditions
        self.ncfile.time_integrator = self.time_integrator
        self.ncfile.minmod_theta = self.minmod_theta
        self.ncfile.coriolis_force = self.coriolis_force
        self.ncfile.wind_stress_type = self.wind_stress.type
        self.ncfile.eddy_viscosity_coefficient = self.eddy_viscosity_coefficient
        self.ncfile.g = g  
        self.ncfile.nx = nx
        self.ncfile.ny = ny
        self.ncfile.dx = dx
        self.ncfile.dy = dy
        self.ncfile.ghost_cells_x = self.ghost_cells_x
        self.ncfile.ghost_cells_y = self.ghost_cells_y
        self.ncfile.bottom_friction_r = self.bottom_friction_r
        
        #Create dimensions 
        self.ncfile.createDimension('time', None) #Unlimited time dimension
        self.ncfile.createDimension('x', nx + 2*self.ghost_cells_x)
        self.ncfile.createDimension('y', ny + 2*self.ghost_cells_y)
        if (not self.ignore_ghostcells) and (self.staggered_grid):
            self.ncfile.createDimension('x_u',   nx + 2*self.ghost_cells_x + 1)
            self.ncfile.createDimension('y_u',   ny + 2*self.ghost_cells_y)
            self.ncfile.createDimension('x_v',   nx + 2*self.ghost_cells_x)
            self.ncfile.createDimension('y_v',   ny + 2*self.ghost_cells_y + 1)

        #Create axis
        self.nc_time = self.ncfile.createVariable('time', np.dtype('float32').char, 'time')
        x = self.ncfile.createVariable('x', np.dtype('float32').char, 'x')
        y = self.ncfile.createVariable('y', np.dtype('float32').char, 'y')

        x.standard_name = "projection_x_coordinate"
        y.standard_name = "projection_y_coordinate"
        x.axis = "X"
        y.axis = "Y"

        if (not self.ignore_ghostcells) and (self.staggered_grid):
            x_u = self.ncfile.createVariable('x_u', np.dtype('float32').char, 'x_u')
            y_u = self.ncfile.createVariable('y_u', np.dtype('float32').char, 'y_u')
            x_u.standard_name = "projection_x_coordinate"
            y_u.standard_name = "projection_y_coordinate"
            x_u.axis = "X"
            y_u.axis = "Y"
            x_v = self.ncfile.createVariable('x_v', np.dtype('float32').char, 'x_v')
            y_v = self.ncfile.createVariable('y_v', np.dtype('float32').char, 'y_v')
            x_v.standard_name = "projection_x_coordinate"
            y_v.standard_name = "projection_y_coordinate"
            x_v.axis = "X"
            y_v.axis = "Y"
            
        #Create bogus projection variable
        self.nc_proj = self.ncfile.createVariable('projection_stere', np.dtype('int32').char)
        self.nc_proj.grid_mapping_name = 'polar_stereographic'
        self.nc_proj.scale_factor_at_projection_origin = 0.9330127018922193
        self.nc_proj.straight_vertical_longitude_from_pole = 70.0
        self.nc_proj.latitude_of_projection_origin = 90.0
        self.nc_proj.earth_radius = 6371000.0
        self.nc_proj.proj4 = '+proj=stere +lat_0=90 +lon_0=70 +lat_ts=60 +units=m +a=6.371e+06 +e=0 +no_defs'

        x[:] = np.linspace(-self.ghost_cells_x*dx + dx/2.0, \
                           (nx + self.ghost_cells_x)*dx - dx/2.0, \
                           nx + 2*self.ghost_cells_x)
        y[:] = np.linspace(-self.ghost_cells_y*dy + dy/2.0, \
                           (ny + self.ghost_cells_y)*dy - dy/2.0, \
                           ny + 2*self.ghost_cells_y)
        if not self.ignore_ghostcells and self.staggered_grid:
            x_u[:] = np.linspace(-self.ghost_cells_x*dx, \
                                 (nx + 2*self.ghost_cells_x)*dx, \
                                 nx + 2*self.ghost_cells_x + 1)
            y_u[:] = np.linspace(-self.ghost_cells_y*dy + dy/2.0, \
                               (ny + self.ghost_cells_y)*dy + dy/2.0, \
                               ny + 2*self.ghost_cells_y)
            x_v[:] = np.linspace(-self.ghost_cells_x*dx + dx/2.0, \
                               (nx + self.ghost_cells_x)*dx + dx/2.0, \
                               nx + 2*self.ghost_cells_x)
            y_v[:] = np.linspace(-self.ghost_cells_y*dy, \
                                 (ny + 2*self.ghost_cells_y)*dy, \
                                 ny + 2*self.ghost_cells_y + 1)
            
        #Set units
        self.nc_time.units = 'seconds since 1970-01-01 00:00:00'
        x.units = 'meter'
        y.units = 'meter'
        if self.staggered_grid:
            x_u.units = 'meter'
            y_u.units = 'meter'
            x_v.units = 'meter'
            y_v.units = 'meter'

        #Create a land mask (with no land)
        self.nc_land = self.ncfile.createVariable('land_binary_mask', np.dtype('float32').char, ('y', 'x'))
        self.nc_land.standard_name = 'land_binary_mask'
        self.nc_land.units = '1'
        self.nc_land[:] = 0

        # Create info about bathymetry
        self.nc_bathymetry = self.ncfile.createVariable('bathymetry', np.dtype('float32').char, ('time', 'y', 'x'), zlib=True)
        self.nc_bathymetry.standard_name = 'bedrock_altitude'
        self.nc_bathymetry.grid_mapping = 'projection_stere'
        self.nc_bathymetry.coordinates = 'lon lat'
        self.nc_bathymetry.units = 'meter'
        self.nc_bathymetry[0, :] = self.bathymetry
        
        
        self.nc_eta = self.ncfile.createVariable('eta', np.dtype('float32').char, ('time', 'y', 'x'), zlib=True)
        if self.staggered_grid:
            self.nc_u = self.ncfile.createVariable('u', np.dtype('float32').char, ('time', 'y_u', 'x_u'), zlib=True)
            self.nc_v = self.ncfile.createVariable('v', np.dtype('float32').char, ('time', 'y_v', 'x_v'), zlib=True)
        else:
            self.nc_u = self.ncfile.createVariable('u', np.dtype('float32').char, ('time', 'y', 'x'), zlib=True)
            self.nc_v = self.ncfile.createVariable('v', np.dtype('float32').char, ('time', 'y', 'x'), zlib=True)
            
        self.nc_eta.standard_name = 'sea_water_elevation'
        self.nc_u.standard_name = 'x_sea_water_velocity'
        self.nc_v.standard_name = 'y_sea_water_velocity'
        self.nc_eta.grid_mapping = 'projection_stere'
        self.nc_u.grid_mapping = 'projection_stere'
        self.nc_v.grid_mapping = 'projection_stere'
        self.nc_eta.coordinates = 'lon lat'
        self.nc_u.coordinates = 'lon lat'
        self.nc_v.coordinates = 'lon lat'

        #Set units
        self.nc_eta.units = 'meter'
        self.nc_u.units = 'meter second-1'
        self.nc_v.units = 'meter second-1'
 
        # Init conditions should be added as the first element in the above arrays!
        self.i = 0
        self.writeTimestep(sim)


        
    def __str__(self):
        msg = ""
        theMap = vars(self)
        for i,j in theMap.items():
            if str(type(j)) == '<type \'str\'>':
                msg += i + ":\t\t" + j
            elif str(type(j)) == '<type \'numpy.ndarray\'>':
                msg += i + ":\t\tnumpy array of shape " + str(j.shape)
            elif str(type(j)) == '<type \'numpy.int32\'>' or \
                 str(type(j)) == '<type \'numpy.float32\'>' or \
                 str(type(j)) == '<type \'int\'>' or \
                 str(type(j)) == '<type \'bool\'>':
                msg += i + ":\t\t" + str(j)
            else:
                msg += i + ":\t\t" + str(type(j)) 
            msg += '\n'
        return msg    
        
        
    def __enter__(self):
        return self
        
        
        
        
        
    def __exit__(self, exc_type, exc_value, traceback):
        print "Closing file " + self.output_file_name +" ..." 
        self.ncfile.close()
        
        

    def writeTimestep(self, sim):
        eta, hu, hv = sim.download()
        if (not self.ignore_ghostcells):
            self.nc_time[self.i] = sim.t
            self.nc_eta[self.i, :] = eta
            self.nc_u[self.i, :] = hu
            self.nc_v[self.i, :] = hv
                       
        self.i += 1

            
    def write(self, t, eta, u, v, eta2=None, u2=None, v2=None):
        if (self.ignore_ghostcells):
            self.nc_time[self.i] = t
            #self.nc_eta[i, :] = eta[1:-1, 1:-1]
            #self.nc_u[i, :] = u[1:-1, 1:-1]
            #self.nc_v[i, :] = v[1:-1, 1:-1]
            self.nc_u[self.i, :] = u[1:-2, 1:-1]
            self.nc_v[self.i, :] = v[1:-1, 1:-2]
        else:
            self.nc_time[self.i] = t
            self.nc_eta[self.i, :] = eta
            self.nc_u[self.i, :] = u
            self.nc_v[self.i, :] = v
            
        if(self.num_layers == 2):
            if (self.ignore_ghostcells):
                self.nc_eta2[self.i, :] = eta2[1:-1, 1:-1]
                self.nc_u2[self.i, :] = u2[1:-1, 1:-1]
                self.nc_v2[self.i, :] = v2[1:-1, 1:-1]
            else:
                self.nc_eta2[self.i, :] = eta2
                self.nc_u2[self.i, :] = u2
                self.nc_v2[self.i, :] = v2

        self.i += 1


    def _addText(self, ax, msg):
        bp = 70 # breakpoint
        if len(msg) > bp:
            rest = '     ' + msg[bp:]
            ax.text(0.1, self.textPos, msg[0:bp])
            self.textPos -= 0.2
            self._addText(ax, rest)
        else:
            ax.text(0.1, self.textPos, msg)
            #print len(msg)
            self.textPos -= 0.2
        
    def infoPlot(self, ax):
        self.textPos = 2.3
        # Ax is the subplot object
        ax.text(1, 2.8, 'NetCDF INFO')
        
        self._addText(ax, 'working directory: ' + self.current_directory)
        self._addText(ax, 'filename: ' + self.output_file_name)
        self._addText(ax, '')
        self._addText(ax, 'git hash: ' + self.git_hash)
        self._addText(ax, '')
        self._addText(ax, 'Simulator: ' + self.simulator_short)
        self._addText(ax, 'BC: ' + str(self.boundary_conditions))
        self._addText(ax, 'f:  ' + str(self.coriolis_force))
        self._addText(ax, 'dt: ' + str(self.dt) + ", dx: " + str(self.sim.dx) + ", dy: " + str(self.sim.dy))
        self._addText(ax, 'wind type: ' + str(self.wind_stress.type))
        
        ax.axis([0, 6, 0, 3])
