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
from datetime import date
from netCDF4 import Dataset

class CTCSNetCDFWriter:
    def __init__(self, outfilename, nx, ny, dx, dy, H=None, num_layers=1, ignore_ghostcells=True, \
                 width=1, height=1):
        self.ncfile = Dataset(outfilename,'w', clobber=True) 
        self.ignore_ghostcells = ignore_ghostcells
        self.num_layers = num_layers

        self.ncfile.Conventions = "CF-1.4"
        
        #Create dimensions 
        self.ncfile.createDimension('time', None) #Unlimited time dimension
        if (self.ignore_ghostcells):
            self.ncfile.createDimension('x_eta', nx)
            self.ncfile.createDimension('y_eta', ny)
            #self.ncfile.createDimension('x_u', nx-1)
            #self.ncfile.createDimension('y_u', ny)
            #self.ncfile.createDimension('x_v', nx)
            #self.ncfile.createDimension('y_v', ny-1)
            self.ncfile.createDimension('x', nx-1)
            self.ncfile.createDimension('y', ny-1)
        else:
            self.ncfile.createDimension('x_eta', nx+2)
            self.ncfile.createDimension('y_eta', ny+2)
            self.ncfile.createDimension('x_u', nx+1)
            self.ncfile.createDimension('y_u', ny+2)
            self.ncfile.createDimension('x_v', nx+2)
            self.ncfile.createDimension('y_v', ny+1)

        #Create axis
        self.nc_time = self.ncfile.createVariable('time', np.dtype('float32').char, 'time')
        #x_eta = self.ncfile.createVariable('x_eta', np.dtype('float32').char, 'x_eta')
        #y_eta = self.ncfile.createVariable('y_eta', np.dtype('float32').char, 'y_eta')
        #x_u = self.ncfile.createVariable('x_u', np.dtype('float32').char, 'x_u')
        #y_u = self.ncfile.createVariable('y_u', np.dtype('float32').char, 'y_u')
        #x_v = self.ncfile.createVariable('x_v', np.dtype('float32').char, 'x_v')
        #y_v = self.ncfile.createVariable('y_v', np.dtype('float32').char, 'y_v')
        x = self.ncfile.createVariable('x', np.dtype('float32').char, 'x')
        y = self.ncfile.createVariable('y', np.dtype('float32').char, 'y')

        self.nc_time.long_name = "time since initialization"
        self.nc_time.calendar = "gregorian"
        self.nc_time.field = "time, scalar, series"
        self.nc_time.axis = "T"
        self.nc_time.standard_name = "time"
        
        x.standard_name = "projection_x_coordinate"
        y.standard_name = "projection_y_coordinate"
        
        x.axis = "X"
        y.axis = "Y"
                
        #Create bogus projection variable
        self.nc_proj = self.ncfile.createVariable('projection_stere', np.dtype('int32').char)
        self.nc_proj.grid_mapping_name = 'polar_stereographic'
        self.nc_proj.scale_factor_at_projection_origin = 0.9330127018922193
        self.nc_proj.straight_vertical_longitude_from_pole = 70.0
        self.nc_proj.latitude_of_projection_origin = 90.0
        self.nc_proj.earth_radius = 6371000.0
        self.nc_proj.proj4 = '+proj=stere +lat_0=90 +lon_0=70 +lat_ts=60 +units=m +a=6.371e+06 +e=0 +no_defs'
        
        #Set axis values/ticks
        if (self.ignore_ghostcells):
            #x_eta[:] = np.linspace(dx/2.0, nx*dx - dx/2.0, nx)
            #y_eta[:] = np.linspace(dy/2.0, ny*dy - dy/2.0, ny)
            #x_u[:] = np.linspace(1, (nx-1)*dx, nx-1)
            #y_u[:] = np.linspace(dy/2.0, ny*dy - dy/2.0, ny)
            #x_v[:] = np.linspace(dx/2.0, nx*dx - dx/2.0, nx)
            #y_v[:] = np.linspace(1, (ny-1)*dy, ny-1)
            x[:] = np.linspace(0, width, nx-1)
            y[:] = np.linspace(0, height, ny-1)
        else:
            x_eta[:] = np.linspace(-dx/2.0, nx*dx + dx/2.0, nx+2)
            y_eta[:] = np.linspace(-dy/2.0, ny*dy + dy/2.0, ny+2)
            x_u[:] = np.linspace(0, nx*dx, nx+1)
            y_u[:] = np.linspace(-dy/2.0, ny*dy + dy/2.0, ny+2)
            x_v[:] = np.linspace(-dx/2.0, nx*dx + dx/2.0, nx+2)
            y_v[:] = np.linspace(0, ny*dy, ny+1)
            
        #Set units
        self.nc_time.units = 'seconds since 1970-01-01 00:00:00'
        #x_eta.units = 'm'
        #y_eta.units = 'm'
        #x_u.units = 'm'
        #y_u.units = 'm'
        #x_v.units = 'm'
        #y_v.units = 'm'
        x.units = 'm'
        y.units = 'm'

        # Reference time does not make sense at this time, but at least we are following
        # the CF conventions.
        #self.nc_time.reference_time = date.today().isoformat()

        #Create output data variables
        #self.nc_eta = self.ncfile.createVariable('eta', np.dtype('float32').char, ('time', 'y_eta', 'x_eta'))
        #self.nc_u = self.ncfile.createVariable('u', np.dtype('float32').char, ('time', 'y_u', 'x_u'))
        #self.nc_v = self.ncfile.createVariable('v', np.dtype('float32').char, ('time', 'y_v', 'x_v'))
        
        #Create a land mask (with no land)
        self.nc_land = self.ncfile.createVariable('land_binary_mask', np.dtype('float32').char, ('y', 'x'))
        self.nc_land.standard_name = 'land_binary_mask'
        self.nc_land.units = '1'
        self.nc_land[:] = 0

        # Write initial conditions to file
        if H is not None:
            self.nc_H = self.ncfile.createVariable('H', np.dtype('float32').char, ('time', 'y_eta', 'x_eta'), zlib=True)
            self.nc_H.standard_name = 'write water_surface_reference_datum_altitude'
            self.nc_H.grid_mapping = 'projection_stere'
            self.nc_H.coordinates = 'lon lat'
            self.nc_H.units = 'meter'
            self.nc_H[0, :] = H[1:-1, 1:-1] # writing initial condition to timestep 0
        
        self.nc_eta = self.ncfile.createVariable('eta', np.dtype('float32').char, ('time', 'y', 'x'), zlib=True)
        self.nc_u = self.ncfile.createVariable('u', np.dtype('float32').char, ('time', 'y', 'x'), zlib=True)
        self.nc_v = self.ncfile.createVariable('v', np.dtype('float32').char, ('time', 'y', 'x'), zlib=True)
        self.nc_eta.standard_name = 'water_surface_height_above_reference_datum'
        self.nc_u.standard_name = 'x_sea_water_velocity'
        self.nc_v.standard_name = 'y_sea_water_velocity'
        self.nc_eta.grid_mapping = 'projection_stere'
        self.nc_u.grid_mapping = 'projection_stere'
        self.nc_v.grid_mapping = 'projection_stere'
        self.nc_eta.coordinates = 'lon lat'
        self.nc_u.coordinates = 'lon lat'
        self.nc_v.coordinates = 'lon lat'
        
        if(num_layers == 2):
            self.nc_eta2 = self.ncfile.createVariable('eta2', np.dtype('float32').char, ('time', 'y_eta', 'x_eta'))
            self.nc_u2 = self.ncfile.createVariable('u2', np.dtype('float32').char, ('time', 'y_u', 'x_u'))
            self.nc_v2 = self.ncfile.createVariable('v2', np.dtype('float32').char, ('time', 'y_v', 'x_v'))
        
        #Set units
        #self.nc_eta.units = 'm'
        #self.nc_u.units = 'm'
        #self.nc_v.units = 'm'
	self.nc_eta.units = 'meter'
        self.nc_u.units = 'meter second-1'
        self.nc_v.units = 'meter second-1'

        
        
        
        
    def __enter__(self):
        return self
        
        
        
        
        
    def __exit__(self, exc_type, exc_value, traceback):
        print "Closing '" + self.ncfile.filepath() + "'"
        self.ncfile.close()
        
        
        
        
        
    def write(self, i, t, eta, u, v, eta2=None, u2=None, v2=None):
        if (self.ignore_ghostcells):
            self.nc_time[i] = t
            #self.nc_eta[i, :] = eta[1:-1, 1:-1]
            #self.nc_u[i, :] = u[1:-1, 1:-1]
            #self.nc_v[i, :] = v[1:-1, 1:-1]
            self.nc_u[i, :] = u[1:-2, 1:-1]
            self.nc_v[i, :] = v[1:-1, 1:-2]
        else:
            self.nc_time[i] = t
            self.nc_eta[i, :] = eta
            self.nc_u[i, :] = u
            self.nc_v[i, :] = v
            
        if(self.num_layers == 2):
            if (self.ignore_ghostcells):
                self.nc_eta2[i, :] = eta2[1:-1, 1:-1]
                self.nc_u2[i, :] = u2[1:-1, 1:-1]
                self.nc_v2[i, :] = v2[1:-1, 1:-1]
            else:
                self.nc_eta2[i, :] = eta2
                self.nc_u2[i, :] = u2
                self.nc_v2[i, :] = v2
