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
from netCDF4 import Dataset


"""
Writes out data to a NetCDF file from the Centered in Time, Centered in Space
numerical scheme.
"""
class CTCSNetCDFWriter:
    def __init__(self, outfilename, nx, ny, dx, dy, ignore_ghostcells=True):
        self.ncfile = Dataset(outfilename,'w') 
        self.ignore_ghostcells = ignore_ghostcells
        
        #Create dimensions 
        self.ncfile.createDimension('time', None) #Unlimited time dimension
        if (self.ignore_ghostcells):
            self.ncfile.createDimension('x_eta', nx)
            self.ncfile.createDimension('y_eta', ny)
            self.ncfile.createDimension('x_u', nx-1)
            self.ncfile.createDimension('y_u', ny)
            self.ncfile.createDimension('x_v', nx)
            self.ncfile.createDimension('y_v', ny-1)
        else:
            self.ncfile.createDimension('x_eta', nx+2)
            self.ncfile.createDimension('y_eta', ny+2)
            self.ncfile.createDimension('x_u', nx+1)
            self.ncfile.createDimension('y_u', ny+2)
            self.ncfile.createDimension('x_v', nx+2)
            self.ncfile.createDimension('y_v', ny+1)

        #Create axis
        self.nc_time = self.ncfile.createVariable('time', np.dtype('float32').char, 'time')
        x_eta = self.ncfile.createVariable('x_eta', np.dtype('float32').char, 'x_eta')
        y_eta = self.ncfile.createVariable('y_eta', np.dtype('float32').char, 'y_eta')
        x_u = self.ncfile.createVariable('x_u', np.dtype('float32').char, 'x_u')
        y_u = self.ncfile.createVariable('y_u', np.dtype('float32').char, 'y_u')
        x_v = self.ncfile.createVariable('x_v', np.dtype('float32').char, 'x_v')
        y_v = self.ncfile.createVariable('y_v', np.dtype('float32').char, 'y_v')
        
        #Set axis values/ticks
        if (self.ignore_ghostcells):
            x_eta[:] = np.linspace(dx/2.0, nx*dx - dx/2.0, nx)
            y_eta[:] = np.linspace(dy/2.0, ny*dy - dy/2.0, ny)
            x_u[:] = np.linspace(1, (nx-1)*dx, nx-1)
            y_u[:] = np.linspace(dy/2.0, ny*dy - dy/2.0, ny)
            x_v[:] = np.linspace(dx/2.0, nx*dx - dx/2.0, nx)
            y_v[:] = np.linspace(1, (ny-1)*dy, ny-1)
        else:
            x_eta[:] = np.linspace(-dx/2.0, nx*dx + dx/2.0, nx+2)
            y_eta[:] = np.linspace(-dy/2.0, ny*dy + dy/2.0, ny+2)
            x_u[:] = np.linspace(0, nx*dx, nx+1)
            y_u[:] = np.linspace(-dy/2.0, ny*dy + dy/2.0, ny+2)
            x_v[:] = np.linspace(-dx/2.0, nx*dx + dx/2.0, nx+2)
            y_v[:] = np.linspace(0, ny*dy, ny+1)

        #Set units
        self.nc_time.units = 's'
        x_eta.units = 'm'
        y_eta.units = 'm'
        x_u.units = 'm'
        y_u.units = 'm'
        x_v.units = 'm'
        y_v.units = 'm'

        

        #Create output data variables
        self.nc_eta = self.ncfile.createVariable('eta', np.dtype('float32').char, ('time', 'y_eta', 'x_eta'))
        self.nc_u = self.ncfile.createVariable('u', np.dtype('float32').char, ('time', 'y_u', 'x_u'))
        self.nc_v = self.ncfile.createVariable('v', np.dtype('float32').char, ('time', 'y_v', 'x_v'))
        
        #Set units
        self.nc_eta.units = 'm'
        self.nc_u.units = 'm'
        self.nc_v.units = 'm'

        
        
        
        
    def __enter__(self):
        return self
        
        
        
        
        
    def __exit__(self, exc_type, exc_value, traceback):
        print "Closing '" + self.ncfile.filepath() + "'"
        self.ncfile.close()
        
        
        
        
        
    def write(self, i, t, eta, u, v):
        if (self.ignore_ghostcells):
            self.nc_time[i] = t
            self.nc_eta[i, :] = eta[1:-1, 1:-1]
            self.nc_u[i, :] = u[1:-1, 1:-1]
            self.nc_v[i, :] = v[1:-1, 1:-1]
        else:
            self.nc_time[i] = t
            self.nc_eta[i, :] = eta
            self.nc_u[i, :] = u
            self.nc_v[i, :] = v