# -*- coding: utf-8 -*-

"""
This python class aids in plotting results from the numerical 
simulations

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


from matplotlib import pyplot as plt
import numpy as np
import time

"""
Class that makes plotting faster by caching the plots instead of recreating them
"""
class PlotHelper:

    def __init__(self, fig, x_coords, y_coords, radius, eta1, u1, v1, interpolation_type='spline36'):
        self.ny, self.nx = eta1.shape
        self.fig = fig;
        
        domain_extent = [ x_coords[0, 0], x_coords[0, -1], y_coords[0, 0], y_coords[-1, 0] ]
        
        ax = self.fig.add_subplot(2,3,1)
        self.sp_eta = plt.imshow(eta1, interpolation=interpolation_type, origin='bottom', vmin=-0.5, vmax=0.5, extent=domain_extent)
        plt.axis('tight')
        ax.set_aspect('equal')
        plt.title('Eta')
        plt.colorbar()
        
        ax = self.fig.add_subplot(2,3,2)
        self.sp_u = plt.imshow(u1, interpolation=interpolation_type, origin='bottom', vmin=-1.5, vmax=1.5, extent=domain_extent)
        plt.axis('tight')
        ax.set_aspect('equal')
        plt.title('U')
        plt.colorbar()
        
        ax = self.fig.add_subplot(2,3,3)
        self.sp_v = plt.imshow(v1, interpolation=interpolation_type, origin='bottom', vmin=-1.5, vmax=1.5, extent=domain_extent)
        plt.axis('tight')
        ax.set_aspect('equal')
        plt.title('V')
        plt.colorbar()
            
        self.fig.add_subplot(2,3,4)
        self.sp_radial, = plt.plot(radius.ravel(), eta1.ravel(), '.')
        plt.axis([0, 500, -1.5, 1])
        plt.title('Eta Radial plot')

        self.fig.add_subplot(2,3,5)
        self.sp_x_axis, = plt.plot(x_coords[self.ny/2,:], eta1[self.ny/2,:], 'k+--', label='x-axis')
        self.sp_y_axis, = plt.plot(y_coords[:,self.nx/2], eta1[:,self.nx/2], 'kx:', label='y-axis')
        plt.axis([-500, 500, -1.5, 1])
        plt.title('Eta along axis')
        plt.legend()

        self.fig.add_subplot(2,3,6)
        self.sp_x_diag, = plt.plot(1.41*np.diagonal(x_coords, offset=-abs(self.nx-self.ny)/2), \
                                   np.diagonal(eta1, offset=-abs(self.nx-self.ny)/2), \
                                   'k+--', label='x = -y')
        self.sp_y_diag, = plt.plot(1.41*np.diagonal(y_coords.T, offset=abs(self.nx-self.ny)/2), \
                                   np.diagonal(eta1.T, offset=abs(self.nx-self.ny)/2), \
                                   'kx:', label='x = y')
        plt.axis([-500, 500, -1.5, 1])
        plt.title('Eta along diagonal')
        plt.legend()
        
        
        
        
        
    def plot(self, eta1, u1, v1):
        self.fig.add_subplot(2,3,1)
        self.sp_eta.set_data(eta1)

        self.fig.add_subplot(2,3,2)
        self.sp_u.set_data(u1)

        self.fig.add_subplot(2,3,3)
        self.sp_v.set_data(v1)
            
        self.fig.add_subplot(2,3,4)
        self.sp_radial.set_ydata(eta1.ravel());

        self.fig.add_subplot(2,3,5)
        self.sp_x_axis.set_ydata(eta1[(self.ny+2)/2,:])
        self.sp_y_axis.set_ydata(eta1[:,(self.nx+2)/2])

        self.fig.add_subplot    (2,3,6)
        self.sp_x_diag.set_ydata(np.diagonal(eta1, offset=-abs(self.nx-self.ny)/2))
        self.sp_y_diag.set_ydata(np.diagonal(eta1.T, offset=abs(self.nx-self.ny)/2))
        
        plt.draw()
        time.sleep(0.1)
        
        
