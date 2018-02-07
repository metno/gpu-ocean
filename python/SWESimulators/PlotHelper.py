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
import matplotlib.gridspec as gridspec
import numpy as np
import time

"""
Class that makes plotting faster by caching the plots instead of recreating them
"""
class PlotHelper:

    def __init__(self, fig, x_coords, y_coords, radius, eta1, u1, v1, eta2=None, u2=None, v2=None, interpolation_type='spline36'):
        self.ny, self.nx = eta1.shape
        self.fig = fig;
        
        fig.set_figheight(15)
        fig.set_figwidth(15)
        
        min_x = np.min(x_coords[:,0]);
        min_y = np.min(y_coords[0,:]);
        
        max_x = np.max(x_coords[0,:]);
        max_y = np.max(y_coords[:,0]);
        
        domain_extent = [ x_coords[0, 0], x_coords[0, -1], y_coords[0, 0], y_coords[-1, 0] ]
        
        if (eta2 is not None):
            assert(u2 is not None)
            assert(v2 is not None)
            self.gs = gridspec.GridSpec(3, 3)
        else:
            self.gs = gridspec.GridSpec(2, 3)
        
        ax = self.fig.add_subplot(self.gs[0, 0])
        self.sp_eta = plt.imshow(eta1, interpolation=interpolation_type, origin='bottom', vmin=-0.05, vmax=0.05, extent=domain_extent)
        plt.axis('tight')
        ax.set_aspect('equal')
        plt.title('Eta')
        plt.colorbar()
        self.particles = plt.scatter(x=None, y=None, color='blue')
        self.observations = plt.scatter(x=None, y=None, color='red')
        
        ax = self.fig.add_subplot(self.gs[0, 1])
        self.sp_u = plt.imshow(u1, interpolation=interpolation_type, origin='bottom', vmin=-1.5, vmax=1.5, extent=domain_extent)
        plt.axis('tight')
        ax.set_aspect('equal')
        plt.title('U')
        plt.colorbar()
        
        ax = self.fig.add_subplot(self.gs[0, 2])
        self.sp_v = plt.imshow(v1, interpolation=interpolation_type, origin='bottom', vmin=-1.5, vmax=1.5, extent=domain_extent)
        plt.axis('tight')
        ax.set_aspect('equal')
        plt.title('V')
        plt.colorbar()
            
        ax = self.fig.add_subplot(self.gs[1, 0])
        self.sp_radial1, = plt.plot(radius.ravel(), eta1.ravel(), '.')
        plt.axis([0, min(max_x, max_y), -1.5, 1])
        plt.title('Eta Radial plot')

        ax = self.fig.add_subplot(self.gs[1, 1])
        self.sp_x_axis1, = plt.plot(x_coords[self.ny/2,:], eta1[self.ny/2,:], 'k+--', label='x-axis')
        self.sp_y_axis1, = plt.plot(y_coords[:,self.nx/2], eta1[:,self.nx/2], 'kx:', label='y-axis')
        plt.axis([max(min_x, min_y), min(max_x, max_y), -1.5, 1])
        plt.title('Eta along axis')
        plt.legend()

        ax = self.fig.add_subplot(self.gs[1, 2])
        self.sp_x_diag1, = plt.plot(1.41*np.diagonal(x_coords, offset=-abs(self.nx-self.ny)/2), \
                                   np.diagonal(eta1, offset=-abs(self.nx-self.ny)/2), \
                                   'k+--', label='x = -y')
        self.sp_y_diag1, = plt.plot(1.41*np.diagonal(y_coords.T, offset=abs(self.nx-self.ny)/2), \
                                   np.diagonal(eta1.T, offset=abs(self.nx-self.ny)/2), \
                                   'kx:', label='x = y')
        plt.axis([max(min_x, min_y), min(max_x, max_y), -1.5, 1])
        plt.title('Eta along diagonal')
        plt.legend()
        
        
        if (eta2 is not None):
            ax = self.fig.add_subplot(self.gs[2, 0])
            self.sp_radial2, = plt.plot(radius.ravel(), eta2.ravel(), '.')
            plt.axis([0, min(max_x, max_y), -1.5, 1])
            plt.title('Eta2 Radial plot')

            ax = self.fig.add_subplot(self.gs[2, 1])
            self.sp_x_axis2, = plt.plot(x_coords[self.ny/2,:], eta2[self.ny/2,:], 'k+--', label='x-axis')
            self.sp_y_axis2, = plt.plot(y_coords[:,self.nx/2], eta2[:,self.nx/2], 'kx:', label='y-axis')
            plt.axis([max(min_x, min_y), min(max_x, max_y), -1.5, 1])
            plt.title('Eta2 along axis')
            plt.legend()

            ax = self.fig.add_subplot(self.gs[2, 2])
            self.sp_x_diag2, = plt.plot(1.41*np.diagonal(x_coords, offset=-abs(self.nx-self.ny)/2), \
                                       np.diagonal(eta2, offset=-abs(self.nx-self.ny)/2), \
                                       'k+--', label='x = -y')
            self.sp_y_diag2, = plt.plot(1.41*np.diagonal(y_coords.T, offset=abs(self.nx-self.ny)/2), \
                                       np.diagonal(eta2.T, offset=abs(self.nx-self.ny)/2), \
                                       'kx:', label='x = y')
            plt.axis([max(min_x, min_y), min(max_x, max_y), -1.5, 1])
            plt.title('Eta2 along diagonal')
            plt.legend()
        
        
        
        
        
    def plot(self, eta1, u1, v1, eta2=None, u2=None, v2=None):
        self.fig.add_subplot(self.gs[0, 0])
        self.sp_eta.set_data(eta1)

        self.fig.add_subplot(self.gs[0, 1])
        self.sp_u.set_data(u1)

        self.fig.add_subplot(self.gs[0, 2])
        self.sp_v.set_data(v1)
            
        self.fig.add_subplot(self.gs[1, 0])
        self.sp_radial1.set_ydata(eta1.ravel());

        self.fig.add_subplot(self.gs[1, 1])
        self.sp_x_axis1.set_ydata(eta1[(self.ny+2)/2,:])
        self.sp_y_axis1.set_ydata(eta1[:,(self.nx+2)/2])

        self.fig.add_subplot(self.gs[1, 2])
        self.sp_x_diag1.set_ydata(np.diagonal(eta1, offset=-abs(self.nx-self.ny)/2))
        self.sp_y_diag1.set_ydata(np.diagonal(eta1.T, offset=abs(self.nx-self.ny)/2))
        
        if (eta2 is not None):
            self.fig.add_subplot(self.gs[2, 0])
            self.sp_radial2.set_ydata(eta2.ravel());

            self.fig.add_subplot(self.gs[2, 1])
            self.sp_x_axis2.set_ydata(eta2[(self.ny+2)/2,:])
            self.sp_y_axis2.set_ydata(eta2[:,(self.nx+2)/2])

            self.fig.add_subplot(self.gs[2, 2])
            self.sp_x_diag2.set_ydata(np.diagonal(eta2, offset=-abs(self.nx-self.ny)/2))
            self.sp_y_diag2.set_ydata(np.diagonal(eta2.T, offset=abs(self.nx-self.ny)/2))
        
        plt.draw()
        time.sleep(0.001)
        
        
    def showParticles(self, particlePositions, observationPositions=None):
        
        self.particles.set_offsets(particlePositions)
        if observationPositions is not None:
            self.observations.set_offsets(observationPositions)
        
        plt.draw()
        
        
"""
For easily creating a plot of values on a 2D domain
"""        
class SinglePlot:
    
    def __init__(self, fig, x_coords, y_coords, data, interpolation_type='spline36', title='Data'):
        self.ny, self.nx = data.shape
        self.fig = fig;
        
        fig.set_figheight(5)
        fig.set_figwidth(5)
        
        min_x = np.min(x_coords[:,0]);
        min_y = np.min(y_coords[0,:]);
        
        max_x = np.max(x_coords[0,:]);
        max_y = np.max(y_coords[:,0]);
        
        domain_extent = [ x_coords[0, 0], x_coords[0, -1], y_coords[0, 0], y_coords[-1, 0] ]
        
        self.gs = gridspec.GridSpec(1,1)
        
        maxValue = np.max(data)
        minValue = np.min(data)
        
        ax = self.fig.add_subplot(self.gs[0, 0])
        self.sp_eta = plt.imshow(data, interpolation=interpolation_type, origin='bottom', vmin=minValue, vmax=maxValue, extent=domain_extent)
        plt.axis('tight')
        ax.set_aspect('equal')
        plt.title(title)
        plt.colorbar()
        