# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2019 Norwegian Meteorological Institute
Copyright (C) 2019 SINTEF Digital

This python module implements saving shallow water simulations to a
netcdf file.

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
from SWESimulators import Common, WindStress, OceanographicUtilities


def getBoundaryConditionsData(source_url, timestep_indices, timesteps, x0, x1, y0, y1):
    """
    timestep_indices => index into netcdf-array, e.g. [1, 3, 5]
    timestep => time at timestep, e.g. [1800, 3600, 7200]
    """
    nt = len(timesteps)
    if (timestep_indices is None):
        timestep_indices = range(len(timesteps))

    bc_eta = {}
    bc_eta['north'] = np.empty((nt, x1-x0), dtype=np.float32)
    bc_eta['south'] = np.empty((nt, x1-x0), dtype=np.float32)
    bc_eta['east'] = np.empty((nt, y1-y0), dtype=np.float32)
    bc_eta['west'] = np.empty((nt, y1-y0), dtype=np.float32)

    bc_hu = {}
    bc_hu['north'] = np.empty((nt, x1-x0), dtype=np.float32)
    bc_hu['south'] = np.empty((nt, x1-x0), dtype=np.float32)
    bc_hu['east'] = np.empty((nt, y1-y0), dtype=np.float32)
    bc_hu['west'] = np.empty((nt, y1-y0), dtype=np.float32)

    bc_hv = {}
    bc_hv['north'] = np.empty((nt, x1-x0), dtype=np.float32)
    bc_hv['south'] = np.empty((nt, x1-x0), dtype=np.float32)
    bc_hv['east'] = np.empty((nt, y1-y0), dtype=np.float32)
    bc_hv['west'] = np.empty((nt, y1-y0), dtype=np.float32)
    
    try:
        ncfile = Dataset(source_url)

        H = ncfile.variables['h'][y0-1:y1+1, x0-1:x1+1]

        for i, timestep_index in enumerate(timestep_indices):
            zeta = ncfile.variables['zeta'][timestep_index, y0-1:y1+1, x0-1:x1+1]
            zeta = zeta.filled(0)
            bc_eta['north'][i] = zeta[-1, 1:-1]
            bc_eta['south'][i] = zeta[0, 1:-1]
            bc_eta['east'][i] = zeta[1:-1, -1]
            bc_eta['west'][i] = zeta[ 1:-1, 0]

            h = H + zeta

            hu = ncfile.variables['ubar'][timestep_index, y0-1:y1+1, x0-1:x1+1]
            hu = hu.filled(0) #zero on land
            hu = h*hu

            bc_hu['north'][i] = hu[-1, 1:-1]
            bc_hu['south'][i] = hu[0, 1:-1]
            bc_hu['east'][i] = hu[1:-1, -1]
            bc_hu['west'][i] = hu[1:-1, 0]

            hv = ncfile.variables['vbar'][timestep_index, y0-1:y1+1, x0-1:x1+1]
            hv = hv.filled(0) #zero on land
            hv = h*hv

            bc_hv['north'][i] = hv[-1, 1:-1]
            bc_hv['south'][i] = hv[0, 1:-1]
            bc_hv['east'][i] = hv[1:-1, -1]
            bc_hv['west'][i] = hv[1:-1, 0]

    except Exception as e:
        raise e
    finally:
        ncfile.close()

    bc_data = Common.BoundaryConditionsData(timesteps, 
        north=Common.SingleBoundaryConditionData(bc_eta['north'], bc_hu['north'], bc_hv['north']),
        south=Common.SingleBoundaryConditionData(bc_eta['south'], bc_hu['south'], bc_hv['south']),
        east=Common.SingleBoundaryConditionData(bc_eta['east'], bc_hu['east'], bc_hv['east']),
        west=Common.SingleBoundaryConditionData(bc_eta['west'], bc_hu['west'], bc_hv['west']))
    
    return bc_data


def getWindSourceterm(source_url, timestep_indices, timesteps, x0, x1, y0, y1):
    """
    timestep_indices => index into netcdf-array, e.g. [1, 3, 5]
    timestep => time at timestep, e.g. [1800, 3600, 7200]
    """
    
    if (timestep_indices is None):
        timestep_indices = range(len(timesteps))
        
    try:
        ncfile = Dataset(source_url)
        u_wind = ncfile.variables['Uwind'][timestep_indices, y0:y1, x0:x1]
        v_wind = ncfile.variables['Vwind'][timestep_indices, y0:y1, x0:x1]
    except Exception as e:
        raise e
    finally:
        ncfile.close()

    u_wind = u_wind.filled(0)
    v_wind = v_wind.filled(0)
    
    wind_speed = np.sqrt(np.power(u_wind, 2) + np.power(v_wind, 2))

    # C_drag as defined by Engedahl (1995)
    #(See "Documentation of simple ocean models for use in ensemble predictions. Part II: Benchmark cases"
    #at https://www.met.no/publikasjoner/met-report/met-report-2012 for details.) /
    def computeDrag(wind_speed):
        C_drag = np.where(wind_speed < 11, 0.0012, 0.00049 + 0.000065*wind_speed)
        return C_drag
    C_drag = computeDrag(wind_speed)

    rho_a = 1.225 # Density of air
    rho_w = 1025 # Density of water

    #Wind stress is then 
    # tau_s = rho_a * C_drag * |W|W
    wind_stress = C_drag * wind_speed * rho_a / rho_w
    wind_stress_u = wind_stress*u_wind
    wind_stress_v = wind_stress*v_wind
    
    wind_source = WindStress.WindStress(t=timesteps, X=wind_stress_u, Y=wind_stress_v)
    
    return wind_source


def getInitialConditions(source_url, x0, x1, y0, y1, timestep_indices=None, land_value=5.0, iterations=10, sponge_cells=[80, 80, 80, 80]):
    ic = {}
    
    try:
        ncfile = Dataset(source_url)
        H_m = ncfile.variables['h'][y0-1:y1+1, x0-1:x1+1]
        eta0 = ncfile.variables['zeta'][0, y0-1:y1+1, x0-1:x1+1]
        u0 = ncfile.variables['ubar'][0, y0:y1, x0:x1]
        v0 = ncfile.variables['vbar'][0, y0:y1, x0:x1]
        angle = ncfile.variables['angle'][y0:y1, x0:x1]
        latitude = ncfile.variables['lat'][y0, x0] #Latitude of first cell - used in beta plane
        x = ncfile.variables['X'][x0:x1]
        y = ncfile.variables['Y'][y0:y1]
        
        if (timestep_indices is not None):
            timesteps = ncfile.variables['time'][timestep_indices[:]]
        else:
            timesteps = ncfile.variables['time'][:]
            timestep_indices = range(len(timesteps))
        
    except Exception as e:
        raise e
    finally:
        ncfile.close()

    #Generate timesteps in reference to t0
    t0 = min(timesteps)
    assert(np.all(np.diff(timesteps)>=0))
    timesteps = timesteps - t0
    
    #Generate intersections bathymetry
    H_m = np.ma.array(H_m, mask=eta0.mask.copy())
    H_i, _ = OceanographicUtilities.midpointsToIntersections(H_m, land_value=land_value, iterations=iterations)
    eta0 = eta0[1:-1, 1:-1]
    h0 = OceanographicUtilities.intersectionsToMidpoints(H_i) + eta0
    
    #Generate physical variables
    eta0 = eta0.filled(0)
    hu0 = h0*u0.filled(0)
    hv0 = h0*v0.filled(0)
    
    #Initial reference time and all timesteps
    ic['t0'] = t0
    ic['timesteps'] = timesteps
    
    #Number of cells
    ic['sponge_cells'] = sponge_cells
    ic['NX'] = x1 - x0
    ic['NY'] = y1 - y0
    ic['nx'] = ic['NX'] - sponge_cells[1] - sponge_cells[3]
    ic['ny'] = ic['NY'] - sponge_cells[0] - sponge_cells[2]
    
    #Dx and dy
    #FIXME: Assumes equal for all.. .should check
    ic['dx'] = np.average(x[1:] - x[:-1])
    ic['dy'] = np.average(y[1:] - y[:-1])
    
    #Gravity and friction
    #FIXME: Friction coeff from netcdf?
    ic['g'] = 9.81
    ic['r'] = 3.0e-3
    
    #Physical variables
    ic['H'] = H_i
    ic['eta0'] = eta0
    ic['hu0'] = hu0
    ic['hv0'] = hv0
    
    #Coriolis angle and beta
    ic['angle'] = angle
    ic['f'], ic['coriolis_beta'] = OceanographicUtilities.calcCoriolisParams(OceanographicUtilities.degToRad(latitude))
    
    #Boundary conditions
    ic['boundary_conditions_data'] = getBoundaryConditionsData(source_url, timestep_indices, timesteps, x0, x1, y0, y1)
    ic['boundary_conditions'] = Common.BoundaryConditions(north=3, south=3, east=3, west=3, spongeCells=sponge_cells)
    
    #Wind stress
    ic['wind_stress'] = getWindSourceterm(source_url, timestep_indices, timesteps, x0, x1, y0, y1)
    
    #Note
    ic['note'] = datetime.datetime.now().isoformat() + ": Generated from " + source_url
    
    return ic


def rescaleInitialConditions(old_ic, scale):
    ic = old_ic.copy()
    
    ic['NX'] = int(old_ic['NX']*scale)
    ic['NY'] = int(old_ic['NY']*scale)
    ic['nx'] = ic['NX'] - old_ic['sponge_cells'][1] - old_ic['sponge_cells'][3]
    ic['ny'] = ic['NY'] - old_ic['sponge_cells'][0] - old_ic['sponge_cells'][2]
    ic['dx'] = old_ic['dx']/scale
    ic['dy'] = old_ic['dy']/scale
    _, _, ic['H'] = OceanographicUtilities.rescaleIntersections(old_ic['H'], ic['NX']+1, ic['NY']+1)
    _, _, ic['eta0'] = OceanographicUtilities.rescaleMidpoints(old_ic['eta0'], ic['NX'], ic['NY'])
    _, _, ic['hu0'] = OceanographicUtilities.rescaleMidpoints(old_ic['hu0'], ic['NX'], ic['NY'])
    _, _, ic['hv0'] = OceanographicUtilities.rescaleMidpoints(old_ic['hv0'], ic['NX'], ic['NY'])
    if (old_ic['angle'].shape == old_ic['eta0']):
        ic['angle'] = OceanographicUtilities.rescaleMidpoints(old_ic['angle'], ic['NX'], ic['NY'])
    #Not touched:
    #"boundary_conditions": 
    #"boundary_conditions_data": 
    #"wind_stress": 
    ic['note'] = old_ic['note'] + "\n" + datetime.datetime.now().isoformat() + ": Rescaled by factor " + str(scale)

    return ic


def removeMetadata(old_ic):
    ic = old_ic.copy()
    
    ic.pop('note', None)
    ic.pop('NX', None)
    ic.pop('NY', None)
    ic.pop('sponge_cells', None)
    ic.pop('t0', None)
    ic.pop('timesteps', None)
    
    return ic