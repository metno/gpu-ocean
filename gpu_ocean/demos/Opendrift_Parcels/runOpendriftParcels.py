#Import packages we need
import numpy as np
from netCDF4 import Dataset
import datetime
from IPython.display import display
import matplotlib
from matplotlib import pyplot as plt
import importlib
from datetime import timedelta
import pyproj

#For GPUOcean
#from SWESimulators import CDKLM16, Common, IPythonMagic, NetCDFInitialization
from SWESimulators import GPUDrifterCollection, Observation
#from SWESimulators import DataAssimilationUtils as dautils
#from demos.realisticSimulations import norkyst_plotting #Trengs denne?
#from SWESimulators import PlotHelper
from demos.Opendrift_Parcels.GPUOcean_runDeterministic import *

#For Opendrift
from opendrift.readers import reader_netCDF_CF_generic
from opendrift.models.oceandrift import OceanDrift

#For OceanParcels
from parcels import FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4, plotTrajectoriesFile, Field


"""
This python module implements functions facilitating simulating and plotting
drift trajectories wth GPU Ocean, OceanParcels and Opendrift within the framework.   
"""

#Funtions for transforming objects

def opendriftObj2gpuoceanObs(opendrift_obj, x0, x1,y0,y1, land_mask = None):
    """
    Takes in opendrift object and GPU Ocean domain coordinates. Returns GPU Ocean Observation object. 
    """        
    t = opendrift_obj.get_time_array()[1]
    for i in range(len(t)):
        t[i] = t[i].total_seconds()
    t = np.array(t)
    
    lon, lat = opendrift_obj.get_lonlats()
    x,y = opendrift_obj.lonlat2xy(lon,lat)
    
    #Get X, Y
    reader = next(iter(opendrift_obj.readers.items()))[1].Dataset
    X, Y = reader.X, reader.Y
    
    res = int(X[1]-X[0]) #Finding grid-resolution (assumed same in both horizontal directions)
    
    x = x - X[x0 +2] # in m
    y = y- Y[y0 +2] # in m

    #TODO: Remove drifters ouside the domain

    nx = (x1-x0-4) #num_cells
    ny = (y1-y0-4) #num_cells
    
    obs = Observation.Observation(domain_size_x = nx*res,
                                    domain_size_y = ny*res,
                                    nx=x1-x0-4, ny=y1-y0-4, land_mask = land_mask)

    obs.add_observations_from_arrays(t,x,y) #x,y i m fra origin i gpuocean ?
    
    return obs

def parcelsFile2gpuoceanObs(filename, source_url, x0, x1, y0,y1, X= None, Y= None, proj = None, res = 800, land_mask = None):
    """
    Takes in filename of parcels file, netCDF-file path and GPU Ocean domain coordinates. Returns GPU Ocean Observation object. 
    If grid resolution of netCDF-file is not 800, this needs to be specified.
    """   
    pfile = Dataset(filename)
    t = pfile.variables['time'][0,:]
    lon = np.ma.filled(pfile.variables['lon'], np.nan)
    lat = np.ma.filled(pfile.variables['lat'], np.nan)
    pfile.close()

    x, y = lonlat2xygpuocean(source_url, lon, lat, x0, y0, X = X, Y = Y, proj = proj)
    
    #TODO: Remove drifters ouside the domain

    nx = (x1-x0-4) #num_cells
    ny = (y1-y0-4) #num_cells
    
    obs = Observation.Observation(domain_size_x = nx*res,
                                    domain_size_y = ny*res,
                                    nx=x1-x0-4, ny=y1-y0-4, land_mask = land_mask)
    
    obs.add_observations_from_arrays(t,x,y)
    
    return obs


#Functions for running simulations

def simulate_parcels(source_url, output_filename, lat, lon, 
                     wind_drift_factor= 0.0, 
                     velocity_average= True, duration=24):
    """
    source_url: local file or list of local files with fielddata in NetCDF-format
    output_filename: name of file in which to save calculated trajectory
    lat, lon: initial coordinates of single drifter or lists with coordinates for multiple drifters
    wind_drift_factor: fraction of wind-speed at which objects will be advected. Default is 0 (no direct wind-drift)
    velocity_average: Boolean variable deciding whether averaged horisontal velocities or surface velocities will be used. 
                      Default is average which is consistent with GPU Ocean
    duration: duration of the simulation in hours. Default is 24 hours.
    TODO: Add functionality to start drifters at a later time in the simulation like in GPU Ocean.
    """
    filenames = {'U' : source_url, 'V': source_url}
    dimensions = {'lat': 'lat','lon': 'lon','time': 'time'}

    if velocity_average: 
        variables = {'U': 'ubar', 'V': 'vbar'}
    else:
        variables = {'U': 'u', 'V': 'v'}

    fieldset = FieldSet.from_netcdf(filenames, variables, dimensions, interp_method = 'cgrid_velocity')
    
    if wind_drift_factor:
        Uwind = Field.from_netcdf(source_url, ('U', 'Uwind'), dimensions, field_chunksize='auto', interp_method = 'cgrid_velocity')
        Vwind = Field.from_netcdf(source_url, ('V', 'Vwind'), dimensions, field_chunksize='auto', interp_method = 'cgrid_velocity')
        Uwind.set_scaling_factor(wind_drift_factor)
        Vwind.set_scaling_factor(wind_drift_factor)
        fieldset = FieldSet(U = fieldset.U+ Uwind,V = fieldset.V+ Vwind)

    pset = ParticleSet.from_list(fieldset = fieldset, pclass = JITParticle, lon=lon, lat=lat)
    output_file = pset.ParticleFile(name = output_filename, outputdt = timedelta(minutes=15))

    pset.execute(AdvectionRK4, runtime = timedelta(hours = duration), dt = timedelta(minutes=5), output_file = output_file)

    output_file.export()    
    
    
def simulate_opendrift(source_url, lat, lon, 
                       wind_drift_factor = 0.0, 
                       velocity_average = True, duration=24):
    """
    source_url: url or local file or list of either with fielddata in NetCDF-format
    output_filename: name of file in which to save calculated trajectory
    lat, lon: initial coordinates of single drifter or lists with coordinates for multiple drifters.
    wind_drift_factor: fraction of wind-speed at which objects will be advected. Default is 0 (no direct wind-drift)
    velocity_average: Boolean variable deciding whether averaged horisontal velocities or surface velocities will be used. 
                      Default is average which is consistent with GPU Ocean
    duration: duration of the simulation in hours. Default is 24 hours.
    TODO: Add functionality to start drifters at a later time in the simulation like in GPU Ocean.
          Can quite easily make random distribution of starting position if comparing with GPU Monte Carlo. 
    """    
    reader_norkyst = reader_netCDF_CF_generic.Reader(source_url)
    o = OceanDrift(loglevel=20)
    
    if velocity_average:
        reader_norkyst.variable_mapping['x_sea_water_velocity'] = 'ubar'
        reader_norkyst.variable_mapping['y_sea_water_velocity'] = 'vbar'
    
    o.add_reader(reader_norkyst, variables=['x_sea_water_velocity', 'y_sea_water_velocity', 'x_wind', 'y_wind'])
        
    o.seed_elements(lon= lon, lat=lat, time=reader_norkyst.start_time, wind_drift_factor = wind_drift_factor)
    
    o.set_config('drift:scheme', 'runge-kutta4') #Set to runge-kutta4, which is the same as Parcels. Default is euler. 
    
    o.run(duration = timedelta(hours=duration), time_step = 300, time_step_output = 900)
    
    return o



