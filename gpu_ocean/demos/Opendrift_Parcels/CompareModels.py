%matplotlib inline
%config IPCompleter.greedy=True

#Import packages we need
import numpy as np
from netCDF4 import Dataset
import datetime
from IPython.display import display
import matplotlib
from matplotlib import pyplot as plt
import importlib
from datetime import timedelta

#Path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../../')))

#For GPUOcean
from SWESimulators import CDKLM16, Common, IPythonMagic, NetCDFInitialization
from SWESimulators import GPUDrifterCollection, Observation
from SWESimulators import DataAssimilationUtils as dautils
from demos.realisticSimulations import norkyst_plotting

#For Opendrift
from opendrift.readers import reader_netCDF_CF_generic
from opendrift.models.oceandrift import OceanDrift

#For OceanParcels
from parcels import FieldSet, ParticleSet, Variable, JITParticle, AdvectionRK4, plotTrajectoriesFile, Field


"""
This python module implements functions facilitating simulating and plotting
drift trajectories wth GPU Ocean, OceanParcels and Opendrift within the framework.   
"""

def simulate_parcels(source_url, output_filename, lat, lon, 
                     wind_drift_percentage= 0.0, 
                     velocity_average= True, duration=24):
    """
    source_url: local file or list of local files with fielddata in NetCDF-format
    output_filename: name of file in which to save calculated trajectory
    lat, lon: initial coordinates of single drifter or lists with coordinates for multiple drifters
    wind_drift_percentage: fraction of wind-speed at which objects will be advected. Default is 0 (no direct wind-drift)
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
    
    if wind_drift_percentage:
        Uwind = Field.from_netcdf(source_url, ('U', 'Uwind'), dimensions, field_chunksize='auto', interp_method = 'cgrid_velocity')
        Vwind = Field.from_netcdf(source_url, ('V', 'Vwind'), dimensions, field_chunksize='auto', interp_method = 'cgrid_velocity')
        Uwind.set_scaling_factor(wind_drift_percentage)
        Vwind.set_scaling_factor(wind_drift_percentage)
        fieldset = FieldSet(U = fieldset.U+ Uwind,V = fieldset.V+ Vwind)

    pset = ParticleSet.from_list(fieldset = fieldset, pclass = JITParticle, lon=lon, lat=lat)
    output_file = pset.ParticleFile(name = output_filename, outputdt = timedelta(minutes=15))

    pset.execute(AdvectionRK4, runtime = timedelta(hours = duration), dt = timedelta(minutes=5), output_file = output_file)

    output_file.export()

    
    
    
    
def simulate_opendrift(source_url, lat, lon, 
                       wind_drift_percentage = 0.0, 
                       velocity_average = True, duration=24):
    """
    source_url: url or local file or list of either with fielddata in NetCDF-format
    output_filename: name of file in which to save calculated trajectory
    lat, lon: initial coordinates of single drifter or lists with coordinates for multiple drifters.
    wind_drift_percentage: fraction of wind-speed at which objects will be advected. Default is 0 (no direct wind-drift)
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
        
    o.seed_elements(lon= lon, lat=lat, time=reader_norkyst.start_time, wind_drift_factor = wind_drift_percentage)
    
    o.set_config('drift:scheme', 'runge-kutta4') #Set to runge-kutta4, which is the same as Parcels. Default is euler. 
    
    o.run(duration = timedelta(hours=duration), time_step = 300, time_step_output = 900)
    
    return o






def simulate_gpuocean_deterministic(source_url, domain, initx, inity, 
                                    sim_args, erode_land = 1,
                                    observation_type = dautils.ObservationType.UnderlyingFlow,
                                    outfolder = None,  rescale=0, 
                                    start_forecast_hours = 0, forecast_duration = 23 ):
    """
    source_url: url or local file or list of either with fielddata in NetCDF-format
    domain: array/list on form [x0,x1,y0,y1] defining the domain for the simulation
    initx, inity = initial coordinates of single drifter or lists with coordinates for multiple drifters. 
                   In local cartesian coordinates of simulation-domain. 
    sim_args, erode_land, observation_type: arguments needed for simulator and observation object. sim_args must be given.
    outfolder: outfolder
    start_forecast_hours = number hours after which to start simulating drifttrajectories(ocean model starts at beginning of field-data)
    forecast_duration = duration of simulation(including possibly only ocean simulation). DEfault is 24 hours.
    """
    end_forecast_hours = start_forecast_hours + forecast_duration
    
    #Create simulator
    data_args = NetCDFInitialization.getInitialConditions(source_url, domain[0], domain[1], domain[2],domain[3] , 
                     timestep_indices = None, erode_land = erode_land, download_data = False)
    
    if rescale:
        data_args = NetCDFInitialization.rescaleInitialConditions(data_args, scale=rescale)

    importlib.reload(CDKLM16)
    sim = CDKLM16.CDKLM16(**sim_args, **NetCDFInitialization.removeMetadata(data_args))
    
    #Forecast
    observation_args = {'observation_type': observation_type,
                    'nx': sim.nx, 'ny': sim.ny,
                    'domain_size_x': sim.nx*sim.dx,
                    'domain_size_y': sim.ny*sim.dy,
                    'land_mask': sim.getLandMask()
                   }

    trajectory_forecast = Observation.Observation(**observation_args)

    if outfolder is not None:
        out_folder = os.path.abspath(outfolder)
        os.makedirs(out_folder, exist_ok=True)
        trajectory_forecast_filename = 'trajectory_forecast_'+str(start_forecast_hours)+'_to_'+str(end_forecast_hours)+'.pickle'
        trajectory_forecast_path = os.path.join(out_folder, trajectory_forecast_filename)
    
    #Drifters
    #Assumes initx, inity same format/shape
    if type(initx) is not list:
        initx = [initx]
        inity = [inity]
    
    num_drifters = len(initx)
    
    drifters = GPUDrifterCollection.GPUDrifterCollection(gpu_ctx, num_drifters, 
                                                     boundaryConditions = sim.boundary_conditions,
                                                     domain_size_x = trajectory_forecast.domain_size_x,
                                                     domain_size_y = trajectory_forecast.domain_size_y,
                                                     gpu_stream = sim.gpu_stream)
    
    drifter_pos_init = np.array([initx, inity]).T
    
    #Run simulation
    num_total_hours = end_forecast_hours
    
    five_mins_in_an_hour = 12
    sub_dt = 5*60 # five minutes
    
    progress = Common.ProgressPrinter(5)
    pp = display(progress.getPrintString(0), display_id=True)

    for hour in range(num_total_hours):
        
        if hour == start_forecast_hours:
            # Attach drifters
            drifters.setDrifterPositions(drifter_pos_init)
            sim.attachDrifters(drifters)
            trajectory_forecast.add_observation_from_sim(sim)
        
        for mins in range(five_mins_in_an_hour):
            t = sim.step(sub_dt)
            if hour >= start_forecast_hours:
                trajectory_forecast.add_observation_from_sim(sim)
        
        pp.update(progress.getPrintString(hour/(end_forecast_hours-1)))
    
    if outfolder is not None:
        trajectory_forecast.to_pickle(trajectory_forecast_path)
    
    return trajectory_forecast