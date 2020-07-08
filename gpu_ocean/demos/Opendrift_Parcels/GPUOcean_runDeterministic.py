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
from SWESimulators import CDKLM16, Common, IPythonMagic, NetCDFInitialization
from SWESimulators import GPUDrifterCollection, Observation
from SWESimulators import DataAssimilationUtils as dautils
from demos.realisticSimulations import norkyst_plotting #Trengs denne?
from SWESimulators import PlotHelper


"""
This python module implements functions facilitating simulating and plotting
drift trajectories wth GPU Ocean as well as tranforming lon/lat to x/y coordinates. 
"""

#Funtions for transforming coordinates

def getXYproj(source_url):
    """
    Given netCDF-file, returns X, Y arrays and projection.
    """
    try:
        ncfile = Dataset(source_url)
        var = ncfile.variables['projection_stere']
        proj4 = var.__getattr__('proj4')
        X = ncfile.variables['X'][:]
        Y = ncfile.variables['Y'][:]
    except Exception as e:
        raise e
    finally:
        ncfile.close()
    
    proj = pyproj.Proj(proj4)
    
    return X, Y, proj

def initlonlat2initgpuocean(source_url, lon, lat, X= None, Y= None, proj = None, num_cells_x = 100, num_cells_y = 100):
    """
    Given netCDF-file, takes in longitude and latitude coordinates(single or lists) 
    and returns necessary variables for initilazing a GPUOcean simulation.
    Returns domain coordinates as well as initial positions of drifters. Default domain size = 100 cells in both directions.
    If X, Y and proj are given, netCDF-file is not opened.
    """
    if (X is None) or (Y is None) or (proj is None):
        X, Y, proj = getXYproj(source_url)
    
    res = int(X[1]-X[0]) #Finding grid-resolution (assumed same in both horizontal directions)
    
    #Finding tentative x,y(not for a specific domain)
    x, y = proj(lon,lat, inverse = False)
    
    #Given x,y, num_cells_x, num_cells_y and resolution: specify domain in gpuocean
    x0, x1 = x//res - num_cells_x//2, x//res + num_cells_x//2 
    y0, y1 = y//res - num_cells_y//2, y//res + num_cells_y//2
    
    #Find new x,y in gpuocean coordinates for initial position
    xinit = x - X[int(x0) + 2]
    yinit = y- Y[int(y0) + 2]
    
    return xinit, yinit, int(x0), int(x1), int(y0), int(y1)

def lonlat2xygpuocean(source_url, lon, lat, x0, y0, X= None, Y= None, proj = None):
    """
    Takes in NetCDF-file, x, y coordinates(single or lists) and x0, y0 of GPU Ocean-domain. 
    Returns x, y projection of lon, lat.
    If X, Y and proj are given, netCDF-file is not opened.
    """
    if (X is None) or (Y is None) or (proj is None):
        X, Y, proj = getXYproj(source_url)
    
    res = int(X[1]-X[0]) #Finding grid-resolution (assumed same in both horizontal directions)
    
    #Finding tentative x,y(not for a specific domain)
    x, y = proj(lon,lat, inverse = False)

    #Find new x,y in gpuocean coordinates for initial position
    x = x - X[int(x0) + 2]
    y = y- Y[int(y0) + 2]
    
    return x, y


def xygpuocean2lonlat(source_url, x, y, x0, y0, X= None, Y= None, proj = None):
    """
    Takes in NetCDF-file, longitude and latitude coordinates and x0, y0 of GPU Ocean-domain. 
    Returns lon, lat equivalent of x, y. 
    If X, Y and proj are given, netCDF-file is not opened.
    """
    if (X is None) or (Y is None) or (proj is None):
        X, Y, proj = getXYproj(source_url)
        
    x = x + X[x0+2]
    y = y + Y[y0+2]
    
    lon, lat = proj(x, y, inverse = True)

    return lon, lat


#Function for running simulation

def simulate_gpuocean_deterministic(source_url, domain, initx, inity, 
                                    sim_args, erode_land = 1, 
                                    observation_type = dautils.ObservationType.UnderlyingFlow,
                                    wind_drift_factor = 0.0, rescale=0,
                                    outfolder = None, start_forecast_hours = 0, forecast_duration = 24 ):
    """
    source_url: url or local file or list of either with fielddata in NetCDF-format
    domain: array/list on form [x0,x1,y0,y1] defining the domain for the simulation
    initx, inity = initial coordinates of single drifter or lists with coordinates for multiple drifters. 
                   In local cartesian coordinates of simulation-domain. 
    sim_args, erode_land, observation_type: arguments needed for simulator and observation object. sim_args must be given.
    wind_drift_factor: fraction of wind-speed at which objects will be advected. Default is 0 (no direct wind-drift)
    rescale: factor setting resolution of simulation-grid. 0 indicates no rescaling(original resolution), 
             while any other number changes the resolution (ie 2 gives double resolution)
    outfolder: outfolder
    start_forecast_hours = number hours after which to start simulating drifttrajectories(ocean model starts at beginning of field-data)
                           Default is at beginning of field-data. 
    forecast_duration = duration of simulation(including possibly only ocean simulation). Default is 24 hours.
    """

    end_forecast_hours = start_forecast_hours + forecast_duration
    
    #Create simulator
    data_args = NetCDFInitialization.getInitialConditions(source_url, domain[0], domain[1], domain[2],domain[3] , 
                     timestep_indices = None, erode_land = erode_land, download_data = False)
    
    if wind_drift_factor:
        wind_data = data_args.pop('wind', None)
    else:
        wind_data = None
        
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
        trajectory_forecast_filename = 'trajectory_forecast_'+str(start_forecast_hours)+'_to_'+str(end_forecast_hours)+'.pickle'
    
    #Drifters
    #Assumes initx, inity same format/shape
    if type(initx) is not list:
        initx = [initx]
        inity = [inity]
    
    num_drifters = len(initx)
    
    drifters = GPUDrifterCollection.GPUDrifterCollection(sim_args['gpu_ctx'], num_drifters, wind = wind_data, 
                                                         wind_drift_factor = wind_drift_factor,
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





#Functions for plotting

def getVfromReference(source_url,domain, hour):
    x0,x1,y0,y1 = domain
    
    ncfile = None
    try:
        ncfile = Dataset(source_url)
        H_m = ncfile.variables['h'][y0:y1, x0:x1]
        eta = ncfile.variables['zeta'][hour, y0:y1, x0:x1]
        hu = ncfile.variables['ubar'][hour, y0:y1, x0:x1]
        hv = ncfile.variables['vbar'][hour, y0:y1, x0:x1]
        
        hu = hu * (H_m + eta)
        hv = hv * (H_m + eta)
        
    except Exception as e:
        raise e
    finally:
        ncfile.close()
        
    H_m.mask = eta.mask
    
    V = PlotHelper.genVelocity(eta + H_m, hu, hv)
    V.mask = eta.mask
    return V



def createForecastCanvas(observation, background= False, url=None, domain = None, drifter_id = 0,  zoom = 1, hour=23):
    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(111)

    domain_size_x = observation.domain_size_x
    domain_size_y = observation.domain_size_y

    extent=np.array([0, domain_size_x, 0, domain_size_y]) 

    if background:
        assert(url is not None and domain is not None), 'Url or domain missing for background'
        v_cmap = plt.cm.YlOrRd

        land_color = 'grey'
        v_cmap.set_bad(land_color, alpha = 1.0)
        V = getVfromReference(url,domain, hour)

        ax.imshow(V, origin = 'lower', extent = extent, cmap = v_cmap, vmin = 0, vmax = 0.6)
    else:
        ax.imshow(observation.land_mask, origin="lower", extent=extent, cmap='binary')
    
    if (zoom!=1):
        path_x = observation.get_drifter_path(drifter_id, 0, hour*3600, in_km = False)[:][:,0]
        path_y = observation.get_drifter_path(drifter_id, 0, hour*3600, in_km = False)[:][:,1]
        
        xmin, xmax, ymin, ymax = np.nanmin(path_x), np.nanmax(path_x), np.nanmin(path_y), np.nanmax(path_y)
        
        xcenter, ycenter = xmin + (xmax-xmin)/2, ymin + (ymax-ymin)/2

        xlim_min = max(xcenter - 0.5 * domain_size_x * 1/zoom, 0)
        xlim_max = min(xcenter + 0.5 * domain_size_x * 1/zoom, domain_size_x)
        ylim_min = max(ycenter - 0.5 * domain_size_y * 1/zoom, 0)
        ylim_max = min(ycenter + 0.5 * domain_size_y * 1/zoom, domain_size_y)
    else:
        xlim_min, xlim_max, ylim_min, ylim_max = 0,domain_size_x,0, domain_size_y

    ax.set_xlim([xlim_min, xlim_max])
    ax.set_ylim([ylim_min, ylim_max])

    return ax


def plotAllDrifters(obs, drifter_ids=None,background = False, url = None, color_id = 2, domain = None, ax = None, zoom_element = 0, start = 0, end = 23, zoom = 1):
    """background: True if velocity field as background(absolute value of velocity). Default is False. 
                Opendrift/Norkyst800 used currently. GPUOcean? Just use data from generated netcdf-file?
        url needed if background = True, domain needed if background = True
    """
    color_combinations = [
        # [true color, forecast color]
        #['xkcd:dark grey blue', 'xkcd:light blue grey', 'xkcd:wine'],
        ['xkcd:scarlet', 'xkcd:light blue grey', 'xkcd:dark blue grey'],
        ['xkcd:viridian', 'xkcd:foam green','xkcd:light blue grey'],
        ['g', 'xkcd:foam green', 'xkcd:dark grey green']
    ]
    colors = color_combinations[0]
    
    if drifter_ids is None:
        drifter_ids = np.arange(obs.get_num_drifters(ignoreBuoys=True))
    
    num_drifters = len(drifter_ids)
    
    drifter_paths = [None]*num_drifters

    forecast_start_t = 0
    forecast_end_t = end*3600

    for i in range(num_drifters):
        drifter_paths[i] = obs.get_drifter_path(drifter_ids[i], forecast_start_t, forecast_end_t, in_km = False)

    if ax is None:
        ax = createForecastCanvas(obs, background = background,url = url,domain = domain, hour = end, drifter_id = zoom_element, zoom = zoom)
        
    for drifter_path in drifter_paths:
        for path in drifter_path:
            ax.plot(path[:,0], path[:,1], color=colors[color_id], zorder=5)

            # Mark start and end of true path
            start_pos = drifter_path[0][0,:]
            end_pos   = drifter_path[-1][-1,:]

            circ_start = matplotlib.patches.Circle((start_pos[0], start_pos[1]), 
                                                   20, color = 'xkcd:scarlet',
                                                   fill=False, zorder=10)
            ax.add_patch(circ_start)
            ax.plot(end_pos[0], end_pos[1], 'x', color='k', zorder=11)
