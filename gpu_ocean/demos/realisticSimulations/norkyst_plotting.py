import numpy as np
import datetime
from netCDF4 import Dataset

from matplotlib import animation, rc, colors, colorbar
from matplotlib import pyplot as plt

from IPython.display import display
from IPython.display import Video

from SWESimulators import PlotHelper, Common, OceanographicUtilities

def plotSolution(fig, 
                 eta, hu, hv, h, dx, dy, 
                 t, comment,
                 h_min=-1.5, h_max=1.5, 
                 uv_min=-0.05, uv_max=0.05, 
                 add_extra=0,
                 ax=None, sp=None,
                 rotate=False, downsample=None,
                 drifter_locations=None,
                 drifter_locations_original=None,
                 buoy_indices=None):    
    from datetime import timedelta
    
    fig.suptitle("Time = {:0>8} ({:s})".format(str(timedelta(seconds=int(t))), comment), 
                 fontsize=18,
                 horizontalalignment='left')
    
    x_plots = 1
    y_plots = 1

    fig_size = fig.get_size_inches()*fig.dpi
    if (downsample is not None):
        downsample = np.maximum(np.int32([1, 1]), np.int32(0.5 * np.float64(eta.shape) / np.float64(fig_size)))
        
        eta = eta[::downsample[0], ::downsample[1]]
        h = h[::downsample[0], ::downsample[1]]
        hu = hu[::downsample[0], ::downsample[1]]
        hv = hv[::downsample[0], ::downsample[1]]
    
    if (add_extra == 1):
        x_plots=3
        y_plots=1
    elif (add_extra == 2):
        x_plots=3
        y_plots=2
        
    if (add_extra == 2):
        V_max = 5 * (uv_max-uv_min) / np.max(h)
        R_min = -V_max/2
        R_max = V_max/2
        
        V = PlotHelper.genVelocity(h, hu, hv)
        R = PlotHelper.genColors(h, hu, hv, plt.cm.seismic, R_min, R_max)
            
    ny, nx = eta.shape
    
    # scale drifter and buoy locations according to the domain
    # 
    if drifter_locations is not None:
        drifter_locations = drifter_locations/1000
    if drifter_locations_original is not None:
        drifter_locations_original = drifter_locations_original/1000
    if buoy_indices is not None:
        buoy_indices[:, 0] = buoy_indices[:, 0]*dx/1000
        buoy_indices[:, 1] = buoy_indices[:, 1]*dy/1000
        
        
    if (rotate):
        domain_extent = [0, ny*dy/1000, 0, nx*dx/1000]
        eta = np.rot90(eta, 3)
        h = np.rot90(h, 3)
        hu = np.rot90(hu, 3)
        hv = np.rot90(hv, 3)
        
        # Locations in the domain must be alteret as well
        # x values becomes ny-y, and y values becomes x 
        if drifter_locations is not None:
            drifter_locations_copy = drifter_locations.copy()
            drifter_locations[:,0] = ny*dy/1000 - drifter_locations_copy[:, 1]
            drifter_locations[:,1] = drifter_locations_copy[:,0]
        if drifter_locations_original is not None:
            drifter_locations_original_copy = drifter_locations_original.copy()
            drifter_locations_original[:,0] = ny*dy/1000 - drifter_locations_original_copy[:, 1]
            drifter_locations_original[:,1] = drifter_locations_original_copy[:,0]
        if buoy_indices is not None:
            buoy_indices_copy = buoy_indices.copy()
            buoy_indices[:,0] = ny*dx/1000 - buoy_indices_copy[:, 1]
            buoy_indices[:,1] = buoy_indices_copy[:,0]
        
        if (add_extra == 2):
            V = np.rot90(V, 3)
            R = np.rot90(R, 3)
    else:
        domain_extent = [0, nx*dx/1000, 0, ny*dy/1000]
    
    
    if (ax is None):
        ax = [None]*x_plots*y_plots
        sp = [None]*x_plots*y_plots
        
        ax[0] = plt.subplot(y_plots, x_plots, 1)
        sp[0] = ax[0].imshow(eta, interpolation="none", origin='bottom', 
                             cmap=plt.cm.coolwarm, 
                             vmin=h_min, vmax=h_max, 
                             extent=domain_extent)
        
        plt.colorbar(sp[0], shrink=0.9)
        plt.axis('image')
        plt.title("$\eta{}$")
        
        # Show drifters
        if drifter_locations is not None:
            ax[0].scatter(x=drifter_locations[:,0], y=drifter_locations[:,1], color='xkcd:lime green')    
        if drifter_locations_original is not None:
            ax[0].scatter(x=drifter_locations_original[:,0], y=drifter_locations_original[:,1], color='xkcd:lime green', marker='x')    
        if buoy_indices is not None:
            ax[0].scatter(x=buoy_indices[:,0], y=buoy_indices[:,1], color='xkcd:black', marker='^')  # yellow  
            
        if (add_extra > 0):
            ax[1] = plt.subplot(y_plots, x_plots, 2)
            sp[1] = ax[1].imshow(hu, interpolation="none", origin='bottom', 
                                 cmap=plt.cm.coolwarm, 
                                 vmin=uv_min, vmax=uv_max, 
                                 extent=domain_extent)
            plt.colorbar(sp[1], shrink=0.9)
            plt.axis('image')
            plt.title("$hu$")

            ax[2] = plt.subplot(y_plots, x_plots, 3)
            sp[2] = ax[2].imshow(hv, interpolation="none", origin='bottom', 
                                 cmap=plt.cm.coolwarm, 
                                 vmin=uv_min, vmax=uv_max, 
                                 extent=domain_extent)
            plt.colorbar(sp[2], shrink=0.9)
            plt.axis('image')
            plt.title("$hv$")

        if (add_extra > 1):
            ax[3] = plt.subplot(y_plots, x_plots, 4)
            sp[3] = ax[3].imshow(V, interpolation="none", origin='bottom', 
                               cmap=plt.cm.Oranges, 
                               vmin=0, vmax=V_max, 
                               extent=domain_extent)
            plt.colorbar(sp[3], shrink=0.9)
            plt.axis('image')
            plt.title("Particle velocity magnitude")

            ax[4] = plt.subplot(y_plots, x_plots, 5)
            sp[4] = ax[4].imshow(R, interpolation="none", 
                               origin='bottom', 
                               extent=domain_extent)
            sm = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=R_min, vmax=R_max), cmap=plt.cm.seismic)
            sm.set_array([])
            fig.colorbar(sm, shrink=0.9)
            #colorbar.Colorbar(ax[4], cmap=plt.cm.seismic, mappable=plt.cm.ScalarMappable(norm=colors.Normalize(vmin=R_min, vmax=R_max), cmap=plt.cm.seismic)colors.Normalize(vmin=R_min, vmax=R_max), orientation='horizontal', shrink=0.9)
            plt.axis('image')
            plt.title("Vorticity magnitude")
            
    else:        
        #Update plots
        fig.sca(ax[0])
        sp[0].set_data(eta)
        
        if (add_extra > 0):
            fig.sca(ax[1])
            sp[1].set_data(hu)

            fig.sca(ax[2])
            sp[2].set_data(hv)
        
        if (add_extra > 1):
            fig.sca(ax[3])
            sp[3].set_data(V)

            fig.sca(ax[4])
            sp[4].set_data(R)
    
    return ax, sp


def plotStatistics(filename): 
    try:
        ncfile = Dataset(filename)
        t = ncfile.variables['time'][:]
        num_iterations = ncfile.variables['num_iterations'][:]

        num_timesteps = len(t)
        max_abs_u = np.zeros(num_timesteps)
        max_abs_v = np.zeros(num_timesteps)
        
        H_m = ncfile.variables['Hm'][:,:]
        
        for i in range(num_timesteps):
            eta = ncfile.variables['eta'][i,:,:]
            mask = eta.mask.copy()
            
            h = H_m + eta
            h[H_m < 5] = np.ma.masked
            
            eps = 1.0e-5
            hu = ncfile.variables['hu'][i,:,:]
            hv = ncfile.variables['hv'][i,:,:]
            u = OceanographicUtilities.desingularise(h, hu, eps)
            v = OceanographicUtilities.desingularise(h, hv, eps)
            
            max_abs_u[i] = np.max(np.abs(u))
            max_abs_v[i] = np.max(np.abs(v))
            
        
    except Exception as e:
        print("Something went wrong:" + str(e))
        raise e
    finally:
        ncfile.close()

    plt.title("Statistics")
    plt.subplot(2,1,1)
    plt.plot(0.5*(t[1:] + t[:-1])/3600, np.diff(t)/np.diff(num_iterations), label="Average Dt")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(t/3600, max_abs_u, label="Max $|u|$")
    plt.plot(t/3600, max_abs_v, label="Max $|v|$")
    plt.legend()
    
    
    
    
def ncAnimation(filename, title=None, movie_frames=None, create_movie=True, fig=None, save_movie=True, **kwargs):    
    if (title is None):
        title = filename.replace('_', ' ').replace('.nc', '')

    #Create figure and plot initial conditions
    if fig is None:
        fig = plt.figure(figsize=(14, 4))

    try:
        ncfile = Dataset(filename)
        x = ncfile.variables['x'][:]
        y = ncfile.variables['y'][:]
        t = ncfile.variables['time'][:]

        H_m = ncfile.variables['Hm'][:,:]
        eta = ncfile.variables['eta'][:,:,:]
        hu = ncfile.variables['hu'][:,:,:]
        hv = ncfile.variables['hv'][:,:,:]
    except Exception as e:
        raise e
    finally:
        ncfile.close()

    if movie_frames is None:
        movie_frames = len(t)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    progress = Common.ProgressPrinter(5)
    pp = display(progress.getPrintString(0),display_id=True)

    if (create_movie):
        ax, sp = plotSolution(fig, 
                              eta[0],
                              hu[0],
                              hv[0],
                              H_m+eta[0],
                              dx, dy, 
                              t[0], title,
                              **kwargs)
    else:
        ax, sp = plotSolution(fig, 
                              eta[-1],
                              hu[-1],
                              hv[-1],
                              H_m+eta[-1],
                              dx, dy, 
                              t[-1], title,
                              **kwargs)
        return

    #Helper function which simulates and plots the solution    
    def animate(i):
        t_now = t[0] + (i / (movie_frames-1)) * (t[-1] - t[0]) 

        k = np.searchsorted(t, t_now)
        if (k >= eta.shape[0]):
            k = eta.shape[0] - 1
        j = max(0, k-1)
        if (j == k):
            k += 1
        s = (t_now - t[j]) / (t[k] - t[j])

        plotSolution(fig, 
                     (1-s)*eta[j] + s*eta[k], 
                     (1-s)*hu[j]  + s*hu[k], 
                     (1-s)*hv[j]  + s*hv[k], 
                     H_m+(1-s)*eta[j] + s*eta[k], 
                     dx, dy, 
                     t_now, title, 
                     **kwargs, ax=ax, sp=sp)

        pp.update(progress.getPrintString(i / (movie_frames-1)))

    #Matplotlib for creating an animation
    anim = animation.FuncAnimation(fig, animate, range(movie_frames), interval=100)
    if (save_movie):
        anim.save(filename + '.mp4')
        plt.close(fig)
        return Video(filename + '.mp4')
    else:
        plt.close(fig)
        return anim
    
    
    
    
    
def refAnimation(source_url, case, movie_frames=None, timestep_indices=None, create_movie=True, fig=None, save_movie=True, **kwargs): 
    
    #Create figure and plot initial conditions
    if fig is None:
        fig = plt.figure(figsize=(14, 4))
        
    try:
        ncfile = Dataset(source_url)
        
        if (timestep_indices is not None):
            timesteps = ncfile.variables['time'][timestep_indices[:]]
        else:
            timesteps = ncfile.variables['time'][:]
            timestep_indices = range(len(timesteps))
            
        if (not create_movie):
            timestep_indices = [timestep_indices[0]] + [timestep_indices[-1]]
            
        t0 = min(timesteps)
        timesteps = timesteps - t0
        
        H_m = ncfile.variables['h'][case['y0']:case['y1'], case['x0']:case['x1']]
        eta = ncfile.variables['zeta'][timestep_indices[:], case['y0']:case['y1'], case['x0']:case['x1']]
        hu = ncfile.variables['ubar'][timestep_indices[:], case['y0']:case['y1'], case['x0']:case['x1']]
        hv = ncfile.variables['vbar'][timestep_indices[:], case['y0']:case['y1'], case['x0']:case['x1']]
        x = ncfile.variables['X'][case['x0']:case['x1']]
        y = ncfile.variables['Y'][case['y0']:case['y1']]
        
        dx = np.average(x[1:] - x[:-1])
        dy = np.average(y[1:] - y[:-1])
        
        for i in range(len(timestep_indices[:])):
            hu[i] = hu[i] * (H_m + eta[i])
            hv[i] = hv[i] * (H_m + eta[i])
    except Exception as e:
        raise e
    finally:
        ncfile.close()

    if movie_frames is None:
        movie_frames = len(timestep_indices)
    
    if (create_movie):
        ax, sp = plotSolution(fig, 
                              eta[0],
                              hu[0],
                              hv[0],
                              H_m+eta[0],
                              dx, dy, 
                              timesteps[0], "Reference solution $t_0$=" + datetime.datetime.fromtimestamp(t0).isoformat(timespec='seconds'),
                              **kwargs)
    else:
        ax, sp = plotSolution(fig, 
                              eta[-1],
                              hu[-1],
                              hv[-1],
                              H_m+eta[-1],
                              dx, dy,
                              timesteps[-1], "Reference solution $t_0$=" + datetime.datetime.fromtimestamp(t0).isoformat(timespec='seconds'),
                              **kwargs)
        return
    
    
    progress = Common.ProgressPrinter(5)
    pp = display(progress.getPrintString(0),display_id=True)
    
    #Helper function which simulates and plots the solution    
    def animate(i):
        t_now = timesteps[0] + (i / (movie_frames-1)) * (timesteps[-1] - timesteps[0]) 
        
        k = np.searchsorted(timesteps, t_now)
        if (k >= eta.shape[0]):
            k = eta.shape[0] - 1
        j = max(0, k-1)
        if (j == k):
            k += 1
        s = (t_now - timesteps[j]) / (timesteps[k] - timesteps[j])
        
        plotSolution(fig, 
                     (1-s)*eta[j] + s*eta[k], 
                     (1-s)*hu[j]  + s*hu[k], 
                     (1-s)*hv[j]  + s*hv[k], 
                     H_m+(1-s)*eta[j] + s*eta[k], 
                     dx, dy, 
                     t_now, "Reference solution $t_0$=" + datetime.datetime.fromtimestamp(t0).isoformat(timespec='seconds'),
                     **kwargs, ax=ax, sp=sp)
        
        pp.update(progress.getPrintString(i / (movie_frames-1)))

    #Matplotlib for creating an animation
    anim = animation.FuncAnimation(fig, animate, range(movie_frames), interval=100)
    
    if (save_movie):
        anim.save('reference.mp4')
        plt.close(fig)
        print("Saving to reference.mp4")
        return Video('reference.mp4')
    else:
        plt.close(fig)
        return anim