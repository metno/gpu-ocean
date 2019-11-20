# -*- coding: utf-8 -*-

"""
This python class represents an ensemble of ocean models with slightly
perturbed states. Runs on multiple nodes using MPI

Copyright (C) 2018  SINTEF ICT

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
from mpi4py import MPI
import gc, os, sys
import time

# Needed for clean exit with logging enabled
import pycuda.driver as cuda

#Import our simulator
file_dir = os.path.dirname(os.path.realpath(__file__)) 
sys.path.insert(0, os.path.abspath(os.path.join(file_dir, '../')))

from SWESimulators import CDKLM16, PlotHelper, Common, WindStress, OceanographicUtilities, MPIOceanModelEnsemble, NetCDFInitialization
from SWESimulators import DataAssimilationUtils as dautils

def testMPI():
    comm = MPI.COMM_WORLD

    print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))

    #Make results deterministic
    np.random.seed(seed=(42 + comm.rank))
    comm.Barrier()   
    
    
    

def generateDrifterPositions(data_args, num_drifters):
    """Randomly placed drifters, avoiding land."""
    assert(MPI.COMM_WORLD.rank == 0)
    
    drifters = np.empty((num_drifters, 2))
    
    for id in range(num_drifters):
        land = True
        while(land is True):
            x = np.random.randint(0, data_args['nx'])
            y = np.random.randint(0, data_args['ny'])
            
            if(data_args['hu0'][y][x] is not np.ma.masked and
               data_args['hu0'][y-1][x] is not np.ma.masked and data_args['hu0'][y+1][x] is not np.ma.masked and
               data_args['hu0'][y][x-1] is not np.ma.masked and data_args['hu0'][y][x+1] is not np.ma.masked):
                drifters[id, 0]  = x * data_args['dx']
                drifters[id, 1]  = y * data_args['dy']
                land = False

    return drifters
    
    
    
    
#FIXME: WARNING: this performs superfluous work just for plotting
def gatherPlottingInfo(ensemble):
    
    range_x = np.sqrt(ensemble.ensemble.observation_variance)*20
    
    #Gather innovatoins and then remove root / master
    local_innovations = ensemble._localGetInnovations()
    local_innovations_norm = np.linalg.norm(np.linalg.norm(local_innovations, axis=2), axis=1).astype(np.float32)
    global_innovations_norm = None
    if (ensemble.comm.rank == 0):
        global_innovations_norm = np.empty((ensemble.num_nodes+1, ensemble.local_ensemble_size), dtype=np.float32)
    ensemble.comm.Gather(local_innovations_norm, global_innovations_norm, root=0)
    if (ensemble.comm.rank == 0):
        global_innovations_norm = global_innovations_norm[1:].ravel()
        

    #Gather gaussian pdf and then remove root / master
    use_true_gaussian_pdf=False
    if (use_true_gaussian_pdf):
        local_gaussian_pdf = ensemble._localGetGaussianPDF(local_innovations)
        global_gaussian_pdf = None
        if (ensemble.comm.rank == 0):
            global_gaussian_pdf = np.empty((ensemble.num_nodes+1, ensemble.local_ensemble_size))
        ensemble.comm.Gather(local_gaussian_pdf, global_gaussian_pdf, root=0)
        if (ensemble.comm.rank == 0):
            global_gaussian_pdf = global_gaussian_pdf[1:].ravel()
    else:
        #FIXME: This is butt ugly but gives the analytical gaussian distribution
        global_gaussian_pdf = None
        if (ensemble.comm.rank == 0):
            x = np.zeros((100, ensemble.num_drifters, 2))
            x[:,0,0] = np.linspace(0, (range_x), 100)
            global_gaussian_pdf = ensemble._localGetGaussianPDF(x)
    
    
    #Get the normalized weights
    global_normalized_weights = ensemble.getNormalizedWeights()
    
    return {
        'range_x': range_x,
        'gauss_pdf': global_gaussian_pdf,
        'gauss_weights': global_normalized_weights,
        'innovations': global_innovations_norm
    }
    
    
    
    
def mainLoop(ensemble, resampling_times, outfilename):
    if (ensemble.comm.rank == 0):
        plotting_info = []
        
    #Perform actual data assimilation        
    t = 0
    for i, resampling_time in enumerate(resampling_times):
        #Step all nodes in time to next assimilation stage
        assimilation_dt = resampling_time - t
        
        step_size = 60
        #FIXME: Assert: assimilation_dt divisable with step_size
        sub_steps = int(assimilation_dt // step_size)
        
        drifter_cells = ensemble.getDrifterCells()
        
        for j in range(sub_steps):
            t = ensemble.modelStep(step_size)
            #Make ParticleInfo for writing to file 
            ensemble.dumpStateSample(drifter_cells)

        
        
        
        # Get info before step for plotting
        # FIXME: Expensive and only for plotting
        ##ensemble_stats_pre = gatherPlottingInfo(ensemble)
            
            
            
            
        
        #Gather the gaussian weights from all nodes to a global vector on rank 0
        global_normalized_weights = ensemble.getNormalizedWeights()
        
        #Resample the particles
        ensemble.resampleParticles(global_normalized_weights)
        
        #Get info for plotting
        #FIXME: Expensive and only for plotting
        ##ensemble_stats_post = gatherPlottingInfo(ensemble)
        ##if ensemble.comm.rank == 0:
        ##    plotting_info.append([resampling_time, ensemble_stats_pre, ensemble_stats_post])
            
            
        print(str(ensemble.comm.rank) + ", ", end="", flush=True)
    
    #Save to file
    ##if (ensemble.comm.rank == 0):
    ##    print("Saving data to " + outfilename)
    ##    np.savez(outfilename, plotting_info=plotting_info)
    
    ensemble.dumpParticleInfosToFiles("particle_info")
    
    
    
    
    
def makePlots(filename):
    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec

    import numpy as np

    def plotDistanceInfo(ensemble_info, ax0, ax1):
        # With observation 
        x = np.linspace(0, ensemble_info['range_x'], num=len(ensemble_info['gauss_pdf']))
        ax0.plot(x, ensemble_info['gauss_pdf'], 'g', label="pdf directly from innovations")
        ax0.set_ylabel('Particle PDF')
        #plt.legend()
        #plt.title("Distribution of particle innovations")

        #hisograms:
        ax0_1 = ax0.twinx()
        ax0_1.hist(ensemble_info['innovations'], bins=30, \
                 range=(0, ensemble_info['range_x']),\
                 normed=True, label="particle innovations (norm)")
        ax0_1.set_ylabel('Innovations (hist)')

        
        
        
        # PLOT SORTED DISTANCES FROM OBSERVATION
        indices_sorted_by_observation = ensemble_info['innovations'].argsort()
        ax1.plot(ensemble_info['gauss_weights'][indices_sorted_by_observation]/np.max(ensemble_info['gauss_weights']),\
                 'g', label="Weight directly from innovations")
        ax1.set_ylabel('Weights directly from innovations', color='g')
        ax1.grid()
        ax1.set_ylim(0,1.4)
        #plt.legend(loc=7)
        ax1.set_xlabel('Particle ID')

        ax1_1 = ax1.twinx()
        ax1_1.plot(ensemble_info['innovations'][indices_sorted_by_observation], label="innovations")
        ax1_1.set_ylabel('Innovations', color='b')



    with np.load(filename) as data:
        for time, pre, post in data['plotting_info']:
            # Only rank 0 creates a figure:
            fig = None
            fig = plt.figure(figsize=(16, 8))
            
            ax0 = plt.subplot(2,2,1)
            ax1 = plt.subplot(2,2,2)
            ax2 = plt.subplot(2,2,3)
            ax3 = plt.subplot(2,2,4)
            
            plotDistanceInfo(pre, ax0, ax1)
            plotDistanceInfo(post, ax2, ax3)
            
            plt.suptitle("t=" + str(time) + " before (top) and after (bottom) resampling")
        plt.show()

def getInitialConditions(source_url, x0, x1, y0, y1):
    cache_folder = 'netcdf_cache'
    os.makedirs(cache_folder, exist_ok=True)

    filename = os.path.abspath(os.path.join(cache_folder, os.path.basename(source_url)))

    if (os.path.isfile(filename)):
        source_url = filename

    else:
        import requests
        download_url = source_url.replace("dodsC", "fileServer")

        req = requests.get(download_url, stream = True)
        filesize = int(req.headers.get('content-length'))

        progress = Common.ProgressPrinter()
        print(progress.getPrintString(0) + "\r")

        print("Downloading data to local file (" + str(filesize // (1024*1024)) + " MB)")
        with open(filename, "wb") as outfile:
            for chunk in req.iter_content(chunk_size = 10*1024*1024):
                if chunk:
                    outfile.write(chunk)
                    print(progress.getPrintString(outfile.tell() / filesize) + "\r")
                    print("")

        source_url = filename

    print("Source is: " + source_url)
    
    return NetCDFInitialization.removeMetadata(NetCDFInitialization.getInitialConditions(source_url, x0, x1, y0, y1))
    
    
def setupLogger(args):
    import logging

    #Get root logger
    logger = logging.getLogger('')
    logger.setLevel(args.log_level)

    #Add log to screen
    ch = logging.StreamHandler()
    ch.setLevel(args.log_level)
    logger.addHandler(ch)
    logger.log(args.log_level, "Console logger using level %s", logging.getLevelName(args.log_level))
            
    #Get the logfilename (try to evaluate if Python expression...)
    try:
        log_file = eval(args.logfile, self.shell.user_global_ns, self.shell.user_ns)
    except:
        log_file = args.log_file
            
    #Add log to file
    logger.log(args.log_level, "File logger using level %s to %s", logging.getLevelName(args.log_level), log_file)
    fh = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    fh.setLevel(args.log_level)
    logger.addHandler(fh)
        
    logger.info("Python version %s", sys.version)
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run an MPI cluster for sequential importance resampling. Run this file using e.g., using mpiexec -n 4 python sequential_importance_resampling.py to run a simulation. Plot results using the --make_plots option (without MPI!)')
    parser.add_argument('-dt', type=float, default=0.0)
    parser.add_argument('--outfile', type=str, default="sir_" + str(MPI.COMM_WORLD.rank) + ".npz")
    parser.add_argument('--num_drifters', type=int, default=-1) #TODO: Use only num_drifters of the entries in observation_file
    parser.add_argument('--per_node_ensemble_size', type=int, default=5)
    parser.add_argument('--observation_variance', type=float, default=1)#0.01**2)
    parser.add_argument('--initialization_variance_factor_ocean_field', type=float, default=0.0)
    parser.add_argument('--observation_file', type=str, default=None)
    parser.add_argument('--make_plots', action='store_true', default=False)
    parser.add_argument('--log_file', type=str, default="sequential_importance_resampling.log")
    parser.add_argument('--log_level', type=int, default=20)
    
    args = parser.parse_args()
    
    logger = setupLogger(args)
    
    
    if (args.make_plots):
        print("Making plots should not be run with MPI")
        makePlots(args.outfile)
        
        
    else:
        #Test that MPI works
        testMPI()
        
        # FIXME: Hardcoded parameters
        observation_type = dautils.ObservationType.StaticBuoys
        
        kwargs = {}
        #Generate initial conditions on rank 0
        if (MPI.COMM_WORLD.rank == 0):
            
            #Size of ensemble per node
            kwargs['local_ensemble_size'] = args.per_node_ensemble_size
            
            #Initial conditions
            # FIXME: Hardcoded parameters
            source_url = norkyst800_url = 'https://thredds.met.no/thredds/dodsC/fou-hi/norkyst800m-1h/NorKyst-800m_ZDEPTHS_his.an.2019071600.nc'
            x0 = 25
            x1 = 2575
            y0 = 25
            y1 = 875
            
            dt = args.dt
            
            t0 = time.time()
            data_args = getInitialConditions(source_url, x0, x1, y0, y1)
            t1 = time.time()
            total = t1-t0
            print("Fetched initial conditions in " + str(total) + " s")
            
            sim_args = {
                "dt": dt,
                "rk_order": 2,
                "desingularization_eps": 1.0,
                "small_scale_perturbation": True,
                "small_scale_perturbation_amplitude": 1.0e-5, #1.0e-5,
                "small_scale_perturbation_interpolation_factor": 1,
                "write_netcdf": True,
            }
            
            kwargs['data_args'] = data_args
            kwargs['sim_args'] = sim_args
            
        #Arguments sent to the ensemble (OceanModelEnsemble)
        kwargs['ensemble_args'] = {
            'observation_variance': args.observation_variance, 
            'initialization_variance_factor_ocean_field': args.initialization_variance_factor_ocean_field
        }            
            
        #Create ensemble on all nodes
        t0 = time.time()
        ensemble = MPIOceanModelEnsemble.MPIOceanModelEnsemble(MPI.COMM_WORLD, args.observation_file, observation_type, **kwargs)
        t1 = time.time()
        total = t1-t0
        print("Initialized MPI ensemble on rank " + str(MPI.COMM_WORLD.rank) + " in " + str(total) + " s")
        
        #Run main loop
        resampling_times = np.array([1, 2, 3, 4]) * 15*60
        #resampling_times[0] = 5*60
        
        if (ensemble.comm.rank == 0):
            print("Will resample at times: ", resampling_times)
            
        t0 = time.time()
        mainLoop(ensemble, resampling_times, args.outfile)
        t1 = time.time()
        total = t1-t0
        print("Main loop on rank " + str(MPI.COMM_WORLD.rank) + " finished in " + str(total) + " s")
        
        # Handle CUDA context when exiting python
        import atexit
        def exitfunc():
            import logging
            logger =  logging.getLogger(__name__)
            logger.info("[" + str(MPI.COMM_WORLD.rank) + "]: Exitfunc: Resetting CUDA context stack")
            while (cuda.Context.get_current() != None):
                context = cuda.Context.get_current()
                logger.info("`-> Popping <%s>", str(context.handle))
                cuda.Context.pop()
            logger.debug("==================================================================")
            logging.shutdown()
            
        atexit.register(exitfunc)
