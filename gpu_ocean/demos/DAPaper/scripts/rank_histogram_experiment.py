# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2019 SINTEF Digital

This python program is used to set up and run data assimilation experiments
that are used to create rank histograms.

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


import sys, os, json, datetime, time, shutil
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))

if os.path.isdir(os.path.abspath(os.path.join(current_dir, '../../../SWESimulators'))):
        sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../../../')))


        
#--------------------------------------------------------------
# PARAMETERS
#--------------------------------------------------------------
# We aim to minimize the number of required input parameters, in order to run the 
# same experiment each time.

import argparse
parser = argparse.ArgumentParser(description='Generate an ensemble.')
parser.add_argument('--experiments', type=int, default=None)
parser.add_argument('--output_folder', type=str, default="/media/havahol/Seagate Backup Plus Drive/gpu_ocean")

const_args = {
    'ensemble_size' : 40,
    'method' : 'iewpf2',
    'observation_interval' : 1,
    'observation_variance' : 1,
    'observation_type' : 'buoys',
    'buoy_area' : 'west'
}




args = parser.parse_args()

# Checking input args
if args.experiments is None:
    print("Number of experiments missing, please provide a --experiments argument.")
    sys.exit(-1)
elif args.experiments < 1:
    parser.error("Illegal number of experiments: " + str(args.ensemble_size))
    sys.exit(-1)


###-----------------------------------------
## Define files for ensemble and truth.
##
ensemble_init_path = os.path.abspath('double_jet_ensemble_init/')
assert len(os.listdir(ensemble_init_path)) == 102, "Ensemble init folder has wrong number of files"

#truth_path = os.path.abspath('double_jet_truth/')
truth_path = os.path.abspath('truth_2019_06_06-09_23_41/')
assert len(os.listdir(truth_path)) == 4, "Truth folder has wrong number of files"


media_dir = args.output_folder
if not os.path.isdir(media_dir):
    print("Output directory does not exist. Please provide an existing folder as --output_folder argument.")
    sys.exit(-1)

timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
destination_dir = os.path.join(media_dir, "rank_histogram_experiments_" +  timestamp + "/")
os.makedirs(destination_dir)


# Define misc filenames
log_file = os.path.join(destination_dir, 'description.txt')



with open(log_file, 'w') as f:
    f.write('Rank histogram experiment ' + timestamp + '\n')
    f.write('----------------------------------------------' + '\n')

    
def log(msg, screen=True):
    with open(log_file, 'a') as f:
        f.write(msg + '\n')
    if screen:
        print(msg)
        
    
def logParams():
    log('Input arguments:')
    for arg in vars(args):
        log('\t' + str((arg, getattr(args, arg))))
    log('Constant arguments:')
    log('\t' + str(const_args))
    log('\nPath to initial conditions for ensemble:')
    log('\t' + ensemble_init_path)
    log('destination folder:')
    log('\t' + destination_dir)
    

logParams()
        
method = const_args['method']
if not method == 'iewpf2':
    log('---> WRONG METHOD: ' + method)
    log('Exiting!')
    sys.exit(-1)
log(' ----> Using IEWPF 2 stage method')


###--------------------------------
# Import required packages
#
tic = time.time()
# For GPU contex:
from SWESimulators import Common
# For the ensemble:
from SWESimulators import EnsembleFromFiles, Observation
# For data assimilation:
from SWESimulators import IEWPFOcean
# For ObservationType:
from SWESimulators import DataAssimilationUtils as dautils

toc = time.time()
log("\n{:02.4f} s: ".format(toc-tic) + 'GPU Ocean packages imported', True)

# Create CUDA context
tic = time.time()
gpu_ctx = Common.CUDAContext()
device_name = gpu_ctx.cuda_device.name()
toc = time.time()
log("{:02.4f} s: ".format(toc-tic) + "Created context on " + device_name, True)


###--------------------------
# Set parameters
#


if not const_args['observation_type'] == 'buoys':
    log('---> WRONG OBSERVATION TYPE: ' + const_args['observation_type'])
    log('Exiting!')
    sys.exit(-1)
    
    
log('Observation type sat to StaticBuoys!')
observation_type = dautils.ObservationType.StaticBuoys



master_tic = time.time()

numHours = 6
forecastHours = 1
x_index =  100
hours_to_store = [1, 6, 7, 12, 18, 24, 48, 72]
ensemble_size = const_args['ensemble_size']
use_lcg = False

log('---------- Starting simulation --------------') 
log('--- numHours:       ' + str(numHours))
log('--- hours_to_store: ' + str(hours_to_store))
log('--- forecastHours:  ' + str(forecastHours))
log('--- use_lcg:        ' + str(use_lcg))
log('---------------------------------------------') 




###-------------------------------------------------
# Functions for creating filenames
# 
def experiment_filename(experiment_id, hour):
    # Generate unique filename
    filename = 'hour_' + str(hour).zfill(3) + '_rank_hist_experiment_' + str(experiment_id).zfill(5) + '.npz'
    return os.path.join(destination_dir, filename)


### ----------------------------------------------
#   RANK HISTOGRAM LOOP
#


for run_id in range(args.experiments):
    
    log('----------- Rank histogram experiment ' + str(run_id) + " ---------------------------")
    
    
    ## TODO: Create new truth
    # Function that takes input: end_time, destination_dir
    #               and returns: truth_dir
    
    ## TODO: Random initial ensemble 
    #  Add functionality to sample initial ensemble 
    #  if ensemble_size < num_files in ensemble_init_path.
    
    ## TODO: Perturb truth. 
    #  In the truth, add a new column as perturbed_observation
    #  and use a boolean parameter in get_observation to select 
    #  perturbed or true observations
    
    ### Initialize ensemble
    tic = time.time()
    ensemble = EnsembleFromFiles.EnsembleFromFiles(gpu_ctx, ensemble_size, \
                                                   ensemble_init_path, truth_path, \
                                                   const_args['observation_variance'],
                                                   cont_write_netcdf = False,
                                                   use_lcg = use_lcg,
                                                   observation_type=observation_type)

    # Configure observations according to the selected drifters/buoys:
    ensemble.configureObservations(observationInterval = const_args['observation_interval'],
                                   buoy_area = const_args['buoy_area'])
    
    toc = time.time()
    log("{:02.4f} s: ".format(toc-tic) + "Ensemble is loaded and created", True)
    
    
    # Initialize IEWPF class (if needed)
    tic = time.time()
    iewpf = IEWPFOcean.IEWPFOcean(ensemble)
    toc = time.time()
    log("{:02.4f} s: ".format(toc-tic) + "Data assimilation class initiated", True)
    
    obstime = 3*24*60*60

    da_tic = time.time()
    
    ### DATA ASSIMILATION 
    log('-------- Starting data assimilation ')

    for hour in range(numHours+forecastHours):
        time_in_hours = hour + 1
        
        for fiveMin in range(12):

            drifter_cells = ensemble.getDrifterCells()
                
            for minute in range(5):
                obstime += 60

                forecast_instead_of_da = (time_in_hours in hours_to_store and fiveMin == 11) or time_in_hours > numHours
                apply_model_error = minute < 4 or forecast_instead_of_da

                ensemble.stepToObservation(obstime, model_error_final_step=apply_model_error)

                if minute == 4 and not forecast_instead_of_da:
                    iewpf.iewpf_2stage(ensemble, perform_step=False)
                
                #ensemble.registerStateSample(drifter_cells)
            # Done minutes

        # Done five minutes
                
        toc = time.time()
        log("{:04.1f} s: ".format(toc-da_tic) + " Done simulating hour " + str(time_in_hours))
        
        num_active_particles = ensemble.getNumActiveParticles()
        if num_active_particles < ensemble_size:
            log('-------> Found dead particles! Only ' + str(num_active_particles) + ' of ' + str(ensemble_size) + ' active.')
            outfile = experiment_filename(run_id, time_in_hours)
            outfile = outfile.replace('.npz', '.txt')
            np.savetxt(outfile, [obstime])
            break 
        
        if time_in_hours in hours_to_store:
            log('-------> Storing ensemble ensemble data at hour ' + str(time_in_hours))

            outfile = experiment_filename(run_id, time_in_hours)
            log('-------> outfile: ' + outfile)
            
            hu  = np.zeros((ensemble.ny, ensemble_size))
            hv  = np.zeros((ensemble.ny, ensemble_size))
            eta = np.zeros((ensemble.ny, ensemble_size))

            ### Store the results
            for particle_id in range(ensemble_size):

                p_eta, p_hu, p_hv  = ensemble.downloadParticleOceanState(particle_id)

                hu[:, particle_id]  = p_hu[:, x_index]
                hv[:, particle_id]  = p_hv[:, x_index]
                eta[:, particle_id] = p_eta[:, x_index]

            ## TODO: Store truth
            # make hu_truth, hv_truth, eta_truth and store along hu, hv and eta.
            
            ## TODO: Use stored truth
            # In the rank histogram, use stored truth if avilable in each file. 

            np.savez(outfile, hu=hu, hv=hv, eta=eta, t=obstime)
        # end if time_in_hours in hours_to_store
    
    # Done hours
    toc = time.time()
    log("{:04.1f} s: ".format(toc-da_tic) + " Rank histogram experiment stored at time " + str(obstime))


    # Clean up simulation and close netcdf file
    tic = time.time()
    sim = None
    ensemble.cleanUp()
    toc = time.time()
    print("\n{:02.4f} s: ".format(toc-tic) + "Clean up simulator done.")
    print("{:05.1f} s".format(toc-master_tic) + " since starting the program.")

log('Done! Only checking is left. There should be a "yes, done" in the next line')


assert(numHours == 6), 'Simulated with wrong number of hours'
assert(forecastHours == 1), 'Simulated with the wrong forecast hours'
#assert(obstime == 4*24*60*60), 'Forecast did not reach goal time'

log('Yes, done!')


exit(0)




