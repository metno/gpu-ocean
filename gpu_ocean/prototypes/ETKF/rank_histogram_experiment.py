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
import subprocess

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
parser.add_argument('--output_folder', type=str, default="rank_histogram_experiments")
parser.add_argument('--method', type=str, default='ETKF')
parser.add_argument('--observation_variance', type=float, default=1)

const_args = {
    'ensemble_size' : 40,
    #'method' : 'iewpf2',
    'observation_interval' : 1,
    'observation_type' : 'buoys',
    'buoy_area' : 'all'
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
ensemble_init_path = os.path.abspath('../presented_data/ensemble_init/')
assert len(os.listdir(ensemble_init_path)) == 100 or len(os.listdir(ensemble_init_path)) == 101, \
    "Ensemble init folder has wrong number of files: " + str(len(os.listdir(ensemble_init_path)))

truth_path = os.path.abspath('../presented_data/true_state/')
assert len(os.listdir(truth_path)) == 2 or len(os.listdir(truth_path)) == 3, \
    "Truth folder has wrong number of files"

media_dir = args.output_folder
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
        
# Reading and checking method
method = str(args.method).lower()
if method == 'iewpf2':
    log(' ----> Using IEWPF 2 stage method')
elif method == 'etkf':
    log(' ----> Using IEWPF 2 stage method')
elif method == 'none':
    log(' ----> No data assimilation')
else:
    log('Illegal method: ' + str(method))
    sys.exit(-1)
    
###------------------------------
## Git info
##
try:
    git_hash = str.strip(str(subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf8")[0:-1]))
    git_branch = str.strip(str(subprocess.check_output(["git","symbolic-ref", "--short", "HEAD"]).decode("utf8")[0:-1]))
except:
    git_hash = "git info missing..."
    git_branch = "git branch missing..."
    
log('------Git info---------')
log('Branch: ' + str(git_branch))
log('Hash:   ' + str(git_hash))
log('--------------------------')



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
import ETKF
# For ObservationType:
from SWESimulators import DataAssimilationUtils as dautils
# For generating new truths
from SWESimulators import DoubleJetExperimentUtils as djeutils

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
use_lcg = True
randomize_initial_ensemble = True

log('---------- Starting simulation --------------') 
log('--- numHours:       ' + str(numHours))
log('--- hours_to_store: ' + str(hours_to_store))
log('--- forecastHours:  ' + str(forecastHours))
log('--- use_lcg:        ' + str(use_lcg))
log('--- randomize_initial_ensemble: ' + str(randomize_initial_ensemble))
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
    
    
    ### Generating a new truth
    tic = time.time()
    truth_name = 'truth_run_' + str(run_id).zfill(4) + '_attempt_'
    truth_path = None
    truth_attempt = 0
    while truth_path is None:
        log('Generating truth - attempt ' + str(truth_attempt))
        try:
            truth_path = djeutils.generateTruth(gpu_ctx, destination_dir,
                                                duration_in_days=3,
                                                duration_in_hours=numHours+forecastHours,
                                                folder_name=truth_name+str(truth_attempt))
        except Exception as e:
            truth_attempt += 1
            log('Got exception in truth: ' + str(e))
            
    toc = time.time()
    log("{:02.4f} s: ".format(toc-tic) + "Generated truth " + str(truth_path))
            
    
    
    ## TODO: Perturb truth. 
    #  In the truth, add a new column as perturbed_observation
    #  and use a boolean parameter in get_observation to select 
    #  perturbed or true observations
    
    ### Initialize ensemble
    tic = time.time()
    ensemble = EnsembleFromFiles.EnsembleFromFiles(gpu_ctx, ensemble_size, \
                                                   ensemble_init_path, truth_path, \
                                                   args.observation_variance,
                                                   cont_write_netcdf = False,
                                                   use_lcg = use_lcg,
                                                   observation_type=observation_type,
                                                   randomize_initial_ensemble=randomize_initial_ensemble)

    # Configure observations according to the selected drifters/buoys:
    ensemble.configureObservations(observationInterval = const_args['observation_interval'],
                                   buoy_area = const_args['buoy_area'])
    
    toc = time.time()
    log("{:02.4f} s: ".format(toc-tic) + "Ensemble is loaded and created", True)
    
    
    # Initialize IEWPF class (if needed)
    if method == 'iewpf2':
        tic = time.time()
        iewpf = IEWPFOcean.IEWPFOcean(ensemble)
        toc = time.time()
        log("{:02.4f} s: ".format(toc-tic) + "Data assimilation class initiated", True)
    elif method == 'etkf':
        tic = time.time()
        etkf = ETKFOcean.ETKFOcean(ensemble)
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
                apply_model_error = minute < 4 or forecast_instead_of_da or method=='none'

                if method == 'iewpf2':
                    ensemble.stepToObservation(obstime, model_error_final_step=apply_model_error)
                    if minute == 4 and not forecast_instead_of_da:
                        iewpf.iewpf_2stage(ensemble, perform_step=False)
                elif method == 'etkf':
                    ensemble.stepToObservation(obstime)
                    if minute == 4 and not forecast_instead_of_da :
                        etkf.etkf(ensemble)
                
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

            
            ## TODO: Add observed truth to npz file
            observations = ensemble.observeTrueState()
            observation_variance = ensemble.getObservationVariance()
            
            ### Store truth
            true_eta, true_hu, true_hv, true_t = ensemble.true_state_reader.getTimeStep(time_in_hours)
            true_eta = true_eta[:, x_index]
            true_hu  = true_hu[:, x_index]
            true_hv  = true_hv[:, x_index]
            
            np.savez(outfile, t=obstime,
                     eta=eta, hu=hu, hv=hv,
                     true_eta=true_eta, true_hu=true_hu, true_hv=true_hv,
                     observations=observations, observation_variance=observation_variance)
        # end if time_in_hours in hours_to_store
    
    # Done hours
    toc = time.time()
    log("{:04.1f} s: ".format(toc-da_tic) + " Rank histogram experiment stored at time " + str(obstime))
    
    
    # Clean up simulation and close netcdf file
    tic = time.time()
    sim = None
    ensemble.cleanUp()
    toc = time.time()
    log("\n{:02.4f} s: ".format(toc-tic) + "Clean up simulator done.")
    log("{:07.1f} s".format(toc-master_tic) + " since starting the program.")

log('Done! Only checking is left. There should be a "yes, done" in the next line')


assert(numHours == 6), 'Simulated with wrong number of hours'
assert(forecastHours == 1), 'Simulated with the wrong forecast hours'
#assert(obstime == 4*24*60*60), 'Forecast did not reach goal time'

log('Yes, done!')


exit(0)




