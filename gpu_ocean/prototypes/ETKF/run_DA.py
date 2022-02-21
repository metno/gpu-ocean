# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2019 SINTEF Digital

This python program is used to set up and run a data-assimilation 
and drift trajectory forecasting experiment.

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

current_dir = os.getcwd()

if os.path.isdir(os.path.abspath(os.path.join(current_dir, '../../SWESimulators'))):
        sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../../')))

# Knonw hosts: havvarsel, PPI.
host = os.uname()[1] 
if host in ["r740-5hdn2s2-ag-gcompute", "r740-5hcv2s2-ag-gcompute", "r740-dsxm2t2-ag-gcompute", "r740-dsws2t2-ag-gcompute"]:
    host = "ppi"

#--------------------------------------------------------------
# PARAMETERS
#--------------------------------------------------------------
# Read input parameters and check that they are good

import argparse
parser = argparse.ArgumentParser(description='Generate an ensemble.')
parser.add_argument('-N', '--ensemble_size', type=int, default=100)
parser.add_argument('--method', type=str, default='LETKF')
parser.add_argument('--inflation_factor', type=float, default=1.0)
parser.add_argument('--observation_interval', type=int, default=1)
parser.add_argument('--observation_variance', type=float, default=1.0)
parser.add_argument('--observation_type', type=str, default='buoys')
parser.add_argument('--buoy_area', type=str, default='all')
if host == "ppi":
    parser.add_argument('--media_dir', type=str, default='/lustre/storeB/users/florianb/forecasting_results/')
elif host == "havvarsel":
    parser.add_argument('--media_dir', type=str, default='/sintef/forecasting_results/')
else:
    parser.add_argument('--media_dir', type=str)
parser.add_argument('--num_days', type=int, default=7) 
parser.add_argument('--num_hours', type=int, default=24) 
parser.add_argument('--forecast_days', type=int, default=3)
parser.add_argument('--profiling', action='store_true')

parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--iewpf2beta', type=float, default=None)

args = parser.parse_args()


# Checking input args
if args.ensemble_size is None:
    print("Ensemble size missing, please provide a --ensemble_size argument.")
    sys.exit(-1)
elif args.ensemble_size < 1:
    parser.error("Illegal ensemble size " + str(args.ensemble_size))

profiling = args.profiling


###-----------------------------------------
## Define files for ensemble and truth.
##

if host == "ppi":
    ensemble_init_path = '/lustre/storeB/users/florianb/data/ensemble_init/'
elif host == "havvarsel":
    ensemble_init_path = '/sintef/data/ensemble_init/'
else:
    ensemble_init_path = 'Give path here'
assert len(os.listdir(ensemble_init_path)) == 100 or len(os.listdir(ensemble_init_path)) == 101,\
    "Ensemble init folder has wrong number of files: " + str(len(os.listdir(ensemble_init_path)))

if host == "ppi":
    truth_path = '/lustre/storeB/users/florianb/data/true_state/'
elif host == "havvarsel":
    truth_path = '/sintef/data/true_state/'
else:
    truth_path = 'Give path here'
assert len(os.listdir(truth_path)) == 2 or len(os.listdir(truth_path)) == 3,\
    "Truth folder has wrong number of files"


timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
media_dir = args.media_dir
destination_dir = os.path.join(media_dir, "da_experiment_" +  timestamp + "/")
os.makedirs(destination_dir)

# Copy the truth into the destination folder
shutil.copytree(truth_path, os.path.join(destination_dir, 'truth'))

# Define misc filenames
log_file = os.path.join(destination_dir, 'description.txt')

particleInfoPrefix = os.path.join(destination_dir, 'particle_info_')
forecastFileBase = os.path.join(destination_dir, 'forecast_member_')


with open(log_file, 'w') as f:
    f.write('Data Assimilation experiment ' + timestamp + '\n')
    f.write('----------------------------------------------' + '\n')

def logParams():
    log('Input arguments:')
    for arg in vars(args):
        log('\t' + str((arg, getattr(args, arg))))
    log('\nPath to initial conditions for ensemble:')
    log('\t' + ensemble_init_path)
    log('Path to true state:')
    log('\t' + truth_path)
    log('destination folder:')
    log('\t' + destination_dir)
    log('Path to particle info:')
    log('\t' + particleInfoPrefix)
    log('Path to forecast members:')
    log('\t' + forecastFileBase)

def log(msg, screen=True):
    with open(log_file, 'a') as f:
        f.write(msg + '\n')
    if screen:
        print(msg)
        
logParams()
        
    
# Reading and checking method
method = str(args.method).lower()
if method == 'iewpf2':
    log(' ----> Using IEWPF 2 stage method')
elif method == 'enkf':
    log(' ----> Using EnKF')
elif method == 'etkf':
    log(' ----> Using ETKF')
elif method == 'letkf':
    log(' ----> Using ETKF with localisation')
elif method == 'none':
    log(' ----> No data assimilation')
else:
    log('Illegal method: ' + str(method))
    sys.exit(-1)
    
    
# Time parameters
start_time      =  3*24*60*60 #  3 days in seconds
simulation_time = 10*24*60*60 # 10 days in seconds (three days spin up is prior to this)fa
end_time        = 13*24*60*60 # 13 days in seconds


# Based on truth from June 25th 2019
drifterSet = [ 2, 7, 12, 24, 29, 35, 41, 48, 53, 60]
#drifterSet = [ 2, 24, 60]

# Log extra information for the ensemble state for the following cells:
extraCells = np.array([[254, 241], # Cross with two trajectories
                       [249, 246], # northwest of above
                       [259, 236], # southeast of above
                       [343, 131], # Closed circle of same drifter
                       [196,  245], # Middle of single trajectory
                       [150,  250], # Middle of single trajectory, later than above
                       [102, 252], # On the same trajectory as the above, but later, and also in a intersection
                       [ 388, 100], # Unobserved area just north of southern jet
                       [ 388, 80],  # Unobserved area in southern jet
                       [ 388, 150], # Unobserved area in calm area
                      ])


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
# The prototyping packages:
#import EnKFOcean #(different prototype folder)
import ETKFOcean
import SkillScore
# For forcasting:
from SWESimulators import GPUDrifterCollection
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
# Initiate the ensemble
#

observation_type = dautils.ObservationType.UnderlyingFlow
if args.observation_type == 'buoys':
    observation_type = dautils.ObservationType.StaticBuoys
    log('Observation type changed to StaticBuoys!')
elif args.observation_type == 'all_drifters':
    drifterSet = 'all'
    log('Using all drifters for DA experiment')

print(observation_type)

cont_write_netcdf = True and not profiling

tic = time.time()
ensemble = EnsembleFromFiles.EnsembleFromFiles(gpu_ctx, 
                                                args.ensemble_size,                                                
                                                ensemble_init_path, 
                                                truth_path,                                                
                                                args.observation_variance,
                                                cont_write_netcdf = cont_write_netcdf,
                                                use_lcg = False, xorwow_seed = args.seed,
                                                write_netcdf_directory = destination_dir,
                                                observation_type=observation_type)

# Configure observations according to the selected drifters:
ensemble.configureObservations(drifterSet=drifterSet, 
                               observationInterval = args.observation_interval,
                               buoy_area = args.buoy_area)
ensemble.configureParticleInfos(extraCells)
toc = time.time()
log("{:02.4f} s: ".format(toc-tic) + "Ensemble is loaded and created", True)
log("Using drifterSet:\n" + str(drifterSet))
if args.observation_type == 'buoys':
    log('buoys to read:')
    log(str(ensemble.observations.read_buoy))


dt_ref = ensemble.particles[-1].dt


### -------------------------------
# Initialize DA class (if needed)
#
tic = time.time()
iewpf = None
if method.startswith('iewpf'):
    iewpf = IEWPFOcean.IEWPFOcean(ensemble, beta=args.iewpf2beta, write_betas=True)
    toc = time.time()
    log("{:02.4f} s: ".format(toc-tic) + "Data assimilation class IEWPFOcean initiated", True)
elif method.startswith('enkf'):
    enkf = EnKFOcean.EnKFOcean(ensemble, args.inflation_factor)
    toc = time.time()
    log("{:02.4f} s: ".format(toc-tic) + "Data assimilation class EnKFOcean initiated", True)
elif method.startswith('etkf') or method.startswith('letkf'):
    etkf = ETKFOcean.ETKFOcean(ensemble, args.inflation_factor)
    toc = time.time()
    log("{:02.4f} s: ".format(toc-tic) + "Data assimilation class ETKFOcean initiated", True)
else:
    toc = time.time()
    log("{:02.4f} s: ".format(toc-tic) + "Skipping creation of a DA class", True)

scores = ["MSE", "bias", "CRPS"]
skillScore = SkillScore.SkillScore(ensemble, scores)

### ----------------------------------------------
#   DATA ASSIMILATION
#

obstime = start_time # time in seconds (starting after spin-up phase)

master_tic = time.time()

numDays = args.num_days 
numHours = args.num_hours 
forecast_days = args.forecast_days


log('---------- Starting simulation --------------') 
log('--- numDays:       ' + str(numDays))
log('--- numHours:      ' + str(numHours))
log('--- forecast_days: ' + str(forecast_days))
log('---------------------------------------------') 

for day in range(numDays):
    log('-------- Starting day ' + str(day))
    
    for hour in range(numHours):
        
        for fiveMin in range(12):
            
            drifter_cells = ensemble.getDrifterCells()
            
            for minute in range(5):
                obstime += 60
                if method == 'iewpf2':
                    ensemble.stepToObservation(obstime, model_error_final_step=(minute<4))
                    if minute == 4:
                        skillScore.assess(ensemble, perturb=True)
                        iewpf.iewpf_2stage(ensemble, perform_step=False)
                else:
                    ensemble.stepToObservation(obstime)
                    if minute == 4:
                        skillScore.assess(ensemble)
                        if method == 'enkf':
                            enkf.EnKF(ensemble)
                        if method == 'etkf':
                            etkf.ETKF(ensemble)
                        if method == 'letkf':
                            etkf.LETKF(ensemble)
                    

                ensemble.registerStateSample(drifter_cells)

                ensemble.deactivateDegeneratedParticles(0.5*dt_ref, 1.5*dt_ref)

                # Done minutes

        # Done five minutes
    
        toc = time.time()
        log("{:04.1f} s: ".format(toc-master_tic) + " Done simulating hour " + str(hour + 1) + " of day " + str(day + 3))
    # Done hours

    ensemble.dumpParticleInfosToFile(particleInfoPrefix)
    
    ensemble.writeEnsembleToNetCDF()

if method == "iewpf2":
    np.savetxt(os.path.join(destination_dir, 'IEWPF2betas.txt'), iewpf.betas)
    
# Done days
avg_scores = skillScore.evaluate(destination_dir)
for score in scores: 
    log("Skill score (" + score + ") = " + str(avg_scores[score]))


### -------------------------------------------------
#   Start forecast
#


log('-----------------------------------------------------------')
log('-----------   STARTING FORECAST              --------------')
log('-----------------------------------------------------------')

forecast_start_time = obstime

# Read all drifters (even those that are not used in the assimilation)
drifter_start_positions = ensemble.observeTrueDrifters(applyDrifterSet=False, ignoreBuoys=True)
num_drifters = len(drifter_start_positions)

forecast_end_time = forecast_start_time + forecast_days*numHours*60*60

observation_intervals = 5*60
netcdf_intervals = numHours*60*60

netcdf_iterations = int((forecast_end_time - forecast_start_time)/netcdf_intervals)
observations_iterations = int(netcdf_intervals/observation_intervals)


for particle_id in range(ensemble.getNumParticles()):
    
    if ensemble.particlesActive[particle_id]:

        sim = ensemble.particles[particle_id]

        tic = time.time()
        next_obs_time = sim.t


        drifters = GPUDrifterCollection.GPUDrifterCollection(gpu_ctx, num_drifters,
                                                             boundaryConditions=ensemble.getBoundaryConditions(), 
                                                             domain_size_x=ensemble.getDomainSizeX(), domain_size_y=ensemble.getDomainSizeY())
        drifters.setDrifterPositions(drifter_start_positions)
        sim.attachDrifters(drifters)

        forecast_file_name = forecastFileBase + str(particle_id).zfill(4) + ".bz2"

        observations = Observation.Observation()
        observations.add_observation_from_sim(sim)

        for netcdf_it in range(netcdf_iterations):

            for obs_it in range(observations_iterations):
                next_obs_time += observation_intervals

                # Step until next observation 
                sim.dataAssimilationStep(next_obs_time, write_now=False)

                # Store observation
                observations.add_observation_from_sim(sim)

            sim.writeState()

        # Write forecast to file    
        observations.to_pickle(forecast_file_name)

        toc = time.time()
        log("{:04.1f} s: ".format(toc-tic) + " Forecast for particle " + str(particle_id) + " done")
        log("      Forecast written to " + forecast_file_name)
    
    else:
        log("Skipping forecast for particle " + str(particle_id) + ", as this particle is dead")



# Clean up simulation and close netcdf file
tic = time.time()
sim = None
ensemble.cleanUp()
toc = time.time()
print("\n{:02.4f} s: ".format(toc-tic) + "Clean up simulator done.")

log('Done! Only checking is left. There should be a "yes, done" in the next line')

if not profiling:
    assert(numDays == 7), 'Simulated with wrong number of days!'
    assert(numHours == 24), 'Simulated with wrong number of hours'
    assert(forecast_end_time == 13*24*60*60), 'Forecast did not reach goal time'

log('Yes, done!')



