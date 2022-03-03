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


if host == "ppi":
    media_dir='/lustre/storeB/users/florianb/forecasting_results/'
elif host == "havvarsel":
    media_dir='/sintef/forecasting_results/'
else:
    print("Unknown host")
    sys.exit(0)


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
destination_dir = os.path.join(media_dir, "performance_experiment_" +  timestamp + "/")
os.makedirs(destination_dir)


###--------------------------------
# Import required packages
#
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
# For ObservationType:
from SWESimulators import DataAssimilationUtils as dautils

# Create CUDA context
gpu_ctx = Common.CUDAContext()
device_name = gpu_ctx.cuda_device.name()

#----------------------------------------------------
# Performance set-up

methods = ["MC", "IEWPF2", "LETKF"]
ensemble_sizes = [50, 100, 250]

numDays = 1 
numHours = 1
numFiveMin = 3

performance = np.zeros((3,3,numDays*numHours))

for m in range(len(methods)):
    method = methods[m].lower()
    print("----------------------------------")
    print(time.strftime("%H:%M:%S",time.gmtime()) + ": Running " + method)
    for e in range(len(ensemble_sizes)):
        ensemble_size = ensemble_sizes[e]
        ###--------------------------
        # Initiate the ensemble
        print(time.strftime("%H:%M:%S",time.gmtime()) + ": Ensemble size " + str(ensemble_size))
        print("Initializing ensemble...")

        ensemble = EnsembleFromFiles.EnsembleFromFiles(gpu_ctx, 
                                                        ensemble_size,                                                
                                                        ensemble_init_path, 
                                                        truth_path,                                                
                                                        1.0,
                                                        cont_write_netcdf = False,
                                                        use_lcg = False, xorwow_seed = np.random.randint(1,10000),
                                                        write_netcdf_directory = destination_dir,
                                                        observation_type=dautils.ObservationType.StaticBuoys)


        ### -------------------------------
        # Initialize DA class (if needed)
        print("Initializing data assimilation...")
        if method.startswith('iewpf'):
            iewpf = IEWPFOcean.IEWPFOcean(ensemble)
            print("... IEWPF loaded")
        elif method.startswith('etkf') or method.startswith('letkf'):
            etkf = ETKFOcean.ETKFOcean(ensemble)
            print("... LETKF loaded")
        else:
            print("... MC does not require a DA class.")

        ### ----------------------------------------------
        #   DATA ASSIMILATION
        #
        print(time.strftime("%H:%M:%S", time.gmtime()) + ": Starting simulation")
        h = 0

        obstime = 3*24*60*60 # time in seconds (starting after spin-up phase)
        for day in range(numDays):
            
            gpu_ctx.synchronize()    
            tic = time.time()
            
            for hour in range(numHours):
                
                for fiveMin in range(numFiveMin):
                                    
                    for minute in range(5):
    
                        obstime += 60
                        if method == 'iewpf2':
                            ensemble.stepToObservation(obstime, model_error_final_step=(minute<4))
                            if minute == 4:
                                iewpf.iewpf_2stage(ensemble, perform_step=False)
                        else:
                            ensemble.stepToObservation(obstime)
                            if minute == 4:
                                if method == 'etkf':
                                    etkf.ETKF(ensemble)
                                if method == 'letkf':
                                    etkf.LETKF(ensemble)
                    print("Simulation at " + time.strftime("%j %H:%M:%S", time.gmtime(obstime)))
                    # Done minute
                # Done five minutes
            # Done hour
            gpu_ctx.synchronize()    
            toc = time.time()

            performance[m,e,h] = toc - tic
            h = h+1 
                
        print("Cleaning...")
        # Clean up simulation and close netcdf file
        ensemble.cleanUp()


np.save(destination_dir+"computation_times", performance)
