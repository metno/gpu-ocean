# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2018-2019 SINTEF Digital
Copyright (C) 2018-2019 Norwegian Meteorological Institute

This python program initializes a given number of ensemble members
based on the DoubleJetCase parameters and initial conditions.

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

import sys, os, json, datetime
current_dir = os.path.dirname(os.path.realpath(__file__))

if os.path.isdir(os.path.abspath(os.path.join(current_dir, '../../../SWESimulators'))):
        sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../../../')))

import argparse
parser = argparse.ArgumentParser(description='Generate an ensemble.')
parser.add_argument('-N', '--ensemble_size', type=int, default=10)

args = parser.parse_args()


# Checking input args
if args.ensemble_size > 100:
    print("Are you really sure that you want to generate more than 100 ensemble members?")
    print("They will nonethereless look similar to each other, and two identical members " + \
          "will still diverge due to model errors soon enough.")
    sys.exit(-1)
elif args.ensemble_size < 1:
    parser.error("Illegal ensemble size " + str(args.ensemble_size))

    
# Define suitable folder for creating the ensemble in
folder_name = "ensemble_init_" + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + "/"
print("------ Generating initial ensemble ---------------")
print("Writing ensemble to director: " + folder_name)
print("Making " + str(args.ensemble_size) + " ensemble members")



# Import timing utilities
import time
tic = time.time();

# Import packages we need
import numpy as np
from SWESimulators import CDKLM16, Common, DoubleJetCase, GPUDrifterCollection

toc = time.time()
print("{:02.4f} s: ".format(toc-tic) + "Imported packages")



# Create CUDA context
tic = time.time()
gpu_ctx = Common.CUDAContext()
device_name = gpu_ctx.cuda_device.name()
toc = time.time()
print("{:02.4f} s: ".format(toc-tic) + "Created context on " + device_name)


sim = None

#
# Initialize and spinup all ensemble members
#
for ensemble_member in range(args.ensemble_size):
    
    print("Creating ensemble member " + str(ensemble_member))
    tic = time.time()
    
    # Generate parameters and initial conditions (which includes spin up time)
    doubleJetCase = DoubleJetCase.DoubleJetCase(gpu_ctx,
                                                DoubleJetCase.DoubleJetPerturbationType.IEWPFPaperCase)
    
    doubleJetCase_args, doubleJetCase_init = doubleJetCase.getInitConditions()

    if (np.any(np.isnan(doubleJetCase_init["eta0"]))):
        print(" `-> ERROR: Not a number in spinup, aborting!")
        sys.exit(-1)
    
    toc = time.time()
    print("\n{:02.4f} s: ".format(toc-tic) + "Generated initial conditions for member " + str(ensemble_member))

    netcdf_filename = folder_name + 'double_jet_case_' + str(ensemble_member).zfill(2)
    netcdf_args = {'write_netcdf': True, 'netcdf_filename': netcdf_filename}

    tic = time.time()
    sim = CDKLM16.CDKLM16(**doubleJetCase_args, **doubleJetCase_init, **netcdf_args)
    sim.cleanUp()
    
    toc = time.time()
    print("\n{:02.4f} s: ".format(toc-tic) + "Generated NetCDF file for member " + str(ensemble_member))

    
    
print("Spin-up of " + str(args.ensemble_size) + " members completed!\n")
