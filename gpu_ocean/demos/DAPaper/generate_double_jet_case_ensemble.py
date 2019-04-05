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

import sys, os
current_dir = os.path.dirname(os.path.realpath(__file__))

if os.path.isdir(os.path.abspath(os.path.join(current_dir, '../../SWESimulators'))):
        sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../../')))
if os.path.isdir(os.path.abspath(os.path.join(current_dir, '../../../python/SWESimulators'))):
        sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../../../python/')))

import argparse
parser = argparse.ArgumentParser(description='Generate an ensemble.')
parser.add_argument('--members', type=int, default=10)
parser.add_argument('--spinup', type=int, default=259200) # default: 3 days spinup
parser.add_argument('--initialization_time', type=int, default=604800) # default: forecast period starts at day 10 (including spinup)
parser.add_argument('--forecast_length', type=int, default=259200) # default: forecast length of 3 days
parser.add_argument('--nx', type=int, default=512)
parser.add_argument('--ny', type=int, default=512)
parser.add_argument('--block_width', type=int)
parser.add_argument('--block_height', type=int)
parser.add_argument('--drifters', type=int, default=10)
parser.add_argument('--steps_per_download', type=int, default=100)
parser.add_argument('--steps_per_observation', type=int, default=100)
parser.add_argument('--simulator', type=str, default="CDKLM")

args = parser.parse_args()

if(args.steps_per_observation % args.steps_per_download):
    print(" `-> ERROR: Choose steps_per_observation as a multiple of steps_per_download, aborting!")
    sys.exit(-1)

# TODO: Write command/arguments + seed + git commit hash to netCDF-file(s) -> enables recreation of data?
# TODO: Adapt to var dt
# TODO: Convienency function: allow time given in days
# TODO: Write observations to file

print("=== Domain size [{:02d} x {:02d}], block size [{:s} x {:s}] ===".format(args.nx, args.ny, str(args.block_width), str(args.block_height)))

# Import timing utilities
import time
tic = time.time();

# Import packages we need
import numpy as np
from SWESimulators import FBL, CTCS, KP07, CDKLM16, Common, DoubleJetCase, GPUDrifterCollection

toc = time.time()
print("{:02.4f} s: ".format(toc-tic) + "Imported packages")

# Create CUDA context
tic = time.time()
gpu_ctx = Common.CUDAContext()
device_name = gpu_ctx.cuda_device.name()
toc = time.time()
print("{:02.4f} s: ".format(toc-tic) + "Created context on " + device_name)

def initCDKLM(ensemble_member):
    """
    Initializes the CDKLM simulator
    """
    tic = time.time()

    netcdf_filename = 'double_jet_case_' + str(ensemble_member)
    kwargs = {'write_netcdf': True, 'netcdf_filename': netcdf_filename}
    if (args.block_width != None):
        kwargs['block_width'] = args.block_width
    if (args.block_height != None):
        kwargs['block_height'] = args.block_height

    sim = CDKLM16.CDKLM16(**doubleJetCase_args, **doubleJetCase_init, **kwargs)

    toc = time.time()
    print("{:02.4f} s: ".format(toc-tic) + "Created CDKLM simulator")

    return sim

sim = None

#
# Initialize and spinup all ensemble members
#
for n in range(args.members-1,-1,-1):
    tic = time.time()
    # Get parameters and initial conditions
    doubleJetCase = DoubleJetCase.DoubleJetCase(gpu_ctx,
                                            DoubleJetCase.DoubleJetPerturbationType.LowFrequencySpinUp, 
                                            model_error=True, commonSpinUpTime = 0)
    doubleJetCase_args, doubleJetCase_init = doubleJetCase.getInitConditions()
    toc = time.time()
    print("\n{:02.4f} s: ".format(toc-tic) + "Generated initial conditions")

    if (args.simulator == "KP"):
        print("ERROR: Not yet implemented, aborting")
        sys.exit(-1)
    elif (args.simulator == "CDKLM"): 
        sim = initCDKLM(n)
    elif (args.simulator == "FBL"):
        print("ERROR: Not yet implemented, aborting")
        sys.exit(-1)
    elif (args.simulator == "CTCS"):
        print("ERROR: Not yet implemented, aborting")
        sys.exit(-1)
    else:
        print("ERROR: Not yet implemented, aborting")
        sys.exit(-1)

    tic = time.time()

    sim.step(args.spinup)

    eta1, u1, v1 = sim.download()
    toc = time.time()
    print("{:02.4f} s: ".format(toc-tic) + "Spinup of member " + str(n))

    if (np.any(np.isnan(eta1))):
        print(" `-> ERROR: Not a number in spinup, aborting!")
        sys.exit(-1)

print("Spin-up of " + str(args.members) + " members completed!\n")

#
# Continue with member 0 as ground truth: Attach drifters and run until initialization time (forecast time = 0) + forecast length
#

# Define mid-points for the different drifters 
# Decompose the domain, so that we spread the drifters as much as possible
sub_domains_y = np.int(np.round(np.sqrt(args.drifters)))
sub_domains_x = np.int(np.ceil(1.0*args.drifters/sub_domains_y))
midPoints = np.empty((args.drifters, 2))
for sub_y in range(sub_domains_y):
    for sub_x in range(sub_domains_x):
        drifter_id = sub_y*sub_domains_x + sub_x
        if drifter_id >= args.drifters:
            break
        midPoints[drifter_id, 0]  = (sub_x + 0.5)*sim.nx*sim.dx/sub_domains_x
        midPoints[drifter_id, 1]  = (sub_y + 0.5)*sim.ny*sim.dy/sub_domains_y
# (using default observation_variance)
drifters = GPUDrifterCollection.GPUDrifterCollection(gpu_ctx, args.drifters,
                                                 boundaryConditions=doubleJetCase_args['boundary_conditions'],
                                                 domain_size_x=sim.nx*sim.dx, domain_size_y=sim.ny*sim.dy)
initPos = np.empty((args.drifters, 2))
for d in range(args.drifters):
    initPos[d,:] = np.random.multivariate_normal(midPoints[d,:], np.zeros((2,2)))
drifters.setDrifterPositions(initPos)
sim.attachDrifters(drifters)

# initial observation here!
drifter_obs = drifters.getDrifterPositions()
drifter_obs_formatted = {}
for i in drifter_obs:
    drifter_obs_formatted[i[0]]=i[1]

with open('drifter_obs.json', 'w') as f:
    json.dump(drifter_obs_formatted, f, sort_keys=True, indent=4, ensure_ascii=False, separators=(',',':'))

total_iterations = int((args.initialization_time+args.forecast_length) / sim.dt + 1)
max_mcells = 0;

downloads = int(total_iterations / args.steps_per_download)
for i in range(downloads):
    print("{:03.0f} %".format(100*(i+1) / downloads))

    gpu_ctx.synchronize()
    tic = time.time()

    t = sim.step(args.steps_per_download*sim.dt)

    gpu_ctx.synchronize()
    toc = time.time()

    mcells = args.nx*args.ny*args.steps_per_download/(1e6*(toc-tic))
    max_mcells = max(mcells, max_mcells);
    print(" `-> {:02.4f} s: ".format(toc-tic) + "Step, " + "{:02.4f} mcells/sec".format(mcells))
    tic = time.time()
    eta1, u1, v1 = sim.download()
    toc = time.time()
    print(" `-> {:02.4f} s: ".format(toc-tic) + "Download")
    print(" '->max(u): " + str(np.max(u1)))

    if (np.any(np.isnan(eta1))):
        print(" `-> ERROR: Not a number in simulation, aborting!")
        sys.exit(-1)

    print(" `-> t_sim={:02.4f}".format(t) + ", u_max={:02.4f}".format(np.max(u1)))
    
    if(not ((i*args.steps_per_download) % args.steps_per_observation)):
        # observe here!
        print("OBSERVE!")

print(" === Maximum megacells: {:02.8f} ===".format(max_mcells))

print("AFTER")
print(drifters.getDrifterPositions()[0])
