# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2018-2019 SINTEF Digital
Copyright (C) 2018-2019 Norwegian Meteorological Institute

This python program generate a truth simulation that will be the subject for 
data assimilation experiments. It is based on the DoubleJetCase parameters 
and initial conditions, and is spun up for 3 days before starting to write
its state to file. The generated data set should cover time range from
day 3 to day 13.

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

import sys, os, json
current_dir = os.path.dirname(os.path.realpath(__file__))

if os.path.isdir(os.path.abspath(os.path.join(current_dir, '../../../SWESimulators'))):
        sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../../../')))

# This file takes no parameters, as it is should clearly define a specific truth.
# If we come to a time where we need to specify a lot of different truths, we can introduce argparser again.

# Parameters:
num_drifters = 100
simulation_time = 10*24*60*60 # 10 days (three days spin up is prior to this)
observation_frequency = 5*60 # 5 minutes
netcdf_frequency = 10*24 # every hour

folder = "truth_" + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + "/" 
netcdf_filename  = folder + "double_jet_case_truth.nc"
drifter_filename = folder + "drifter_observations.json"

if not os.path.isdir(folder):
    os.makedirs(folder)
else:
    print(" `-> ERROR: The folder " + folder + " already exists!")
    sys.exit(-1)    


print("------ Generating initial ensemble ---------------")
print("Writing truth to file: " + netcdf_filename)


# Import timing utilities
import time
tic = time.time();

# Import packages we need
import numpy as np
from SWESimulators import CDKLM16, Common, DoubleJetCase, GPUDrifterCollection

#

toc = time.time()
print("{:02.4f} s: ".format(toc-tic) + "Imported packages")

# Create CUDA context
tic = time.time()
gpu_ctx = Common.CUDAContext()
device_name = gpu_ctx.cuda_device.name()
toc = time.time()
print("{:02.4f} s: ".format(toc-tic) + "Created context on " + device_name)



#--------------------------------------------------------------
# Creating the Case (including spin up)
#--------------------------------------------------------------
print("Initializing the truth")

tic = time.time()
sim = None

doubleJetCase = DoubleJetCase.DoubleJetCase(gpu_ctx,
                                            DoubleJetCase.DoubleJetPerturbationType.IEWPFPaperCase)

doubleJetCase_args, doubleJetCase_init = doubleJetCase.getInitConditions()

if (np.any(np.isnan(doubleJetCase_init["eta0"]))):
    print(" `-> ERROR: Not a number in spinup, aborting!")
    sys.exit(-1)

toc = time.time()
print("\n{:02.4f} s: ".format(toc-tic) + "Generated initial conditions at day 3")





#--------------------------------------------------------------
# Initialize the truth from the Case (spin up is done in the Case)
#--------------------------------------------------------------


toc = time.time()
print("\n{:02.4f} s: ".format(toc-tic) + "Generated initial conditions for member " + str(ensemble_member))

netcdf_args = {'write_netcdf': True, 'netcdf_filename': netcdf_filename}

tic = time.time()
sim = CDKLM16.CDKLM16(**doubleJetCase_args, **doubleJetCase_init, **netcdf_args)

toc = time.time()
print("\n{:02.4f} s: ".format(toc-tic) + "Truth simulator initiated " + str(ensemble_member))



#### TODO: Set up the loop that runs dataAssimilationStep
####       while writing drifters to json-file, 
####       and writing state to netcdf-file.






sys.exit(0)



#
# Initialize and spinup all ensemble members
#
for n in range(args.members-1,-1,-1):
    tic = time.time()
    
    # Generate parameters and initial conditions
    toc = time.time()
    print("\n{:02.4f} s: ".format(toc-tic) + "Generated initial conditions")


    doubleJetCase = DoubleJetCase.DoubleJetCase(gpu_ctx,
                                                DoubleJetCase.DoubleJetPerturbationType.IEWPFPaperCase)
    
    doubleJetCase_args, doubleJetCase_init = doubleJetCase.getInitConditions()

    sim = initCDKLM(n)
    
    tic = time.time()


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

drifter_obs_formatted_all = {}

# initial observation here!
drifter_obs = drifters.getDrifterPositions()
drifter_obs_formatted = {}
for i,d in enumerate(drifter_obs):
    drifter_obs_formatted[i] = d.tolist()

drifter_obs_formatted_all[sim.t] = drifter_obs_formatted

t = sim.t
while t < end_time:
    gpu_ctx.synchronize()
    tic = time.time()

    sim.dataAssimilationStep(t + args.observe_every)
    t += sim.t

    gpu_ctx.synchronize()
    toc = time.time()

    tic = time.time()
    eta1, u1, v1 = sim.download()
    toc = time.time()
    print(" `-> {:02.4f} s: ".format(toc-tic) + "Download")
    print(" '->max(u): " + str(np.max(u1)))

    if (np.any(np.isnan(eta1))):
        print(" `-> ERROR: Not a number in simulation, aborting!")
        sys.exit(-1)

    print(" `-> t_sim={:02.4f}".format(t) + ", u_max={:02.4f}".format(np.max(u1)))
    
    # observe here!
    drifter_obs = drifters.getDrifterPositions()
    drifter_obs_formatted = {}
    for i,d in enumerate(drifter_obs):
        drifter_obs_formatted[i] = d.tolist()

    drifter_obs_formatted_all[sim.t] = drifter_obs_formatted

# TODO: add meaningful id to filename or content
with open('drifter_obs.json', 'w') as f:
    json.dump(drifter_obs_formatted_all, f, sort_keys=True, indent=4, ensure_ascii=False, separators=(',',':'))
    f.write("\n\n")
