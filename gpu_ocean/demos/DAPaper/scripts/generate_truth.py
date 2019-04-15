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

import sys, os, json, datetime
current_dir = os.path.dirname(os.path.realpath(__file__))

if os.path.isdir(os.path.abspath(os.path.join(current_dir, '../../../SWESimulators'))):
        sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../../../')))


        
#--------------------------------------------------------------
# PARAMETERS
#--------------------------------------------------------------
# This file takes no parameters, as it is should clearly define a specific truth.
# If we come to a time where we need to specify a lot of different truths, we can introduce argparser again.

# Time parameters
start_time      =  3*24*60*60 #  3 days
simulation_time = 10*24*60*60 # 10 days (three days spin up is prior to this)
end_time        = 13*24*60*60 # 13 days
observation_frequency = 5*60  # 5 minutes
netcdf_frequency = 60*60      # every hour

# Drifter parameters:
num_drifters = 64

# File parameters:
folder = "truth_" + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + "/" 
netcdf_filename  = folder + "double_jet_case_truth.nc"
drifter_filename = folder + "drifter_observations.pickle"








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
from SWESimulators import CDKLM16, Common, DoubleJetCase, GPUDrifterCollection, Observation

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
print("\n{:02.4f} s: ".format(toc-tic) + "Generated initial conditions for truth")


#netcdf_args = {}
netcdf_args = {'write_netcdf': True, 'netcdf_filename': netcdf_filename}

tic = time.time()
sim = CDKLM16.CDKLM16(**doubleJetCase_args, **doubleJetCase_init, **netcdf_args)

toc = time.time()
print("\n{:02.4f} s: ".format(toc-tic) + "Truth simulator initiated")

t = sim.t
print("Three days = " + str(start_time) + " seconds, but sim.t = " + str(sim.t) + ". Same? " + str(start_time == sim.t))



#--------------------------------------------------------------
# Create drifters at t = 3 days
#--------------------------------------------------------------
drifters = GPUDrifterCollection.GPUDrifterCollection(gpu_ctx, num_drifters,
                                                     boundaryConditions=doubleJetCase_args['boundary_conditions'],
                                                     domain_size_x=sim.nx*sim.dx, domain_size_y=sim.ny*sim.dy)
sim.attachDrifters(drifters)



#--------------------------------------------------------------
# Create observation object and write the first drifter positions
#--------------------------------------------------------------
observations = Observation.Observation()
observations.add_observation_from_sim(sim)

netcdf_iterations = int(simulation_time/netcdf_frequency)
observation_sub_iterations = int(netcdf_frequency/observation_frequency)

print("We will now make " + str(netcdf_iterations) + " iterations with writing netcdf, and  " + \
      str(observation_sub_iterations) + " subiterations with registering drifter positions.")

theoretical_time = start_time


########################
# Main simulation loop #
########################
tic = time.time()

time_error = 0

for netcdf_it in range(netcdf_iterations):
    for observation_sub_it in range(observation_sub_iterations):

        next_obs_time = t + observation_frequency
        
        # Step until next observation 
        sim.dataAssimilationStep(next_obs_time, write_now=False)
        
        # Store observation
        observations.add_observation_from_sim(sim)
        
        t = sim.t
        
        time_error += t-next_obs_time
        theoretical_time += observation_frequency
      
    sim.writeState()
    
    if netcdf_it % 10 == 0:
        
        # Check that everything looks okay
        eta, hu, hv = sim.download()
        if (np.any(np.isnan(eta))):
            print(" `-> ERROR: Not a number in spinup, aborting!")
            sys.exit(-1)
        
        sub_t = time.time() - tic
        print("{:02.4f} s into loop: ".format(sub_t) + "Done with netcdf iteration " + str(netcdf_it) + " of " + str(netcdf_iterations))
        
        
        
toc = time.time()
print("{:02.4f} s: ".format(toc-tic) + "Done with generating thruth")

print("sim.t:            " + str(sim.t))
print("end_time:         " + str(end_time))


########################
# Simulation loop DONE #
########################
        
# Dump drifter observations to file:
tic = time.time()
observations.to_pickle(drifter_filename)
toc = time.time()
print("\n{:02.4f} s: ".format(toc-tic) + "Drifter observations written to " + drifter_filename)

# Clean up simulation and close netcdf file
tic = time.time()
sim.cleanUp()
toc = time.time()
print("\n{:02.4f} s: ".format(toc-tic) + "Clean up simulator done.")





