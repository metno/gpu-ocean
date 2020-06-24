# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2018-2019 SINTEF Digital
Copyright (C) 2018-2019 Norwegian Meteorological Institute

This python program generate a truth simulation and related observations 
that will be the subject for data-assimilation experiments. It is based on
the DoubleJetCase parameters and initial conditions, and is spun up for 
3 days before starting to write its state to file. The generated data set
should cover time range from day 3 to day 13. 

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



# Import timing utilities
import time
tic = time.time();

# Import packages we need
import numpy as np
from SWESimulators import Common
from SWESimulators import DoubleJetExperimentUtils as djeutils

toc = time.time()
print("{:02.4f} s: ".format(toc-tic) + "Imported packages")

# Create CUDA context
tic = time.time()
gpu_ctx = Common.CUDAContext()
device_name = gpu_ctx.cuda_device.name()
toc = time.time()
print("{:02.4f} s: ".format(toc-tic) + "Created context on " + device_name)

#--------------------------------------------------------------
# PARAMETERS
#--------------------------------------------------------------
# This file takes no parameters, as it is should clearly define a specific truth.
# If we come to a time where we need to specify a lot of different truths, we can introduce argparser again.


length_of_truth_in_days = 13

#----------------------------------------------------
# Call function to generate truth
#----------------------------------------------------
djeutils.generateTruth(gpu_ctx, current_dir, log_to_screen=True)

