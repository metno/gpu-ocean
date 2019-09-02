# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2019 SINTEF Digital

This python program creates npz-files that contains the results
from the manual power consumption experiments made on the 840M and
GeForce GTX780 GPUs, so that these results can be plotted and analyzed
along with the automatic experiments from the V100 GPU.

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

import re
import numpy as np
import pandas as pd
import subprocess
import os, stat
import sys
import os.path
import time
import tempfile
import shutil
import logging
import urllib
import json




    
max_temperature_key    = 'max_temperature'
min_temperature_key    = 'min_temperature'
cumsum_temperature_key = 'cumsum_temperature'
max_power_key          = 'max_power'
min_power_key          = 'min_power'
total_power_key        = 'total_power'
mean_power_key         = 'mean_power'
max_utilization_key    = 'max_utilization'
min_utilization_key    = 'min_utilization'
mean_utilization_key   = 'mean_utilization'

smi_statistics = [max_temperature_key, min_temperature_key, cumsum_temperature_key,
                  max_power_key, min_power_key, total_power_key, mean_power_key,
                  max_utilization_key, min_utilization_key, mean_utilization_key
                 ]


    
    
# Setup logging
logging.getLogger("").setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logging.getLogger("").addHandler(ch)
logger = logging.getLogger("main")

current_dir = os.path.dirname(os.path.realpath(__file__))

# Folder for storing 
current_time = time.strftime("%Y_%m_%d-%H_%M_%S")
out_folder_name = os.path.join(current_dir, 'manual_experiments_'+current_time)
if not os.path.isdir(out_folder_name):
    os.mkdir(out_folder_name)


def store_results(filename, 
                  git_commits, megacells, 
                  mean_power, max_power):

    assert(filename.endswith('.npz')), 'Filename must be a .npz file'
    
    assert(len(git_commits) == 2), str(len(git_commits))+' git_commits given, expected 2'
    assert(len(mean_power) == 2), str(len(mean_power))+' mean_power given, expected 2'
    assert(len(max_power) == 2), str(len(max_power))+' max_power given, expected 2'
    assert(len(megacells) == 2), str(len(megacells))+' megacells given, expected 2'

    #Save results to file, using the same signature as the automatic experiments
    git_commits = np.array(git_commits)
    megacells   = np.array(megacells)
    max_temperature    = np.full(2, np.nan)
    min_temperature    = np.full(2, np.nan)
    cumsum_temperature    = np.full(2, np.nan)
    max_power   = np.array(max_power)
    min_power   = np.full(2, np.nan)
    total_power = np.full(2, np.nan)
    mean_power  = np.array(mean_power)
    max_utilization  = np.full(2, np.nan)
    min_utilization  = np.full(2, np.nan)
    mean_utilization = np.full(2, np.nan)
    nvidia_smi_files = np.full(2, np.nan)
    labels = np.full(2, np.nan)


    outfile = os.path.join(out_folder_name, filename)
    logger.debug("Writing results to " + outfile)
    np.savez(outfile, 
             versions=git_commits, 
             labels=labels, 
             megacells=megacells,
             max_temperature=max_temperature,
             min_temperature=min_temperature,
             cumsum_temperature=cumsum_temperature,
             max_power=max_power,
             min_power=min_power,
             total_power=total_power,
             mean_power=mean_power,
             max_utilization=max_utilization,
             min_utilization=min_utilization,
             mean_utilization=mean_utilization,
             nvidia_smi_files=nvidia_smi_files,
             timestamp=current_time)

# Signature:
# store_results(filename, git_commits, 
#               megacells, mean_power, max_power

store_results('cdklm_cuda_laptop.npz', 
              ['a126bab97e690b7c642814e3c8b96f9879adf487',
               '12536844bdc4459dcf4cc92776faea3a81d0a32c'],
              [37.1, 129.2], [20.0, 27.0], [31.0, 37.9])
store_results('cdklm_opencl_laptop.npz', 
              ['4113be8accf34aa57ce186f3e236d1c0c14ffd5b',
               '964e98a5831950724002674b216dfe28f2d7ffd2'],
              [62.5, 111.6], [24.5, 31.9], [33.7, 40.6])
store_results('ctcs_cuda_laptop.npz', 
              ['9507b86aa57bdcedccdf3840435b31b20005dc11',
               'c39c8ba8701fdf708dadafd12cc76f809aeb8cb0'],
              [124.3, 299.1], [21.2, 2.8], [33.5, 30.8])
store_results('ctcs_opencl_laptop.npz', 
              ['4113be8accf34aa57ce186f3e236d1c0c14ffd5b',
               '964e98a5831950724002674b216dfe28f2d7ffd2'],
              [116.4, 172.1], [29.6, 27.6], [37.8, 35.0])
store_results('fbl_cuda_laptop.npz', 
              ['9507b86aa57bdcedccdf3840435b31b20005dc11',
               '38ff9b268a84e3f4a0805c67041b336f396e9a31'],
              [220.9, 445.7], [21.1, 22.6], [31.6, 32.8])
store_results('fbl_opencl_laptop.npz', 
              ['4113be8accf34aa57ce186f3e236d1c0c14ffd5b',
               '964e98a5831950724002674b216dfe28f2d7ffd2'],
              [186.7, 397.9], [31.1, 28.0], [38.8, 37.0])
               
               
store_results('cdklm_cuda_desktop.npz', 
              ['a126bab97e690b7c642814e3c8b96f9879adf487',
               '12536844bdc4459dcf4cc92776faea3a81d0a32c'],
              [92.8, 512.7], [181.7, 223.3], [184.2, 249.4])
store_results('cdklm_opencl_desktop.npz', 
              ['4113be8accf34aa57ce186f3e236d1c0c14ffd5b',
               '964e98a5831950724002674b216dfe28f2d7ffd2'],
              [169.2, 522.7], [177.3, 223.6], [181.0, 250.2])
store_results('ctcs_cuda_desktop.npz', 
              ['9507b86aa57bdcedccdf3840435b31b20005dc11',
               'c39c8ba8701fdf708dadafd12cc76f809aeb8cb0'],
              [671.9, 1713.5], [229.3, 225.9], [254.6, 253.3])
store_results('ctcs_opencl_desktop.npz', 
              ['4113be8accf34aa57ce186f3e236d1c0c14ffd5b',
               '964e98a5831950724002674b216dfe28f2d7ffd2'],
              [1000.5, 1895.4], [227.3, 227.9], [254.4, 253.7])
store_results('fbl_cuda_desktop.npz', 
              ['9507b86aa57bdcedccdf3840435b31b20005dc11',
               '38ff9b268a84e3f4a0805c67041b336f396e9a31'],
              [1779.3, 2721.6], [228.6, 229.2], [252.4, 255.8])
store_results('fbl_opencl_desktop.npz', 
              ['4113be8accf34aa57ce186f3e236d1c0c14ffd5b',
               '964e98a5831950724002674b216dfe28f2d7ffd2'],
              [1900.3, 2655.5], [227.9, 226.3], [251.9, 254.1])
               