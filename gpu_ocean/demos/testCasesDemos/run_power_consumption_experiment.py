# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2018, 2019 SINTEF Digital
Copyright (C) 2018 Norwegian Meteorological Institute

This python program runs a short program while reporting execution
and hardware metrics, and is intended to be used for meassuring
computational performance and power consumption.

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

import argparse
parser = argparse.ArgumentParser(description='Find ideal block sizes for a given simulator/scheme.')
parser.add_argument('--simulator', type=str)
args = parser.parse_args()

import re
import numpy as np
import pandas as pd
import subprocess
import os
import os.path
import sys
import time

#import matplotlib.pyplot as plt
#from matplotlib import cm
#from mpl_toolkits.mplot3d import axes3d, Axes3D

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(current_dir))

main_log_filename='power_consumptions.log'
shell = False

def runBenchmark(folder_name, simulator):
    #sim = np.array(["FBL", "CTCS", "KP", "CDKLM"])
    sim = np.array([simulator])
    if simulator.lower() == 'all':
        sim = np.array(["FBL", "CTCS", "CDKLM"])
    #block_width = np.array([4, 8, 12, 16, 24, 32])
    #block_height = np.array([4, 8, 12, 16, 24, 32])
    block_width = np.array([16, 24]) #, 12, 16, 24, 32])
    block_height = np.array([16, 24]) #, 12, 16, 24, 32])
    #block_width=list(range(2,33,1))
    #block_height=list(range(2,33,1))
    
    print("Running with the following:")
    print("Widths: " + str(block_width))
    print("Heights: " + str(block_height))
    print("Simulator(s): " + str(sim))

    block_width, block_height = np.meshgrid(block_width, block_height)
    test_filename = os.path.join(folder_name, main_log_filename)

    
    with open(test_filename, 'w') as test_file:
        for k in range(len(sim)):
            
            test_file.write("##########################################################################\n")
            test_file.write("Using simulator " + sim[k] + ".\n")
            test_file.write("##########################################################################\n")
            for j in range(block_width.shape[1]):
                for i in range(block_width.shape[0]):
                    
                    start_info = sim[k] + " [{:02d} x {:02d}]\n".format(block_width[i,j], block_height[i,j])
                    test_file.write("=========================================\n")
                    test_file.write(start_info)
                    test_file.write("-----------------------------------------\n")
                    print('\n'+start_info)
                    
                    # Start nvidia-smi to file f
                    smi_report_filename = 'nvidia_smi_w'+str(block_width[i,j])+'_h'+str(block_height[i,j])+'.log'
                    smi_report_file = os.path.join(folder_name, smi_report_filename)
                    
                    smi_cmd = [
                        'nvidia-smi',
                        '--query-gpu=timestamp,'+\
                                    'temperature.gpu,'+\
                                    'memory.free,'+\
                                    'fan.speed,'+\
                                    'utilization.gpu,'+\
                                    'power.draw,'+\
                                    'clocks.current.sm,'+\
                                    'clocks.current.graphics,'+\
                                    'clocks.current.memory',
                        '--format=csv',
                        '--loop-ms=500',
                        '--filename='+str(smi_report_file)
                    ]
                    print('nvidia_smi_file='+str(smi_report_file))
                    
                    smi_process = subprocess.Popen(smi_cmd, shell=shell, 
                                                   stdin=subprocess.PIPE, 
                                                   stdout=subprocess.PIPE, 
                                                   stderr=subprocess.STDOUT)
                    
                    
                    # Sleep 3 sec
                    time.sleep(3)
                    
                    # Run benchmark
                    print('starting benchmark... ', end='')
                    tic = time.time()
                    
                    test_file.write("=========================================\n")
                    cmd = [ "python3", "run_benchmark.py", "--block_width", str(block_width[i,j]), \
                           "--block_height", str(block_height[i,j]), "--simulator", sim[k],
                           "--steps_per_download", "300", "--iterations", "10"]
                    p = subprocess.Popen(cmd, shell=shell, 
                                         stdin=subprocess.PIPE, 
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.STDOUT)
                    output = p.stdout.read()
                    test_file.write('nvidia_smi_file='+str(smi_report_file)+', ')
                    test_file.write('nvidia_smi_cmd='+str(smi_cmd)+', ')
                    test_file.write(str(output) + "\n")
                    test_file.write("=========================================\n")
                    test_file.write("\n")
                    print('benchmark finished!')

                    toc = time.time()
                    
                    # Sleep 3 sec
                    time.sleep(3)
                    
                    # Kill nvidia-smi process.
                    smi_process.terminate()
                    
                    
                    infostr = sim[k] + " [{:02d} x {:02d}] completed in {:.02f} s\n".format(block_width[i,j], block_height[i,j], (toc-tic))
                    test_file.write(infostr)
                    print(infostr)
                    
            test_file.write("\n\n\n")
            test_file.flush()
            os.fsync(test_file)

def getData(filename):
    # State variables
    simulator = None
    block_width = None
    block_height = None
    megacells = None
    max_temperature = None
    min_temperature = None
    cumsum_temperature = None

    data = np.empty((0, 7))

    with open(filename) as origin_file:
        for line in origin_file:
            line = str(line)

            # Find simulator
            match = re.match('Using simulator (.+)\.', line)
            if match:
                simulator = match.group(1)
                
            if simulator == None:
                continue

            # Find block size
            match = re.match(simulator + ' \[(\\d+) x (\\d+)\]$', line)
            if match:
                block_width = match.group(1)
                block_height = match.group(2)

            # Find simulator megacells
            match = re.match('.*Maximum megacells: (\\d+(\.\\d+)?)', line)
            if match:
                megacells = match.group(1)
                
            # Find nvidia-smi report
            match = re.match('nvidia_smi_file=(.+).log,', line)
            if match:
                smi_log_file = os.path.abspath(match.group(1)+'.log')
                
               
                smi_log = pd.read_csv(smi_log_file)
                
                # Remove whitespaces from column names:
                smi_log.rename(columns=lambda x: x.strip(), inplace=True)
                
                
                #print(smi_log)
                
                # Find min and max 
                max_temperature = smi_log['temperature.gpu'].max()
                min_temperature = smi_log['temperature.gpu'].min()

                # temperature*seconds
                all_sum_temperature = smi_log['temperature.gpu'].sum()*0.5

                # Drop first and last three seconds (0.5 sec loggin)
                # This compensates for the sleep commands above.
                smi_log.drop(smi_log.tail(6).index,inplace=True)
                smi_log.drop(smi_log.tail(6).index,inplace=True)
                cumsum_temperature = smi_log['temperature.gpu'].sum()*0.5
                
                #print('max/min temperature: ', max_temperature, min_temperature)
                #print('all_sum_temperature: ', all_sum_temperature)
                #print('cumsum_temperature:  ', cumsum_temperature)
                
                data = np.append(data, [[simulator, 
                                         block_width, block_height, megacells,
                                         max_temperature, min_temperature, cumsum_temperature]], 
                                 axis=0)
                
                block_width = None
                block_height = None
                max_temperature = None 
                min_temperature = None
                cumsum_temperature = None
                megacells = None 
                

    return data

if __name__ == "__main__":
    # Generate unique folder name
    folder_name = os.path.abspath("power_block_size_0")
    folder_test = 0
    while (os.path.isdir(folder_name)):
        folder_name = os.path.abspath("power_block_size_" + str(folder_test))
        folder_test += 1

    print("Storing data in folder " + folder_name)
    
    os.mkdir(folder_name)
    runBenchmark(folder_name, args.simulator)
        
    main_log_file = os.path.join(folder_name, main_log_filename)
    data = getData(main_log_file)
    print(data)
    
    simulators = np.unique(data[:,0])
    block_widths = np.unique(data[:,1])
    block_heights = np.unique(data[:,2])

    fields = ['simulator', 
              'block_width', 'block_height', 'megacells',
              'max_temperature', 'min_temperature', 'cumsum_temperature']

    labels = ['simulator', 
              'block_width', 'block_height', 'Megacells/s',
              'Max temperature [C]', 'Min temperature [C]', 'Accumulated temperature [Cs]']


    for field_id in range(3,7):
        for simulator in simulators:
            print(simulator + " - " + labels[field_id])

            df = pd.DataFrame(index=block_widths, columns=block_heights)

            # Pick this simulator data only and sort into a dataframe
            columns = data[:,0] == simulator
            for record in data[columns]:
                block_width = record[1]
                block_height = record[2]
                resulting_values = np.float32(record[field_id])
                df[block_height][block_width] = resulting_values

            maximum = np.nanmax(df.values)
            minimum = np.nanmin(df.values)
            mean = np.nanmean(df.values)
            print(df)
            print("Maximum={:.2f}".format(maximum))
            print("Minimum={:.2f}".format(minimum))
            print("Mean={:.2f}".format(mean))

