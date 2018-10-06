# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2018 SINTEF Digital
Copyright (C) 2018 Norwegian Meteorological Institute

This python program runs a short program while reporting execution
metrics, and is intended to be used for meassuring computational 
performance and throughput.

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

def runBenchmark(filename, simulator):
    #sim = np.array(["FBL", "CTCS", "KP", "CDKLM"])
    sim = np.array([simulator])
    block_width = np.array([4, 8, 12, 16, 24, 32])
    block_height = np.array([4, 8, 12, 16, 24, 32])
    #block_width=list(range(2,33,1))
    #block_height=list(range(2,33,1))
    
    print("Running with the following:")
    print("Widths: " + str(block_width))
    print("Heights: " + str(block_height))

    block_width, block_height = np.meshgrid(block_width, block_height)
    
    with open(test_filename, 'w') as test_file:
        for k in range(len(sim)):
            test_file.write("##########################################################################\n")
            test_file.write("Using simulator " + sim[k] + ".\n")
            test_file.write("##########################################################################\n")
            for j in range(block_width.shape[1]):
                for i in range(block_width.shape[0]):
                    
                    tic = time.time()
                    
                    test_file.write("=========================================\n")
                    test_file.write(sim[k] + " [{:02d} x {:02d}]\n".format(block_width[i,j], block_height[i,j]))
                    test_file.write("-----------------------------------------\n")
                    cmd = [ "python3", "run_benchmark.py", "--block_width", str(block_width[i,j]), \
                           "--block_height", str(block_height[i,j]), "--simulator", sim[k]]
                    p = subprocess.Popen(cmd, shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    output = p.stdout.read()
                    test_file.write(str(output) + "\n")
                    test_file.write("=========================================\n")
                    test_file.write("\n")
                    
                    toc = time.time()
                    
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

    data = np.empty((0, 4))

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
                data = np.append(data, [[simulator, block_width, block_height, megacells]], axis=0)
                
                block_width = None
                block_height = None

    return data

if __name__ == "__main__":
    # Generate unique filename
    test_filename = "blocksize_benchmark_run_0.txt"
    file_test = 0
    while (os.path.isfile(test_filename)):

        test_filename = "blocksize_benchmark_run_" + str(file_test) + ".txt"
        file_test += 1

    print("Storing data in " + test_filename)
    
    if not (os.path.isfile(test_filename)):
        runBenchmark(test_filename, args.simulator)
    else:
        print("Using existing run in " + test_filename)
        
    data = getData(test_filename)
    print(data)
    
    simulators = np.unique(data[:,0])
    block_widths = np.unique(data[:,1])
    block_heights = np.unique(data[:,2])

    print(block_widths)

    for simulator in simulators:
        print(simulator)

        df = pd.DataFrame(index=block_widths, columns=block_heights)

        # Pick this simulator data only and sort into a dataframe
        columns = data[:,0] == simulator
        for record in data[columns]:
            block_width = record[1]
            block_height = record[2]
            megacells = np.float32(record[3])
            df[block_height][block_width] = megacells

        maximum = np.nanmax(df.values)
        minimum = np.nanmin(df.values)
        mean = np.nanmean(df.values)
        print(df)
        print("Maximum={:.2f}".format(maximum))
        print("Minimum={:.2f}".format(minimum))
        print("Mean={:.2f}".format(mean))
        