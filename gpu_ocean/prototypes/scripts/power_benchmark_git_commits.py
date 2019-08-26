# -*- coding: utf-8 -*-

"""
This software is part of GPU Ocean. 

Copyright (C) 2018 SINTEF Digital
Copyright (C) 2018 Norwegian Meteorological Institute

This python program checks out a series of git commits, and reports 
execution metrics for each commit. The purpose is to benchmark code 
optimalization on different platforms.

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






# Helper function for running git commands
def safe_call(cmd, **kwargs):
    if (os.name == 'nt'):
        shell = True
    else:
        shell = False
    
    
    stdout = None
    try:
        logger.debug("Safe call: " + str(cmd) + " -- " + str(kwargs))
        stdout = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=shell, **kwargs)
    except subprocess.CalledProcessError as e:
        output = e.output
        if isinstance(output, bytes):
            output = output.decode("utf-8")
        logger.error("Failed, return code " + str(e.returncode) + "\n" + output)
        
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8")
    return stdout
    
    
# Helper function to monitor the benchmark program with nvidia-smi    
def monitor_benchmark(benchmark_cmd, smi_report_file, **kwargs):
    if (os.name == 'nt'):
        shell = True
    else:
        shell = False
    
    with tempfile.TemporaryFile() as tmpfile:

        # Start nvidia-smi to file f
        #smi_report_filename = 'nvidia_smi_w'+str(block_width[i,j])+'_h'+str(block_height[i,j])+'.log'
        #smi_report_file = os.path.join(folder_name, smi_report_filename)

        logger.debug('')
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
        logger.debug("=========================================\n")
        logger.debug('nvidia_smi_file='+str(smi_report_file)+', ')
        logger.debug('nvidia_smi_cmd='+str(smi_cmd)+', ')
        
        smi_process = subprocess.Popen(smi_cmd, shell=shell, 
                                       stdin=subprocess.PIPE, 
                                       stdout=tmpfile, 
                                       stderr=subprocess.STDOUT)


        # Sleep 3 sec
        time.sleep(3)

        # Run benchmark
        stdout = safe_call(benchmark_cmd, **kwargs)
        
        # Sleep 3 sec
        time.sleep(3)

        # Kill nvidia-smi process.
        smi_process.terminate()

    return stdout
    
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

# Helper function for reading values from the nvidia-smi file
def read_smi_file(smi_log_file):
    print('Reading file: ', smi_log_file)
    smi_log = pd.read_csv(smi_log_file)
    temp_key  = 'temperature.gpu'
    power_key = 'power.draw [W]'
    utilization_key = 'utilization.gpu [%]'
    fan_key = 'fan.speed [%]'
    
    # Remove whitespaces from column names:
    smi_log.rename(columns=lambda x: x.strip(), inplace=True)

    #for col in smi_log.columns:
    #    print(col)
        
    # Parse missing values
    smi_log = smi_log.replace(' [Not Supported]', np.NaN)
    
    # Helper function for parcing percent values
    def parse_percent(df, key):
        if not pd.isnull(df[key].iloc[0]):
            #print('fixing ' + key)
            df[key] = df[key].str.replace(' ','')
            df[key] = df[key].str.replace('%', '')
            df[key] = pd.to_numeric(smi_log[key])
            
    parse_percent(smi_log, utilization_key)
    parse_percent(smi_log, fan_key)
    
    # Find min and max 
    max_temperature = smi_log[temp_key].max()
    min_temperature = smi_log[temp_key].min()
    min_power_draw  = smi_log[power_key].min()
    max_power_draw  = smi_log[power_key].max()
    min_utilization = smi_log[utilization_key].min()
    max_utilization = smi_log[utilization_key].max()
    
    # temperature*seconds
    all_sum_temperature = smi_log[temp_key].sum()*0.5

    # Drop first and last three seconds (0.5 sec loggin)
    # This compensates for the sleep commands above.
    smi_log.drop(smi_log.head(6).index,inplace=True)
    smi_log.drop(smi_log.tail(6).index,inplace=True)
    
    cumsum_temperature = smi_log[temp_key].sum()*0.5
    total_power = smi_log[power_key].sum()*0.5
    if np.isnan(max_power_draw) and np.isnan(min_power_draw):
        total_power = np.NaN
    mean_power = smi_log[power_key].mean()
    mean_utilization = smi_log[utilization_key].mean()
        
    smi_values = {max_temperature_key: max_temperature,
                  min_temperature_key: min_temperature,
                  cumsum_temperature_key: cumsum_temperature,
                  max_power_key: max_power_draw,
                  min_power_key: min_power_draw,
                  total_power_key: total_power,
                  mean_power_key: mean_power,
                  max_utilization_key: max_utilization,
                  min_utilization_key: min_utilization,
                  mean_utilization_key: mean_utilization
                 }

    
    return smi_values

    
    
# Setup logging
logging.getLogger("").setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logging.getLogger("").addHandler(ch)
logger = logging.getLogger("main")

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../../')))

logger.info("Added " + current_dir + " to path")



#Parse arguments
import argparse
parser = argparse.ArgumentParser(description='Benchmark a simulator across git commits.')
parser.add_argument('--run_benchmark_opts', type=str, default=None, required=True, help="\"--simulator=CDKLM --steps_per_download 100 --iterations 3\" (note the quotation marks)")
parser.add_argument('csv_file', default=None, help="CSV file with columns git_commit,label,block_width,block_height (note no spaces in column names)")
parser.add_argument('--add_exe_path', action='append', type=str, default=[])
parser.add_argument('--outfile_basename', type=str, default=None, help="The basename (in os.path.basename terms) of the filename to write to")
parser.add_argument('--python', type=str, default='python', help="Path to python executable")
args = parser.parse_args()
logger.info(args)

# Base name for output file
basename = args.outfile_basename
if (basename == None):
    basename = os.path.splitext(os.path.basename(args.csv_file))[0]
logger.debug('basename: ', basename)

# Folder for storing 
current_time = time.strftime("%Y_%m_%d-%H_%M_%S")
smi_report_folder = os.path.join(current_dir, basename+'_nvidia-smi-reports')
if not os.path.isdir(smi_report_folder):
    os.mkdir(smi_report_folder)




#Read CSV file
df = pd.read_csv(args.csv_file, comment='#')
logger.info(df)


# Commits where we need the new_fbl flag
commits_requireing_new_fbl = [
    "fa2dc111750a08760a27d2d47c2aaebc3aded911", 
    "38ff9b268a84e3f4a0805c67041b336f396e9a31",
    "964e98a5831950724002674b216dfe28f2d7ffd2",
    "92353c0254c69ab9025cb594e9e7165b9535d2ed"
]


#Set git options
if (os.name == 'nt'):
    git_command = "git.exe"
else:
    git_command = "git"
    
# Create modified environment for git to execute in
my_env = None
if (len(args.add_exe_path) > 0):
    my_env = os.environ
for path in args.add_exe_path:
    my_env["PATH"] = my_env["PATH"] + ";" + path
    
    
    
    
#Get URL of repository
git_url = safe_call([git_command, "rev-parse", "--show-toplevel"], cwd=current_dir, env=my_env)
git_url = git_url.strip()
logger.info(git_url)

#Create temporary directory to clone into
tmpdir = tempfile.mkdtemp(prefix='git_tmp')
logger.debug("Writing to " + tmpdir)

# Clone this git repo to tmp    
git_clone = os.path.join(tmpdir, "git_clone")
stdout = safe_call([git_command, "clone", git_url, git_clone], cwd=tmpdir, env=my_env)
logger.debug(stdout)





#Set options for benchmark script
benchmark_script_relpath = os.path.normpath("gpu_ocean/demos/testCasesDemos/run_benchmark.py")
benchmark_script_abspath = os.path.join(git_clone, benchmark_script_relpath)
benchmark_script_version = "6cc7c23c7a244de4a32e4eadc602b30fdc30708c"


# Need to manipulate working directory in order to compile opencl code
benchmark_working_dir = os.path.join(git_clone, "gpu_ocean/")



# Loop through the git_versions and run each benchmark
for index, row in df.iterrows():
    logger.debug("= Start new benchmark ==========================================")
    print("Benchmark number " + str(index+1) + " / " + str(len(df.index)))
    print(row)
    print(row['git_commit'])
    logger.debug(str(index) + ": " + str(row['label']))
    stdout = safe_call([git_command, "checkout", "--force", '-q', row['git_commit']], cwd=git_clone, env=my_env)
    logger.debug("stdout: \n" + str(stdout))
    
    logger.debug("Checkout " + benchmark_script_relpath)
    stdout = safe_call([git_command, "checkout", "--force", '-q', benchmark_script_version, "--", benchmark_script_relpath], cwd=git_clone, env=my_env)
    logger.debug("stdout: \n" + str(stdout))
        
    options = args.run_benchmark_opts.split(' ')
    options += ["--output", "../../benchmark_" + row['git_commit'] + ".npz"] 
    options += ["--block_width", str(row['block_width']), "--block_height", str(row['block_height'])]

    # Check if we need the new fbl flag
    if row['git_commit'] in commits_requireing_new_fbl:
        options += ["--new_fbl", "1"]

    benchmark_cmd = [args.python, benchmark_script_abspath] + options
    
    smi_report_filename = 'nvidia_smi_'+current_time+'_'+row['git_commit']+'.log'
    smi_report_file = os.path.join(smi_report_folder, smi_report_filename)

    stdout = monitor_benchmark(benchmark_cmd, smi_report_file, cwd=benchmark_working_dir, env=my_env)
    
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8")
    logger.debug("stdout:\n" + stdout)
            
logger.debug("*************** benchmarking finished **********************")
time.sleep(3) # Need to give some time to the last nvidia-smi report to be wrapped up. 



#Save results to file
megacells   = np.full((len(df.index)), np.nan)
max_temperature    = np.full((len(df.index)), np.nan)
min_temperature    = np.full((len(df.index)), np.nan)
cumsum_temperature    = np.full((len(df.index)), np.nan)
max_power   = np.full((len(df.index)), np.nan)
min_power   = np.full((len(df.index)), np.nan)
total_power = np.full((len(df.index)), np.nan)
mean_power  = np.full((len(df.index)), np.nan)
max_utilization  = np.full((len(df.index)), np.nan)
min_utilization  = np.full((len(df.index)), np.nan)
mean_utilization = np.full((len(df.index)), np.nan)
nvidia_smi_files = np.full((len(df.index)), '')

for index, row in df.iterrows():
    filename = os.path.join(tmpdir, "benchmark_" + row['git_commit'] + ".npz")
    with np.load(filename) as version_data:
        megacells[index] = version_data['megacells']
        
    smi_report_file = os.path.join(smi_report_folder, 'nvidia_smi_'+current_time+'_'+row['git_commit']+'.log')
    
    # Read smi files here and fill above arrays 
    nvidia_smi_files[index] = smi_report_file
    smi_values = read_smi_file(smi_report_file)
    
    max_temperature[index]    = smi_values[max_temperature_key]
    min_temperature[index]    = smi_values[min_temperature_key]
    cumsum_temperature[index] = smi_values[cumsum_temperature_key]
    max_power[index]          = smi_values[max_power_key]
    min_power[index]          = smi_values[min_power_key]
    total_power[index]        = smi_values[total_power_key]
    mean_power[index]         = smi_values[mean_power_key]
    max_utilization[index]    = smi_values[max_utilization_key]
    min_utilization[index]    = smi_values[min_utilization_key]
    mean_utilization[index]   = smi_values[mean_utilization_key]
    

outfile = os.path.join(os.getcwd(), basename + "_" + current_time + ".npz")
logger.debug("Writing results to " + outfile)
np.savez(outfile, versions=df['git_commit'], labels=df['label'], 
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
         args=json.dumps(vars(args)), timestamp=current_time)



#Remove temporary directory
def remove_readonly(func, path, _):
    "Clear the readonly bit and reattempt the removal"
    os.chmod(path, stat.S_IWRITE)
    func(path)

shutil.rmtree(tmpdir, onerror=remove_readonly)
logger.debug("Removed tempdir " + tmpdir)
