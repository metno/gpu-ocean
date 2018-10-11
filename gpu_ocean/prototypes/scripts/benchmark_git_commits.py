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
benchmark_script_version = "77b50ac6165f77f500a56ebc7467a6f91ea6fc22"


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

    cmd = [args.python, benchmark_script_abspath] + options
    stdout = safe_call(cmd, cwd=benchmark_working_dir, env=my_env)
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8")
    logger.debug("stdout:\n" + stdout)
    
    # Cleaning git folder so that it can be deleted
    #stdout = git(git_clone, ["-n", ])
        
logger.debug("*************** benchmarking finished **********************")





#Save results to file
megacells = np.full((len(df.index)), np.nan)

for index, row in df.iterrows():
    filename = os.path.join(tmpdir, "benchmark_" + row['git_commit'] + ".npz")
    with np.load(filename) as version_data:
        megacells[index] = version_data['megacells']

current_time = time.strftime("%Y_%m_%d-%H_%M_%S")
basename = args.outfile_basename
if (basename == None):
    basename = os.path.splitext(os.path.basename(args.csv_file))[0]
outfile = os.path.join(os.getcwd(), basename + "_" + current_time + ".npz")
logger.debug("Writing results to " + outfile)
np.savez(outfile, versions=df['git_commit'], labels=df['label'], megacells=megacells, args=json.dumps(vars(args)), timestamp=current_time)




#Remove temporary directory
def remove_readonly(func, path, _):
    "Clear the readonly bit and reattempt the removal"
    os.chmod(path, stat.S_IWRITE)
    func(path)

shutil.rmtree(tmpdir, onerror=remove_readonly)
logger.debug("Removed tempdir " + tmpdir)
