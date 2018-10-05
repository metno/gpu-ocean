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
import subprocess
import os, stat
import sys
import os.path
import time
import tempfile
import shutil
import logging
import urllib
import enum

#current_dir = os.path.dirname(os.path.realpath(__file__))
#sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../../')))

import argparse
parser = argparse.ArgumentParser(description='Benchmark a simulator across git commits.')
#parser.add_argument('--nx', type=int, default=1000)
#parser.add_argument('--ny', type=int, default=1000)
#parser.add_argument('--block_width', type=int)
#parser.add_argument('--block_height', type=int)
#parser.add_argument('--steps_per_download', type=int, default=2000)
#parser.add_argument('--iterations', type=int, default=1)
parser.add_argument('--simulator', type=str, required=True)
#parser.add_argument('--output', type=str, default=None)
parser.add_argument('--log_file', type=str, default="")
parser.add_argument('--git_commits', type=str, default=None)
parser.add_argument('--architecture_type', type=str, default="desktop")


args = parser.parse_args()


if not args.log_file == "":
	print("ERROR: I haven't implemented writing log to file yet... Sorry :/")
	sys.exit(-1)

logging.getLogger("").setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logging.getLogger("").addHandler(ch)

logger = logging.getLogger("main")
logger.debug("Test")


if args.git_commits is not None:
	print("ERROR: Reading this list from file is still not implemented... Sorry :/")
	sys.exit(-1)
else:
    git_versions = [
        #(git hash, git commit message, block size laptop, block size desktop, block size supercomputer)
        ("a126bab97e690b7c642814e3c8b96f9879adf487", "original (per arch optimized block size)", (32,4), (12,12), (16, 16)), #Change blocksize here!
        ("5d817bb7cd2b369039117d19aae7d669a9a2e53a", "Optimized shared memory use Hm", (32,4), (12,12), (16, 16)),
        ("2e5da01457874ad5087398da77b9102ee991fb94", "Optimized shared memory use RHx/RHy", (32,4), (12,12), (16, 16)),
        ("fcd145c8c29f3d38a024685fdd0fc3cf9580366c", "Optimized shared memory use Q", (32,4), (12,12), (16, 16)),
        ("74e957bb41e391a5c5fbb19c3cac393079272dd3", "Optimized shared memory use F", (32,4), (12,12), (16, 16)),
        ("5aba525d1d64937c03e8d2b33bb7f6d80b97a81a", "Optimized shared memory use G", (32,4), (12,12), (16, 16)),
        ("addb061fe1cfccebb6fab70f9826be3752162b0b", "Optimized shared memory use Qy", (32,4), (12,12), (16, 16)),
        ("aab115045fd6d313940fa37be3149155ba4ead70", "Made variables const", (32,4), (12,12), (16, 16)),
        ("f6a911368b429df608eb5e1218c28bc14d6ffbe2", "Updated blocksize experiment code", (32,4), (12,12), (16, 16)),
        ("339eb0c9a3bf04875507fab24d8924e24718a2eb", "Changed default block size", (16,8), (24,8), (16, 16)), #Change blocksize here!
        ("1319edf92c20b8d5fd9fc8c0e0ed58176f9cc2ba", "Tried to reduce register use", (16,8), (24,8), (16, 16)),
        ("6de871265b820aecc63d68a2e1f858605eb367c7", "Refactored to optimize register use", (16,8), (24,8), (16, 16)),
        ("f6d1727d747b486203df91079b83a9e717668852", "Added logger to CDKLM", (16,8), (24,8), (16, 16)),
        ("40f9d4817fa084de7cebfd3cf6d20254b93e7393", "Added fast math compilation flag", (16,8), (24,8), (16, 16)),
        #("e2af2159becbd3c8769903eb4bea602a96d6c3a1", "Compilation flags experimentation", (16,8), (24,8), (16, 16)),
        ("12536844bdc4459dcf4cc92776faea3a81d0a32c", "New optimal block size", (32,8), (32,12), (16, 16)) #Change blocksize here!
    ]



class Architecture(enum.Enum):
    LAPTOP = 1
    DESKTOP = 2
    SUPERCOMPUTER = 3
	
architecture = None
try:
	architecture = Architecture[args.architecture_type.upper()]
except KeyError:
	print("ERROR: architecture_type must be one of the following: [laptop, desktop, supercomputer]")
	sys.exit(-1)

if not (args.simulator == "CDKLM" or args.simulator == "FBL" or args.simulator == "CTCS"):
	print("ERROR: Unknown simulator type '" + args.simulator + "', aborting")
	sys.exit(-1)
logger.info("Using simulator: " + args.simulator)


git_url = None
git_command = "git"
my_env = os.environ
shell = True

# If we are on windows, we might need to modify the path and git command:
if os.name == 'nt':
    my_env["PATH"] = "C:\\Program Files\\Git\\mingw64\\bin\\;C:\\Program Files\\Git\\usr\\bin\\;" + my_env["PATH"]
    git_command = "git.exe"
if os.name == 'posix':
    shell = False

try:
    git_url = subprocess.check_output([git_command, "rev-parse", "--show-toplevel"], stderr=subprocess.STDOUT, shell=shell)
    git_url = os.path.abspath(os.path.join(git_url.decode("utf-8").rstrip(), ".git"))
except subprocess.CalledProcessError as e:
    logger.error("Failed, return code " + str(e.returncode) + "\n" + str(e.output))
logger.info(git_url)
    
    
benchmark_script_relpath = "gpu_ocean/demos/testCasesDemos/run_benchmark.py"
cdklm_python_class_relpath = "gpu_ocean/SWESimulators/CDKLM16.py"
benchmark_script_version = "5c2214573269367fbdb6d67ace07f2ffa57d37ba"
benchmark_script_options = ["--simulator", "CDKLM", '--steps_per_download', '10', '--iterations', '3']
outfile = os.path.join(os.getcwd(), "output_" + time.strftime("%Y_%m_%d-%H_%M_%S") + ".npz")

for version, log, laptop, desktop, supercomputer in git_versions:
    logger.info(version + " - " + log)


#Create temporary directory
tmpdir = tempfile.mkdtemp(prefix='git_tmp')
logger.debug("Writing to " + tmpdir)

# Function for running git commands
def git(cwd, options):
    a = None
    try:
        cmd = [git_command] + options
        logger.debug("Git cloning: " + cwd + " -- " + str(cmd))
        a = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=shell, cwd=cwd, env=my_env)
    except subprocess.CalledProcessError as e:
        logger.error("Failed, return code " + str(e.returncode) + "\n" + str(e.output))
    return a.decode("utf-8") 

# Clone this git repo to tmp    
git_clone = os.path.join(tmpdir, "git_clone")
stdout = git(tmpdir, ["clone", git_url, git_clone])
logger.debug(stdout)



# Loop through the git_versions and run each benchmark
for version, log, laptop_block, desktop_block, supercomputer_block in git_versions:
    logger.debug("checkout " + version)
    logger.debug("    with commit msg: " + log)
    stdout = git(git_clone, ["checkout", "--force", '-q', version])
    logger.debug("stdout: " + str(stdout))
    
    logger.debug("Checkout " + benchmark_script_relpath)
    stdout = git(git_clone, ["checkout", "--force", '-q', benchmark_script_version, "--", benchmark_script_relpath])
    logger.debug("stdout: " + str(stdout))
        
    if architecture == Architecture.DESKTOP or architecture == Architecture.SUPERCOMPUTER:
        block_size_options = None
        
        if architecture == Architecture.LAPTOP:
            block_size_options = ["--block_width", str(laptop_block[0]), "--block_height", str(laptop_block[1])] 

            
        if architecture == Architecture.DESKTOP:
            block_size_options = ["--block_width", str(desktop_block[0]), "--block_height", str(desktop_block[1])]
            
        if architecture == Architecture.SUPERCOMPUTER:
            block_size_options = ["--block_width", str(supercomputer_block[0]), "--block_height", str(supercomputer_block[1])]
                   
    a = None
    try:
        #cmd = ["python", "--version"]
        cmd = ["python", os.path.join(git_clone, benchmark_script_relpath)] + benchmark_script_options + ["--output", "benchmark_" + version + ".npz"] + block_size_options
        a = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=shell, cwd=tmpdir, env=my_env)
    except subprocess.CalledProcessError as e:
        logger.error("Failed, return code " + str(e.returncode) + "\n" + str(e.output))
    logger.debug("Output:\n" + a.decode("utf-8"))
    
    # Cleaning git folder so that it can be deleted
    #stdout = git(git_clone, ["-n", ])
        
print("benchmarking finished")



versions, labels, laptop_block, desktop_block, supercomputer_block = list(zip(*git_versions))
megacells = np.full((len(versions)), np.nan)

for i, ver in enumerate(git_versions):
    version, log, l_block, d_block, hpc_block  = ver
    filename = os.path.join(tmpdir, "benchmark_" + version + ".npz")
    with np.load(filename) as version_data:
        megacells[i] = version_data['CDKLM']

np.savez(outfile, versions=versions, labels=labels, megacells=megacells)

