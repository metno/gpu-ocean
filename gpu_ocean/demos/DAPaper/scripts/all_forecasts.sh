!/bin/bash

#output_dir="/media/havahol/Seagate Backup Plus Drive/gpu_ocean/all_forecasts_$(date +%Y_%m_%d-%H_%M_%S)"
output_dir="/data/gpu_ocean/demos/DAPaper/scripts/all_forecasts_$(date +%Y_%m_%d-%H_%M_%S)"
echo $output_dir

#gpuoceanpython=/home/havahol/miniconda3/envs/gpuocean/bin/python
gpuoceanpython=python


which $gpuoceanpython

mkdir "${output_dir}"

#python rank_histogram_experiment.py --experiments 1 --output_folder $output_dir

# Dry run
$gpuoceanpython run_experiment.py -N 100 --method none --media_dir "${output_dir}"

# Drifter set 
$gpuoceanpython run_experiment.py -N 100

# All drifters
$gpuoceanpython run_experiment.py -N 100 --observation_type all_drifters --media_dir "${output_dir}"

# All buoys
$gpuoceanpython run_experiment.py -N 100 --observation_type buoys --media_dir "${output_dir}"

# Western buoys
$gpuoceanpython run_experiment.py -N 100 --observation_type buoys --buoy_area west --media_dir "${output_dir}"

# Northern buoys
$gpuoceanpython run_experiment.py -N 100 --observation_type buoys --buoy_area south --media_dir "${output_dir}"




