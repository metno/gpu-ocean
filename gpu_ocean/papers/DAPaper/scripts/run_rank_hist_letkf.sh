#!/bin/bash

output_dir="tmp/rank_hist_iewpf_$(date +%Y_%m_%d-%H_%M_%S)"
echo $output_dir

which python

mkdir $output_dir
python rank_histogram_experiment.py --method letkf --experiment 1 --output_folder $output_dir

scp -r $output_dir ubuntu@gpu-ocean.met.no:
