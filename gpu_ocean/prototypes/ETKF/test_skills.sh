#!/bin/bash

conda activate gpuocean

python run_DA.py --method none --forecast_days 0 
python run_DA.py --method iewpf2 --forecast_days 0
python run_DA.py --method letkf --seed 31291 --forecast_days 0 
