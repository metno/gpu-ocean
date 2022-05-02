#!/bin/bash

python run_DA.py --method iewpf2
python run_DA.py --method iewpf2 --iewpf2beta 0.25
python run_DA.py --method iewpf2 --iewpf2beta 0.5
python run_DA.py --method iewpf2 --iewpf2beta 0.75
python run_DA.py --method iewpf2 --iewpf2beta 1.0
python run_DA.py --method iewpf2 --iewpf2beta 1.25