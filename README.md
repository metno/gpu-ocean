[![Build Status](https://travis-ci.org/metno/gpu-ocean.svg?branch=master)](https://travis-ci.org/metno/gpu-ocean)
[![Coverage Status](https://coveralls.io/repos/github/metno/gpu-ocean/badge.svg?branch=master)](https://coveralls.io/github/metno/gpu-ocean?branch=master)

# gpu-ocean
GPU Ocean codebase.

# Installation
In order to run this code, you need to have access to a CUDA enabled GPU, with CUDA toolkit and appropriate drivers installed. If you are on Windows, you also need to have installed Visual Studios and add the path to its bin folder in PATH. This is so that pycuda can find a C++ compiler.

We recommend that you set up your python environment using Conda as follows:
- Install [miniconda](https://conda.io/miniconda.html) (which is a minimal subset of Anaconda)
- Install jupyter notebook (unless you already have it installed on your system) by opening a terminal (or Anaconda prompt if on Windows) and type
    ```
    conda install -c conda-forge jupyter
    ```
- Install the conda extensions that allows jupyter notebook to select conda environments as kernels:
    ```
    conda install -c conda-forge nb_conda_kernels
    ```
- Create a new conda environment according to the environment file in this repository
    ```
    conda env create -f conda_environment.yml
    ```
- Activate the new environment
    ```
    conda activate gpuocean
    ```
- Install pycuda (but none of its dependencies) using pip:
    ```
    pip3 install --no-deps -U pycuda
    ```

You should now be able to start a jupyter notebook server, open one of our notebooks, select the conda environment 'gpuocean' as kernel, and run the code. 

Have fun!

# Download all data files
```
cd <project root directory>
wget -r -np -nH -R "index.html*" http://gpu-ocean.met.no:9000/gpu_ocean
```
