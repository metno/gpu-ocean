[![Build Status](https://travis-ci.org/metno/gpu-ocean.svg?branch=master)](https://travis-ci.org/metno/gpu-ocean)
[![Coverage Status](https://coveralls.io/repos/github/metno/gpu-ocean/badge.svg?branch=master)](https://coveralls.io/github/metno/gpu-ocean?branch=master)

# gpu-ocean
GPU Ocean codebase.

# Installation

## Requirements
In order to run this code, you need to have access to a CUDA enabled GPU, with CUDA toolkit and appropriate drivers installed.

## Preparation steps on Windows

If you are on Windows, you also need to have installed Visual Studios and add the path to its bin folder in PATH. This is so that pycuda can find a C++ compiler. The following steps are an example how to yield those steps:

-   Install [NVIDIA CUDA Toolbox](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) 
-   Install [Visual Studio 2019 (Community version)](https://visualstudio.microsoft.com/vs/community/)
-   Add a C++ compiler to the PATH variable of Windows
    1.  Find folder which contains compiler (check `C:\Program Files\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\Hostx64\x64`)
    2.  Open control panel "Edit the system environment variables"
    3.  Click on "Environment variables"
    4.  Select "Path" from the user variables and choose edit
    5.  Add the folder from above as new path

## Set-up
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
    pip3 install --trusted-host files.pythonhosted.org --no-deps -U pycuda
    ```

You should now be able to start a jupyter notebook server, open one of our notebooks, select the conda environment 'gpuocean' as kernel, and run the code. 

Have fun!

# Download all data files
```
cd <project root directory>
wget -r -np -nH -R "index.html*" http://gpu-ocean.met.no:9000/gpu_ocean
```

# For contributors 

More information can be found in the [wiki pages](https://github.com/metno/gpu-ocean/wiki/)
