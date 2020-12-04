# Installation of GPU-Ocean Software

## Pre-Requirements
-   Install Visual Studio 2019 (Community version)
-   Install NVIDIA CUDA Toolbox (https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
-   Install Git for Windows

## Pre-Steps
-   Add a C++ compiler to the PATH variable of Windows
    1.  Find folder which contains compiler (C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\Hostx64\x64)
    2.  Open control panel "Edit the system environment variables"
    3.  Click on "Environment variables"
    4.  Select "Path" from the user variables and choose edit
    5.  Add the folder from above as new path

## Requirement
-   Install Miniconda3

## Installation
-   Clone a fork of the gpu-ocean repository to a local folder via the Git CMD (Consider: https://github.com/metno/gpu-ocean/wiki/Suggsted-workflow)
-   Open Anaconda Prompt (Miniconda3)
-   Install jupyter notebook in conda
    ```
    conda install -c conda-forge jupyter
    ```
-   Install the conda extensions that allows jupyter notebook to select conda environments as kernels:
    ```
    conda install -c conda-forge nb_conda_kernels
    ```
-   Browse to the repository in the Anaconda Prompt window using ```cd```
-   Modify ```conda_environment.yml``` such that mpi4py is commented out with ```#```
-   Create a new conda environment according to the environment file in this repository
    ```
    conda env create -f conda_environment.yml
    ```
-   Activate the new environment
    ```
    conda activate gpuocean
    ```
-   Install the missing package mpi4py using pip:
    ```
    pip3 install --trusted-host files.pythonhosted.org mpi4py
-   Install pycuda (but none of its dependencies) using pip:
    ```
    pip3 install --trusted-host files.pythonhosted.org --no-deps -U pycuda
    ```

## Get started 
-   Launch a Jupter notebook
    ```
    jupyer-notebook
    ```
-   Select a ```.ipynb``` notebook and run code!

