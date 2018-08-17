/*
These OpenCL kernels generates random numbers for creating model error 
based on given covariance relations.

Copyright (C) 2018  SINTEF ICT

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
*/

#include "common.cu"
#include "../config.h"

//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/**
  *  Generates two uniform random numbers based on the ANSIC Linear Congruential 
  *  Generator.
  */
__device__ float2 ansic_lcg(ulong* seed_ptr) {
    ulong seed = (*seed_ptr);
    double denum = 2147483648.0;
    ulong modulo = 2147483647;

    seed = ((seed * 1103515245) + 12345) % modulo; //% 0x7fffffff;
    float u1 = seed / denum;

    seed = ((seed * 1103515245) + 12345) % modulo; //0x7fffffff;
    float u2 = seed / denum;

    (*seed_ptr) = seed;

    float2 out;
    out.x = u1;
    out.y = u2;

    return out;
    //return make_float2(u1, u2);
}

/**
  *  Generates two random numbers, drawn from a normal distribtion with mean 0 and
  *  variance 1. Based on the Box Muller transform.
  */
__device__ float2 boxMuller(ulong* seed) {
    float2 u = ansic_lcg(seed);
    
    float r = sqrt(-2.0f*log(u.x));
    float n1 = r*cospi(2*u.y);
    float n2 = r*sinpi(2*u.y);

    float2 out;
    out.x = n1;
    out.y = n2;
    return out;
    
    //return make_float2(n1, n2);
}

/**
  * Kernel that generates uniform random numbers.
  */
__global__ void uniformDistribution(
        // Size of data
        int seed_nx_, int seed_ny_,        
        int random_nx_,
        
        //Data
        ulong* seed_ptr_, int seed_pitch_,
        float* random_ptr_, int random_pitch_
    ) {

    //Index of cell within domain
    const int ti = (blockDim.x * blockIdx.x) + threadIdx.x;
    const int tj = (blockDim.y * blockIdx.y) + threadIdx.y;

    // Each thread computes and writes two uniform numbers.

    if ((ti < seed_nx_) && (tj < seed_ny_)) {
    
        //Compute pointer to current row in the U array
        ulong* const seed_row = (ulong*) ((char*) seed_ptr_ + seed_pitch_*tj);
        float* const random_row = (float*) ((char*) random_ptr_ + random_pitch_*tj);
        
        ulong seed = seed_row[ti];
        float2 u = ansic_lcg(&seed);

        seed_row[ti] = seed;

        if (2*ti + 1 < random_nx_) {
            random_row[2*ti    ] = u.x;
            random_row[2*ti + 1] = u.y;
        }
        else if (2*ti == random_nx_) {
            random_row[2*ti    ] = u.x;
        }
    }
}


/**
  * Kernel that generates normal distributed random numbers with mean 0 and variance 1.
  */
__global__ void normalDistribution(
        // Size of data
        int seed_nx_, int seed_ny_,
        int random_nx_, 
        
        //Data
        ulong* seed_ptr_, int seed_pitch_,
        float* random_ptr_, int random_pitch_
    ) {
    
    //Index of cell within domain
    const int ti = (blockDim.x * blockIdx.x) + threadIdx.x;
    const int tj = (blockDim.y * blockIdx.y) + threadIdx.y;

    // Each thread computes and writes two uniform numbers.

    if ((ti < seed_nx_) && (tj < seed_ny_)) {
    
        //Compute pointer to current row in the U array
        ulong* const seed_row = (ulong*) ((char*) seed_ptr_ + seed_pitch_*tj);
        float* const random_row = (float*) ((char*) random_ptr_ + random_pitch_*tj);
        
        ulong seed = seed_row[ti];
        float2 u = boxMuller(&seed);

        seed_row[ti] = seed;

        if (2*ti + 1 < random_nx_) {
            random_row[2*ti    ] = u.x;
            random_row[2*ti + 1] = u.y;
        }
        else if (2*ti == random_nx_) {
            random_row[2*ti    ] = u.x;
        }
    }
}


/**
  * Local function calculating the SOAR function given two grid locations
  */
__device__ float soar_covariance(int a_x, int a_y, int b_x, int b_y,
                      float dx, float dy, float soar_q0, float soar_L) {
    const float dist = sqrt( dx*dx*(a_x - b_x)*(a_x - b_x) +
                             dy*dy*(a_y - b_y)*(a_y - b_y) );
    return soar_q0*( 1.0f + dist/soar_L)*exp(-dist/soar_L);
}



/**
  * Kernel that adds a perturbation to the input field eta.
  * The perturbation is based on a SOAR covariance function using a cut-off value of 2.
  */
__global__ void perturbOcean(
        // Size of data
        int nx_, int ny_,
        float dx_, float dy_,
        int ghost_cells_x_, int ghost_cells_y_,
        
        // physical parameters
        float g_, float f_, float beta_, float y0_reference_cell_,
        
        // Parameter for the SOAR function
        float soar_q0_, float soar_L_, 
        
        // Periodic domain
        int periodic_north_south_, int periodic_east_west_,
        
        // random data
        float* random_ptr_, int random_pitch_,

        // Ocean data
        float* eta_ptr_, int eta_pitch_,
        float* hu_ptr_, int hu_pitch_,
        float* hv_ptr_, int hv_pitch_,
        float* Hi_ptr_, int Hi_pitch_
    ) {

    //Index of cell within block
    const int tx = threadIdx.x; 
    const int ty = threadIdx.y;

    //Index of start of block within domain
    const int bx = blockDim.x * blockIdx.x;
    const int by = blockDim.y * blockIdx.y;

    //Index of cell within domain
    const int ti = bx + tx;
    const int tj = by + ty;

    const int cutoff = 2;

    // Local storage for xi (the random numbers)
    __shared__ float xi[block_height+6][block_width+6];

    // Local storage for d_eta (also used for H)
    __shared__ float d_eta[block_height+2][block_width+2];


    // Use local memory for d_eta to compute H_mid for given thread id
    for (int j = ty; j < block_height+1; j += blockDim.y) {
        const int global_j = clamp(by+j, 0, ny_+1);
        float* const Hi_row = (float*) ((char*) Hi_ptr_ + Hi_pitch_*(global_j+ghost_cells_y_));
        for (int i = tx; i < block_width+1; i += blockDim.x) {
            const int global_i = clamp(bx+i, 0, nx_+1);
            d_eta[j][i] = Hi_row[global_i+ghost_cells_x_];
        }
    }
    
    __syncthreads();
    
    const float H_mid = 0.25f*(d_eta[ty  ][tx] + d_eta[ty  ][tx+1] +
                              d_eta[ty+1][tx] + d_eta[ty+1][tx+1]   );
    
    // Read random numbers into local memory:
    for (int j = ty; j < block_height+6; j += blockDim.y) {
        int global_j = 0;
        if (periodic_north_south_) {
            global_j = (by + j - cutoff - 1 + ny_) % ny_;
        } else {
            global_j = clamp(by + j, 0, ny_+6);
        }
        float* const random_row = (float*) ((char*) random_ptr_ + random_pitch_*global_j);
        for (int i = tx; i < block_width+6; i += blockDim.x) {
            int global_i = 0;
            if (periodic_east_west_) {
                global_i = (bx + i - cutoff - 1 + nx_) % nx_;
            } else {
                global_i = clamp(bx + i, 0, nx_+6);
            }
            xi[j][i] = random_row[global_i];
        }
    }

    __syncthreads();

    // Compute d_eta using the SOAR covariance function, and store in local memory
    // All reads are from local memory
    for (int j = ty; j < block_height+2; j += blockDim.y) {
        for(int i = tx; i < block_width+2; i += blockDim.x) {
            const int a_x = i + cutoff;
            const int a_y = j + cutoff;
            int b_x = i;
            int b_y = j;
            float Qxi = 0.0f;
            for (int b_y = j; b_y < a_y + cutoff + 1; b_y++) {
                for (int b_x = i; b_x < a_x + cutoff + 1; b_x++) {
                    const float Q = soar_covariance(a_x, a_y, b_x, b_y,
                                                    dx_, dy_, soar_q0_, soar_L_);
                    Qxi += Q*xi[b_y][b_x];
                }
            }
            d_eta[j][i] = Qxi;
        }
    }

    __syncthreads();

    // Evaluate geostrophic balance and write eta, hu and hv to global memory
    if ((ti < nx_) && (tj < ny_)) {

        //Compute pointer to current row in the U array
        float* const eta_row = (float*) ((char*) eta_ptr_ + eta_pitch_*(tj+ghost_cells_y_));
        float* const hu_row  = (float*) ((char*) hu_ptr_  + hu_pitch_*(tj+ghost_cells_y_));
        float* const hv_row = (float*) ((char*)  hv_ptr_ + hv_pitch_*(tj+ghost_cells_y_));
        
        const int eta_tx = tx+1;
        const int eta_ty = ty+1;

        const float coriolis = f_ + beta_*(tj - y0_reference_cell_ + ghost_cells_y_)*dy_;

        // Total water depth in the given cell (H + eta + d_eta)
        const float h_mid = d_eta[eta_ty][eta_tx] + H_mid + eta_row[ti+ghost_cells_x_];

        // Slope of perturbation of eta
        const float eta_diff_x = (d_eta[eta_ty][eta_tx+1] - d_eta[eta_ty][eta_tx-1]) / (2.0f*dx_);
        const float eta_diff_y = (d_eta[eta_ty+1][eta_tx] - d_eta[eta_ty-1][eta_tx]) / (2.0f*dy_);

        // perturbation of hu and hv
        const float d_hu = -(g_/coriolis)*h_mid*eta_diff_y;
        const float d_hv =  (g_/coriolis)*h_mid*eta_diff_x;        

        if (true) {
            eta_row[ti+ghost_cells_x_] += d_eta[ty+1][tx+1];
             hu_row[ti+ghost_cells_x_] += d_hu;
             hv_row[ti+ghost_cells_x_] += d_hv;
        } else {
            eta_row[ti+ghost_cells_x_] = d_eta[ty+1][tx+1];
            hu_row[ti+ghost_cells_x_] = d_hu;
            hv_row[ti+ghost_cells_x_] = d_hv;
        }
    }
}

