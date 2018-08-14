/*
This OpenCL kernel implements part of the Forward Backward Linear 
numerical scheme for the shallow water equations, described in 
L. P. Røed, "Documentation of simple ocean models for use in ensemble
predictions", Met no report 2012/3 and 2012/5 .

Copyright (C) 2016  SINTEF ICT

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

#include "../config.h"

/**
  * Kernel that evolves eta one step in time.
  */
__global__ void computeEtaKernel(
        //Discretization parameters
        int nx_, int ny_,
        float dx_, float dy_, float dt_,
    
        //Physical parameters
        float g_, //< Gravitational constant
        float f_, //< Coriolis coefficient
	float beta_, //< Coriolis force f_ + beta_*(y-y0)
	float y_zero_reference_cell_, // the cell row representing y0 (y0 at southern face)
        float r_, //< Bottom friction coefficient
    
        //Data
        float* H_ptr_, int H_pitch_,
        float* U_ptr_, int U_pitch_,
        float* V_ptr_, int V_pitch_,
        float* eta_ptr_, int eta_pitch_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    //Index of block within domain
    const int bx = blockDim.x * blockIdx.x;
    const int by = blockDim.y * blockIdx.y;

    //Index of cell within domain
    const int ti = bx + tx; 
    const int tj = by + ty;
    
    __shared__ float U_shared[block_height][block_width+1];
    __shared__ float V_shared[block_height+1][block_width];
    
    //Compute pointer to current row in the U array
    float* const eta_row = (float*) ((char*) eta_ptr_ + eta_pitch_*tj);

    //Read current eta
    float eta_current = 0.0f;
    if (ti < nx_ && tj < ny_) {
        eta_current = eta_row[ti];
    }
    
    //Read U into shared memory
    for (int j=ty; j<block_height; j+=blockDim.y) {
        const unsigned int l = by + j;
        
        //Compute the pointer to current row in the V array
        float* const U_row = (float*) ((char*) U_ptr_ + U_pitch_*l);
        
        for (int i=tx; i<block_width+1; i+=blockDim.x) {
            const unsigned int k = bx + i;
            if (k < nx_+1 && l < ny_) {
                U_shared[j][i] = U_row[k];
            }
            else {
                U_shared[j][i] = 0.0f;
            }
        }
    }
    
    //Read V into shared memory
    for (int j=ty; j<block_height+1; j+=blockDim.y) {
        const unsigned int l = by + j;
        //Compute the pointer to current row in the V array
        float* const V_row = (float*) ((char*) V_ptr_ + V_pitch_*l);
        for (int i=tx; i<block_width; i+=blockDim.x) {
            const unsigned int k = bx + i;
            if (k < nx_ && l < ny_+1) {
                V_shared[j][i] = V_row[k];
            }
            else {
                V_shared[j][i] = 0.0f;
            }
        }
    }

    //Make sure all threads have read into shared mem
    __syncthreads();

    //Compute the eta at the next timestep
    float eta_next = eta_current - dt_/dx_ * (U_shared[ty][tx+1] - U_shared[ty][tx])
                                 - dt_/dy_ * (V_shared[ty+1][tx] - V_shared[ty][tx]);
    
    //Write to main memory
    if (ti < nx_ && tj < ny_) {
        eta_row[ti] = eta_next;
    }
}
