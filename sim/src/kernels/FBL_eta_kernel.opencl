/*
This software is part of GPU Ocean. 

Copyright (C) 2018 SINTEF Digital
Copyright (C) 2018 Norwegian Meteorological Institute

This OpenCL kernel implements part of the Forward Backward Linear 
numerical scheme for the shallow water equations, described in 
L. P. Røed, "Documentation of simple ocean models for use in ensemble
predictions", Met no report 2012/3 and 2012/5 .

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

//#include "../config.h"

#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#define CLK_LOCAL_MEM_FENCE
#endif

/**
  * Kernel that evolves eta one step in time.
  */
__kernel void computeEtaKernel(
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
        __global float* H_ptr_, int H_pitch_,
        __global float* U_ptr_, int U_pitch_,
        __global float* V_ptr_, int V_pitch_,
        __global float* eta_ptr_, int eta_pitch_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of block within domain
    const int bx = get_local_size(0) * get_group_id(0);
    const int by = get_local_size(1) * get_group_id(1);

    //Index of cell within domain
    const int ti = get_global_id(0); 
    const int tj = get_global_id(1);
    
    __local float U_shared[block_height][block_width+1];
    __local float V_shared[block_height+1][block_width];
    
    //Compute pointer to current row in the U array
    __global float* const eta_row = (__global float*) ((__global char*) eta_ptr_ + eta_pitch_*tj);

    //Read current eta
    float eta_current = 0.0f;
    if (ti < nx_ && tj < ny_) {
        eta_current = eta_row[ti];
    }
    
    //Read U into shared memory
    for (int j=ty; j<block_height; j+=get_local_size(1)) {
        const unsigned int l = by + j;
        
        //Compute the pointer to current row in the V array
        __global float* const U_row = (__global float*) ((__global char*) U_ptr_ + U_pitch_*l);
        
        for (int i=tx; i<block_width+1; i+=get_local_size(0)) {
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
    for (int j=ty; j<block_height+1; j+=get_local_size(1)) {
        const unsigned int l = by + j;
        //Compute the pointer to current row in the V array
        __global float* const V_row = (__global float*) ((__global char*) V_ptr_ + V_pitch_*l);
        for (int i=tx; i<block_width; i+=get_local_size(0)) {
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
    barrier(CLK_LOCAL_MEM_FENCE);

    //Compute the eta at the next timestep
    float eta_next = eta_current - dt_/dx_ * (U_shared[ty][tx+1] - U_shared[ty][tx])
                                 - dt_/dy_ * (V_shared[ty+1][tx] - V_shared[ty][tx]);
    
    //Write to main memory
    if (ti < nx_ && tj < ny_) {
        eta_row[ti] = eta_next;
    }
}
