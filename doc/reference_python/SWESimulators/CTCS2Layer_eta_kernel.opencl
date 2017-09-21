/**
This OpenCL kernel implements part of the Centered in Time, Centered 
in Space (leapfrog) numerical scheme for the shallow water equations, 
described in 
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

#define block_height 8
#define block_width 8

typedef __local float u_shmem[block_height][block_width+1];
typedef __local float v_shmem[block_height+1][block_width];


/**
  * Kernel that evolves eta one step in time.
  */
__kernel void computeEtaKernel(
        //Discretization parameters
        int nx_, int ny_,
        float dx_, float dy_, float dt_,
        
        //Data for layer 1
        __global float* eta1_0_ptr_, int eta1_0_pitch_, //eta_1^n-1 (also used as output, that is eta_1^n+1)
        __global float* U1_1_ptr_, int U1_1_pitch_, // U^n
        __global float* V1_1_ptr_, int V1_1_pitch_, // V^n
        
        //Data for layer 2
        __global float* eta2_0_ptr_, int eta2_0_pitch_, //eta_2^n-1 (also used as output, that is eta_2^n+1)
        __global float* U2_1_ptr_, int U2_1_pitch_, // U^n
        __global float* V2_1_ptr_, int V2_1_pitch_ // V^n
        ) {
    
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Start of block within domain
    const int bx = get_local_size(0) * get_group_id(0) + 1; //Skip global ghost cells
    const int by = get_local_size(1) * get_group_id(1) + 1; //Skip global ghost cells

    //Index of cell within domain
    const int ti = bx + tx;
    const int tj = by + ty;
    
    //Layer 1
    u_shmem U1_1_shared;
    v_shmem V1_1_shared;
    
    //Layer 2
    u_shmem U2_1_shared;
    v_shmem V2_1_shared;
    
    //Compute pointer to current row in the eta arrays
    __global float* eta1_0_row = (__global float*) ((__global char*) eta1_0_ptr_ + eta1_0_pitch_*tj);
    __global float* eta2_0_row = (__global float*) ((__global char*) eta2_0_ptr_ + eta2_0_pitch_*tj);

    //Read current eta
    float eta1_0 = 0.0f;
    float eta2_0 = 0.0f;
    if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_+1) {
        eta1_0 = eta1_0_row[ti];
        eta2_0 = eta2_0_row[ti];
    }
    
    //Read U into shared memory
    for (int j=ty; j<block_height; j+=get_local_size(1)) {
        const int l = clamp(by + j, 1, ny_); // fake ghost cells
        
        //Compute the pointer to current row in the U array
        __global float* const U1_1_row = (__global float*) ((__global char*) U1_1_ptr_ + U1_1_pitch_*l);
        __global float* const U2_1_row = (__global float*) ((__global char*) U2_1_ptr_ + U2_1_pitch_*l);
        
        for (int i=tx; i<block_width+1; i+=get_local_size(0)) {
            const int k = clamp(bx + i - 1, 0, nx_); // prevent out of bounds
            
            U1_1_shared[j][i] = U1_1_row[k];
            U2_1_shared[j][i] = U2_1_row[k];
        }
    }
    
    //Read V into shared memory
    for (int j=ty; j<block_height+1; j+=get_local_size(1)) {
        const int l = clamp(by + j - 1, 0, ny_); // prevent out of bounds
        
        //Compute the pointer to current row in the V array
        __global float* const V1_1_row = (__global float*) ((__global char*) V1_1_ptr_ + V1_1_pitch_*l);
        __global float* const V2_1_row = (__global float*) ((__global char*) V2_1_ptr_ + V2_1_pitch_*l);
        
        for (int i=tx; i<block_width; i+=get_local_size(0)) {
            const int k = clamp(bx + i, 1, nx_); // fake ghost cells
            
            V1_1_shared[j][i] = V1_1_row[k];
            V2_1_shared[j][i] = V2_1_row[k];
        }
    }

    //Make sure all threads have read into shared mem
    barrier(CLK_LOCAL_MEM_FENCE);

    //Compute the H at the next timestep
    float eta1_2 = eta1_0 - 2.0f*dt_/dx_ * (U1_1_shared[ty][tx+1] - U1_1_shared[ty][tx])
                          - 2.0f*dt_/dy_ * (V1_1_shared[ty+1][tx] - V1_1_shared[ty][tx]);
    float eta2_2 = eta2_0 - 2.0f*dt_/dx_ * (U2_1_shared[ty][tx+1] - U2_1_shared[ty][tx])
                          - 2.0f*dt_/dy_ * (V2_1_shared[ty+1][tx] - V2_1_shared[ty][tx]);
    
    //Write to main memory
    if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_+1) {
        eta1_0_row[ti] = eta1_2;
        eta2_0_row[ti] = eta2_2;
    }
}