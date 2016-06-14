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
        float r_, //< Bottom friction coefficient
    
        //Data
        __global float* eta0_ptr_, int eta0_pitch_, //eta^n-1 (also used as output, that is eta^n+1)
        __global float* U1_ptr_, int U1_pitch_, // U^n
        __global float* V1_ptr_, int V1_pitch_ // V^n
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
    
    __local float U1_shared[block_height][block_width+1];
    __local float V1_shared[block_height+1][block_width];
    
    //Compute pointer to current row in the U array
    __global float* eta0_row = (__global float*) ((__global char*) eta0_ptr_ + eta0_pitch_*tj);

    //Read current eta
    float eta0 = 0.0f;
    if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_+1) {
        eta0 = eta0_row[ti];
    }
    
    //Read U into shared memory
    for (int j=ty; j<block_height; j+=get_local_size(1)) {
        const int l = clamp(by + j, 1, ny_); // fake ghost cells
        
        //Compute the pointer to current row in the V array
        __global float* const U1_row = (__global float*) ((__global char*) U1_ptr_ + U1_pitch_*l);
        
        for (int i=tx; i<block_width+1; i+=get_local_size(0)) {
            const int k = clamp(bx + i - 1, 0, nx_); // prevent out of bounds
            
            U1_shared[j][i] = U1_row[k];
        }
    }
    
    //Read V into shared memory
    for (int j=ty; j<block_height+1; j+=get_local_size(1)) {
        const int l = clamp(by + j - 1, 0, ny_); // prevent out of bounds
        
        //Compute the pointer to current row in the V array
        __global float* const V1_row = (__global float*) ((__global char*) V1_ptr_ + V1_pitch_*l);
        
        for (int i=tx; i<block_width; i+=get_local_size(0)) {
            const int k = clamp(bx + i, 1, nx_); // fake ghost cells
            
            V1_shared[j][i] = V1_row[k];
        }
    }

    //Make sure all threads have read into shared mem
    __syncthreads();

    //Compute the H at the next timestep
    float eta2 = eta0 - 2.0f*dt_/dx_ * (U1_shared[ty][tx+1] - U1_shared[ty][tx])
                      - 2.0f*dt_/dy_ * (V1_shared[ty+1][tx] - V1_shared[ty][tx]);
    
    //Write to main memory
    if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_+1) {
        eta0_row[ti] = eta2;
    }
}