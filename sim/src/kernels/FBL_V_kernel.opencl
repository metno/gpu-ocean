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


#include "common.opencl"




/**
  * Kernel that evolves V one step in time.
  */
__kernel void computeVKernel(
        //Discretization parameters
        int nx_, int ny_,
        float dx_, float dy_, float dt_,
    
        //Physical parameters
        float g_, //< Gravitational constant
        float f_, //< Coriolis coefficient
        float r_, //< Bottom friction coefficient
    
        //Data
        __global float* H_ptr_, int H_pitch_,
        __global float* U_ptr_, int U_pitch_,
        __global float* V_ptr_, int V_pitch_,
        __global float* eta_ptr_, int eta_pitch_,
    
        // Wind stress parameters
        int wind_stress_type_, 
        float tau0_, float rho_, float alpha_, float xm_, float Rc_,
        float x0_, float y0_,
        float u0_, float v0_,
        float t_) {
        
    __local float H_shared[block_height+1][block_width];
    __local float U_shared[block_height+1][block_width+1];
    __local float eta_shared[block_height+1][block_width];

    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of block within domain
    const int bx = get_local_size(0) * get_group_id(0);
    const int by = get_local_size(1) * get_group_id(1);

    //Index of cell within domain
    const int ti = get_global_id(0); 
    const int tj = get_global_id(1);
    
    //Compute pointer to current row in the V array
    __global float* const V_row = (__global float*) ((__global char*) V_ptr_ + V_pitch_*tj);

    //Read current V
    float V_current = 0.0f;
    if (ti < nx_ && tj < ny_+1) {
        V_current = V_row[ti];
    }

    //Read H and eta into shared memory
    for (int j=ty; j<block_height+1; j+=get_local_size(1)) {
        const int l = by + j - 1;
        
        //Compute the pointer to current row in the H and eta arrays
        __global float* const H_row = (__global float*) ((__global char*) H_ptr_ + H_pitch_*l);
        __global float* const eta_row = (__global float*) ((__global char*) eta_ptr_ + eta_pitch_*l);
        
        for (int i=tx; i<block_width; i+=get_local_size(0)) {
            const int k = bx + i;
            if (k < nx_ && l >= 0 && l < ny_+1) {
                H_shared[j][i] = H_row[k];
                eta_shared[j][i] = eta_row[k];
            }
            else {
                H_shared[j][i] = 0.0f;
                eta_shared[j][i] = 0.0f;
            }
        }
    }

    //Read U into shared memory
    for (int j=ty; j<block_height+1; j+=get_local_size(1)) {
        const int l = by + j - 1;
        
        //Compute the pointer to current row in the V array
        __global float* const U_row = (__global float*) ((__global char*) U_ptr_ + U_pitch_*l);
        
        for (int i=tx; i<block_width+1; i+=get_local_size(0)) {
            const int k = bx + i;
            if (k < nx_+1 && l >= 0 && l < ny_) {
                U_shared[j][i] = U_row[k];
            }
            else {
                U_shared[j][i] = 0.0f;
            }
        }
    }

    //Make sure all threads have read into shared mem
    barrier(CLK_LOCAL_MEM_FENCE);

    //Reconstruct H at the V position
    float H_m = 0.5f*(H_shared[ty][tx] + H_shared[ty+1][tx]);

    //Reconstruct U at the V position
    float U_m;
    if (ti==0) {
        U_m = 0.5f*(U_shared[ty][tx+1] + U_shared[ty+1][tx+1]);
    }
    else if (ti==nx_-1) {
        U_m = 0.5f*(U_shared[ty][tx] + U_shared[ty+1][tx]);
    }
    else {
        U_m = 0.25f*(U_shared[ty][tx] + U_shared[ty][tx+1]
                + U_shared[ty+1][tx] + U_shared[ty+1][tx+1]);
    }

    //Calculate the friction coefficient
    float B = H_m/(H_m + r_*dt_);

    //Calculate the gravitational effect
    float P = g_*H_m*(eta_shared[ty][tx] - eta_shared[ty+1][tx])/dy_;

    //Calculate the wind shear stress
    float Y = windStressY(
        wind_stress_type_, 
        dx_, dy_, dt_,
        tau0_, rho_, alpha_, xm_, Rc_,
        x0_, y0_,
        u0_, v0_,
        t_);
    
    //Compute the V at the next timestep
    float V_next = B*(V_current + dt_*(-f_*U_m + P + Y) );

    //Write to main memory
    if (ti < nx_ && tj > 0 && tj < ny_ ) {
        V_row[ti] = V_next;
    }

    // TODO:
    // Currently, boundary conditions are individual kernels.
    // They should be moved to be within-kernel functions.
}
