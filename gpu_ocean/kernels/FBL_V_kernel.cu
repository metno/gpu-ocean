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

#include "common.cu"

// Finds the coriolis term based on the linear Coriolis force
// f = \tilde{f} + beta*(y-y0)
__device__ float linear_coriolis_term(const float f, const float beta,
			   const float tj, const float dy,
			   const float y_zero_reference_cell) {
    // y_0 is at the southern face of the row y_zero_reference_cell.
    float y = (tj-y_zero_reference_cell + 0.0f)*dy;
    return f + beta * y;
}


/**
  * Kernel that evolves V one step in time.
  */
extern "C" {
__global__ void computeVKernel(
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
        float* eta_ptr_, int eta_pitch_,
    
        // Wind stress parameters
        float wind_stress_t_) {
        
    __shared__ float H_shared[block_height+1][block_width];
    __shared__ float U_shared[block_height+1][block_width+1];
    __shared__ float eta_shared[block_height+1][block_width];

    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    //Index of block within domain
    const int bx = blockDim.x * blockIdx.x;
    const int by = blockDim.y * blockIdx.y;

    //Index of cell within domain
    const int ti = bx + tx;
    const int tj = by + ty;

    //Compute pointer to current row in the V array
    float* const V_row = (float*) ((char*) V_ptr_ + V_pitch_*tj);

    //Read current V
    float V_current = 0.0f;
    if (ti < nx_ && tj < ny_+1) {
        V_current = V_row[ti];
    }

    //Read H and eta into shared memory
    for (int j=ty; j<block_height+1; j+=blockDim.y) {
        const int l = by + j - 1;
        
        //Compute the pointer to current row in the H and eta arrays
        float* const H_row = (float*) ((char*) H_ptr_ + H_pitch_*l);
        float* const eta_row = (float*) ((char*) eta_ptr_ + eta_pitch_*l);
        
        for (int i=tx; i<block_width; i+=blockDim.x) {
            const int k = bx + i;
            if (k < nx_ && l >= 0 && l < ny_) {
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
    for (int j=ty; j<block_height+1; j+=blockDim.y) {
        const int l = by + j - 1;
        
        //Compute the pointer to current row in the V array
        float* const U_row = (float*) ((char*) U_ptr_ + U_pitch_*l);
        
        for (int i=tx; i<block_width+1; i+=blockDim.x) {
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
    __syncthreads();

    //Reconstruct H at the V position
    float H_m = 0.5f*(H_shared[ty][tx] + H_shared[ty+1][tx]);

    // Coriolis forces at V position and U positions
    float f_v   = linear_coriolis_term(f_, beta_, tj,      dy_, y_zero_reference_cell_);
    float f_u_p = linear_coriolis_term(f_, beta_, tj+0.5f, dy_, y_zero_reference_cell_);
    float f_u_m = linear_coriolis_term(f_, beta_, tj-0.5f, dy_, y_zero_reference_cell_); 
    
    //Reconstruct f*U at the V position
    float fU_m;
    if (ti==0) {
	// Using Coriolis at V postiion
        fU_m = 0.5f*f_v*(U_shared[ty][tx+1] + U_shared[ty+1][tx+1]);
    }
    else if (ti==nx_-1) {
	// Using Coriolis at V postiion
        fU_m = 0.5f*f_v*(U_shared[ty][tx] + U_shared[ty+1][tx]);
    }
    else {
        fU_m = 0.25f*( f_u_m*(U_shared[ty  ][tx] + U_shared[ty  ][tx+1])
		     + f_u_p*(U_shared[ty+1][tx] + U_shared[ty+1][tx+1]) );
    }

    //Calculate the friction coefficient
    float B = H_m/(H_m + r_*dt_);

    //Calculate the gravitational effect
    float P = g_*H_m*(eta_shared[ty][tx] - eta_shared[ty+1][tx])/dy_;

    //FIXME Check coordinates (ti_, tj_) here!!!
    //TODO Check coordinates (ti_, tj_) here!!!
    //WARNING Check coordinates (ti_, tj_) here!!!
    float Y = windStressY(wind_stress_t_, ti+0.5, tj, nx_, ny_);

    //Compute the V at the next timestep
    float V_next = B*(V_current + dt_*(-fU_m + P + Y) );

    //Write to main memory
    if (ti < nx_ && tj > 0 && tj < ny_ ) {
        V_row[ti] = V_next;
    }

    // TODO:
    // Currently, boundary conditions are individual kernels.
    // They should be moved to be within-kernel functions.
}
} // extern "C" 