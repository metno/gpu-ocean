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
    float y = (tj-y_zero_reference_cell + 0.5f)*dy;
    return f + beta * y;
}



/**
  * Kernel that evolves U one step in time.
  */
extern "C" {
__global__ void computeUKernel(
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
    
    __shared__ float H_shared[block_height][block_width+1];
    __shared__ float V_shared[block_height+1][block_width+1];
    __shared__ float eta_shared[block_height][block_width+1];

    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    //Index of block within domain
    const int bx = blockDim.x * blockIdx.x;
    const int by = blockDim.y * blockIdx.y;

    //Index of cell within domain
    const int ti = bx + tx;
    const int tj = by + ty;
    
    //Compute pointer to row "tj" in the U array
    float* const U_row = (float*) ((char*) U_ptr_ + U_pitch_*tj);

    //Read current U
    float U_current = 0.0f;
    if (ti < nx_ + 1 && tj < ny_) {
        U_current = U_row[ti];
    }

    //Read H and eta into local memory
    for (int j=ty; j<block_height; j+=blockDim.y) {
        const int l = by + j;
        
        //Compute the pointer to row "l" in the H and eta arrays
        float* const H_row = (float*) ((char*) H_ptr_ + H_pitch_*l);
        float* const eta_row = (float*) ((char*) eta_ptr_ + eta_pitch_*l);
        
        for (int i=tx; i<block_width+1; i+=blockDim.x) {
            const int k = bx + i - 1;
            
            if (k >= 0 && k < nx_ && l < ny_) {
                H_shared[j][i] = H_row[k];
                eta_shared[j][i] = eta_row[k];
            }
            else {
                H_shared[j][i] = 0.0f;
                eta_shared[j][i] = 0.0f;
            }
        }
    }

    //Read V into shared memory
    for (int j=ty; j<block_height+1; j+=blockDim.y) {
        const int l = by + j;
        
        //Compute the pointer to current row in the V array
        float* const V_row = (float*) ((char*) V_ptr_ + V_pitch_*l);
        
        for (int i=tx; i<block_width+1; i+=blockDim.x) {
            const int k = bx + i - 1;
            
            if (k >= 0 && k < nx_ && l < ny_+1) {
                V_shared[j][i] = V_row[k];
            }
            else {
                V_shared[j][i] = 0.0f;
            }
        }
    }

    //Make sure all threads have read into shared mem
    __syncthreads();

    //Reconstruct H at the U position
    float H_m = 0.5f*(H_shared[ty][tx] + H_shared[ty][tx+1]);

    // Coriolis forces at U position and V positions
    float f_u =   linear_coriolis_term(f_, beta_, tj,      dy_, y_zero_reference_cell_);
    float f_v_p = linear_coriolis_term(f_, beta_, tj+0.5f, dy_, y_zero_reference_cell_);
    float f_v_m = linear_coriolis_term(f_, beta_, tj-0.5f, dy_, y_zero_reference_cell_);
    
    //Reconstruct f*V at the U position
    float fV_m = 0.0f;
    if (tj==0) {
	// Using Coriolis at U position
        fV_m = 0.5f*f_u*(V_shared[ty+1][tx] + V_shared[ty+1][tx+1]);
    }
    else if (tj==ny_-1) {
	// Using Coriolis at U position
        fV_m = 0.5f*f_u*(V_shared[ty][tx] + V_shared[ty][tx+1]);
    }
    else {
        fV_m = 0.25f*( f_v_m*(V_shared[ty  ][tx] + V_shared[ty  ][tx+1])
		     + f_v_p*(V_shared[ty+1][tx] + V_shared[ty+1][tx+1]) );
    }

    //Calculate the friction coefficient
    float B = H_m/(H_m + r_*dt_);

    //Calculate the gravitational effect
    float P = g_*H_m*(eta_shared[ty][tx] - eta_shared[ty][tx+1])/dx_;
    
    //FIXME Check coordinates (ti_, tj_) here!!!
    //TODO Check coordinates (ti_, tj_) here!!!
    //WARNING Check coordinates (ti_, tj_) here!!!
    float X = windStressX(wind_stress_t_, ti, tj+0.5, nx_, ny_);

    //Compute the U at the next timestep
    float U_next = B*(U_current + dt_*(fV_m + P + X) );

    //Write to main memory for internal cells
    if (ti > 0 && ti < nx_ && tj < ny_) {
        U_row[ti] = U_next;
    }

    // TODO:
    // Currently, boundary conditions are individual kernels.
    // They should be moved to be within-kernel functions.

}
} // extern "C"