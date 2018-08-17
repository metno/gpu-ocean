/**
This OpenCL kernel implements part of the Centered in Time, Centered 
in Space (leapfrog) numerical scheme for the shallow water equations, 
described in 
L. P. Røed, "Documentation of simple ocean models for use in ensemble
predictions", Met no report 2012/3 and 2012/5.

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
        int bc_north_, int bc_south_,
        float dx_, float dy_, float dt_,
    
        //Physical parameters
        float g_, //< Gravitational constant
        float f_, //< Coriolis coefficient
        float beta_, //< Coriolis force f_ + beta_*(y-y0)
        float y_zero_reference_cell_, // the cell row representing y0 (y0 at southern face)
        float r_, //< Bottom friction coefficient
    
        //Numerical diffusion
        float A_,
    
        //Data
        float* H_ptr_, int H_pitch_,
        float* eta1_ptr_, int eta1_pitch_, // eta^n
        float* U1_ptr_, int U1_pitch_, // U^n
        float* V0_ptr_, int V0_pitch_, // V^n-1, also output V^n+1
        float* V1_ptr_, int V1_pitch_, // V^n
    
        // Wind stress parameters
		float wind_stress_t_) {
        
    __shared__ float H_shared[block_height+1][block_width+2];
    __shared__ float eta1_shared[block_height+1][block_width+2];
    __shared__ float U1_shared[block_height+1][block_width+1];
    __shared__ float V1_shared[block_height+2][block_width+2];

    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int closed_boundary_cell_north = (int)(bc_north_ == 1);
    const int closed_boundary_cell_south = (int)(bc_south_ == 1);
    
    //Start of block within domain
    const int bx = blockDim.x * blockIdx.x + 1; //Skip global ghost cells
    const int by = blockDim.y * blockIdx.y + 1 + closed_boundary_cell_south; //Skip global ghost cells

    //Index of cell within domain
    const int ti = bx + tx;
    const int tj = by + ty;
    
    //Compute pointer to current row in the V array
    float* const V0_row = (float*) ((char*) V0_ptr_ + V0_pitch_*tj);

    //Read current V
    float V0 = 0.0f;
    // if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_) {
    //if ( ti >= 1 && ti <= nx_ && tj >= 2 && tj <= ny_) {
    if (ti >= 1 && ti <= nx_ && tj >= 1+closed_boundary_cell_south && tj <= ny_+1-closed_boundary_cell_north) {        
        V0 = V0_row[ti];
    }

    //Read H and eta into shared memory: (nx+2)*(ny+1) cells
    for (int j=ty; j<block_height+1; j+=blockDim.y) {
        // "fake" global ghost cells by clamping
        // const int l = clamp(by + j, 1, ny_);
        const int l = by + j - 1;
        if (l >= 0 && l <= ny_+1) {
        
            //Compute the pointer to current row in the H and eta arrays
            float* const H_row = (float*) ((char*) H_ptr_ + H_pitch_*l);
            float* const eta1_row = (float*) ((char*) eta1_ptr_ + eta1_pitch_*l);

            for (int i=tx; i<block_width+2; i+=blockDim.x) {
                // "fake" global ghost cells by clamping
                // const int k = clamp(bx + i - 1, 1, nx_);
                const int k = bx + i - 1;
                if (k >= 0 && k <= nx_+1) {
                    H_shared[j][i] = H_row[k];
                    eta1_shared[j][i] = eta1_row[k];
                }
            }
        }
    }

    //Read U into shared memory: (nx+1)*(ny+1) cells
    for (int j=ty; j<block_height+1; j+=blockDim.y) {
        // "fake" ghost cells by clamping
        // const int l = clamp(by + j, 1, ny_);
        const int l = by + j - 1;
        if ( l >= 0 && l <= ny_+1) {

            //Compute the pointer to current row in the U array
            float* const U1_row = (float*) ((char*) U1_ptr_ + U1_pitch_*l);

            for (int i=tx; i<block_width+1; i+=blockDim.x) {
                // Prevent out-of-bounds
                // const int k = clamp(bx + i - 1, 0, nx_);
                const int k = bx + i;
                if ( k >= 0 && k <= nx_+2) {                
                    U1_shared[j][i] = U1_row[k];
                }
            }
        }
    }
    

    //Read V into shared memory: (nx+2)*(ny+2) cells
    for (int j=ty; j<block_height+2; j+=blockDim.y) {
        // Prevent out-of-bounds
        // const int l = clamp(by + j - 1, 0, ny_);
        const int l = by + j - 1;
        if (l >= 0 && l <= ny_+2) {

            //Compute the pointer to current row in the U array
            float* const V1_row = (float*) ((char*) V1_ptr_ + V1_pitch_*l);

            for (int i=tx; i<block_width+2; i+=blockDim.x) {
                // "fake" ghost cells by clamping
                // const int k = clamp(bx + i - 1, 1, nx_);
                const int k = bx + i - 1;
                if (k >= 0 && k <= nx_+1) {
                    V1_shared[j][i] = V1_row[k];
                }
            }
        }
    }

    //Make sure all threads have read into shared mem
    __syncthreads();
    
    /**
      * Now get all our required variables as short-hands
      * here we use the notation of
      *  Var_00 as var_i,j
      *  Var_p0 as var_i+1,j
      *  Var_0m as var_i,j-1
      * etc
      */
    const float V_00 = V1_shared[ty+1][tx+1]; //V at "center"
    const float V_0p = V1_shared[ty+2][tx+1]; //V at "north"
    const float V_0m = V1_shared[ty  ][tx+1]; //V at "south"
    const float V_p0 = V1_shared[ty+1][tx+2]; //V at "east"
    const float V_m0 = V1_shared[ty+1][tx  ]; //V at "west"
    
    const float U_00 = U1_shared[ty  ][tx+1];
    const float U_0p = U1_shared[ty+1][tx+1];
    const float U_m0 = U1_shared[ty  ][tx  ];
    const float U_mp = U1_shared[ty+1][tx  ];
    
    const float H_m0 = H_shared[ty  ][tx  ]; 
    const float H_00 = H_shared[ty  ][tx+1]; 
    const float H_p0 = H_shared[ty  ][tx+2];
    const float H_mp = H_shared[ty+1][tx  ];
    const float H_0p = H_shared[ty+1][tx+1]; 
    const float H_pp = H_shared[ty+1][tx+2];
    
    const float eta_m0 = eta1_shared[ty  ][tx  ]; 
    const float eta_00 = eta1_shared[ty  ][tx+1]; 
    const float eta_p0 = eta1_shared[ty  ][tx+2];
    const float eta_mp = eta1_shared[ty+1][tx  ]; 
    const float eta_0p = eta1_shared[ty+1][tx+1]; 
    const float eta_pp = eta1_shared[ty+1][tx+2];

    // Coriolis at U positions:
    const float glob_thread_y = blockIdx.y * blockDim.y + threadIdx.y;
    const float f_u_0 = linear_coriolis_term(f_, beta_, glob_thread_y-0.5f, dy_, y_zero_reference_cell_);
    const float f_u_p = linear_coriolis_term(f_, beta_, glob_thread_y+0.5f, dy_, y_zero_reference_cell_);

    //Reconstruct H_bar and H_y (at the V position)
    const float H_bar_m0 = 0.25f*(H_m0 + H_mp + H_00 + H_0p);
    const float H_bar_00 = 0.25f*(H_00 + H_0p + H_p0 + H_pp);
    const float H_y = 0.5f*(H_00 + H_0p);
    
    //Reconstruct Eta_bar at the V position
    const float eta_bar_m0 = 0.25f*(eta_m0 + eta_mp + eta_00 + eta_0p);
    const float eta_bar_00 = 0.25f*(eta_00 + eta_0p + eta_p0 + eta_pp);

    //Reconstruct fU at the V position
    const float fU_bar = 0.25f*( f_u_0*(U_m0 + U_00) + f_u_p*(U_mp + U_0p) );

    //Calculate the friction coefficient
    const float C = 1.0f + 2.0f*r_*dt_/H_y + 2.0f*A_*dt_*(dx_*dx_ + dy_*dy_)/(dx_*dx_*dy_*dy_);

    //Calculate the pressure/gravitational effect
    const float h_0p = H_0p + eta_0p;
    const float h_00 = H_00 + eta_00;
    const float h_y = 0.5f*(h_00 + h_0p); //Could possibly use h for pressure terms instead of H
    const float P_y_hat = -0.5f*g_*(eta_0p*eta_0p - eta_00*eta_00);
    const float P_y = -g_*h_y*(eta_0p - eta_00) + P_y_hat;
    
    //Calculate nonlinear effects
    const float N_a = (V_0p + V_00)*(V_0p + V_00) / (H_0p + eta_0p);
    const float N_b = (V_00 + V_0m)*(V_00 + V_0m) / (H_00 + eta_00);
    const float N_c = (U_0p + U_00)*(V_p0 + V_00) / (H_bar_00 + eta_bar_00);
    const float N_d = (U_mp + U_m0)*(V_00 + V_m0) / (H_bar_m0 + eta_bar_m0);
    float N = 0.25f*( N_a - N_b + (dy_/dx_)*(N_c - N_d) );
    
    //Calculate eddy viscosity term
    float E = (V_p0 - V0 + V_m0)/(dx_*dx_) + (V_0p - V0 + V_0m)/(dy_*dy_);

    //Calculate the wind shear stress
    //FIXME Check coordinates (ti_, tj_) here!!!
    //TODO Check coordinates (ti_, tj_) here!!!
    //WARNING Check coordinates (ti_, tj_) here!!!
    float Y = windStressY(wind_stress_t_, ti+0.5, tj, nx_, ny_);
    
    // Finding the contribution from Coriolis
    float global_thread_y = blockIdx.y * blockDim.y + threadIdx.y;
    float coriolis_f = linear_coriolis_term(f_, beta_, global_thread_y, dy_, y_zero_reference_cell_);

    //Compute the V at the next timestep
    float V2 = (V0 + 2.0f*dt_*(-fU_bar + (N + P_y)/dy_ + Y + A_*E) ) / C;

    //Write to main memory for internal cells
    //if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_) {
    //if ( ti >= 1 && ti <= nx_ && tj >= 2 && tj <= ny_) {
    if (ti >= 1 && ti <= nx_ && tj >= 1+closed_boundary_cell_south && tj <= ny_+1-closed_boundary_cell_north) {        
        V0_row[ti] = V2;
    }
}
} // extern "C" {