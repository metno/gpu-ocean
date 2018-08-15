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
        float* eta0_ptr_, int eta0_pitch_, //eta^n-1 (also used as output, that is eta^n+1)
        float* U1_ptr_, int U1_pitch_, // U^n
        float* V1_ptr_, int V1_pitch_ // V^n
        ) {
    
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    //Start of block within domain
    const int bx = blockDim.x * blockIdx.x + 1; //Skip global ghost cells
    const int by = blockDim.y * blockIdx.y + 1; //Skip global ghost cells

    //Index of cell within domain
    const int ti = bx + tx;
    const int tj = by + ty;
    
    __shared__ float U1_shared[block_height][block_width+1];
    __shared__ float V1_shared[block_height+1][block_width];
    
    //Compute pointer to current row in the U array
    float* eta0_row = (float*) ((char*) eta0_ptr_ + eta0_pitch_*tj);

    //Read current eta
    float eta0 = 0.0f;
    // if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_+1) {
    if ( ti >= 1 && ti <= nx_ && tj >= 1 && tj <= ny_) {
        eta0 = eta0_row[ti];
    }
    
    //Read U into shared memory
    for (int j=ty; j<block_height; j+=blockDim.y) {
        //const int l = clamp(by + j, 1, ny_); // fake ghost cells
        const int l = by + j;
        if (l >= 0 && l <= ny_+1) {

            //Compute the pointer to current row in the V array
            float* const U1_row = (float*) ((char*) U1_ptr_ + U1_pitch_*l);

            for (int i=tx; i<block_width+1; i+=blockDim.x) {
                //const int k = clamp(bx + i - 1, 0, nx_); // prevent out of bounds
                const int k = bx + i;
                if (k >= 0 && k <= nx_+2) {
                    U1_shared[j][i] = U1_row[k];
                }
            }
        }
    }
    
    //Read V into shared memory
    for (int j=ty; j<block_height+1; j+=blockDim.y) {
        //const int l = clamp(by + j - 1, 0, ny_); // prevent out of bounds
        const int l = by + j;
        if (l >= 0 && l <= ny_+2) {

            //Compute the pointer to current row in the V array
            float* const V1_row = (float*) ((char*) V1_ptr_ + V1_pitch_*l);

            for (int i=tx; i<block_width; i+=blockDim.x) {
                //const int k = clamp(bx + i, 1, nx_); // fake ghost cells
                const int k = bx + i;
                if (k > 0 && k <= nx_+1) {
                    V1_shared[j][i] = V1_row[k];
                }
            }
        }
    }

    //Make sure all threads have read into shared mem
    __syncthreads();

    //Compute the H at the next timestep
    float eta2 = eta0 - 2.0f*dt_/dx_ * (U1_shared[ty][tx+1] - U1_shared[ty][tx])
                      - 2.0f*dt_/dy_ * (V1_shared[ty+1][tx] - V1_shared[ty][tx]);
    
    //Write to main memory
    //if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_+1) {
    if ( ti >= 1 && ti <= nx_ && tj >= 1 && tj <= ny_ ) {
        eta0_row[ti] = eta2;
    }
}
