/*
This software is part of GPU Ocean.

Copyright (C) 2019  SINTEF Digital

Implements kernels for obtaining maximum step size for finite-volume
schemes implemented in GPU Ocean.

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


// Requires that the following names are defined when the kernel is built:
// NUM_THREADS 
// block_height
// block_width


#define FLT_MAX 10000.0

extern "C" {
/*
 * Find the maximum dt allowed within each block based on the current ocean state.
 */
__global__ void per_block_max_dt(
        //Discretization parameters
        const int nx_, const int ny_,
        const float dx_, const float dy_,
        const float g_,
        float* eta_ptr_, const int eta_pitch_,
        float* hu_ptr_,  const int hu_pitch_,
        float* hv_ptr_,  const int hv_pitch_,
        float* Hm_ptr_,  const int Hm_pitch_,
        float* dt_ptr_,  const int dt_pitch_)
{
    
    // Block ID for output
    const int block_id_x = blockIdx.x;
    const int block_id_y = blockIdx.y;
    
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    //Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x + 2; //Skip global ghost cells, i.e., +2
    const int tj = blockIdx.y * blockDim.y + threadIdx.y + 2;
    
    __shared__ float shared_dt[block_height][block_width];

    if ((ti < nx_+2) && (tj < ny_+2)) {
        float* const eta_row = (float*) ((char*) eta_ptr_ + eta_pitch_*tj);
        float* const hu_row  = (float*) ((char*) hu_ptr_  +  hu_pitch_*tj);
        float* const hv_row  = (float*) ((char*) hv_ptr_  +  hv_pitch_*tj);
        float* const Hm_row  = (float*) ((char*) Hm_ptr_  +  Hm_pitch_*tj);
        
        float const eta = eta_row[ti];
        float const h   = eta + Hm_row[ti];
        float const u   = hu_row[ti]/h;
        float const v   = hv_row[ti]/h;
        
        float const gravity_wave = sqrt(g_*h);
        
        float dt_thread =          0.25f*dx_/abs(u + gravity_wave);
        dt_thread = min(dt_thread, 0.25f*dx_/abs(u - gravity_wave));
        dt_thread = min(dt_thread, 0.25f*dy_/abs(v + gravity_wave));
        dt_thread = min(dt_thread, 0.25f*dy_/abs(v - gravity_wave));
        
        shared_dt[ty][tx] = dt_thread;
    }
    else {
        shared_dt[ty][tx] = FLT_MAX;
    }
    
    __syncthreads();
  
    
    if ( (tx == 0) && (ty == 0)) {
        
        float min_dt = shared_dt[0][0];
        for (int j=0; j < blockDim.y; j++) {
            for (int i=0; i < blockDim.x; i++) {
                min_dt = min(min_dt, shared_dt[j][i]);
            }
        }
        
        float* const dt_row = (float*) ((char*) dt_ptr_ + dt_pitch_*block_id_y);
        dt_row[block_id_x] = min_dt;
    }
    
}
} // extern "C"
   
