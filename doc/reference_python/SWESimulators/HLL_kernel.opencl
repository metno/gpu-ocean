/*
This OpenCL kernel implements the HLL flux

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



typedef __local float cell_shmem[block_height+2][block_width+2];
typedef __local float u_edge_shmem[block_height+2][block_width+1];
typedef __local float v_edge_shmem[block_height+1][block_width+2];









__kernel void swe_2D(
        int nx_, int ny_,
        float dx_, float dy_, float dt_,
        float g_,
        
        //Input h^n
        __global float* h0_ptr_, int h0_pitch_,
        __global float* hu0_ptr_, int hu0_pitch_,
        __global float* hv0_ptr_, int hv0_pitch_,
        
        //Output h^{n+1}
        __global float* h1_ptr_, int h1_pitch_,
        __global float* hu1_ptr_, int hu1_pitch_,
        __global float* hv1_ptr_, int hv1_pitch_) {
        
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of block within domain
    const int bx = get_local_size(0) * get_group_id(0);
    const int by = get_local_size(1) * get_group_id(1);

    //Index of cell within domain
    const int ti = get_global_id(0) + 1; //Skip global ghost cells, i.e., +1
    const int tj = get_global_id(1) + 1;
    
    //Conserved variables
    cell_shmem h;
    cell_shmem hu;
    cell_shmem hv;
    
    //Intermediate flux variables
    u_edge_shmem h_f_flux;
    u_edge_shmem hu_f_flux;
    u_edge_shmem hv_f_flux;
    
    v_edge_shmem h_g_flux;
    v_edge_shmem hu_g_flux;
    v_edge_shmem hv_g_flux;
    
    
    //Read into shared memory
    for (int j=ty; j<block_height+2; j+=get_local_size(1)) {
        const int l = clamp(by + j, 0, ny_+1); // Out of bounds
        
        //Compute the pointer to current row in the arrays
        __global float* const h_row = (__global float*) ((__global char*) h0_ptr_ + h0_pitch_*l);
        __global float* const hu_row = (__global float*) ((__global char*) hu0_ptr_ + hu0_pitch_*l);
        __global float* const hv_row = (__global float*) ((__global char*) hv0_ptr_ + hv0_pitch_*l);
        
        for (int i=tx; i<block_width+2; i+=get_local_size(0)) {
            const int k = clamp(bx + i, 0, nx_+1); // Out of bounds
            
            h[j][i] = h_row[k];
            hu[j][i] = hu_row[k];
            hv[j][i] = hv_row[k];
        }
    }
    
    //Make sure all threads have read into shared mem
    __syncthreads();
    

    {
        const int i = tx + 1; //Skip local ghost cells, i.e., +1
        const int j = ty + 1;
        
        //Fix boundary conditions
        if (ti == 1) {
            h[j][i-1]  =   h[j][i];
            hu[j][i-1] = -hu[j][i];
            hv[j][i-1] =  hv[j][i];
        }
        if (ti == nx_) {
            h[j][i+1]  =   h[j][i];
            hu[j][i+1] = -hu[j][i];
            hv[j][i+1] =  hv[j][i];
        }
        if (tj == 1) {
            h[j-1][i]  =   h[j][i];
            hu[j-1][i] =  hu[j][i];
            hv[j-1][i] = -hv[j][i];
        }
        if (tj == ny_) {
            h[j+1][i]  =   h[j][i];
            hu[j+1][i] =  hu[j][i];
            hv[j+1][i] = -hv[j][i];
        }
        
    }
    __syncthreads();
    
    
    
    //Compute F flux
    for (int j=ty; j<block_height+2; j+=get_local_size(1)) {        
        for (int i=tx; i<block_width+1; i+=get_local_size(0)) { //Note: only compute for edges, thus width+1 only.
            const float3 Q_l  = (float3)(h[j][i], hu[j][i], hv[j][i]);
            const float3 Q_r  = (float3)(h[j][i+1], hu[j][i+1], hv[j][i+1]);
            
            const float3 flux = HLL_flux(Q_l, Q_r, g_);
            
            //Write to shared memory
            h_f_flux[j][i] = flux.x;
            hu_f_flux[j][i] = flux.y;
            hv_f_flux[j][i] = flux.z;
        }
    }
    
    //Compute G flux
    for (int j=ty; j<block_height+1; j+=get_local_size(1)) { //Note: only compute for edges, thus width+1 only.
        for (int i=tx; i<block_width+2; i+=get_local_size(0)) { 
            //NOte that hu and hv are swapped ("transposing" the domain)!
            const float3 Q_l  = (float3)(h[j][i], hv[j][i], hu[j][i]);
            const float3 Q_r  = (float3)(h[j+1][i], hv[j+1][i], hu[j+1][i]);
            
            const float3 flux = HLL_flux(Q_l, Q_r, g_);
            
            //Write to shared memory
            //Note that we here swap hu and hv back to the original
            h_g_flux[j][i] = flux.x;
            hu_g_flux[j][i] = flux.z;
            hv_g_flux[j][i] = flux.y;
        }
    }
        
    

    //Make sure all threads have finished computing fluxes
    __syncthreads();
    
    
    
    //Compute for all internal cells
    if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_+1) {
        const int i = tx + 1; //Skip local ghost cells, i.e., +1
        const int j = ty + 1;
        
        const float h_r = (h_f_flux[j][i-1] - h_f_flux[j][i]) / dx_
                        + (h_g_flux[j-1][i] - h_g_flux[j][i]) / dy_;
        const float hu_r = (hu_f_flux[j][i-1] - hu_f_flux[j][i]) / dx_
                         + (hu_g_flux[j-1][i] - hu_g_flux[j][i]) / dy_;
        const float hv_r = (hv_f_flux[j][i-1] - hv_f_flux[j][i]) / dx_
                         + (hv_g_flux[j-1][i] - hv_g_flux[j][i]) / dy_;
        
        const float h1 = h[j][i] + dt_*h_r;
        const float hu1 = hu[j][i] + dt_*hu_r;
        const float hv1 = hv[j][i] + dt_*hv_r;

        __global float* const h_row  = (__global float*) ((__global char*) h1_ptr_ + h1_pitch_*tj);
        __global float* const hu_row = (__global float*) ((__global char*) hu1_ptr_ + hu1_pitch_*tj);
        __global float* const hv_row = (__global float*) ((__global char*) hv1_ptr_ + hv1_pitch_*tj);
        
        h_row[ti] = h1;
        hu_row[ti] = hu1;
        hv_row[ti] = hv1;
    }
}