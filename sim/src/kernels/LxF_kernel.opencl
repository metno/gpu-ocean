/*
This OpenCL kernel implements the classical Lax-Friedrichs scheme
for the shallow water equations, with edge fluxes.

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
  * Computes the flux along the x axis for all faces
  */
void computeFluxF(__local float Q[3][block_height+2][block_width+2],
                  __local float F[3][block_height][block_width+1],
                  const float g_, const float dx_, const float dt_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    for (int j=ty; j<block_height; j+=get_local_size(1)) {
        const int l = j + 1; //Skip ghost cells
        for (int i=tx; i<block_width+1; i+=get_local_size(0)) {
            const int k = i;
            
            // Q at interface from the right and left
            const float3 Qp = (float3)(Q[0][l][k+1],
                                       Q[1][l][k+1],
                                       Q[2][l][k+1]);
            const float3 Qm = (float3)(Q[0][l][k],
                                       Q[1][l][k],
                                       Q[2][l][k]);
                                       
            // Computed flux
            const float3 flux = LxF_2D_flux(Qm, Qp, g_, dx_, dt_);
            F[0][j][i] = flux.x;
            F[1][j][i] = flux.y;
            F[2][j][i] = flux.z;
        }
    }
}


/**
  * Computes the flux along the y axis for all faces
  */
void computeFluxG(__local float Q[3][block_height+2][block_width+2],
                  __local float G[3][block_height+1][block_width],
                  const float g_, const float dy_, const float dt_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    for (int j=ty; j<block_height+1; j+=get_local_size(1)) {
        const int l = j;
        for (int i=tx; i<block_width; i+=get_local_size(0)) {
            const int k = i + 1; //Skip ghost cells
            
            // Q at interface from the right and left
            // Note that we swap hu and hv
            const float3 Qp = (float3)(Q[0][l+1][k],
                                       Q[2][l+1][k],
                                       Q[1][l+1][k]);
            const float3 Qm = (float3)(Q[0][l][k],
                                       Q[2][l][k],
                                       Q[1][l][k]);

            // Computed flux
            // Note that we swap back
            const float3 flux = LxF_2D_flux(Qm, Qp, g_, dy_, dt_);
            G[0][j][i] = flux.x;
            G[1][j][i] = flux.z;
            G[2][j][i] = flux.y;
        }
    }  
}


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
            
    //Index of cell within domain
    const int ti = get_global_id(0) + 1; //Skip global ghost cells, i.e., +1
    const int tj = get_global_id(1) + 1;
    
    __local float Q[3][block_height+2][block_width+2];
    __local float F[3][block_height][block_width+1];
    __local float G[3][block_height+1][block_width];
    
    //Read into shared memory
    readBlock1(h0_ptr_, h0_pitch_,
               hu0_ptr_, hu0_pitch_,
               hv0_ptr_, hv0_pitch_,
               Q, nx_, ny_);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //Set boundary conditions
    noFlowBoundary1(Q, nx_, ny_);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    //Compute fluxes along the x and y axis
    computeFluxF(Q, F, g_, dx_, dt_);
    computeFluxG(Q, G, g_, dy_, dt_);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    //Evolve for all internal cells
    if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_+1) {
        //Index of thread within block
        const int tx = get_local_id(0);
        const int ty = get_local_id(1);
        
        const int i = tx + 1; //Skip local ghost cells, i.e., +1
        const int j = ty + 1;
        
        const float h1  = Q[0][j][i] + (F[0][ty][tx] - F[0][ty  ][tx+1]) * dt_ / dx_ 
                                     + (G[0][ty][tx] - G[0][ty+1][tx  ]) * dt_ / dy_;
        const float hu1 = Q[1][j][i] + (F[1][ty][tx] - F[1][ty  ][tx+1]) * dt_ / dx_ 
                                     + (G[1][ty][tx] - G[1][ty+1][tx  ]) * dt_ / dy_;
        const float hv1 = Q[2][j][i] + (F[2][ty][tx] - F[2][ty  ][tx+1]) * dt_ / dx_ 
                                     + (G[2][ty][tx] - G[2][ty+1][tx  ]) * dt_ / dy_;

        __global float* const h_row  = (__global float*) ((__global char*) h1_ptr_ + h1_pitch_*tj);
        __global float* const hu_row = (__global float*) ((__global char*) hu1_ptr_ + hu1_pitch_*tj);
        __global float* const hv_row = (__global float*) ((__global char*) hv1_ptr_ + hv1_pitch_*tj);
        
        h_row[ti] = h1;
        hu_row[ti] = hu1;
        hv_row[ti] = hv1;
    }
}