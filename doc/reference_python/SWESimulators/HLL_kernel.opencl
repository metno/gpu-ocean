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





/**
  * Computes the flux along the x axis for all faces
  */
void computeFluxF(__local float Q[3][block_height+2][block_width+2],
                  __local float F[3][block_height+1][block_width+1],
                  const float g_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    for (int j=ty; j<block_height; j+=get_local_size(1)) {   
        const int l = j + 1; //Skip ghost cells     
        for (int i=tx; i<block_width+1; i+=get_local_size(0)) { 
            const int k = i;
            
            const float3 Q_l  = (float3)(Q[0][l][k  ], Q[1][l][k  ], Q[2][l][k  ]);
            const float3 Q_r  = (float3)(Q[0][l][k+1], Q[1][l][k+1], Q[2][l][k+1]);
            
            const float3 flux = HLL_flux(Q_l, Q_r, g_);
            
            //Write to shared memory
            F[0][j][i] = flux.x;
            F[1][j][i] = flux.y;
            F[2][j][i] = flux.z;
        }
    }
}





/**
  * Computes the flux along the x axis for all faces
  */
void computeFluxG(__local float Q[3][block_height+2][block_width+2],
                  __local float G[3][block_height+1][block_width+1],
                  const float g_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    for (int j=ty; j<block_height+1; j+=get_local_size(1)) {
        const int l = j;
        for (int i=tx; i<block_width; i+=get_local_size(0)) {            
            const int k = i + 1; //Skip ghost cells
            
            //NOte that hu and hv are swapped ("transposing" the domain)!
            const float3 Q_l = (float3)(Q[0][l  ][k], Q[2][l  ][k], Q[1][l  ][k]);
            const float3 Q_r = (float3)(Q[0][l+1][k], Q[2][l+1][k], Q[1][l+1][k]);
                                       
            // Computed flux
            const float3 flux = HLL_flux(Q_l, Q_r, g_);
            
            //Write to shared memory
            //Note that we here swap hu and hv back to the original
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
    //Shared memory variables
    __local float Q[3][block_height+2][block_width+2];
    __local float F[3][block_height+1][block_width+1];
    
    
    //Read into shared memory
    readBlock1(h0_ptr_, h0_pitch_,
               hu0_ptr_, hu0_pitch_,
               hv0_ptr_, hv0_pitch_,
               Q, nx_, ny_);
    barrier(CLK_LOCAL_MEM_FENCE);

    noFlowBoundary1(Q, nx_, ny_);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //Compute F flux
    computeFluxF(Q, F, g_);
    barrier(CLK_LOCAL_MEM_FENCE);
    evolveF1(Q, F, nx_, ny_, dx_, dt_);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //Set boundary conditions
    noFlowBoundary1(Q, nx_, ny_);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //Compute G flux
    computeFluxG(Q, F, g_);
    barrier(CLK_LOCAL_MEM_FENCE);
    evolveG1(Q, F, nx_, ny_, dy_, dt_);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    
    
    // Write to main memory for all internal cells
    writeBlock1(h1_ptr_, h1_pitch_,
                hu1_ptr_, hu1_pitch_,
                hv1_ptr_, hv1_pitch_,
                Q, nx_, ny_);
}