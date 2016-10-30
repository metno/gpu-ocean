/*
This OpenCL kernel implements the Kurganov-Petrova numerical scheme 
for the shallow water equations, described in 
A. Kurganov & Guergana Petrova
A Second-Order Well-Balanced Positivity Preserving Central-Upwind
Scheme for the Saint-Venant System Communications in Mathematical
Sciences, 5 (2007), 133-160. 

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



void computeFluxF(__local float Q[3][block_height+4][block_width+4],
                  __local float Qx[3][block_height+2][block_width+2],
                  __local float F[3][block_height+1][block_width+1],
                  const float g_, const float dx_, const float dt_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    for (int j=ty; j<block_height; j+=get_local_size(1)) {
        const int l = j + 2; //Skip ghost cells
        for (int i=tx; i<block_width+1; i+=get_local_size(0)) {
            const int k = i + 1;
            // Reconstruct point values of Q at the left and right hand side 
            // of the cell for both the left (i) and right (i+1) cell 
            const float3 Q_rl = (float3)(Q[0][l][k+1] - 0.5f*Qx[0][j][i+1],
                                         Q[1][l][k+1] - 0.5f*Qx[1][j][i+1],
                                         Q[2][l][k+1] - 0.5f*Qx[2][j][i+1]);
            const float3 Q_rr = (float3)(Q[0][l][k+1] + 0.5f*Qx[0][j][i+1],
                                         Q[1][l][k+1] + 0.5f*Qx[1][j][i+1],
                                         Q[2][l][k+1] + 0.5f*Qx[2][j][i+1]);
                                         
            const float3 Q_ll = (float3)(Q[0][l][k] - 0.5f*Qx[0][j][i],
                                         Q[1][l][k] - 0.5f*Qx[1][j][i],
                                         Q[2][l][k] - 0.5f*Qx[2][j][i]);
            const float3 Q_lr = (float3)(Q[0][l][k] + 0.5f*Qx[0][j][i],
                                         Q[1][l][k] + 0.5f*Qx[1][j][i],
                                         Q[2][l][k] + 0.5f*Qx[2][j][i]);
                                    
            //Evolve half a timestep (predictor step)
            const float3 Q_r_bar = Q_rl + dt_/(2.0f*dx_) * (F_func(Q_rl, g_) - F_func(Q_rr, g_));
            const float3 Q_l_bar = Q_lr + dt_/(2.0f*dx_) * (F_func(Q_ll, g_) - F_func(Q_lr, g_));

            // Compute flux based on prediction
            const float3 flux = CentralUpwindFlux(Q_l_bar, Q_r_bar, g_);
            
            //Write to shared memory
            F[0][j][i] = flux.x;
            F[1][j][i] = flux.y;
            F[2][j][i] = flux.z;
        }
    }    
}

void computeFluxG(__local float Q[3][block_height+4][block_width+4],
                  __local float Qy[3][block_height+2][block_width+2],
                  __local float G[3][block_height+1][block_width+1],
                  const float g_, const float dy_, const float dt_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    for (int j=ty; j<block_height+1; j+=get_local_size(1)) {
        const int l = j + 1;
        for (int i=tx; i<block_width; i+=get_local_size(0)) {            
            const int k = i + 2; //Skip ghost cells
            // Reconstruct point values of Q at the left and right hand side 
            // of the cell for both the left (i) and right (i+1) cell 
            //NOte that hu and hv are swapped ("transposing" the domain)!
            const float3 Q_rl = (float3)(Q[0][l+1][k] - 0.5f*Qy[0][j+1][i],
                                         Q[2][l+1][k] - 0.5f*Qy[2][j+1][i],
                                         Q[1][l+1][k] - 0.5f*Qy[1][j+1][i]);
            const float3 Q_rr = (float3)(Q[0][l+1][k] + 0.5f*Qy[0][j+1][i],
                                         Q[2][l+1][k] + 0.5f*Qy[2][j+1][i],
                                         Q[1][l+1][k] + 0.5f*Qy[1][j+1][i]);
                                        
            const float3 Q_ll = (float3)(Q[0][l][k] - 0.5f*Qy[0][j][i],
                                         Q[2][l][k] - 0.5f*Qy[2][j][i],
                                         Q[1][l][k] - 0.5f*Qy[1][j][i]);
            const float3 Q_lr = (float3)(Q[0][l][k] + 0.5f*Qy[0][j][i],
                                         Q[2][l][k] + 0.5f*Qy[2][j][i],
                                         Q[1][l][k] + 0.5f*Qy[1][j][i]);
                                     
            //Evolve half a timestep (predictor step)
            const float3 Q_r_bar = Q_rl + dt_/(2.0f*dy_) * (F_func(Q_rl, g_) - F_func(Q_rr, g_));
            const float3 Q_l_bar = Q_lr + dt_/(2.0f*dy_) * (F_func(Q_ll, g_) - F_func(Q_lr, g_));
            
            // Compute flux based on prediction
            const float3 flux = CentralUpwindFlux(Q_l_bar, Q_r_bar, g_);
            
            //Write to shared memory
            //Note that we here swap hu and hv back to the original
            G[0][j][i] = flux.x;
            G[1][j][i] = flux.z;
            G[2][j][i] = flux.y;
        }
    }
}




/**
  * This unsplit kernel computes the 2D numerical scheme with a TVD RK2 time integration scheme
  */
__kernel void swe_2D(
        int nx_, int ny_,
        float dx_, float dy_, float dt_,
        float g_,
        
        float theta_,
        
        int step_,
        
        //Input h^n
        __global float* h0_ptr_, int h0_pitch_,
        __global float* hu0_ptr_, int hu0_pitch_,
        __global float* hv0_ptr_, int hv0_pitch_,
        
        //Output h^{n+1}
        __global float* h1_ptr_, int h1_pitch_,
        __global float* hu1_ptr_, int hu1_pitch_,
        __global float* hv1_ptr_, int hv1_pitch_) {
        
        
    //Shared memory variables
    __local float Q[3][block_height+4][block_width+4];
    __local float Qx[3][block_height+2][block_width+2];
    __local float F[3][block_height+1][block_width+1];
    
    
    
    //Read into shared memory
    readBlock2(h0_ptr_, h0_pitch_,
               hu0_ptr_, hu0_pitch_,
               hv0_ptr_, hv0_pitch_,
               Q, nx_, ny_);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    //Fix boundary conditions
    noFlowBoundary2(Q, nx_, ny_);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    
    //Step 0 => evolve x first, then y
    if (step_ == 0) {
        //Compute fluxes along the x axis and evolve
        minmodSlopeX(Q, Qx, theta_);
        barrier(CLK_LOCAL_MEM_FENCE);
        computeFluxF(Q, Qx, F, g_, dx_, dt_);
        barrier(CLK_LOCAL_MEM_FENCE);
        evolveF2(Q, F, nx_, ny_, dx_, dt_);
        barrier(CLK_LOCAL_MEM_FENCE);
        
        //Set boundary conditions
        noFlowBoundary2(Q, nx_, ny_);
        barrier(CLK_LOCAL_MEM_FENCE);
        
        //Compute fluxes along the y axis and evolve
        minmodSlopeY(Q, Qx, theta_);
        barrier(CLK_LOCAL_MEM_FENCE);
        computeFluxG(Q, Qx, F, g_, dy_, dt_);
        barrier(CLK_LOCAL_MEM_FENCE);
        evolveG2(Q, F, nx_, ny_, dy_, dt_);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    //Step 1 => evolve y first, then x
    else {
        //Compute fluxes along the y axis and evolve
        minmodSlopeY(Q, Qx, theta_);
        barrier(CLK_LOCAL_MEM_FENCE);
        computeFluxG(Q, Qx, F, g_, dy_, dt_);
        barrier(CLK_LOCAL_MEM_FENCE);
        evolveG2(Q, F, nx_, ny_, dy_, dt_);
        barrier(CLK_LOCAL_MEM_FENCE);
        
        //Set boundary conditions
        noFlowBoundary2(Q, nx_, ny_);
        barrier(CLK_LOCAL_MEM_FENCE);
        
        //Compute fluxes along the x axis and evolve
        minmodSlopeX(Q, Qx, theta_);
        barrier(CLK_LOCAL_MEM_FENCE);
        computeFluxF(Q, Qx, F, g_, dx_, dt_);
        barrier(CLK_LOCAL_MEM_FENCE);
        evolveF2(Q, F, nx_, ny_, dx_, dt_);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    
    // Write to main memory for all internal cells
    writeBlock2(h1_ptr_, h1_pitch_,
                hu1_ptr_, hu1_pitch_,
                hv1_ptr_, hv1_pitch_,
                Q, nx_, ny_);
}