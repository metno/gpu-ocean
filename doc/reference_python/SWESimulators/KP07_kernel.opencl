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
                  const float g_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    for (int j=ty; j<block_height; j+=get_local_size(1)) {
        const int l = j + 2; //Skip ghost cells
        for (int i=tx; i<block_width+1; i+=get_local_size(0)) {
            const int k = i + 1;
            // Q at interface from the right and left
            const float3 Qp = (float3)(Q[0][l][k+1] - 0.5f*Qx[0][j][i+1],
                                       Q[1][l][k+1] - 0.5f*Qx[1][j][i+1],
                                       Q[2][l][k+1] - 0.5f*Qx[2][j][i+1]);
            const float3 Qm = (float3)(Q[0][l][k  ] + 0.5f*Qx[0][j][i  ],
                                       Q[1][l][k  ] + 0.5f*Qx[1][j][i  ],
                                       Q[2][l][k  ] + 0.5f*Qx[2][j][i  ]);
                                       
            // Computed flux
            const float3 flux = CentralUpwindFlux(Qm, Qp, g_);
            F[0][j][i] = flux.x;
            F[1][j][i] = flux.y;
            F[2][j][i] = flux.z;
        }
    }    
}

void computeFluxG(__local float Q[3][block_height+4][block_width+4],
                  __local float Qy[3][block_height+2][block_width+2],
                  __local float G[3][block_height+1][block_width+1],
                  const float g_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    for (int j=ty; j<block_height+1; j+=get_local_size(1)) {
        const int l = j + 1;
        for (int i=tx; i<block_width; i+=get_local_size(0)) {            
            const int k = i + 2; //Skip ghost cells
            // Q at interface from the right and left
            // Note that we swap hu and hv
            const float3 Qp = (float3)(Q[0][l+1][k] - 0.5f*Qy[0][j+1][i],
                                       Q[2][l+1][k] - 0.5f*Qy[2][j+1][i],
                                       Q[1][l+1][k] - 0.5f*Qy[1][j+1][i]);
            const float3 Qm = (float3)(Q[0][l  ][k] + 0.5f*Qy[0][j  ][i],
                                       Q[2][l  ][k] + 0.5f*Qy[2][j  ][i],
                                       Q[1][l  ][k] + 0.5f*Qy[1][j  ][i]);
                                       
            // Computed flux
            // Note that we swap back
            const float3 flux = CentralUpwindFlux(Qm, Qp, g_);
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
        
        float f_, //< Coriolis coefficient
        float r_, //< Bottom friction coefficient
        
        int step_,
        
        //Input h^n
        __global float* h0_ptr_, int h0_pitch_,
        __global float* hu0_ptr_, int hu0_pitch_,
        __global float* hv0_ptr_, int hv0_pitch_,
        
        //Output h^{n+1}
        __global float* h1_ptr_, int h1_pitch_,
        __global float* hu1_ptr_, int hu1_pitch_,
        __global float* hv1_ptr_, int hv1_pitch_,
        
        //Wind stress parameters
        int wind_stress_type_, 
        float tau0_, float rho_, float alpha_, float xm_, float Rc_,
        float x0_, float y0_,
        float u0_, float v0_,
        float t_) {
        
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of block within domain
    const int bx = get_local_size(0) * get_group_id(0);
    const int by = get_local_size(1) * get_group_id(1);

    //Index of cell within domain
    const int ti = get_global_id(0) + 2; //Skip global ghost cells, i.e., +2
    const int tj = get_global_id(1) + 2;
    
    //Shared memory variables
    __local float Q[3][block_height+4][block_width+4];
    
    //The following slightly wastes memory, but enables us to reuse the 
    //funcitons in common.opencl
    __local float Qx[3][block_height+2][block_width+2];
    __local float Qy[3][block_height+2][block_width+2];
    __local float F[3][block_height+1][block_width+1];
    __local float G[3][block_height+1][block_width+1];
    
    
    
    //Read into shared memory
    readBlock2(h0_ptr_, h0_pitch_,
               hu0_ptr_, hu0_pitch_,
               hv0_ptr_, hv0_pitch_,
               Q, nx_, ny_);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    //Fix boundary conditions
    noFlowBoundary2(Q, nx_, ny_);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    //Reconstruct slopes along x and axis
    minmodSlopeX(Q, Qx, theta_);
    minmodSlopeY(Q, Qy, theta_);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    //Compute fluxes along the x and y axis
    computeFluxF(Q, Qx, F, g_);
    computeFluxG(Q, Qy, G, g_);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    //Sum fluxes and advance in time for all internal cells
    if (ti > 1 && ti < nx_+2 && tj > 1 && tj < ny_+2) {
        const int i = tx + 2; //Skip local ghost cells, i.e., +2
        const int j = ty + 2;
        
        const float X = windStressX(
            wind_stress_type_, 
            dx_, dy_, dt_,
            tau0_, rho_, alpha_, xm_, Rc_,
            x0_, y0_,
            u0_, v0_,
            t_);
        const float Y = windStressY(
            wind_stress_type_, 
            dx_, dy_, dt_,
            tau0_, rho_, alpha_, xm_, Rc_,
            x0_, y0_,
            u0_, v0_,
            t_);
        
        const float h1  = Q[0][j][i] + (F[0][ty][tx] - F[0][ty  ][tx+1]) * dt_ / dx_ 
                                     + (G[0][ty][tx] - G[0][ty+1][tx  ]) * dt_ / dy_;
        const float hu1 = Q[1][j][i] + (F[1][ty][tx] - F[1][ty  ][tx+1]) * dt_ / dx_ 
                                     + (G[1][ty][tx] - G[1][ty+1][tx  ]) * dt_ / dy_
                                     + dt_*X - dt_*f_*Q[2][j][i];
        const float hv1 = Q[2][j][i] + (F[2][ty][tx] - F[2][ty  ][tx+1]) * dt_ / dx_ 
                                     + (G[2][ty][tx] - G[2][ty+1][tx  ]) * dt_ / dy_
                                     + dt_*Y + dt_*f_*Q[1][j][i];

        __global float* const h_row  = (__global float*) ((__global char*) h1_ptr_ + h1_pitch_*tj);
        __global float* const hu_row = (__global float*) ((__global char*) hu1_ptr_ + hu1_pitch_*tj);
        __global float* const hv_row = (__global float*) ((__global char*) hv1_ptr_ + hv1_pitch_*tj);
        
        const float C = 2.0f*r_*dt_/Q[0][j][i];
                    
        if  (step_ == 0) {
            //First step of RK2 ODE integrator
            
            h_row[ti] = h1;
            hu_row[ti] = hu1 / (1.0f + C);
            hv_row[ti] = hv1 / (1.0f + C);
        }
        else if (step_ == 1) {
            //Second step of RK2 ODE integrator
            
            //First read Q^n
            const float h_a  = h_row[ti];
            const float hu_a = hu_row[ti];
            const float hv_a = hv_row[ti];
            
            //Compute Q^n+1
            const float h_b  = 0.5f*(h_a + h1);
            const float hu_b = 0.5f*(hu_a + hu1);
            const float hv_b = 0.5f*(hv_a + hv1);
            
            //Write to main memory
            h_row[ti] = h_b;
            hu_row[ti] = hu_b / (1.0f + 0.5f*C);
            hv_row[ti] = hv_b / (1.0f + 0.5f*C);
        }
    }
}