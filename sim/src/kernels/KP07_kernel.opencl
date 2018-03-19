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
#include "../config.h"

#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#define CLK_LOCAL_MEM_FENCE
#endif

// Finds the coriolis term based on the linear Coriolis force
// f = \tilde{f} + beta*(y-y0)
float linear_coriolis_term(const float f, const float beta,
			   const float tj, const float dy,
			   const float y_zero_reference_cell) {
    // y_0 is at the southern face of the row y_zero_reference_cell.
    float y = (tj-y_zero_reference_cell + 0.5f)*dy;
    return f + beta * y;
}

void reconstructHx(__local float RHx[block_height+4][block_width+4],
		   __local float  Hi[block_height+4][block_width+4],
		   const int p,
		   const int q) {
    RHx[q][p] = 0.5f*(Hi[q][p]+Hi[q+1][p]); //0.5*(down+up)
}

void reconstructHy(__local float RHy[block_height+4][block_width+4],
		   __local float  Hi[block_height+4][block_width+4],
		   const int p,
		   const int q) {
    RHy[q][p] = 0.5f*(Hi[q][p]+Hi[q][p+1]); //0.5*(left+right)
}

// Reconstruct depth on the cell faces, store them in RHx and RHy
void reconstructH(__local float RHx[block_height+4][block_width+4],
		  __local float RHy[block_height+4][block_width+4],
		  __local float  Hi[block_height+4][block_width+4]) {
    const int p = get_local_id(0) + 2;
    const int q = get_local_id(1) + 2;
    reconstructHx(RHx, Hi, p, q);
    reconstructHy(RHy, Hi, p, q);

    //Use one warp to perform the extra reconstructions needed
    if (get_local_id(1) == 0) { 
        reconstructHy(RHy, Hi, p, 1);//second row
        reconstructHy(RHy, Hi, p, block_height+2);//second last row
        reconstructHy(RHy, Hi, p, block_height+3);//last row
        if (get_local_id(0) < block_height) {
            reconstructHx(RHx, Hi, 1, p);//second column
            reconstructHx(RHx, Hi, block_width+2, p); //second last column
            reconstructHx(RHx, Hi, block_width+3, p);//last column
        }
    }
}


void adjustSlopeUx(__local float Qx[3][block_height+2][block_width+2],
		   __local float RHx[block_height+4][block_width+4],
		   __local float Q[3][block_height+4][block_width+4],
		   const int p, const int q) {
    // define indices in the Qx world:
    const int pQx = p - 1;
    const int qQx = q - 2;
    
    Qx[0][qQx][pQx] = (Q[0][q][p]-Qx[0][qQx][pQx] < -RHx[q][p]) ?
        (Q[0][q][p] + RHx[q][p]) : Qx[0][qQx][pQx];
    Qx[0][qQx][pQx] = (Q[0][q][p]+Qx[0][qQx][pQx] < -RHx[q][p+1]) ?
        (-RHx[q][p+1] - Q[0][q][p]) : Qx[0][qQx][pQx];
}

void adjustSlopeUy(__local float Qy[3][block_height+2][block_width+2],
		   __local float RHy[block_height+4][block_width+4],
		   __local float Q[3][block_height+4][block_width+4],
		   const int p, const int q) {
    // define indices in the Qy world:
    const int pQy = p - 2;
    const int qQy = q - 1;

    Qy[0][qQy][pQy] = (Q[0][q][p]-Qy[0][qQy][pQy] < -RHy[q][p]) ?
        (Q[0][q][p] + RHy[q][p]) : Qy[0][qQy][pQy];
    Qy[0][qQy][pQy] = (Q[0][q][p]+Qy[0][qQy][pQy] < -RHy[q+1][p]) ?
        (-RHy[q+1][p] - Q[0][q][p]) : Qy[0][qQy][pQy];
}

void adjustSlopes(__local float Qx[3][block_height+2][block_width+2],
		  __local float Qy[3][block_height+2][block_width+2],
		  __local float RHx[block_height+4][block_width+4],
		  __local float RHy[block_height+4][block_width+4],
		  __local float Q[3][block_height+4][block_width+4] ) {
    const int p = get_local_id(0) + 2;
    const int q = get_local_id(1) + 2;

    adjustSlopeUx(Qx, RHx, Q, p, q);
    adjustSlopeUy(Qy, RHy, Q, p, q);

    // Use one warp to perform the extra adjustments
    if (get_local_id(1) == 0) {
        adjustSlopeUy(Qy, RHy, Q, p, 1);
        adjustSlopeUy(Qy, RHy, Q, p, block_height+2);

        if (get_local_id(0) < block_height) {
            adjustSlopeUx(Qx, RHx, Q, 1, p);
            adjustSlopeUx(Qx, RHx, Q, block_width+2, p);
        }
    }
}

float computeDt(const float3 Qp, const float3 Qm, float H, const float g_, const float dx) {
    float hp = Qp.x+H;
    float hm = Qm.x+H;

    // u = hu/h
    float up = (hp > 0.0f) ? Qp.y / hp : 0.0f;
    float um = (hm > 0.0f) ? Qm.y / hm : 0.0f;
    
    // sqrt(gh)
    float cp = sqrt(g_*hp);
    float cm = sqrt(g_*hm);
    
    // max (potential) wave speed, u_max
    float ap = max(max(um+cm, up+cp), 0.0f);
    float am = min(min(um-cm, up-cp), 0.0f);
    float u_max = max(ap, -am);
    
    // cfl = 0.25*dx/u_max
    return (u_max > 0.0f) ? (0.25f*dx / u_max) : FLT_MAX;
}
float computeFluxF(__local float Q[3][block_height+4][block_width+4],
                  __local float Qx[3][block_height+2][block_width+2],
                  __local float F[3][block_height+1][block_width+1],
                  __local float RHx[block_height+4][block_width+4],
                  const float g_, const float dx_) {
    float dt = FLT_MAX;
    
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    for (int j=ty; j<block_height; j+=get_local_size(1)) {
        const int l = j + 2; //Skip ghost cells
        for (int i=tx; i<block_width+1; i+=get_local_size(0)) {
            const int k = i + 1;
            // Q at interface from the right and left
            // In CentralUpwindFlux we need [eta, hu, hv]
            // Subtract the bottom elevation on the relevant face in Q[0]
            float3 Qp = (float3)(Q[0][l][k+1] - Qx[0][j][i+1],
                                 Q[1][l][k+1] - Qx[1][j][i+1],
                                 Q[2][l][k+1] - Qx[2][j][i+1]);
            float3 Qm = (float3)(Q[0][l][k  ] + Qx[0][j][i  ],
                                 Q[1][l][k  ] + Qx[1][j][i  ],
                                 Q[2][l][k  ] + Qx[2][j][i  ]);
                                       
            // Computed flux
            const float3 flux = CentralUpwindFluxBottom(Qm, Qp, RHx[l][k+1], g_);
            F[0][j][i] = flux.x;
            F[1][j][i] = flux.y;
            F[2][j][i] = flux.z;
            
            dt = min(dt, computeDt(Qp, Qm, RHx[l][k+1], g_, dx_));
        }
    }
    
    return dt;
}

float computeFluxG(__local float Q[3][block_height+4][block_width+4],
                  __local float Qy[3][block_height+2][block_width+2],
                  __local float G[3][block_height+1][block_width+1],
                  __local float RHy[block_height+4][block_width+4],
                  const float g_, const float dy_) {
    float dt = FLT_MAX;
    
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    for (int j=ty; j<block_height+1; j+=get_local_size(1)) {
        const int l = j + 1;
        for (int i=tx; i<block_width; i+=get_local_size(0)) {            
            const int k = i + 2; //Skip ghost cells
            // Q at interface from the right and left
            // Note that we swap hu and hv
            float3 Qp = (float3)(Q[0][l+1][k] - Qy[0][j+1][i],
                                 Q[2][l+1][k] - Qy[2][j+1][i],
                                 Q[1][l+1][k] - Qy[1][j+1][i]);
            float3 Qm = (float3)(Q[0][l  ][k] + Qy[0][j  ][i],
                                 Q[2][l  ][k] + Qy[2][j  ][i],
                                 Q[1][l  ][k] + Qy[1][j  ][i]);
                                       
            // Computed flux
            // Note that we swap back
            const float3 flux = CentralUpwindFluxBottom(Qm, Qp, RHy[l+1][k], g_);
            G[0][j][i] = flux.x;
            G[1][j][i] = flux.z;
            G[2][j][i] = flux.y;
            
            dt = min(dt, computeDt(Qp, Qm, RHy[l+1][k], g_, dy_));
        }
    }
    
    return dt;
}




void init_H_with_garbage(__local float Hi[block_height+4][block_width+4],
			 __local float RHx[block_height+4][block_width+4],
			 __local float RHy[block_height+4][block_width+4] ) {

    if (get_local_id(0) == 0 && get_local_id(1) == 0) {
        for (int j = 0; j < block_height+4; j++) {
            for (int i = 0; i < block_width+4; i++) {
                Hi[j][i]  = 99.0f; //0.1*get_global_id(0);
                RHx[j][i] = 99.0f; //0.1*get_global_id(0);
                RHy[j][i] = 99.0f; //0.1*get_global_id(0);
            }
        }
    }
}

float findMaxDt(float per_thread_dt, int tid, __local float* sdata) {
    const int nthreads = block_width*block_height;
    
    //Store per-thread dt in local mem
    sdata[tid] = per_thread_dt;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //Local memory reduction to find min
    //First reduce to 64 elements with synchronization
    /********************************************************************
      * WARNING: This assumes that we have at least 64 threads to work
      *******************************************************************/
    if (nthreads >= 512) {
        if (tid < 512 && (tid+512) < nthreads) {
            sdata[tid] = fmin(sdata[tid], sdata[tid + 512]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (nthreads >= 256) {
        if (tid < 256 && (tid+256) < nthreads) {
            sdata[tid] = fmin(sdata[tid], sdata[tid + 256]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (nthreads >= 128) {
        if (tid < 128 && (tid+128) < nthreads) {
            sdata[tid] = fmin(sdata[tid], sdata[tid + 128]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (nthreads >= 64) {
        if (tid < 64 && (tid+64) < nthreads) {
            sdata[tid] = fmin(sdata[tid], sdata[tid + 64]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
#ifdef NVIDIA
    // Then use the 32-wide SIMD-execution to reduce from 64 to 1
    if (tid < 32) {
        if (nthreads >=  64) { 
            sdata[tid] = fmin(sdata[tid], sdata[tid + 32]);
        }
        if (nthreads >=  32) { 
            sdata[tid] = fmin(sdata[tid], sdata[tid + 16]);
        }
        if (nthreads >=  16) { 
            sdata[tid] = fmin(sdata[tid], sdata[tid + 8]);
        }
        if (nthreads >=  8) { 
            sdata[tid] = fmin(sdata[tid], sdata[tid + 4]);
        }
        if (nthreads >=  4) { 
            sdata[tid] = fmin(sdata[tid], sdata[tid + 2]);
        }
        if (nthreads >=  2) { 
            sdata[tid] = fmin(sdata[tid], sdata[tid + 1]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    if (nthreads >=  64) {
        if (tid < 32) { 
            sdata[tid] = fmin(sdata[tid], sdata[tid + 32]); 
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (nthreads >=  32) { 
        if (tid < 16) {
            sdata[tid] = fmin(sdata[tid], sdata[tid + 16]); 
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (nthreads >=  16) { 
        if (tid < 8) {
            sdata[tid] = fmin(sdata[tid], sdata[tid + 8]);  
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (nthreads >=  8) { 
        if (tid < 4) {
            sdata[tid] = fmin(sdata[tid], sdata[tid + 4]);  
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (nthreads >=  4) { 
        if (tid < 2) {
            sdata[tid] = fmin(sdata[tid], sdata[tid + 2]);  
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (nthreads >=  2) { 
        if (tid < 1) {
            sdata[tid] = fmin(sdata[tid], sdata[tid + 1]);  
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#endif
    
    //Use local mem reduction to find per block minimum
    return sdata[0];
}







/**
  * This unsplit kernel computes the 2D numerical scheme with a TVD RK2 time integration scheme
  */
__kernel void swe_2D(
        int nx_, int ny_,
        float dx_, float dy_, 
        float g_,
        
        float theta_,
        
        float f_, //< Coriolis coefficient
        float beta_, //< Coriolis force f_ + beta_*y
        float y_zero_reference_, // the cell row representing y = 0.5*dy
        
        // Input [w, hu, hv] at time t_n
        __global float* U1_ptr_, int U1_pitch_,
        __global float* U2_ptr_, int U2_pitch_,
        __global float* U3_ptr_, int U3_pitch_,
        
        // Output flux
        __global float* R1_ptr_, int R1_pitch_,
        __global float* R2_ptr_, int R2_pitch_,
        __global float* R3_ptr_, int R3_pitch_,
        
        // Output dt
        __global float* dt_ptr_, 

        // Water depth at cell intersections
        __global float* Hi_ptr_, int Hi_pitch_,
        
        //Wind stress parameters
        int wind_stress_type_, 
        float tau0_, float rho_, float alpha_, float xm_, float Rc_,
        float x0_, float y0_,
        float u0_, float v0_,

        // Boundary conditions (1: wall, 2: periodic, 3: numerical sponge)
        int bc_north_, int bc_east_, int bc_south_, int bc_west_,
	
        float t_, 
        
        int write_dt_
        ) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);

    //Index of cell within domain
    const int ti = get_global_id(0) + 2; //Skip global ghost cells, i.e., +2
    const int tj = get_global_id(1) + 2;
    
    float dt = FLT_MAX;
    
    //Shared memory variables
    __local float Q[3][block_height+4][block_width+4];
    
    //The following slightly wastes memory, but enables us to reuse the 
    //funcitons in common.opencl
    __local float Qx[3][block_height+2][block_width+2];
    __local float Qy[3][block_height+2][block_width+2];
    __local float F[3][block_height+1][block_width+1];
    __local float G[3][block_height+1][block_width+1];

    // Shared memory for bathymetry, and
    // reconstructed bathymetry in both directions
    // We use too much shared memory for RHx and RHy in order
    // to easier follow the implementation from the C++ version
    __local float  Hi[block_height+4][block_width+4];
    __local float RHx[block_height+4][block_width+4];
    __local float RHy[block_height+4][block_width+4];
       
    //Read Q = [eta, hu, hv] into shared memory
    readBlock2(U1_ptr_, U1_pitch_,
               U2_ptr_, U2_pitch_,
               U3_ptr_, U3_pitch_,
               Q, nx_, ny_);
   
    // Read H into sheared memory
    readBlock2single(Hi_ptr_, Hi_pitch_, Hi, nx_, ny_);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reconstruct H at the cell faces
    reconstructH(RHx, RHy, Hi);
    barrier(CLK_LOCAL_MEM_FENCE);

    //Fix boundary conditions
    // TODO: Add if on boundary condition
    if (bc_north_ == 1 || bc_east_ == 1 || bc_south_ == 1 || bc_west_ == 1) {
        noFlowBoundary2Mix(Q, nx_, ny_, bc_north_, bc_east_, bc_south_, bc_west_);
        barrier(CLK_LOCAL_MEM_FENCE);
        // Looks scary to have fence within if, but the bc parameters are static between threads.
    }
    
    //Reconstruct slopes along x and axis
    // The Qx is here dQ/dx*0.5*dx
    minmodSlopeX(Q, Qx, theta_);
    minmodSlopeY(Q, Qy, theta_);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Adjust the slopes to avoid negative values at integration points
    adjustSlopes(Qx, Qy, RHx, RHy, Q);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Compute fluxes along the x and y axis
    dt = min(dt, computeFluxF(Q, Qx, F, RHx, g_, dx_));
    dt = min(dt, computeFluxG(Q, Qy, G, RHy, g_, dy_));
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //Sum fluxes and advance in time for all internal cells
    //Check global indices against global domain
    if (ti > 1 && ti < nx_+2 && tj > 1 && tj < ny_+2) {
        const int i = tx + 2; //Skip local ghost cells, i.e., +2
        const int j = ty + 2;

        // Find bottom topography source terms: S2, S3
        const float ST2 = bottomSourceTerm2(Q, Qx, RHx, g_, i, j);
        const float ST3 = bottomSourceTerm3(Q, Qy, RHy, g_, i, j);
	
        const float X = windStressX(
            wind_stress_type_, 
            dx_, dy_, 0.0f, // FIXME: dt = 0 here
            tau0_, rho_, alpha_, xm_, Rc_,
            x0_, y0_,
            u0_, v0_,
            t_);
        const float Y = windStressY(
            wind_stress_type_, 
            dx_, dy_, 0.0f, // FIXME: dt = 0 here
            tau0_, rho_, alpha_, xm_, Rc_,
            x0_, y0_,
            u0_, v0_,
            t_);

        // Coriolis parameter
        float global_thread_y = tj-2.0f; // Global id including ghost cells
        float coriolis_f = linear_coriolis_term(f_, beta_, global_thread_y,
                            dy_, y_zero_reference_);
        
        __global float* const R1_row = (__global float*) ((__global char*) R1_ptr_ + R1_pitch_*tj);
        __global float* const R2_row = (__global float*) ((__global char*) R2_ptr_ + R2_pitch_*tj);
        __global float* const R3_row = (__global float*) ((__global char*) R3_ptr_ + R3_pitch_*tj);
       
        R1_row[ti] = - (F[0][ty  ][tx+1] - F[0][ty][tx]) / dx_
                     - (G[0][ty+1][tx  ] - G[0][ty][tx]) / dy_;
        R2_row[ti] = - (F[1][ty  ][tx+1] - F[1][ty][tx]) / dx_ 
                     - (G[1][ty+1][tx  ] - G[1][ty][tx]) / dy_
                     + (X + coriolis_f*Q[2][j][i] - ST2/dx_);
        R3_row[ti] = - (F[2][ty  ][tx+1] - F[2][ty][tx]) / dx_
                     - (G[2][ty+1][tx  ] - G[2][ty][tx]) / dy_
                     + (Y - coriolis_f*Q[1][j][i] - ST3/dy_);
    }
    else {
        dt = FLT_MAX;
    }
    
    
    // Find maximum dt (use Q as local memory for reduction)
    if (write_dt_ == 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        dt = findMaxDt(dt, ty*block_width+tx, &Q[0][0][0]);
        if ((tx+ty) == 0) {
            int block_index = get_group_id(1)*get_num_groups(0) + get_group_id(0);
            dt_ptr_[block_index] = dt;
        }
    }
}
