/*
This OpenCL kernel implements the Kurganov-Petrova numerical scheme 
for the shallow water equations, described in 
A. Chertock, M. Dudzinski, A. Kurganov & M. Lukacova-Medvidova
Well-Balanced Schemes for the Shallow Water Equations with Coriolis Forces,
Numerische Mathematik 2016 

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
// f = \tilde{f} + beta*y
float linear_coriolis_term(const float f, const float beta,
			   const float tj, const float dy,
			   const float y_zero_reference) {
    // y_zero_reference is the number of ghost cells
    // and represent the tj so that y = 0.5*dy
    float y = (tj-y_zero_reference + 0.5)*dy;
    return f + beta * y;
}


float3 CDKLM16_F_func(const float3 Q, const float g) {
    float3 F;

    F.x = Q.x*Q.y;                        //h*u
    F.y = Q.x*Q.y*Q.y + 0.5f*g*Q.x*Q.x;   //h*u*u + 0.5f*g*h*h;
    F.z = Q.x*Q.y*Q.z;                    //h*u*v;

    return F;
}







/**
  * Note that the input vectors are (h, u, v), thus not the regular
  * (h, hu, hv)
  */
float3 CDKLM16_flux(const float3 Qm, float3 Qp, const float g) {
    const float3 Fp = CDKLM16_F_func(Qp, g);
    const float up = Qp.y;         // u
    const float cp = sqrt(g*Qp.x); // sqrt(g*h)

    const float3 Fm = CDKLM16_F_func(Qm, g);
    const float um = Qm.y;         // u
    const float cm = sqrt(g*Qm.x); // sqrt(g*h)
    
    const float am = min(min(um-cm, up-cp), 0.0f); // largest negative wave speed
    const float ap = max(max(um+cm, up+cp), 0.0f); // largest positive wave speed
    
    float3 F;
    
    F.x = ((ap*Fm.x - am*Fp.x) + ap*am*(Qp.x-Qm.x))/(ap-am);
    F.y = ((ap*Fm.y - am*Fp.y) + ap*am*(Fp.x-Fm.x))/(ap-am);
    F.z = (Qm.y + Qp.y > 0) ? Fm.z : Fp.z; //Upwinding to be consistent
    
    return F;
}

















__kernel void swe_2D(
        int nx_, int ny_,
        float dx_, float dy_, float dt_,
        float g_,
        
        float theta_,
        
        float f_, //< Coriolis coefficient
	float beta_, //< Coriolis force f_ + beta_*y
	float y_zero_reference_, // the cell row representing y = 0.5*dy
	
        float r_, //< Bottom friction coefficient

	int rk_order, // runge kutta order
        int step_,    // runge kutta step
        
        //Input h^n
        __global float* h0_ptr_, int h0_pitch_,
        __global float* hu0_ptr_, int hu0_pitch_,
        __global float* hv0_ptr_, int hv0_pitch_,
        
        //Output h^{n+1}
        __global float* h1_ptr_, int h1_pitch_,
        __global float* hu1_ptr_, int hu1_pitch_,
        __global float* hv1_ptr_, int hv1_pitch_,

	//Bathymery
	__global float* Bi_ptr_, int Bi_pitch_,
	__global float* Bm_ptr_, int Bm_pitch_,
	
        //Wind stress parameters
        int wind_stress_type_, 
        float tau0_, float rho_, float alpha_, float xm_, float Rc_,
        float x0_, float y0_,
        float u0_, float v0_,
        float t_, 
    
        // Boundary conditions (1: wall, 2: periodic, 3: open boundary (flow relaxation scheme))
        int bc_north_, int bc_east_, int bc_south_, int bc_west_,

	// Geostrophic Equilibrium memory buffers
	// The buffers have the same size as input/output
	int report_geostrophical_equilibrium,
	__global float* uxpvy_ptr_, int uxpvy_pitch_,
	__global float* Kx_ptr_, int Kx_pitch_,
	__global float* Ly_ptr_, int Ly_pitch_

    ) {
        
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of block within domain
    const int bx = get_local_size(0) * get_group_id(0);
    const int by = get_local_size(1) * get_group_id(1);

    //Index of cell within domain
    const int ti = get_global_id(0) + 2; //Skip global ghost cells, i.e., +2
    const int tj = get_global_id(1) + 2;
    
    // Our physical variables
    __local float R[3][block_height+4][block_width+4];
    
    // Our reconstruction variables
    __local float Q[2][block_height+4][block_width+4];
    __local float Qx[3][block_height][block_width+2];
    __local float Qy[3][block_height+2][block_width];
    
    // Our fluxes
    __local float F[3][block_height][block_width+1];
    __local float G[3][block_height+1][block_width];
    
    
    // Bathymetry
    __local float  Bi[block_height+1][block_width+1];
    __local float  Bm[block_height+4][block_width+4];
    __local float RBx[block_height  ][block_width+1];
    __local float RBy[block_height+1][block_width  ];


    // theta_ = 1.5f;
    
    //Read into shared memory
    for (int j=ty; j<block_height+4; j+=get_local_size(1)) {
        const int l = clamp(by + j, 0, ny_+3); // Out of bounds
        
        //Compute the pointer to current row in the arrays
        __global float* const h_row = (__global float*) ((__global char*) h0_ptr_ + h0_pitch_*l);
        __global float* const hu_row = (__global float*) ((__global char*) hu0_ptr_ + hu0_pitch_*l);
        __global float* const hv_row = (__global float*) ((__global char*) hv0_ptr_ + hv0_pitch_*l);
        
        for (int i=tx; i<block_width+4; i+=get_local_size(0)) {
            const int k = clamp(bx + i, 0, nx_+3); // Out of bounds
            
            R[0][j][i] = h_row[k];
            R[1][j][i] = hu_row[k];
            R[2][j][i] = hv_row[k];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    

    // Read Bm into shared memory with 4x4 halo
    for (int j=ty; j < block_height+4; j+=get_local_size(1)) {
	// Ensure that we read from correct domain
	// We never read outermost halo of Bm
	const int l = clamp(by+j, 0, ny_+3); 
	__global float* const Bm_row = (__global float*) ((__global char*) Bm_ptr_ + Bm_pitch_*l);
	for(int i=tx; i < block_width+4; i+=get_local_size(0)) {
	    const int k = clamp(bx+i, 0, nx_+3);

	    Bm[j][i] = Bm_row[k];
	}
    }

    // Read Bi into shared memory
    // Read intersections on all non-ghost cells
    for(int j=ty; j < block_height+1; j+=get_local_size(1)) {
	// Skip ghost cells and 
	const int l = clamp(by+j+2, 2, ny_+2);
	__global float* const Bi_row = (__global float*) ((__global char*) Bi_ptr_ + Bi_pitch_*l);
	for(int i=tx; i < block_width+1; i+=get_local_size(0)) {
	    const int k = clamp(bx+i+2, 2, nx_+2);

	    Bi[j][i] = Bi_row[k];
	}
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    // Evaluate piecewise bi-linear for RBx 
    for (int j=ty; j<block_height; j+=get_local_size(1)) {
        for (int i=tx; i<block_width+1; i+=get_local_size(0)) {
            RBx[j][i] = 0.5f*( Bi[j][i] + Bi[j+1][i] );
	}
    }
    // Evaluate piecewise bi-linear for RBy
    for (int j=ty; j<block_height+1; j+=get_local_size(1)) {
        for (int i=tx; i<block_width; i+=get_local_size(0)) {
            RBy[j][i] = 0.5f*( Bi[j][i] + Bi[j][i+1] );
	}
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //Fix boundary conditions
    if (bc_north_ == 1 || bc_east_ == 1 || bc_south_ == 1 || bc_west_ == 1)
    {
        // These boundary conditions are dealt with inside shared memory
        
        const int i = tx + 2; //Skip local ghost cells, i.e., +2
        const int j = ty + 2;
        
        if (ti == 2 && bc_west_ == 1) {
	    // Wall boundary on west
	    R[0][j][i-1] =  R[0][j][i];
            R[1][j][i-1] = -R[1][j][i];
            R[2][j][i-1] =  R[2][j][i];
            
            R[0][j][i-2] =  R[0][j][i+1];
            R[1][j][i-2] = -R[1][j][i+1];
            R[2][j][i-2] =  R[2][j][i+1];
	}
        if (ti == nx_+1 && bc_east_ == 1) {
	    // Wall boundary on east 
            R[0][j][i+1] =  R[0][j][i];
            R[1][j][i+1] = -R[1][j][i];
            R[2][j][i+1] =  R[2][j][i];
            
            R[0][j][i+2] =  R[0][j][i-1];
            R[1][j][i+2] = -R[1][j][i-1];
            R[2][j][i+2] =  R[2][j][i-1];
        }
        if (tj == 2 && bc_south_ == 1) {
	    // Wall boundary on south
	    R[0][j-1][i] =  R[0][j][i];
            R[1][j-1][i] =  R[1][j][i];
            R[2][j-1][i] = -R[2][j][i];
            
            R[0][j-2][i] =  R[0][j+1][i];
            R[1][j-2][i] =  R[1][j+1][i];
            R[2][j-2][i] = -R[2][j+1][i];
        }
        if (tj == ny_+1 && bc_north_ == 1) {
	    // Wall boundary on north
            R[0][j+1][i] =  R[0][j][i];
            R[1][j+1][i] =  R[1][j][i];
            R[2][j+1][i] = -R[2][j][i];
            
            R[0][j+2][i] =  R[0][j-1][i];
            R[1][j+2][i] =  R[1][j-1][i];
            R[2][j+2][i] = -R[2][j-1][i];
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
        
    
    //Create our "steady state" reconstruction variables (u, v)
    // K and L are never stored, but computed where needed.
    for (int j=ty; j<block_height+4; j+=get_local_size(1)) {
        for (int i=tx; i<block_width+4; i+=get_local_size(0)) {
            
            const float h = R[0][j][i];
            const float u = R[1][j][i] / h;
            const float v = R[2][j][i] / h;

            Q[0][j][i] = u;
            Q[1][j][i] = v;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    
    
    
    
    //Reconstruct slopes along x axis
    for (int j=ty; j<block_height; j+=get_local_size(1)) {
        const int l = j + 2; //Skip ghost cells
        for (int i=tx; i<block_width+2; i+=get_local_size(0)) {
            const int k = i + 1;
            for (int p=0; p<2; ++p) {
                Qx[p][j][i] = minmodSlope(Q[p][l][k-1], Q[p][l][k], Q[p][l][k+1], theta_);
            }
	    // Qx[2] = Kx, which we need to find differently than ux and vx
	    float left_w   = R[0][l][k-1] + Bm[l][k-1];
	    float center_w = R[0][l][k  ] + Bm[l][k  ];
	    float right_w  = R[0][l][k+1] + Bm[l][k+1];

	    float left_v   = Q[1][l][k-1];
	    float center_v = Q[1][l][k  ];
	    float right_v  = Q[1][l][k+1];

	    float global_thread_y = by + j;
	    float coriolis_f = linear_coriolis_term(f_, beta_, global_thread_y,
						    dy_, y_zero_reference_);
	    float V_constant = dx_*coriolis_f/(2.0f*g_);

	    float backward = theta_*g_*(center_w - left_w   - V_constant*(center_v + left_v ) );
	    float central  =   0.5f*g_*(right_w  - left_w   - V_constant*(right_v + 2*center_v + left_v) ); 
	    float forward  = theta_*g_*(right_w  - center_w - V_constant*(center_v + right_v) );

	    // Qx[2] is really dx*Kx
	    Qx[2][j][i] = minmodRaw(backward, central, forward); 
	    
        }
    }
    
    //Reconstruct slopes along y axis
    for (int j=ty; j<block_height+2; j+=get_local_size(1)) {
        const int l = j + 1;
        for (int i=tx; i<block_width; i+=get_local_size(0)) {            
            const int k = i + 2; //Skip ghost cells
            for (int p=0; p<2; ++p) {
                Qy[p][j][i] = minmodSlope(Q[p][l-1][k], Q[p][l][k], Q[p][l+1][k], theta_);
            }
	    // Qy[2] = Ly, which we need to find differently than uy and vy
	    float lower_w  = R[0][l-1][k] + Bm[l-1][k];
	    float center_w = R[0][l  ][k] + Bm[l  ][k];
	    float upper_w  = R[0][l+1][k] + Bm[l+1][k];

	    float global_thread_y = by + j;
	    float center_coriolis_f = linear_coriolis_term(f_, beta_, global_thread_y,
							   dy_, y_zero_reference_);
	    float lower_coriolis_f  = linear_coriolis_term(f_, beta_, global_thread_y - 1,
							   dy_, y_zero_reference_);
	    float upper_coriolis_f  = linear_coriolis_term(f_, beta_, global_thread_y + 1,
							   dy_, y_zero_reference_);
	   	    
	    float lower_fu  = Q[0][l-1][k]*lower_coriolis_f;
	    float center_fu = Q[0][l  ][k]*center_coriolis_f;
	    float upper_fu  = Q[0][l+1][k]*upper_coriolis_f;

	    float U_constant = dy_/(2.0f*g_);

	    float backward = theta_*g_*(center_w - lower_w  + U_constant*(center_fu + lower_fu ) );
	    float central  =   0.5f*g_*(upper_w  - lower_w  + U_constant*(upper_fu + 2*center_fu + lower_fu) ); 
	    float forward  = theta_*g_*(upper_w  - center_w + U_constant*(center_fu + upper_fu) );

	    // Qy[2] is really dy*Ly
	    Qy[2][j][i] = minmodRaw(backward, central, forward); 
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);



	
    //Compute fluxes along the x axis
    for (int j=ty; j<block_height; j+=get_local_size(1)) {
        const int l = j + 2; //Skip ghost cells (be consistent with reconstruction offsets)
        for (int i=tx; i<block_width+1; i+=get_local_size(0)) {
            const int k = i + 1;

            // (u, v) reconstructed at a cell interface from the right (p) and left (m)
	    const float2 Rp = (float2)(Q[0][l][k+1] - 0.5f*Qx[0][j][i+1],
                                       Q[1][l][k+1] - 0.5f*Qx[1][j][i+1]);
            const float2 Rm = (float2)(Q[0][l][k  ] + 0.5f*Qx[0][j][i  ],
                                       Q[1][l][k  ] + 0.5f*Qx[1][j][i  ]);

            // Variables to reconstruct h from u, v, K, L
            const float vp = Q[1][l][k+1];
            const float vm = Q[1][l][k  ];

	    // Bottom in the cells on each side of the face:
	    const float Bm_p = Bm[l][k+1];
	    const float Bm_m = Bm[l][k  ];
	    // B is RBx on the given face!
	    const float B_face = RBx[j][i];

	    const float h_bar_p = R[0][l][k+1];
	    const float h_bar_m = R[0][l][k  ];

	    // Qx[2] is really dx*Kx
	    const float Kx_p = Qx[2][j][i+1];
            const float Kx_m = Qx[2][j][i  ];

	    // Coriolis parameter
	    float global_thread_y = by + j;
	    float coriolis_f = linear_coriolis_term(f_, beta_, global_thread_y,
						    dy_, y_zero_reference_);
	    
            // Reconstruct h 
            const float hp = h_bar_p + Bm_p - B_face - (Kx_p + dx_*coriolis_f*vp)/(2.0f*g_); 
	    const float hm = h_bar_m + Bm_m - B_face + (Kx_m + dx_*coriolis_f*vm)/(2.0f*g_);
	    
            // Our flux variables Q=(h, u, v)
            const float3 Qp = (float3)(hp, Rp.x, Rp.y);
            const float3 Qm = (float3)(hm, Rm.x, Rm.y);
                                       
            // Computed flux
            const float3 flux = CDKLM16_flux(Qm, Qp, g_);
            F[0][j][i] = flux.x;
            F[1][j][i] = flux.y;
            F[2][j][i] = flux.z;
        }
    }
        
    //Compute fluxes along the y axis
    for (int j=ty; j<block_height+1; j+=get_local_size(1)) {
        const int l = j + 1;
        for (int i=tx; i<block_width; i+=get_local_size(0)) {
            const int k = i + 2; //Skip ghost cells
            // Q at interface from the right and left
	    const float2 Rp = (float2)(Q[0][l+1][k] - 0.5f*Qy[0][j+1][i],
                                       Q[1][l+1][k] - 0.5f*Qy[1][j+1][i]);
	    const float2 Rm = (float2)(Q[0][l  ][k] + 0.5f*Qy[0][j  ][i],
				       Q[1][l  ][k] + 0.5f*Qy[1][j  ][i]);
              
            // Variables to reconstruct h from u, v, K, L
            const float up = Q[0][l+1][k];
            const float um = Q[0][l  ][k];

	    // Bottom in the cells on each side of the face:
	    const float Bm_p = Bm[l+1][k];
	    const float Bm_m = Bm[l  ][k];
	    // B is RBx on the given face!
	    const float B_face = RBy[j][i];

	    const float h_bar_p = R[0][l+1][k];
	    const float h_bar_m = R[0][l  ][k];

	    // Qy[2] is really dy*Ly
	    const float Ly_p = Qy[2][j+1][i];
            const float Ly_m = Qy[2][j  ][i];

	    // Coriolis parameter
	    float global_thread_y = by + j;
	    float coriolis_fm = linear_coriolis_term(f_, beta_, global_thread_y,
						    dy_, y_zero_reference_);
	    float coriolis_fp = linear_coriolis_term(f_, beta_, global_thread_y + 1, 
						    dy_, y_zero_reference_);
	    
            // Reconstruct h 
	    const float hp = h_bar_p + Bm_p - B_face - ( Ly_p - dy_*coriolis_fp*up)/(2.0f*g_); 
	    const float hm = h_bar_m + Bm_m - B_face + ( Ly_m - dy_*coriolis_fm*um)/(2.0f*g_); 

	    // Our flux variables Q=(h, v, u)
            // Note that we swap u and v
            const float3 Qp = (float3)(hp, Rp.y, Rp.x);
            const float3 Qm = (float3)(hm, Rm.y, Rm.x);
            
            // Computed flux
            // Note that we swap back u and v
            const float3 flux = CDKLM16_flux(Qm, Qp, g_);
            G[0][j][i] = flux.x;
            G[1][j][i] = flux.z;
            G[2][j][i] = flux.y;
        }
    }
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

	// Bottom topography source terms!
	const float st1 = -g_*R[0][j][i]*(RBx[ty  ][tx+1] - RBx[ty][tx]);
	const float st2 = -g_*R[0][j][i]*(RBy[ty+1][tx  ] - RBy[ty][tx]);

	// Coriolis parameter
	float global_thread_y = tj-2; // Global id including ghost cells
	float coriolis_f = linear_coriolis_term(f_, beta_, global_thread_y,
						dy_, y_zero_reference_);
	
        const float L1  = - (F[0][ty  ][tx+1] - F[0][ty][tx]) / dx_ 
	                  - (G[0][ty+1][tx  ] - G[0][ty][tx]) / dy_;
        const float L2  = - (F[1][ty  ][tx+1] - F[1][ty][tx]) / dx_ 
                          - (G[1][ty+1][tx  ] - G[1][ty][tx]) / dy_
  	                  + (X + coriolis_f*R[2][j][i] + st1/dx_);
        const float L3  = - (F[2][ty  ][tx+1] - F[2][ty][tx]) / dx_ 
                          - (G[2][ty+1][tx  ] - G[2][ty][tx]) / dy_
	                  + (Y - coriolis_f*R[1][j][i] + st2/dy_);
	
        
	__global float* const h_row  = (__global float*) ((__global char*) h1_ptr_  + h1_pitch_*tj);
        __global float* const hu_row = (__global float*) ((__global char*) hu1_ptr_ + hu1_pitch_*tj);
        __global float* const hv_row = (__global float*) ((__global char*) hv1_ptr_ + hv1_pitch_*tj);

	if (rk_order < 3) {
         
	    const float C = 2.0f*r_*dt_/R[0][j][i];
                     
	    if  (step_ == 0) {
		//First step of RK2 ODE integrator
            
		h_row[ti]  =  R[0][j][i] + dt_*L1;
		hu_row[ti] = (R[1][j][i] + dt_*L2) / (1.0f + C);
		hv_row[ti] = (R[2][j][i] + dt_*L3) / (1.0f + C);
	    }
	    else if (step_ == 1) {
		//Second step of RK2 ODE integrator
            
		//First read Q^n
		const float h_a  = h_row[ti];
		const float hu_a = hu_row[ti];
		const float hv_a = hv_row[ti];
            
		//Compute Q^n+1
		const float h_b  = 0.5f*(h_a  + (R[0][j][i] + dt_*L1));
		const float hu_b = 0.5f*(hu_a + (R[1][j][i] + dt_*L2));
		const float hv_b = 0.5f*(hv_a + (R[2][j][i] + dt_*L3));

				
		//Write to main memory
		h_row[ti] = h_b;
		hu_row[ti] = hu_b / (1.0f + 0.5f*C);
		hv_row[ti] = hv_b / (1.0f + 0.5f*C);
		//hu_row[ti] = RBx[ty][tx];
		//hv_row[ti] = RBy[ty][tx];

	    }
	}
	
	else if (rk_order == 3) {
	    // Third order Runge Kutta - only valid if r_ = 0.0 (no friction)

	    if (step_ == 0) {
		//First step of RK3 ODE integrator
		// q^(1) = q^n + dt*L(q^n)
            
		h_row[ti]  =  R[0][j][i] + dt_*L1;
		hu_row[ti] = (R[1][j][i] + dt_*L2);
		hv_row[ti] = (R[2][j][i] + dt_*L3);

	    } else if (step_ == 1) {
		// Second step of RK3 ODE integrator
		// Q^(2) = 3/4 Q^n + 1/4 ( Q^(1) + dt*L(Q^(1)) )
		// Q^n is here in h1, but will be used in next iteration as well --> write to h0

		// First read Q^n:
		const float h_a  = h_row[ti];
		const float hu_a = hu_row[ti];
		const float hv_a = hv_row[ti];

		// Compute Q^(2):
		const float h_b  = 0.75f*h_a  + 0.25f*(R[0][j][i] + dt_*L1);
		const float hu_b = 0.75f*hu_a + 0.25f*(R[1][j][i] + dt_*L2);
		const float hv_b = 0.75f*hv_a + 0.25f*(R[2][j][i] + dt_*L3);

		// Write output to the input buffer:
		__global float* const h_out_row  = (__global float*) ((__global char*) h0_ptr_  + h0_pitch_*tj);
		__global float* const hu_out_row = (__global float*) ((__global char*) hu0_ptr_ + hu0_pitch_*tj);
		__global float* const hv_out_row = (__global float*) ((__global char*) hv0_ptr_ + hv0_pitch_*tj);
		h_out_row[ti] = h_b;
		hu_out_row[ti] = hu_b;
		hv_out_row[ti] = hv_b;
		
	    } else if (step_ == 2) {
		// Third step of RK3 ODE integrator
		// Q^n+1 = 1/3 Q^n + 2/3 (Q^(2) + dt*L(Q^(2))

		// First read Q^n:
		const float h_a  = h_row[ti];
		const float hu_a = hu_row[ti];
		const float hv_a = hv_row[ti];

		// Compute Q^n+1:
		const float h_b  = (h_a  + 2.0f*(R[0][j][i] + dt_*L1)) / 3.0f;
		const float hu_b = (hu_a + 2.0f*(R[1][j][i] + dt_*L2)) / 3.0f;
		const float hv_b = (hv_a + 2.0f*(R[2][j][i] + dt_*L3)) / 3.0f;

		//Write to main memory
		h_row[ti] = h_b;
		hu_row[ti] = hu_b;
		hv_row[ti] = hv_b;

		__global float* const Kx_row = (__global float*) ((__global char*) Kx_ptr_ + Kx_pitch_*tj);	    
		//Kx_row[ti]    = 4;  // K_x
	    }
	}

	// Write geostrophical equilibrium variables:
	if (report_geostrophical_equilibrium) {

	    __global float* const uxpvy_row  = (__global float*) ((__global char*) uxpvy_ptr_ + uxpvy_pitch_*tj);
	    __global float* const Kx_row = (__global float*) ((__global char*) Kx_ptr_ + Kx_pitch_*tj);
	    __global float* const Ly_row = (__global float*) ((__global char*) Ly_ptr_ + Ly_pitch_*tj);

	    uxpvy_row[ti] = Qx[0][ty][tx+1] + Qy[1][ty+1][tx]; // u_x + v_y
	    Kx_row[ti]    = Qx[2][ty][tx+1];  // K_x
	    Ly_row[ti]    = Qy[2][ty+1][tx];  // L_y
	}
    }

	
    
}
