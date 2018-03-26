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


// Finds the coriolis term based on the linear Coriolis force
// f = \tilde{f} + beta*y
double linear_coriolis_term(const double f, const double beta,
			   const double tj, const double dy,
			   const double y_zero_reference) {
    // y_zero_reference is the number of ghost cells
    // and represent the tj so that y = 0.5*dy
    double y = (tj-y_zero_reference + 0.5)*dy;
    return f + beta * y;
}


double3 CDKLM16_F_func(const double3 Q, const double g) {
    double3 F;

    F.x = Q.x*Q.y;                        //h*u
    F.y = Q.x*Q.y*Q.y + 0.5f*g*Q.x*Q.x;   //h*u*u + 0.5f*g*h*h;
    F.z = Q.x*Q.y*Q.z;                    //h*u*v;

    return F;
}







/**
  * Note that the input vectors are (h, u, v), thus not the regular
  * (h, hu, hv)
  */
double3 CDKLM16_flux(const double3 Qm, double3 Qp, const double g) {
    const double3 Fp = CDKLM16_F_func(Qp, g);
    const double up = Qp.y;         // u
    const double cp = sqrt(g*Qp.x); // sqrt(g*h)

    const double3 Fm = CDKLM16_F_func(Qm, g);
    const double um = Qm.y;         // u
    const double cm = sqrt(g*Qm.x); // sqrt(g*h)
    
    const double am = min(min(um-cm, up-cp), 0.0); // largest negative wave speed
    const double ap = max(max(um+cm, up+cp), 0.0); // largest positive wave speed
    
    double3 F;
    
    F.x = ((ap*Fm.x - am*Fp.x) + ap*am*(Qp.x-Qm.x))/(ap-am);
    F.y = ((ap*Fm.y - am*Fp.y) + ap*am*(Fp.x-Fm.x))/(ap-am);
    F.z = (Qm.y + Qp.y > 0) ? Fm.z : Fp.z; //Upwinding to be consistent
    
    return F;
}

double minmodRawDouble(double backward, double central, double forward) {

    return 0.25
	*copysign(1.0, backward)
	*(copysign(1.0, backward) + copysign(1.0, central))
	*(copysign(1.0, central) + copysign(1.0, forward))
	*min( min(fabs(backward), fabs(central)), fabs(forward) );
}

double minmodSlopeDouble(double left, double center, double right, double theta) {
    const double backward = (center - left) * theta;
    const double central = (right - left) * 0.5;
    const double forward = (right - center) * theta;

    return minmodRawDouble(backward, central, forward);
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
        __global float* eta0_ptr_, int eta0_pitch_,
        __global float* hu0_ptr_, int hu0_pitch_,
        __global float* hv0_ptr_, int hv0_pitch_,
        
        //Output h^{n+1}
        __global float* eta1_ptr_, int eta1_pitch_,
        __global float* hu1_ptr_, int hu1_pitch_,
        __global float* hv1_ptr_, int hv1_pitch_,

        //Bathymery
        __global float* Hi_ptr_, int Hi_pitch_,
        __global float* Hm_ptr_, int Hm_pitch_,
	
        //Wind stress parameters
        __global const wind_stress_params *wind_stress_,

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
    __local double R[3][block_height+4][block_width+4];
    
    // Our reconstruction variables
    __local double Q[2][block_height+4][block_width+4];
    __local double Qx[3][block_height][block_width+2];
    __local double Qy[3][block_height+2][block_width];
    
    // Our fluxes
    __local double F[3][block_height][block_width+1];
    __local double G[3][block_height+1][block_width];
    
    
    // Bathymetry
    __local float  Hi[block_height+1][block_width+1];
    __local float  Hm[block_height+4][block_width+4];
    __local float RHx[block_height  ][block_width+1];
    __local float RHy[block_height+1][block_width  ];


    // theta_ = 1.5f;
    
    //Read into shared memory
    for (int j=ty; j<block_height+4; j+=get_local_size(1)) {
        const int l = clamp(by + j, 0, ny_+3); // Out of bounds
        
        //Compute the pointer to current row in the arrays
        __global float* const eta_row = (__global float*) ((__global char*) eta0_ptr_ + eta0_pitch_*l);
        __global float* const hu_row = (__global float*) ((__global char*) hu0_ptr_ + hu0_pitch_*l);
        __global float* const hv_row = (__global float*) ((__global char*) hv0_ptr_ + hv0_pitch_*l);
        
        for (int i=tx; i<block_width+4; i+=get_local_size(0)) {
            const int k = clamp(bx + i, 0, nx_+3); // Out of bounds
            
            R[0][j][i] = eta_row[k];
            R[1][j][i] = hu_row[k];
            R[2][j][i] = hv_row[k];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    

    // Read Hm into shared memory with 4x4 halo
    for (int j=ty; j < block_height+4; j+=get_local_size(1)) {
	// Ensure that we read from correct domain
	// We never read outermost halo of Bm
	const int l = clamp(by+j, 0, ny_+3); 
	__global float* const Hm_row = (__global float*) ((__global char*) Hm_ptr_ + Hm_pitch_*l);
	for(int i=tx; i < block_width+4; i+=get_local_size(0)) {
	    const int k = clamp(bx+i, 0, nx_+3);

	    Hm[j][i] = Hm_row[k];
	}
    }

    // Read Hi into shared memory
    // Read intersections on all non-ghost cells
    for(int j=ty; j < block_height+1; j+=get_local_size(1)) {
	// Skip ghost cells and 
	const int l = clamp(by+j+2, 2, ny_+2);
	__global float* const Hi_row = (__global float*) ((__global char*) Hi_ptr_ + Hi_pitch_*l);
	for(int i=tx; i < block_width+1; i+=get_local_size(0)) {
	    const int k = clamp(bx+i+2, 2, nx_+2);

	    Hi[j][i] = Hi_row[k];
	}
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    
    // Evaluate piecewise bi-linear for RHx
    // RHx and RHy is then the equilibrium depth on faces
    for (int j=ty; j<block_height; j+=get_local_size(1)) {
        for (int i=tx; i<block_width+1; i+=get_local_size(0)) {
            RHx[j][i] = 0.5f*( Hi[j][i] + Hi[j+1][i] );
	}
    }
    // Evaluate piecewise bi-linear for RHy
    for (int j=ty; j<block_height+1; j+=get_local_size(1)) {
        for (int i=tx; i<block_width; i+=get_local_size(0)) {
            RHy[j][i] = 0.5f*( Hi[j][i] + Hi[j][i+1] );
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
            
            const double h = R[0][j][i] + Hm[j][i]; // h = eta + H
            const double u = R[1][j][i] / h;
            const double v = R[2][j][i] / h;

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
                Qx[p][j][i] = minmodSlopeDouble(Q[p][l][k-1], Q[p][l][k], Q[p][l][k+1], theta_);
            }
	    // Qx[2] = Kx, which we need to find differently than ux and vx
	    double left_eta   = R[0][l][k-1];
	    double center_eta = R[0][l][k  ];
	    double right_eta  = R[0][l][k+1];

	    double left_v   = Q[1][l][k-1];
	    double center_v = Q[1][l][k  ];
	    double right_v  = Q[1][l][k+1];

	    double global_thread_y = by + j;
	    double coriolis_f = linear_coriolis_term(f_, beta_, global_thread_y,
						    dy_, y_zero_reference_);
	    double V_constant = dx_*coriolis_f/(2.0*g_);

	    double backward = theta_*g_*(center_eta - left_eta   - V_constant*(center_v + left_v ) );
	    double central  =    0.5*g_*(right_eta  - left_eta   - V_constant*(right_v + 2*center_v + left_v) ); 
	    double forward  = theta_*g_*(right_eta  - center_eta - V_constant*(center_v + right_v) );

	    // Qx[2] is really dx*Kx
	    Qx[2][j][i] = minmodRawDouble(backward, central, forward); 
	    
        }
    }
    
    //Reconstruct slopes along y axis
    for (int j=ty; j<block_height+2; j+=get_local_size(1)) {
        const int l = j + 1;
        for (int i=tx; i<block_width; i+=get_local_size(0)) {            
            const int k = i + 2; //Skip ghost cells
            for (int p=0; p<2; ++p) {
                Qy[p][j][i] = minmodSlopeDouble(Q[p][l-1][k], Q[p][l][k], Q[p][l+1][k], theta_);
            }
	    // Qy[2] = Ly, which we need to find differently than uy and vy
	    double lower_eta  = R[0][l-1][k];
	    double center_eta = R[0][l  ][k];
	    double upper_eta  = R[0][l+1][k];

	    double global_thread_y = by + j;
	    double center_coriolis_f = linear_coriolis_term(f_, beta_, global_thread_y,
							   dy_, y_zero_reference_);
	    double lower_coriolis_f  = linear_coriolis_term(f_, beta_, global_thread_y - 1,
							   dy_, y_zero_reference_);
	    double upper_coriolis_f  = linear_coriolis_term(f_, beta_, global_thread_y + 1,
							   dy_, y_zero_reference_);
	   	    
	    double lower_fu  = Q[0][l-1][k]*lower_coriolis_f;
	    double center_fu = Q[0][l  ][k]*center_coriolis_f;
	    double upper_fu  = Q[0][l+1][k]*upper_coriolis_f;

	    double U_constant = dy_/(2.0*g_);

	    double backward = theta_*g_*(center_eta - lower_eta  + U_constant*(center_fu + lower_fu ) );
	    double central  =    0.5*g_*(upper_eta  - lower_eta  + U_constant*(upper_fu + 2*center_fu + lower_fu) ); 
	    double forward  = theta_*g_*(upper_eta  - center_eta + U_constant*(center_fu + upper_fu) );

	    // Qy[2] is really dy*Ly
	    Qy[2][j][i] = minmodRawDouble(backward, central, forward); 
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);



	
    //Compute fluxes along the x axis
    for (int j=ty; j<block_height; j+=get_local_size(1)) {
        const int l = j + 2; //Skip ghost cells (be consistent with reconstruction offsets)
        for (int i=tx; i<block_width+1; i+=get_local_size(0)) {
            const int k = i + 1;

            // (u, v) reconstructed at a cell interface from the right (p) and left (m)
	    const double2 Rp = (double2)(Q[0][l][k+1] - 0.5*Qx[0][j][i+1],
                                       Q[1][l][k+1] - 0.5*Qx[1][j][i+1]);
            const double2 Rm = (double2)(Q[0][l][k  ] + 0.5*Qx[0][j][i  ],
                                       Q[1][l][k  ] + 0.5*Qx[1][j][i  ]);

            // Variables to reconstruct h from u, v, K, L
            const double vp = Q[1][l][k+1];
            const double vm = Q[1][l][k  ];

	    // Depth in the cells on each side of the face:
	    const double Hm_p = Hm[l][k+1];
	    const double Hm_m = Hm[l][k  ];
	    // H is RHx on the given face!
	    const double H_face = RHx[j][i];

	    const double eta_bar_p = R[0][l][k+1];
	    const double eta_bar_m = R[0][l][k  ];

	    // Qx[2] is really dx*Kx
	    const double Kx_p = Qx[2][j][i+1];
            const double Kx_m = Qx[2][j][i  ];

	    // Coriolis parameter
	    double global_thread_y = by + j;
	    double coriolis_f = linear_coriolis_term(f_, beta_, global_thread_y,
						    dy_, y_zero_reference_);
	    
            // Reconstruct h 
            const double hp = eta_bar_p + H_face - (Kx_p + dx_*coriolis_f*vp)/(2.0*g_); 
	    const double hm = eta_bar_m + H_face + (Kx_m + dx_*coriolis_f*vm)/(2.0*g_);
	    
            // Our flux variables Q=(h, u, v)
            const double3 Qp = (double3)(hp, Rp.x, Rp.y);
            const double3 Qm = (double3)(hm, Rm.x, Rm.y);
                                       
            // Computed flux
            const double3 flux = CDKLM16_flux(Qm, Qp, g_);
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
	    const double2 Rp = (double2)(Q[0][l+1][k] - 0.5*Qy[0][j+1][i],
                                       Q[1][l+1][k] - 0.5*Qy[1][j+1][i]);
	    const double2 Rm = (double2)(Q[0][l  ][k] + 0.5*Qy[0][j  ][i],
				       Q[1][l  ][k] + 0.5*Qy[1][j  ][i]);
              
            // Variables to reconstruct h from u, v, K, L
            const double up = Q[0][l+1][k];
            const double um = Q[0][l  ][k];

	    // Depth in the cells on each side of the face:
	    const double Hm_p = Hm[l+1][k];
	    const double Hm_m = Hm[l  ][k];
	    // H is RHx on the given face!
	    const double H_face = RHy[j][i];

	    const double eta_bar_p = R[0][l+1][k];
	    const double eta_bar_m = R[0][l  ][k];

	    // Qy[2] is really dy*Ly
	    const double Ly_p = Qy[2][j+1][i];
            const double Ly_m = Qy[2][j  ][i];

	    // Coriolis parameter
	    double global_thread_y = by + j;
	    double coriolis_fm = linear_coriolis_term(f_, beta_, global_thread_y,
						    dy_, y_zero_reference_);
	    double coriolis_fp = linear_coriolis_term(f_, beta_, global_thread_y + 1, 
						    dy_, y_zero_reference_);
	    
            // Reconstruct h 
	    const double hp = eta_bar_p + H_face - ( Ly_p - dy_*coriolis_fp*up)/(2.0*g_); 
	    const double hm = eta_bar_m + H_face + ( Ly_m - dy_*coriolis_fm*um)/(2.0*g_); 

	    // Our flux variables Q=(h, v, u)
            // Note that we swap u and v
            const double3 Qp = (double3)(hp, Rp.y, Rp.x);
            const double3 Qm = (double3)(hm, Rm.y, Rm.x);
            
            // Computed flux
            // Note that we swap back u and v
            const double3 flux = CDKLM16_flux(Qm, Qp, g_);
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
        
        const float X = windStressX(wind_stress_, dx_, dy_, dt_, t_);
        const float Y = windStressY(wind_stress_, dx_, dy_, dt_, t_);

	// Bottom topography source terms!
	// -g*(eta + H)*(-1)*dH/dx   * dx
	const double st1 = g_*(R[0][j][i] + Hm[j][i])*(RHx[ty  ][tx+1] - RHx[ty][tx]);
	const double st2 = g_*(R[0][j][i] + Hm[j][i])*(RHy[ty+1][tx  ] - RHy[ty][tx]);

	// Coriolis parameter
	double global_thread_y = tj-2; // Global id including ghost cells
	double coriolis_f = linear_coriolis_term(f_, beta_, global_thread_y,
						dy_, y_zero_reference_);
	
        const double L1  = - (F[0][ty  ][tx+1] - F[0][ty][tx]) / dx_ 
	                  - (G[0][ty+1][tx  ] - G[0][ty][tx]) / dy_;
        const double L2  = - (F[1][ty  ][tx+1] - F[1][ty][tx]) / dx_ 
                          - (G[1][ty+1][tx  ] - G[1][ty][tx]) / dy_
  	                  + (X + coriolis_f*R[2][j][i] + st1/dx_);
        const double L3  = - (F[2][ty  ][tx+1] - F[2][ty][tx]) / dx_ 
                          - (G[2][ty+1][tx  ] - G[2][ty][tx]) / dy_
	                  + (Y - coriolis_f*R[1][j][i] + st2/dy_);
	
        
	__global float* const eta_row = (__global float*) ((__global char*) eta1_ptr_ + eta1_pitch_*tj);
        __global float* const hu_row  = (__global float*) ((__global char*) hu1_ptr_  +  hu1_pitch_*tj);
        __global float* const hv_row  = (__global float*) ((__global char*) hv1_ptr_  +  hv1_pitch_*tj);

	if (rk_order < 3) {
         
	    const double C = 2.0*r_*dt_/(R[0][j][i] + Hm[j][i]);
                     
	    if  (step_ == 0) {
		//First step of RK2 ODE integrator
            
		eta_row[ti] = (float)(  R[0][j][i] + dt_*L1 );
		hu_row[ti]  = (float)( (R[1][j][i] + dt_*L2) / (1.0 + C) );
		hv_row[ti]  = (float)( (R[2][j][i] + dt_*L3) / (1.0 + C) );
	    }
	    else if (step_ == 1) {
		//Second step of RK2 ODE integrator
            
		//First read Q^n
		const double eta_a = eta_row[ti];
		const double hu_a  =  hu_row[ti];
		const double hv_a  =  hv_row[ti];
            
		//Compute Q^n+1
		const double eta_b = 0.5*(eta_a + (R[0][j][i] + dt_*L1));
		const double hu_b  = 0.5*( hu_a + (R[1][j][i] + dt_*L2));
		const double hv_b  = 0.5*( hv_a + (R[2][j][i] + dt_*L3));

				
		//Write to main memory
		eta_row[ti] = (float)( eta_b );
		hu_row[ti]  = (float)( hu_b / (1.0 + 0.5*C) );
		hv_row[ti]  = (float)( hv_b / (1.0 + 0.5*C) );
		//hu_row[ti] = RBx[ty][tx];
		//hv_row[ti] = RBy[ty][tx];

	    }
	}

	
	else if (rk_order == 3) {
	    // Third order Runge Kutta - only valid if r_ = 0.0 (no friction)

	    if (step_ == 0) {
		//First step of RK3 ODE integrator
		// q^(1) = q^n + dt*L(q^n)
            
		eta_row[ti] = (float)(  R[0][j][i] + dt_*L1);
		hu_row[ti]  = (float)( (R[1][j][i] + dt_*L2));
		hv_row[ti]  = (float)( (R[2][j][i] + dt_*L3));

	    } else if (step_ == 1) {
		// Second step of RK3 ODE integrator
		// Q^(2) = 3/4 Q^n + 1/4 ( Q^(1) + dt*L(Q^(1)) )
		// Q^n is here in h1, but will be used in next iteration as well --> write to h0

		// First read Q^n:
		const double eta_a = eta_row[ti];
		const double hu_a  =  hu_row[ti];
		const double hv_a  =  hv_row[ti];

		// Compute Q^(2):
		const double eta_b = 0.75*eta_a + 0.25*(R[0][j][i] + dt_*L1);
		const double hu_b  = 0.75* hu_a + 0.25*(R[1][j][i] + dt_*L2);
		const double hv_b  = 0.75* hv_a + 0.25*(R[2][j][i] + dt_*L3);

		// Write output to the input buffer:
		__global float* const eta_out_row = (__global float*) ((__global char*) eta0_ptr_ + eta0_pitch_*tj);
		__global float* const hu_out_row  = (__global float*) ((__global char*)  hu0_ptr_ +  hu0_pitch_*tj);
		__global float* const hv_out_row  = (__global float*) ((__global char*)  hv0_ptr_ +  hv0_pitch_*tj);
		eta_out_row[ti] = (float)( eta_b );
		hu_out_row[ti]  = (float)(  hu_b );
		hv_out_row[ti]  = (float)(  hv_b );
		
	    } else if (step_ == 2) {
		// Third step of RK3 ODE integrator
		// Q^n+1 = 1/3 Q^n + 2/3 (Q^(2) + dt*L(Q^(2))

		// First read Q^n:
		const double eta_a = eta_row[ti];
		const double hu_a  =  hu_row[ti];
		const double hv_a  =  hv_row[ti];

		// Compute Q^n+1:
		const double eta_b = (eta_a + 2.0*(R[0][j][i] + dt_*L1)) / 3.0;
		const double hu_b  = ( hu_a + 2.0*(R[1][j][i] + dt_*L2)) / 3.0;
		const double hv_b  = ( hv_a + 2.0*(R[2][j][i] + dt_*L3)) / 3.0;

		//Write to main memory
		eta_row[ti] = (float)( eta_b );
		hu_row[ti]  = (float)(  hu_b );
		hv_row[ti]  = (float)(  hv_b );
		
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

	/*
	float bigone = (float)(dx_*dy_*dy_);
	float smalli = (float)(1.0)/(float)(dx_);
	float float_perted_biggi = bigone + smalli;
	double double_perted_biggi = (double)(bigone) + (double)(smalli);

	float bad_one = float_perted_biggi - bigone;
	double good_one = double_perted_biggi - (double)(bigone);
	
	hu_row[ti]  =  (float)(bad_one);
	hv_row[ti]  =  (float)(good_one);
	*/
    }

	
    
}
