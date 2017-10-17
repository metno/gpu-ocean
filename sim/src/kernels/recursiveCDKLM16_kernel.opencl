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

	//Bathymery
	__global float* Bi_ptr_, int Bi_pitch_,
	__global float* Bm_ptr_, int Bm_pitch_,
	
        //Wind stress parameters
        int wind_stress_type_, 
        float tau0_, float rho_, float alpha_, float xm_, float Rc_,
        float x0_, float y0_,
        float u0_, float v0_,
        float t_, 
    
        // Boundary conditions
        int boundary_conditions_type_ // < 1: wall, 2: periodic, 
                                       //   3: periodicNS, 4: periodicEW
) {
        
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of block within domain
    const int bx = get_local_size(0) * get_group_id(0);
    const int by = get_local_size(1) * get_group_id(1);

    //Index of cell within domain
    const int ti = get_global_id(0) + 3; //Skip global ghost cells, i.e., +3
    const int tj = get_global_id(1) + 3;
    
    // Our physical variables
    __local float R[3][block_height+6][block_width+6];
    
    // Our reconstruction variables
    __local float Q[4][block_height+4][block_width+4];
    __local float Qx[4][block_height][block_width+2];
    __local float Qy[4][block_height+2][block_width];
    
    // Our fluxes
    __local float F[3][block_height][block_width+1];
    __local float G[3][block_height+1][block_width];
    
    
    // Bathymetry
    __local float  Bi[block_height+1][block_width+1];
    __local float  Bm[block_height+4][block_width+4];
    __local float RBx[block_height  ][block_width+1];
    __local float RBy[block_height+1][block_width  ];
    
    
    //Read into shared memory
    for (int j=ty; j<block_height+6; j+=get_local_size(1)) {
        const int l = clamp(by + j, 0, ny_+5); // Out of bounds
        
        //Compute the pointer to current row in the arrays
        __global float* const h_row = (__global float*) ((__global char*) h0_ptr_ + h0_pitch_*l);
        __global float* const hu_row = (__global float*) ((__global char*) hu0_ptr_ + hu0_pitch_*l);
        __global float* const hv_row = (__global float*) ((__global char*) hv0_ptr_ + hv0_pitch_*l);
        
        for (int i=tx; i<block_width+6; i+=get_local_size(0)) {
            const int k = clamp(bx + i, 0, nx_+5); // Out of bounds
            
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
	const int l = clamp(by+j+1, 1, ny_+4); 
	__global float* const Bm_row = (__global float*) ((__global char*) Bm_ptr_ + Bm_pitch_*l);
	for(int i=tx; i < block_width+4; i+=get_local_size(0)) {
	    const int k = clamp(bx+1+i, 1, nx_+4);

	    Bm[j][i] = Bm_row[k];
	}
    }

    // Read Bi into shared memory
    // Read intersections on all non-ghost cells
    for(int j=ty; j < block_height+1; j+=get_local_size(1)) {
	// Skip ghost cells and 
	const int l = clamp(by+j+3, 3, ny_+3);
	__global float* const Bi_row = (__global float*) ((__global char*) Bi_ptr_ + Bi_pitch_*l);
	for(int i=tx; i < block_width+1; i+=get_local_size(0)) {
	    const int k = clamp(bx+i+3, 3, nx_+3);

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
    // Evaluate piecewise bi-linear for RBx 
    for (int j=ty; j<block_height+1; j+=get_local_size(1)) {
        for (int i=tx; i<block_width; i+=get_local_size(0)) {
            RBy[j][i] = 0.5f*( Bi[j][i] + Bi[j][i+1] );
	}
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //Fix boundary conditions
    if (boundary_conditions_type_ != 2)
    {
        // These boundary conditions are dealt with inside shared memory
        
        const int i = tx + 3; //Skip local ghost cells, i.e., +3
        const int j = ty + 3;
        
        if (ti == 3 && boundary_conditions_type_ != 4) {
	    // Wall boundary on east and west
	    R[0][j][i-1] =  R[0][j][i];
            R[1][j][i-1] = -R[1][j][i];
            R[2][j][i-1] =  R[2][j][i];
            
            R[0][j][i-2] =  R[0][j][i+1];
            R[1][j][i-2] = -R[1][j][i+1];
            R[2][j][i-2] =  R[2][j][i+1];
            
            R[0][j][i-3] =  R[0][j][i+2];
            R[1][j][i-3] = -R[1][j][i+2];
            R[2][j][i-3] =  R[2][j][i+2];
        }
        if (ti == nx_+2 && boundary_conditions_type_ != 4) {
	    // Wall boundary on east and west
            R[0][j][i+1] =  R[0][j][i];
            R[1][j][i+1] = -R[1][j][i];
            R[2][j][i+1] =  R[2][j][i];
            
            R[0][j][i+2] =  R[0][j][i-1];
            R[1][j][i+2] = -R[1][j][i-1];
            R[2][j][i+2] =  R[2][j][i-1];
            
            R[0][j][i+3] =  R[0][j][i-2];
            R[1][j][i+3] = -R[1][j][i-2];
            R[2][j][i+3] =  R[2][j][i-2];
        }
        if (tj == 3 && boundary_conditions_type_ != 3) {
	    // Wall boundary on north and south
	    R[0][j-1][i] =  R[0][j][i];
            R[1][j-1][i] =  R[1][j][i];
            R[2][j-1][i] = -R[2][j][i];
            
            R[0][j-2][i] =  R[0][j+1][i];
            R[1][j-2][i] =  R[1][j+1][i];
            R[2][j-2][i] = -R[2][j+1][i];
            
            R[0][j-3][i] =  R[0][j+2][i];
            R[1][j-3][i] =  R[1][j+2][i];
            R[2][j-3][i] = -R[2][j+2][i];
        }
        if (tj == ny_+2 && boundary_conditions_type_ != 3) {
	    // Wall boundary on north and south
            R[0][j+1][i] =  R[0][j][i];
            R[1][j+1][i] =  R[1][j][i];
            R[2][j+1][i] = -R[2][j][i];
            
            R[0][j+2][i] =  R[0][j-1][i];
            R[1][j+2][i] =  R[1][j-1][i];
            R[2][j+2][i] = -R[2][j-1][i];
            
            R[0][j+3][i] =  R[0][j-2][i];
            R[1][j+3][i] =  R[1][j-2][i];
            R[2][j+3][i] = -R[2][j-2][i];
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
        
    
    //Create our "steady state" reconstruction variables (u, v, K, L)
    for (int j=ty; j<block_height+4; j+=get_local_size(1)) {
        const int l = j + 1; //Skip one "ghost cell row" of Q, going from 6x6 to 4x4 "halo"
        for (int i=tx; i<block_width+4; i+=get_local_size(0)) {
            const int k = i + 1;
            
            const float h = R[0][l][k];
            const float u = R[1][l][k] / h;
            const float v = R[2][l][k] / h;

	    // B = Bm for the given cell -> Bm should be in shared mem with a halo of 2.
            const float B = Bm[j][i];
	    
	    // U and V are defined recursivly in Eq (2.5).
	    float Um = 0.0f;
	    float Vm = 0.0f;
	    float Up = 0.0f;
	    float Vp = 0.0f;

	    const int glob_j = bx+k;
	    const int glob_k = by+l;
	    // Add an if-statement to check we are in global domain...
	    if (glob_j > 0 && glob_j < nx_+5 && glob_k > 0 && glob_k < ny_+5) {

		// Summing along a column to get Um:
		for (int glob_l = 3; glob_l < glob_k; ++glob_l) {
		    __global float* const h_row = (__global float*) ((__global char*) h0_ptr_ + h0_pitch_*glob_l);
		    __global float* const hu_row = (__global float*) ((__global char*) hu0_ptr_ + hu0_pitch_*glob_l);
		    Um += (hu_row[glob_j]/h_row[glob_j])*dy_;
		}
		Um = Um * f_/g_;
		Up = Um + (f_/g_)*u*dy_;

		// Summing along a row to get Vm:
		__global float* const h_row = (__global float*) ((__global char*) h0_ptr_ + h0_pitch_*glob_k);
		__global float* const hv_row = (__global float*) ((__global char*) hv0_ptr_ + hv0_pitch_*glob_k);
		for (int glob_m = 3; glob_m < glob_j; ++glob_m) {
		    Vm += (hv_row[glob_m]/h_row[glob_m])*dx_;
		}
		Vm = Vm * f_/g_;
		Vp = Vm + (f_/g_)*v*dx_;

	    }

	    const float U = 0.5f*(Up + Um);
	    const float V = 0.5f*(Vp + Vm);
	    
	    //const float U = Um + u*dy_*f_/(2.0f*g_);
	    //const float V = Vm + v*dx_*f_/(2.0f*g_);
	    	    
            //const float U = 0.25f * f_/g_ * (1.0*R[1][l+1][k]/R[0][l+1][k] + 2.0f*u + 1.0f*R[1][l-1][k]/R[0][l-1][k]);
            //const float V = 0.25f * f_/g_ * (1.0*R[2][l][k+1]/R[0][l][k+1] + 2.0f*v + 1.0f*R[2][l][k-1]/R[0][l][k-1]);
            //const float U = f_/g_ * u;
            //const float V = f_/g_ * v;

	    const float K = h + B - V;
            const float L = h + B + U;
            
            Q[0][j][i] = u;
            Q[1][j][i] = v;
            Q[2][j][i] = K;
            Q[3][j][i] = L;         
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    
    
    
    
    //Reconstruct slopes along x axis
    for (int j=ty; j<block_height; j+=get_local_size(1)) {
        const int l = j + 2; //Skip ghost cells
        for (int i=tx; i<block_width+2; i+=get_local_size(0)) {
            const int k = i + 1;
            for (int p=0; p<4; ++p) {
                Qx[p][j][i] = minmodSlope(Q[p][l][k-1], Q[p][l][k], Q[p][l][k+1], theta_);
            }
        }
    }
    
    //Reconstruct slopes along y axis
    for (int j=ty; j<block_height+2; j+=get_local_size(1)) {
        const int l = j + 1;
        for (int i=tx; i<block_width; i+=get_local_size(0)) {            
            const int k = i + 2; //Skip ghost cells
            for (int p=0; p<4; ++p) {
                Qy[p][j][i] = minmodSlope(Q[p][l-1][k], Q[p][l][k], Q[p][l+1][k], theta_);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

      
    
    
    //Compute fluxes along the x axis
    for (int j=ty; j<block_height; j+=get_local_size(1)) {
        const int l = j + 2; //Skip ghost cells (be consistent with reconstruction offsets)
        for (int i=tx; i<block_width+1; i+=get_local_size(0)) {
            const int k = i + 1;

            // R=(u, v, K, L) reconstructed at a cell interface from the right (p) and left (m)
            const float4 Rp = (float4)(Q[0][l][k+1] - 0.5f*Qx[0][j][i+1],
                                       Q[1][l][k+1] - 0.5f*Qx[1][j][i+1],
                                       Q[2][l][k+1] - 0.5f*Qx[2][j][i+1],
                                       Q[3][l][k+1] - 0.5f*Qx[3][j][i+1]);
            const float4 Rm = (float4)(Q[0][l][k  ] + 0.5f*Qx[0][j][i  ],
                                       Q[1][l][k  ] + 0.5f*Qx[1][j][i  ],
                                       Q[2][l][k  ] + 0.5f*Qx[2][j][i  ],
                                       Q[3][l][k  ] + 0.5f*Qx[3][j][i  ]);

            // Variables to reconstruct h from u, v, K, L
            //const float v = Q[1][l][k];

	    // Finding V for reconstruction of hp and hm
	    float V = 0.0f;
	    
	    const int glob_j = bx+k+1;
	    const int glob_k = by+l+1;
	    // Add an if-statement to check we are in global domain...
	    if (glob_j > 0 && glob_j < nx_+5 && glob_k > 0 && glob_k < ny_+5) {

		// Summing along a row to get Vm:
		__global float* const h_row = (__global float*) ((__global char*) h0_ptr_ + h0_pitch_*glob_k);
		__global float* const hv_row = (__global float*) ((__global char*) hv0_ptr_ + hv0_pitch_*glob_k);
		for (int glob_m = 3; glob_m < glob_j+1; ++glob_m) {
		    V += (f_/g_)*(hv_row[glob_m]/h_row[glob_m])*dx_;
		}
		//V = V * f_/g_;
		//V += (f_/g_)*v*dx_;
	    }
	    
	    // B is RBx on the given face!
	    const float B = RBx[j][i];	    
            
            // Reconstruct h = K/g + V - B
            const float hp = Rp.z + V - B;
            const float hm = Rm.z + V - B;
            
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
            const float4 Rp = (float4)(Q[0][l+1][k] - 0.5f*Qy[0][j+1][i],
                                       Q[1][l+1][k] - 0.5f*Qy[1][j+1][i],
                                       Q[2][l+1][k] - 0.5f*Qy[2][j+1][i],
                                       Q[3][l+1][k] - 0.5f*Qy[3][j+1][i]);
            const float4 Rm = (float4)(Q[0][l  ][k] + 0.5f*Qy[0][j  ][i],
                                       Q[1][l  ][k] + 0.5f*Qy[1][j  ][i],
                                       Q[2][l  ][k] + 0.5f*Qy[2][j  ][i],
                                       Q[3][l  ][k] + 0.5f*Qy[3][j  ][i]);
              
            // Variables to reconstruct h from u, v, K, L
            //const float u = Q[0][l][k];
	    
	    // Finding U on the face for reconstruction of hp and hm
	    float U = 0.0f;
	    
	    const int glob_j = bx+k+1;
	    const int glob_k = by+l+1;
	    // Add an if-statement to check we are in global domain...
	    if (glob_j > 0 && glob_j < nx_+5 && glob_k > 0 && glob_k < ny_+5) {

		// Summing along a column to get Um:
		for (int glob_l = 3; glob_l < glob_k+1; ++glob_l) {
		    __global float* const h_row = (__global float*) ((__global char*) h0_ptr_ + h0_pitch_*glob_l);
		    __global float* const hu_row = (__global float*) ((__global char*) hu0_ptr_ + hu0_pitch_*glob_l);
		    U += (f_/g_)*(hu_row[glob_j]/h_row[glob_j])*dy_;
		}
		//U = U * f_/g_;
		//U += (f_/g_)*u*dy_;
	    }

	    // B is RBy on the given face!
            const float B = RBy[j][i];
            
            // Reconstruct h = L/g - U - B
            const float hp = Rp.w - U - B;
            const float hm = Rm.w - U - B;
            
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
    if (ti > 2 && ti < nx_+3 && tj > 2 && tj < ny_+3) {
        const int i = tx + 3; //Skip local ghost cells, i.e., +2
        const int j = ty + 3;
        
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

	// TODO:
	// Add bottom topography source terms!
	const float st1 = -g_*R[0][j][i]*(RBx[ty  ][tx+1] - RBx[ty][tx]);
	const float st2 = -g_*R[0][j][i]*(RBy[ty+1][tx  ] - RBy[ty][tx]);
        
        const float h1  = R[0][j][i] + (F[0][ty][tx] - F[0][ty  ][tx+1]) * dt_ / dx_ 
                                     + (G[0][ty][tx] - G[0][ty+1][tx  ]) * dt_ / dy_;
        const float hu1 = R[1][j][i] + (F[1][ty][tx] - F[1][ty  ][tx+1]) * dt_ / dx_ 
                                     + (G[1][ty][tx] - G[1][ty+1][tx  ]) * dt_ / dy_
                                     + dt_*X + dt_*f_*R[2][j][i] + st1*dt_/dx_;
        const float hv1 = R[2][j][i] + (F[2][ty][tx] - F[2][ty  ][tx+1]) * dt_ / dx_ 
                                     + (G[2][ty][tx] - G[2][ty+1][tx  ]) * dt_ / dy_
                                     + dt_*Y - dt_*f_*R[1][j][i] + st2*dt_/dy_;

        __global float* const h_row  = (__global float*) ((__global char*) h1_ptr_ + h1_pitch_*tj);
        __global float* const hu_row = (__global float*) ((__global char*) hu1_ptr_ + hu1_pitch_*tj);
        __global float* const hv_row = (__global float*) ((__global char*) hv1_ptr_ + hv1_pitch_*tj);
        
        const float C = 2.0f*r_*dt_/R[0][j][i];
                     
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
            h_row[ti]  = h_b;
            hu_row[ti] = hu_b / (1.0f + 0.5f*C);
            hv_row[ti] = hv_b / (1.0f + 0.5f*C);
	    //hu_row[ti] = RBx[ty][tx];
	    //hv_row[ti] = RBy[ty][tx];

	}
    }
    
}
