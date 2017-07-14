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

void reconstructBx(__local float RBx[block_height+4][block_width+4],
		   __local float  Bi[block_height+4][block_width+4],
		   const int p,
		   const int q) {
    RBx[q][p] = 0.5f*(Bi[q][p]+Bi[q+1][p]); //0.5*(down+up)
}

void reconstructBy(__local float RBy[block_height+4][block_width+4],
		   __local float  Bi[block_height+4][block_width+4],
		   const int p,
		   const int q) {
    RBy[q][p] = 0.5f*(Bi[q][p]+Bi[q][p+1]); //0.5*(left+right)
}

void reconstructB(__local float RBx[block_height+4][block_width+4],
		  __local float RBy[block_height+4][block_width+4],
		  __local float  Bi[block_height+4][block_width+4]) {
    const int p = get_local_id(0) + 2;
    const int q = get_local_id(1) + 2;
    reconstructBx(RBx, Bi, p, q);
    reconstructBy(RBy, Bi, p, q);

    //Use one warp to perform the extra reconstructions needed
    if (get_local_id(1) == 0) { 
	reconstructBy(RBy, Bi, p, 1);//second row
	reconstructBy(RBy, Bi, p, block_height+2);//second last row
	reconstructBy(RBy, Bi, p, block_height+3);//last row
	if (get_local_id(0) < block_height) {
	    reconstructBx(RBx, Bi, 1, p);//second column
	    reconstructBx(RBx, Bi, block_width+2, p); //second last column
	    reconstructBx(RBx, Bi, block_width+3, p);//last column
	}
    }
}


void adjustSlopeUx(__local float Qx[3][block_height+2][block_width+2],
		   __local float RBx[block_height+4][block_width+4],
		   __local float Q[3][block_height+4][block_width+4],
		   const int p, const int q) {
    // define indices in the Qx world:
    const int pQx = p - 1;
    const int qQx = q - 2;
    
    Qx[0][qQx][pQx] = (Q[0][q][p]-Qx[0][qQx][pQx] < RBx[q][p]) ?
	(Q[0][q][p] - RBx[q][p]) : Qx[0][qQx][pQx];
    Qx[0][qQx][pQx] = (Q[0][q][p]+Qx[0][qQx][pQx] < RBx[q][p+1]) ?
	(RBx[q][p+1] - Q[0][q][p]) : Qx[0][qQx][pQx];
    
}

void adjustSlopeUy(__local float Qy[3][block_height+2][block_width+2],
		   __local float RBy[block_height+4][block_width+4],
		   __local float Q[3][block_height+4][block_width+4],
		   const int p, const int q) {
    // define indices in the Qy world:
    const int pQy = p - 2;
    const int qQy = q - 1;

    Qy[0][qQy][pQy] = (Q[0][q][p]-Qy[0][qQy][pQy] < RBy[q][p]) ?
	(Q[0][q][p] - RBy[q][p]) : Qy[0][qQy][pQy];
    Qy[0][qQy][pQy] = (Q[0][q][p]+Qy[0][qQy][pQy] < RBy[q+1][p]) ?
	(RBy[q+1][p] - Q[0][q][p]) : Qy[0][qQy][pQy];
    
}

void adjustSlopes(__local float Qx[3][block_height+2][block_width+2],
		  __local float Qy[3][block_height+2][block_width+2],
		  __local float RBx[block_height+4][block_width+4],
		  __local float RBy[block_height+4][block_width+4],
		  __local float Q[3][block_height+4][block_width+4] ) {
    const int p = get_local_id(0) + 2;
    const int q = get_local_id(1) + 2;

    adjustSlopeUx(Qx, RBx, Q, p, q);
    adjustSlopeUy(Qy, RBy, Q, p, q);

    // Use one warp to perform the extra adjustments
    if (get_local_id(1) == 0) {
	adjustSlopeUy(Qy, RBy, Q, p, 1);
	adjustSlopeUy(Qy, RBy, Q, p, block_height+2);

	if (get_local_id(0) < block_height) {
	    adjustSlopeUx(Qx, RBx, Q, 1, p);
	    adjustSlopeUx(Qx, RBx, Q, block_width+2, p);
	}
    }
    
}



void computeFluxF(__local float Q[3][block_height+4][block_width+4],
                  __local float Qx[3][block_height+2][block_width+2],
                  __local float F[3][block_height+1][block_width+1],
		  __local float RBx[block_height+4][block_width+4],
                  const float g_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    for (int j=ty; j<block_height; j+=get_local_size(1)) {
        const int l = j + 2; //Skip ghost cells
        for (int i=tx; i<block_width+1; i+=get_local_size(0)) {
            const int k = i + 1;
            // Q at interface from the right and left
	    // In CentralUpwindFlux we need [h, hu, hv]
	    // Subtract the bottom elevation on the relevant face in Q[0]
            float3 Qp = (float3)(Q[0][l][k+1] - Qx[0][j][i+1],
                                 Q[1][l][k+1] - Qx[1][j][i+1],
                                 Q[2][l][k+1] - Qx[2][j][i+1]);
            float3 Qm = (float3)(Q[0][l][k  ] + Qx[0][j][i  ],
                                 Q[1][l][k  ] + Qx[1][j][i  ],
                                 Q[2][l][k  ] + Qx[2][j][i  ]);
                                       
            // Computed flux
            const float3 flux = CentralUpwindFluxBottom(Qm, Qp, RBx[l][k+1], g_);
            F[0][j][i] = flux.x;
            F[1][j][i] = flux.y;
            F[2][j][i] = flux.z;
        }
    }    
}

void computeFluxG(__local float Q[3][block_height+4][block_width+4],
                  __local float Qy[3][block_height+2][block_width+2],
                  __local float G[3][block_height+1][block_width+1],
		  __local float RBy[block_height+4][block_width+4],
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
            float3 Qp = (float3)(Q[0][l+1][k] - Qy[0][j+1][i],
                                 Q[2][l+1][k] - Qy[2][j+1][i],
                                 Q[1][l+1][k] - Qy[1][j+1][i]);
            float3 Qm = (float3)(Q[0][l  ][k] + Qy[0][j  ][i],
                                 Q[2][l  ][k] + Qy[2][j  ][i],
                                 Q[1][l  ][k] + Qy[1][j  ][i]);
                                       
            // Computed flux
            // Note that we swap back
            const float3 flux = CentralUpwindFluxBottom(Qm, Qp, RBy[l+1][k], g_);
            G[0][j][i] = flux.x;
            G[1][j][i] = flux.z;
            G[2][j][i] = flux.y;
        }
    }
}




void init_B_with_garbage(__local float Bi[block_height+4][block_width+4],
			 __local float RBx[block_height+4][block_width+4],
			 __local float RBy[block_height+4][block_width+4] ) {

    if (get_local_id(0) == 0 && get_local_id(1) == 0) {
	for (int j = 0; j < block_height+4; j++) {
	    for (int i = 0; i < block_width+4; i++) {
		Bi[j][i]  = -99.0f; //0.1*get_global_id(0);
		RBx[j][i] = -99.0f; //0.1*get_global_id(0);
		RBy[j][i] = -99.0f; //0.1*get_global_id(0);
	    }
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

	// Bathymetry at cell intersections
	__global float* Bi_ptr_, int Bi_pitch_,
	__global float* Bm_ptr_, int Bm_pitch_,
        
        //Wind stress parameters
        int wind_stress_type_, 
        float tau0_, float rho_, float alpha_, float xm_, float Rc_,
        float x0_, float y0_,
        float u0_, float v0_,

	// Boundary condition flag
	int boundary_conditions_type_, // < 1: wall, 2: periodic, 
                                       //   3: periodicNS, 4: periodicEW
	
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

    // Shared memory for bathymetry, and
    // reconstructed bathymetry in both directions
    // We use too much shared memory for RBx and RBy in order
    // to easier follow the implementation from the C++ version
    __local float  Bi[block_height+4][block_width+4];
    __local float RBx[block_height+4][block_width+4];
    __local float RBy[block_height+4][block_width+4];

    //init_B_with_garbage(Bi, RBx, RBy);
    //barrier(CLK_LOCAL_MEM_FENCE);
    
    // Read B in mid-cell:
    __global float* const Bm_row  = (__global float*) ((__global char*) Bm_ptr_ + Bm_pitch_*tj);
    const float Bm = Bm_row[ti];
       
    //Read Q = [h, hu, hv] into shared memory
    readBlock2(h0_ptr_, h0_pitch_,
               hu0_ptr_, hu0_pitch_,
               hv0_ptr_, hv0_pitch_,
               Q, nx_, ny_);
   
    // Read B into sheared memory
    readBlock2single(Bi_ptr_, Bi_pitch_,
		     Bi, nx_, ny_);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reconstruct B at the cell faces
    reconstructB(RBx, RBy, Bi);
    barrier(CLK_LOCAL_MEM_FENCE);

    //Fix boundary conditions
    // TODO: Add if on boundary condition
    noFlowBoundary2(Q, nx_, ny_, boundary_conditions_type_);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    //Reconstruct slopes along x and axis
    minmodSlopeX(Q, Qx, theta_);
    minmodSlopeY(Q, Qy, theta_);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Adjust the slopes to avoid negative values at integration points
    adjustSlopes(Qx, Qy, RBx, RBy, Q);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //Compute fluxes along the x and y axis
    computeFluxF(Q, Qx, F, RBx, g_);
    computeFluxG(Q, Qy, G, RBy, g_);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    //Sum fluxes and advance in time for all internal cells
    //Check global indices against global domain
    if (ti > 1 && ti < nx_+2 && tj > 1 && tj < ny_+2) {
        const int i = tx + 2; //Skip local ghost cells, i.e., +2
        const int j = ty + 2;

	// Find bottom topography source terms: S2, S3
	const float ST2 = bottomSourceTerm2(Q, Qx, RBx, g_, i, j);
	const float ST3 = bottomSourceTerm3(Q, Qy, RBy, g_, i, j);
	
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

	const float R1 =
	    - (F[0][ty  ][tx+1] - F[0][ty][tx]) / dx_
	    - (G[0][ty+1][tx  ] - G[0][ty][tx]) / dy_;
	const float R2 =
	    - (F[1][ty  ][tx+1] - F[1][ty][tx]) / dx_ 
	    - (G[1][ty+1][tx  ] - G[1][ty][tx]) / dy_
	    + (X + f_*Q[2][j][i] - ST2/dx_);
	const float R3 =
	    - (F[2][ty  ][tx+1] - F[2][ty][tx]) / dx_
	    - (G[2][ty+1][tx  ] - G[2][ty][tx]) / dy_
	    + (Y - f_*Q[1][j][i] - ST3/dy_);
						       
	
	__global float* const h_row  = (__global float*) ((__global char*) h1_ptr_ + h1_pitch_*tj);
        __global float* const hu_row = (__global float*) ((__global char*) hu1_ptr_ + hu1_pitch_*tj);
        __global float* const hv_row = (__global float*) ((__global char*) hv1_ptr_ + hv1_pitch_*tj);

	const float C = 2.0f*r_*dt_/(Q[0][j][i]-Bm);
                    
        if  (step_ == 0) {
            //First step of RK2 ODE integrator
            
            h_row[ti]  =  Q[0][j][i] + dt_*R1;
            hu_row[ti] = (Q[1][j][i] + dt_*R2) / (1.0f + C);
            hv_row[ti] = (Q[2][j][i] + dt_*R3) / (1.0f + C);
        }
        else if (step_ == 1) {
            //Second step of RK2 ODE integrator
            
            //First read Q^n
            const float h_a  = h_row[ti];
            const float hu_a = hu_row[ti];
            const float hv_a = hv_row[ti];
            
            //Compute Q^n+1
            const float h_b  = 0.5f*(h_a  + (Q[0][j][i] + dt_*R1));
            const float hu_b = 0.5f*(hu_a + (Q[1][j][i] + dt_*R2));
            const float hv_b = 0.5f*(hv_a + (Q[2][j][i] + dt_*R3));
            
            //Write to main memory
            h_row[ti] = h_b;
            hu_row[ti] = hu_b / (1.0f + 0.5f*C);
            hv_row[ti] = hv_b / (1.0f + 0.5f*C);
        }
    }
}
