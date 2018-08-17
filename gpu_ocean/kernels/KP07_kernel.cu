/*
This CUDA kernel implements the Kurganov-Petrova numerical scheme 
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

#include "common.cu"

// Finds the coriolis term based on the linear Coriolis force
// f = \tilde{f} + beta*(y-y0)
__device__ float linear_coriolis_term(const float f, const float beta,
			   const float tj, const float dy,
			   const float y_zero_reference_cell) {
    // y_0 is at the southern face of the row y_zero_reference_cell.
    float y = (tj-y_zero_reference_cell + 0.5f)*dy;
    return f + beta * y;
}

__device__ void reconstructHx(float RHx[block_height+4][block_width+4],
		   float  Hi[block_height+4][block_width+4],
		   const int p,
		   const int q) {
    RHx[q][p] = 0.5f*(Hi[q][p]+Hi[q+1][p]); //0.5*(down+up)
}

__device__ void reconstructHy(float RHy[block_height+4][block_width+4],
		   float  Hi[block_height+4][block_width+4],
		   const int p,
		   const int q) {
    RHy[q][p] = 0.5f*(Hi[q][p]+Hi[q][p+1]); //0.5*(left+right)
}

// Reconstruct depth on the cell faces, store them in RHx and RHy
__device__ void reconstructH(float RHx[block_height+4][block_width+4],
		  float RHy[block_height+4][block_width+4],
		  float  Hi[block_height+4][block_width+4]) {
    const int p = threadIdx.x + 2;
    const int q = threadIdx.y + 2;
    reconstructHx(RHx, Hi, p, q);
    reconstructHy(RHy, Hi, p, q);

    //Use one warp to perform the extra reconstructions needed
    if (threadIdx.y == 0) { 
	reconstructHy(RHy, Hi, p, 1);//second row
	reconstructHy(RHy, Hi, p, block_height+2);//second last row
	reconstructHy(RHy, Hi, p, block_height+3);//last row
	if (threadIdx.x < block_height) {
	    reconstructHx(RHx, Hi, 1, p);//second column
	    reconstructHx(RHx, Hi, block_width+2, p); //second last column
	    reconstructHx(RHx, Hi, block_width+3, p);//last column
	}
    }
}


__device__ void adjustSlopeUx(float Qx[3][block_height+2][block_width+2],
		   float RHx[block_height+4][block_width+4],
		   float Q[3][block_height+4][block_width+4],
		   const int p, const int q) {
    // define indices in the Qx world:
    const int pQx = p - 1;
    const int qQx = q - 2;
    
    Qx[0][qQx][pQx] = (Q[0][q][p]-Qx[0][qQx][pQx] < -RHx[q][p]) ?
	(Q[0][q][p] + RHx[q][p]) : Qx[0][qQx][pQx];
    Qx[0][qQx][pQx] = (Q[0][q][p]+Qx[0][qQx][pQx] < -RHx[q][p+1]) ?
	(-RHx[q][p+1] - Q[0][q][p]) : Qx[0][qQx][pQx];
    
}

__device__ void adjustSlopeUy(float Qy[3][block_height+2][block_width+2],
		   float RHy[block_height+4][block_width+4],
		   float Q[3][block_height+4][block_width+4],
		   const int p, const int q) {
    // define indices in the Qy world:
    const int pQy = p - 2;
    const int qQy = q - 1;

    Qy[0][qQy][pQy] = (Q[0][q][p]-Qy[0][qQy][pQy] < -RHy[q][p]) ?
	(Q[0][q][p] + RHy[q][p]) : Qy[0][qQy][pQy];
    Qy[0][qQy][pQy] = (Q[0][q][p]+Qy[0][qQy][pQy] < -RHy[q+1][p]) ?
	(-RHy[q+1][p] - Q[0][q][p]) : Qy[0][qQy][pQy];
    
}

__device__ void adjustSlopes(float Qx[3][block_height+2][block_width+2],
		  float Qy[3][block_height+2][block_width+2],
		  float RHx[block_height+4][block_width+4],
		  float RHy[block_height+4][block_width+4],
		  float Q[3][block_height+4][block_width+4] ) {
    const int p = threadIdx.x + 2;
    const int q = threadIdx.y + 2;

    adjustSlopeUx(Qx, RHx, Q, p, q);
    adjustSlopeUy(Qy, RHy, Q, p, q);

    // Use one warp to perform the extra adjustments
    if (threadIdx.y == 0) {
	adjustSlopeUy(Qy, RHy, Q, p, 1);
	adjustSlopeUy(Qy, RHy, Q, p, block_height+2);

	if (threadIdx.x < block_height) {
	    adjustSlopeUx(Qx, RHx, Q, 1, p);
	    adjustSlopeUx(Qx, RHx, Q, block_width+2, p);
	}
    }
    
}



__device__ void computeFluxF(float Q[3][block_height+4][block_width+4],
                  float Qx[3][block_height+2][block_width+2],
                  float F[3][block_height+1][block_width+1],
		  float RHx[block_height+4][block_width+4],
                  const float g_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    for (int j=ty; j<block_height; j+=blockDim.y) {
        const int l = j + 2; //Skip ghost cells
        for (int i=tx; i<block_width+1; i+=blockDim.x) {
            const int k = i + 1;
            // Q at interface from the right and left
	    // In CentralUpwindFlux we need [eta, hu, hv]
	    // Subtract the bottom elevation on the relevant face in Q[0]
            float3 Qp = make_float3(Q[0][l][k+1] - Qx[0][j][i+1],
                                 Q[1][l][k+1] - Qx[1][j][i+1],
                                 Q[2][l][k+1] - Qx[2][j][i+1]);
            float3 Qm = make_float3(Q[0][l][k  ] + Qx[0][j][i  ],
                                 Q[1][l][k  ] + Qx[1][j][i  ],
                                 Q[2][l][k  ] + Qx[2][j][i  ]);
                                       
            // Computed flux
            const float3 flux = CentralUpwindFluxBottom(Qm, Qp, RHx[l][k+1], g_);
            F[0][j][i] = flux.x;
            F[1][j][i] = flux.y;
            F[2][j][i] = flux.z;
        }
    }    
}

__device__ void computeFluxG(float Q[3][block_height+4][block_width+4],
                  float Qy[3][block_height+2][block_width+2],
                  float G[3][block_height+1][block_width+1],
		  float RHy[block_height+4][block_width+4],
                  const float g_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    for (int j=ty; j<block_height+1; j+=blockDim.y) {
        const int l = j + 1;
        for (int i=tx; i<block_width; i+=blockDim.x) {            
            const int k = i + 2; //Skip ghost cells
            // Q at interface from the right and left
            // Note that we swap hu and hv
            float3 Qp = make_float3(Q[0][l+1][k] - Qy[0][j+1][i],
                                 Q[2][l+1][k] - Qy[2][j+1][i],
                                 Q[1][l+1][k] - Qy[1][j+1][i]);
            float3 Qm = make_float3(Q[0][l  ][k] + Qy[0][j  ][i],
                                 Q[2][l  ][k] + Qy[2][j  ][i],
                                 Q[1][l  ][k] + Qy[1][j  ][i]);
                                       
            // Computed flux
            // Note that we swap back
            const float3 flux = CentralUpwindFluxBottom(Qm, Qp, RHy[l+1][k], g_);
            G[0][j][i] = flux.x;
            G[1][j][i] = flux.z;
            G[2][j][i] = flux.y;
        }
    }
}




__device__ void init_H_with_garbage(float Hi[block_height+4][block_width+4],
			 float RHx[block_height+4][block_width+4],
			 float RHy[block_height+4][block_width+4] ) {

    if (threadIdx.x == 0 && threadIdx.y == 0) {
	for (int j = 0; j < block_height+4; j++) {
	    for (int i = 0; i < block_width+4; i++) {
		Hi[j][i]  = 99.0f;
		RHx[j][i] = 99.0f;
		RHy[j][i] = 99.0f;
	    }
	}
    }
}



/**
  * This unsplit kernel computes the 2D numerical scheme with a TVD RK2 time integration scheme
  */
extern "C" {
__global__ void swe_2D(
        int nx_, int ny_,
        float dx_, float dy_, float dt_,
        float g_,
        
        float theta_,
        
        float f_, //< Coriolis coefficient
        float beta_, //< Coriolis force f_ + beta_*(y-y0)
        float y_zero_reference_cell_, // the cell row representing y0 (y0 at southern face)
	
        float r_, //< Bottom friction coefficient
        
        int step_,
        
        //Input h^n
        float* eta0_ptr_, int eta0_pitch_,
        float* hu0_ptr_, int hu0_pitch_,
        float* hv0_ptr_, int hv0_pitch_,
        
        //Output h^{n+1}
        float* eta1_ptr_, int eta1_pitch_,
        float* hu1_ptr_, int hu1_pitch_,
        float* hv1_ptr_, int hv1_pitch_,

        // Depth at cell intersections (i) and mid-points (m)
        float* Hi_ptr_, int Hi_pitch_,
        float* Hm_ptr_, int Hm_pitch_,

        // Boundary conditions (1: wall, 2: periodic, 3: numerical sponge)
        int bc_north_, int bc_east_, int bc_south_, int bc_west_,
	
        float wind_stress_t_) {
        
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    //Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x + 2; //Skip global ghost cells, i.e., +2
    const int tj = blockIdx.y * blockDim.y + threadIdx.y + 2;
    
    //Shared memory variables
    __shared__ float Q[3][block_height+4][block_width+4];
    
    //The following slightly wastes memory, but enables us to reuse the 
    //funcitons in common.opencl
    __shared__ float Qx[3][block_height+2][block_width+2];
    __shared__ float Qy[3][block_height+2][block_width+2];
    __shared__ float F[3][block_height+1][block_width+1];
    __shared__ float G[3][block_height+1][block_width+1];

    // Shared memory for bathymetry, and
    // reconstructed bathymetry in both directions
    // We use too much shared memory for RBx and RBy in order
    // to easier follow the implementation from the C++ version
    __shared__ float  Hi[block_height+4][block_width+4];
    __shared__ float RHx[block_height+4][block_width+4];
    __shared__ float RHy[block_height+4][block_width+4];

    //init_H_with_garbage(Hi, RHx, RHy);
    //__syncthreads();
    
    // Read H in mid-cell:
    float* const Hm_row  = (float*) ((char*) Hm_ptr_ + Hm_pitch_*tj);
    const float Hm = Hm_row[ti];
       
    //Read Q = [eta, hu, hv] into shared memory
    readBlock2(eta0_ptr_, eta0_pitch_,
               hu0_ptr_, hu0_pitch_,
               hv0_ptr_, hv0_pitch_,
               Q, nx_, ny_);
   
    // Read H into sheared memory
    readBlock2single(Hi_ptr_, Hi_pitch_,
		     Hi, nx_, ny_);
    __syncthreads();

    // Reconstruct H at the cell faces
    reconstructH(RHx, RHy, Hi);
    __syncthreads();

    //Fix boundary conditions
    // TODO: Add if on boundary condition
    if (bc_north_ == 1 || bc_east_ == 1 || bc_south_ == 1 || bc_west_ == 1)
    {
	noFlowBoundary2Mix(Q, nx_, ny_, bc_north_, bc_east_, bc_south_, bc_west_);
        __syncthreads();	
	// Looks scary to have fence within if, but the bc parameters are static between threads.
    }
    
    //Reconstruct slopes along x and axis
    // The Qx is here dQ/dx*0.5*dx
    minmodSlopeX(Q, Qx, theta_);
    minmodSlopeY(Q, Qy, theta_);
    __syncthreads();

    // Adjust the slopes to avoid negative values at integration points
    adjustSlopes(Qx, Qy, RHx, RHy, Q);
    __syncthreads();
    
    //Compute fluxes along the x and y axis
    computeFluxF(Q, Qx, F, RHx, g_);
    computeFluxG(Q, Qy, G, RHy, g_);
    __syncthreads();
    
    
    //Sum fluxes and advance in time for all internal cells
    //Check global indices against global domain
    if (ti > 1 && ti < nx_+2 && tj > 1 && tj < ny_+2) {
        const int i = tx + 2; //Skip local ghost cells, i.e., +2
        const int j = ty + 2;

	// Find bottom topography source terms: S2, S3
	const float ST2 = bottomSourceTerm2(Q, Qx, RHx, g_, i, j);
	const float ST3 = bottomSourceTerm3(Q, Qy, RHy, g_, i, j);
	
    const float X = windStressX(wind_stress_t_, ti+0.5, tj+0.5, nx_, ny_);
    const float Y = windStressY(wind_stress_t_, ti+0.5, tj+0.5, nx_, ny_);

	
	// Coriolis parameter
	float global_thread_y = tj-2; // Global id including ghost cells
	float coriolis_f = linear_coriolis_term(f_, beta_, global_thread_y,
						dy_, y_zero_reference_cell_);
	
	const float R1 =
	    - (F[0][ty  ][tx+1] - F[0][ty][tx]) / dx_
	    - (G[0][ty+1][tx  ] - G[0][ty][tx]) / dy_;
	const float R2 =
	    - (F[1][ty  ][tx+1] - F[1][ty][tx]) / dx_ 
	    - (G[1][ty+1][tx  ] - G[1][ty][tx]) / dy_
	    + (X + coriolis_f*Q[2][j][i] - ST2/dx_);
	const float R3 =
	    - (F[2][ty  ][tx+1] - F[2][ty][tx]) / dx_
	    - (G[2][ty+1][tx  ] - G[2][ty][tx]) / dy_
	    + (Y - coriolis_f*Q[1][j][i] - ST3/dy_);
						       
	
	float* const eta_row  = (float*) ((char*) eta1_ptr_ + eta1_pitch_*tj);
        float* const hu_row = (float*) ((char*) hu1_ptr_ + hu1_pitch_*tj);
        float* const hv_row = (float*) ((char*) hv1_ptr_ + hv1_pitch_*tj);

	const float C = 2.0f*r_*dt_/(Q[0][j][i]+Hm);
                    
        if  (step_ == 0) {
            //First step of RK2 ODE integrator
            
            eta_row[ti]  =  Q[0][j][i] + dt_*R1;
            hu_row[ti] = (Q[1][j][i] + dt_*R2) / (1.0f + C);
            hv_row[ti] = (Q[2][j][i] + dt_*R3) / (1.0f + C);
        }
        else if (step_ == 1) {
            //Second step of RK2 ODE integrator
            
            //First read Q^n
            const float eta_a  = eta_row[ti];
            const float hu_a = hu_row[ti];
            const float hv_a = hv_row[ti];
            
            //Compute Q^n+1
            const float eta_b  = 0.5f*(eta_a  + (Q[0][j][i] + dt_*R1));
            const float hu_b = 0.5f*(hu_a + (Q[1][j][i] + dt_*R2));
            const float hv_b = 0.5f*(hv_a + (Q[2][j][i] + dt_*R3));
            
            //Write to main memory
            //h_row[ti] = 59.975 + coriolis_f;
	    eta_row[ti] = eta_b;
            hu_row[ti] = hu_b / (1.0f + 0.5f*C);
            hv_row[ti] = hv_b / (1.0f + 0.5f*C);
        }
    }
}
} // extern "C"