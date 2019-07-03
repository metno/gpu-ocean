/*
This software is part of GPU Ocean. 

Copyright (C) 2018, 2019 SINTEF Digital
Copyright (C) 2018, 2019 Norwegian Meteorological Institute

This CUDA kernel implements the CDKLM numerical scheme
for the shallow water equations, described in
A. Chertock, M. Dudzinski, A. Kurganov & M. Lukacova-Medvidova
Well-Balanced Schemes for the Shallow Water Equations with Coriolis Forces,
Numerische Mathematik 2016

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




__device__ float3 CDKLM16_F_func(const float3 Q, const float g) {
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
__device__ float3 CDKLM16_flux(const float3 Qm, float3 Qp, const float g) {
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



__device__
float3 computeFFaceFlux(const int i, const int j, const int bx, const int nx_,
                float R[3][block_height+4][block_width+4],
                float Qx[3][block_height+2][block_width+2],
                float Hi[block_height+3][block_width+3],
                const float g_, const float coriolis_f, const float dx_,
                const int wall_bc_) {
    const int l = j + 2; //Skip ghost cells (be consistent with reconstruction offsets)
    const int k = i + 1;

    // Skip ghost cells in the Hi buffer
    const int H_i = i+1;
    const int H_j = j+1;
    
    // (u, v) reconstructed at a cell interface from the right (p) and left (m)
    // Variables to reconstruct h from u, v, K, L
    const float eta_bar_p = R[0][l][k+1];
    const float eta_bar_m = R[0][l][k  ];
    const float up = R[1][l][k+1];
    const float um = R[1][l][k  ];
    float vp = R[2][l][k+1];
    float vm = R[2][l][k  ];

    const float2 Rp = make_float2(up - 0.5f*Qx[0][j][i+1], vp - 0.5f*Qx[1][j][i+1]);
    const float2 Rm = make_float2(um + 0.5f*Qx[0][j][i  ], vm + 0.5f*Qx[1][j][i  ]);

    // H is RHx on the given face!
    const float H_face = 0.5f*( Hi[H_j][H_i] + Hi[H_j+1][H_i] );

    // Qx[2] is really dx*Kx
    const float Kx_p = Qx[2][j][i+1];
    const float Kx_m = Qx[2][j][i  ];
    
    // Fix west boundary for reconstruction of eta (corresponding to Kx)
    if ((wall_bc_ & 0x08) && (bx + i + 2 == 2    )) { vm = -vm; }
    // Fix east boundary for reconstruction of eta (corresponding to Kx)
    if ((wall_bc_ & 0x02) && (bx + i + 2 == nx_+2)) { vp = -vp; }
    
    // Reconstruct h
    const float hp = eta_bar_p + H_face - (Kx_p + dx_*coriolis_f*vp)/(2.0f*g_);
    const float hm = eta_bar_m + H_face + (Kx_m + dx_*coriolis_f*vm)/(2.0f*g_);

    // Our flux variables Q=(h, u, v)
    const float3 Qp = make_float3(hp, Rp.x, Rp.y);
    const float3 Qm = make_float3(hm, Rm.x, Rm.y);

    // Computed flux
    return CDKLM16_flux(Qm, Qp, g_);
}




__device__
float3 computeGFaceFlux(const int i, const int j, const int by, const int ny_,
                float R[3][block_height+4][block_width+4],
                float Qy[3][block_height+2][block_width+2],
                float Hi[block_height+3][block_width+3],
                const float g_, const float coriolis_fm, const float coriolis_fp, const float dy_,
                const int wall_bc_) {
    const int l = j + 1;
    const int k = i + 2; //Skip ghost cells
    
    // Skip ghost cells in the Hi buffer
    const int H_i = i+1;
    const int H_j = j+1;
    
    // Q at interface from the right and left
    // Variables to reconstruct h from u, v, K, L
    const float eta_bar_p = R[0][l+1][k];
    const float eta_bar_m = R[0][l  ][k];
    float up = R[1][l+1][k];
    float um = R[1][l  ][k];
    const float vp = R[2][l+1][k];
    const float vm = R[2][l  ][k];

    const float2 Rp = make_float2(up - 0.5f*Qy[0][j+1][i], vp - 0.5f*Qy[1][j+1][i]);
    const float2 Rm = make_float2(um + 0.5f*Qy[0][j  ][i], vm + 0.5f*Qy[1][j  ][i]);

    // H is RHx on the given face!
    const float H_face = 0.5f*( Hi[H_j][H_i] + Hi[H_j][H_i+1] );

    // Qy[2] is really dy*Ly
    const float Ly_p = Qy[2][j+1][i];
    const float Ly_m = Qy[2][j  ][i];

    // Fix south boundary for reconstruction of eta (corresponding to Ly)
    if ((wall_bc_ & 0x04) && (by + j + 2 == 2    )) { um = -um; }
    // Fix north boundary for reconstruction of eta (corresponding to Ly)
    if ((wall_bc_ & 0x01) && (by + j + 2 == ny_+2)) { up = -up; }
    
    // Reconstruct h
    const float hp = eta_bar_p + H_face - ( Ly_p - dy_*coriolis_fp*up)/(2.0f*g_);
    const float hm = eta_bar_m + H_face + ( Ly_m - dy_*coriolis_fm*um)/(2.0f*g_);

    // Our flux variables Q=(h, v, u)
    // Note that we swap u and v
    const float3 Qp = make_float3(hp, Rp.y, Rp.x);
    const float3 Qm = make_float3(hm, Rm.y, Rm.x);

    // Computed flux
    // Note that we swap back u and v
    const float3 flux = CDKLM16_flux(Qm, Qp, g_);
    return make_float3(flux.x, flux.z, flux.y);
}



extern "C" {
__global__ void cdklm_swe_2D(
        const int nx_, const int ny_,
        const float dx_, const float dy_, const float dt_,
        const float g_,

        const float theta_,

        const float f_, //< Coriolis coefficient
        const float beta_, //< Coriolis force f_ + beta_*(y-y0)
        const float y_zero_reference_cell_,  // the cell row representing y0 (y0=0 represent southernmost ghost cell)

        const float r_, //< Bottom friction coefficient

        const int rk_order, // runge kutta order
        const int step_,    // runge kutta step

        //Input h^n
        float* eta0_ptr_, const int eta0_pitch_,
        float* hu0_ptr_, const int hu0_pitch_,
        float* hv0_ptr_, const int hv0_pitch_,

        //Output h^{n+1}
        float* eta1_ptr_, const int eta1_pitch_,
        float* hu1_ptr_, const int hu1_pitch_,
        float* hv1_ptr_, const int hv1_pitch_,

        //Bathymery
        float* Hi_ptr_, const int Hi_pitch_,
        float* Hm_ptr_, const int Hm_pitch_,

        //Wind stress parameters
        const float wind_stress_t_,

        // Boundary conditions (1: wall, 2: periodic, 3: open boundary (flow relaxation scheme))
        // Note: these are packed north, east, south, west boolean bits into an int
        const int wall_bc_) {


    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    //Index of block within domain
    const int bx = blockDim.x * blockIdx.x;
    const int by = blockDim.y * blockIdx.y;

    //Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x + 2; //Skip global ghost cells, i.e., +2
    const int tj = blockIdx.y * blockDim.y + threadIdx.y + 2;

    // Our physical variables
    // Input is [eta, hu, hv]
    // Will store [eta, u, v] (Note u and v are actually computed somewhat down in the code)
    __shared__ float R[3][block_height+4][block_width+4];

    // Our reconstruction variables
    //When computing flux along x-axis, we use
    //Qx = [u_x, v_x, K_x]
    //Then we reuse it as
    //Qx = [u_y, v_y, L_y]
    //to compute the y fluxes
    __shared__ float Qx[3][block_height+2][block_width+2];

    // Bathymetry
    // Need to find H on all faces for the cells in the block (block_height+1, block_width+1)
    // and for one face further out to adjust for the Kx and Ly slope outside of the block
    __shared__ float  Hi[block_height+3][block_width+3];



    // theta_ = 1.5f;

    //Read into shared memory
    for (int j=ty; j<block_height+4; j+=blockDim.y) {
        const int l = clamp(by + j, 0, ny_+3); // Out of bounds

        //Compute the pointer to current row in the arrays
        float* const eta_row = (float*) ((char*) eta0_ptr_ + eta0_pitch_*l);
        float* const hu_row = (float*) ((char*) hu0_ptr_ + hu0_pitch_*l);
        float* const hv_row = (float*) ((char*) hv0_ptr_ + hv0_pitch_*l);

        for (int i=tx; i<block_width+4; i+=blockDim.x) {
            const int k = clamp(bx + i, 0, nx_+3); // Out of bounds

            R[0][j][i] = eta_row[k];
            R[1][j][i] = hu_row[k];
            R[2][j][i] = hv_row[k];
        }
    }
    __syncthreads();
    //Skip local ghost cells, i.e., +2
    const float hu = R[1][ty + 2][tx + 2];
    const float hv = R[2][ty + 2][tx + 2];


    // Read Hi into shared memory
    // Read intersections on all non-ghost cells
    for(int j=ty; j < block_height+3; j+=blockDim.y) {
        // Skip ghost cells and
        const int l = clamp(by+j+1, 1, ny_+4);
        float* const Hi_row = (float*) ((char*) Hi_ptr_ + Hi_pitch_*l);
        for(int i=tx; i < block_width+3; i+=blockDim.x) {
            const int k = clamp(bx+i+1, 1, nx_+4);

            Hi[j][i] = Hi_row[k];
        }
    }
    __syncthreads();
    
    // Compensate for one layer of ghost cells
    const float Hm = 0.25f*(Hi[ty+1][tx+1]+Hi[ty+2][tx+1]+Hi[ty+1][tx+2]+Hi[ty+2][tx+2]);
    
    
    // Compute Coriolis terms needed for fluxes etc.
    // Global id should be including the 
    const float coriolis_f_lower   = f_ + beta_ * ((by+ty+2)-y_zero_reference_cell_ - 1.0f + 0.5f)*dy_;
    const float coriolis_f_central = f_ + beta_ * ((by+ty+2)-y_zero_reference_cell_ +        0.5f)*dy_;
    const float coriolis_f_upper   = f_ + beta_ * ((by+ty+2)-y_zero_reference_cell_ + 1.0f + 0.5f)*dy_;



    //Fix boundary conditions
    if (wall_bc_ != 0) {
        // These boundary conditions are dealt with inside shared memory

        const int i = tx + 2; //Skip local ghost cells, i.e., +2
        const int j = ty + 2;

        // Wall boundary on north
        if (tj == ny_+1 && (wall_bc_ & 0x01)) {
            R[0][j+1][i] =  R[0][j][i];
            R[1][j+1][i] =  R[1][j][i];
            R[2][j+1][i] = -R[2][j][i];

            R[0][j+2][i] =  R[0][j-1][i];
            R[1][j+2][i] =  R[1][j-1][i];
            R[2][j+2][i] = -R[2][j-1][i];
        }
        
        // Wall boundary on east
        if (ti == nx_+1 && (wall_bc_ & 0x02)) {
            R[0][j][i+1] =  R[0][j][i];
            R[1][j][i+1] = -R[1][j][i];
            R[2][j][i+1] =  R[2][j][i];

            R[0][j][i+2] =  R[0][j][i-1];
            R[1][j][i+2] = -R[1][j][i-1];
            R[2][j][i+2] =  R[2][j][i-1];
        }
        
        // Wall boundary on south
        if (tj == 2 && (wall_bc_ & 0x04)) {
            R[0][j-1][i] =  R[0][j][i];
            R[1][j-1][i] =  R[1][j][i];
            R[2][j-1][i] = -R[2][j][i];

            R[0][j-2][i] =  R[0][j+1][i];
            R[1][j-2][i] =  R[1][j+1][i];
            R[2][j-2][i] = -R[2][j+1][i];
        }
        
        // Wall boundary on west
        if (ti == 2 && (wall_bc_ & 0x08)) {
            R[0][j][i-1] =  R[0][j][i];
            R[1][j][i-1] = -R[1][j][i];
            R[2][j][i-1] =  R[2][j][i];

            R[0][j][i-2] =  R[0][j][i+1];
            R[1][j][i-2] = -R[1][j][i+1];
            R[2][j][i-2] =  R[2][j][i+1];
        }
    }

    __syncthreads();



    //Create our "steady state" reconstruction variables (u, v)
    // K and L are never stored, but computed where needed.
    for (int j=ty; j<block_height+4; j+=blockDim.y) {
        const int l = clamp(by+j, 0, ny_+3);
        float* const Hm_row = (float*) ((char*) Hm_ptr_ + Hm_pitch_*l);
        for (int i=tx; i<block_width+4; i+=blockDim.x) {
            const int k = clamp(bx+i, 0, nx_+3);

            //const float h = R[0][j][i] + Hm[j][i]; // h = eta + H
            const float h = R[0][j][i] + Hm_row[k];
            
            // Check if the cell is almost dry
            if (h < KPSIMULATOR_DEPTH_CUTOFF) {
                // Desingularizing u and v
                float h4 = h*h; h4 *= h4;
                R[1][j][i] = SQRT_OF_TWO*h*R[1][j][i]/sqrt(h4 + fmaxf(h4, KPSIMULATOR_FLUX_SLOPE_EPS_4));
                R[2][j][i] = SQRT_OF_TWO*h*R[2][j][i]/sqrt(h4 + fmaxf(h4, KPSIMULATOR_FLUX_SLOPE_EPS_4));
            } 
            else {
                R[1][j][i] /= h;
                R[2][j][i] /= h;
            }
            
        }
    }
    __syncthreads();









    //Reconstruct slopes along x axis
    // Write result into shmem Qx = [u_x, v_x, K_x]
    // Qx is used as if its size was Qx[3][block_height][block_width + 2]
    for (int j=ty; j<block_height; j+=blockDim.y) {
        const int l = j + 2; //Skip ghost cells
        for (int i=tx; i<block_width+2; i+=blockDim.x) {
            const int k = i + 1;

            const float left_eta   = R[0][l][k-1];
            const float center_eta = R[0][l][k  ];
            const float right_eta  = R[0][l][k+1];

            {
                const float left_u   = R[1][l][k-1];
                const float center_u = R[1][l][k  ];
                const float right_u  = R[1][l][k+1];
                Qx[0][j][i] = minmodSlope(left_u, center_u, right_u, theta_);
            }

            float left_v   = R[2][l][k-1];
            float center_v = R[2][l][k  ];
            float right_v  = R[2][l][k+1];
            Qx[1][j][i] = minmodSlope(left_v, center_v, right_v, theta_);
            
            // Enforce wall boundary conditions for Kx:
            int global_thread_id_x = bx + i + 1; // index including ghost cells'
            // Western BC
            if (wall_bc_ & 0x08) {
                if (global_thread_id_x < 3    ) { left_v   = -left_v;   }
                if (global_thread_id_x < 2    ) { center_v = -center_v; }
            }
            // Eastern BC
            if (wall_bc_ & 0x02) {
                if (global_thread_id_x > nx_  ) { right_v  = -right_v;  }
                if (global_thread_id_x > nx_+1) { center_v = -center_v; }
            }

            // by + j + 2 = global thread id + ghost cells
            const float coriolis_f = f_ + beta_ * ((by + j + 2)-y_zero_reference_cell_ + 0.5f)*dy_;
            const float V_constant = dx_*coriolis_f/(2.0f*g_);

            // Qx[2] = Kx, which we need to find differently than ux and vx
            const float backward = theta_*g_*(center_eta - left_eta   - V_constant*(center_v + left_v ) );
            const float central  =   0.5f*g_*(right_eta  - left_eta   - V_constant*(right_v + 2*center_v + left_v) );
            const float forward  = theta_*g_*(right_eta  - center_eta - V_constant*(center_v + right_v) );

            // Qx[2] is really dx*Kx
            Qx[2][j][i] = minmodRaw(backward, central, forward);

        }
    }
    __syncthreads();
        
    // Adjust K_x slopes to avoid negative h = eta + H
    // Need K_x (Qx[2]), coriolis parameter (f, beta), eta (R[0]), v (R[2]), H (Hi), g, dx
    //adjustSlopes_x(Qx, Hi, Q);
    __syncthreads();
    
    // Compute flux along x axis
    float3 flux_diff = (  computeFFaceFlux(tx+1, ty, bx, nx_, R, Qx, Hi, g_, coriolis_f_central, dx_, wall_bc_) 
                        - computeFFaceFlux(tx  , ty, bx, nx_, R, Qx, Hi, g_, coriolis_f_central, dx_, wall_bc_)) / dx_;
    __syncthreads();

    //Reconstruct slopes along y axis
    // Write result into shmem Qx = [u_y, v_y, L_y]
    // Qx is now used as if its size was Qx[3][block_height+2][block_width]

    for (int j=ty; j<block_height+2; j+=blockDim.y) {
        const int l = j + 1;
        for (int i=tx; i<block_width; i+=blockDim.x) {
            const int k = i + 2; //Skip ghost cells
            // Qy[2] = Ly, which we need to find differently than uy and vy
            const float lower_eta  = R[0][l-1][k];
            const float center_eta = R[0][l  ][k];
            const float upper_eta  = R[0][l+1][k];

            float lower_u  = R[1][l-1][k];
            float center_u = R[1][l  ][k];
            float upper_u  = R[1][l+1][k];
            Qx[0][j][i] = minmodSlope(lower_u, center_u, upper_u, theta_);

            {
                const float lower_v  = R[2][l-1][k];
                const float center_v = R[2][l  ][k];
                const float upper_v  = R[2][l+1][k];
                Qx[1][j][i] = minmodSlope(lower_v, center_v, upper_v, theta_);
            }

            // Enforce wall boundary conditions for Ly
            int global_thread_id_y = by + j + 1; // index including ghost cells
            // southern BC
            if (wall_bc_ & 0x04) {
                if (global_thread_id_y < 3    ) { lower_u  = -lower_u;  }
                if (global_thread_id_y < 2    ) { center_u = -center_u; }
            }
            // northern BC
            if (wall_bc_ & 0x01) {
                if (global_thread_id_y > ny_  ) { upper_u  = -upper_u;  }
                if (global_thread_id_y > ny_+1) { center_u = -center_u; }
            }
            
            const float thread_y_diff = by + j - 1 + 2 - y_zero_reference_cell_; // (by + j - 1) + 2 = global cell id + ghost cell
            const float center_coriolis_f = f_ + beta_ * (thread_y_diff        + 0.5f)*dy_;
            const float lower_coriolis_f  = f_ + beta_ * (thread_y_diff - 1.0f + 0.5f)*dy_;
            const float upper_coriolis_f  = f_ + beta_ * (thread_y_diff + 1.0f + 0.5f)*dy_;

            const float lower_fu  = lower_u*lower_coriolis_f;
            const float center_fu = center_u*center_coriolis_f;
            const float upper_fu  = upper_u*upper_coriolis_f;

            const float U_constant = dy_/(2.0f*g_);

            const float backward = theta_*g_*(center_eta - lower_eta  + U_constant*(center_fu + lower_fu ) );
            const float central  =   0.5f*g_*(upper_eta  - lower_eta  + U_constant*(upper_fu + 2*center_fu + lower_fu) );
            const float forward  = theta_*g_*(upper_eta  - center_eta + U_constant*(center_fu + upper_fu) );

            // Qy[2] is really dy*Ly
            Qx[2][j][i] = minmodRaw(backward, central, forward);
        }
    }
    __syncthreads();


    //Compute fluxes along the y axis    
    flux_diff = flux_diff + (  computeGFaceFlux(tx, ty+1, by, ny_, R, Qx, Hi, g_, coriolis_f_central,   coriolis_f_upper, dy_, wall_bc_)
                             - computeGFaceFlux(tx, ty  , by, ny_, R, Qx, Hi, g_,   coriolis_f_lower, coriolis_f_central, dy_, wall_bc_)) / dy_;
    __syncthreads();


    //Sum fluxes and advance in time for all internal cells
    if (ti > 1 && ti < nx_+2 && tj > 1 && tj < ny_+2) {
        //Skip local ghost cells, i.e., +2
        const int i = tx + 2; 
        const int j = ty + 2;
        
        // Skip local ghost cells for Hi
        const int H_i = tx + 1;
        const int H_j = ty + 1;

        const float X = windStressX(wind_stress_t_, ti+0.5, tj+0.5, nx_, ny_);
        const float Y = windStressY(wind_stress_t_, ti+0.5, tj+0.5, nx_, ny_);

        // Bottom topography source terms!
        // -g*(eta + H)*(-1)*dH/dx   * dx
        const float RHxp = 0.5f*( Hi[H_j][H_i+1] + Hi[H_j+1][H_i+1] );
        const float RHxm = 0.5f*( Hi[H_j][H_i  ] + Hi[H_j+1][H_i  ] );
        const float st1 = g_*(R[0][j][i] + Hm)*(RHxp - RHxm);

        const float RHyp = 0.5f*( Hi[H_j+1][H_i] + Hi[H_j+1][H_i+1] );
        const float RHym = 0.5f*( Hi[H_j  ][H_i] + Hi[H_j  ][H_i+1] );
        const float st2 = g_*(R[0][j][i] + Hm)*(RHyp - RHym);

        const float L1  = - flux_diff.x;
        const float L2  = - flux_diff.y + (X + coriolis_f_central*hv + st1/dx_);
        const float L3  = - flux_diff.z + (Y - coriolis_f_central*hu + st2/dy_);

        float* const eta_row = (float*) ((char*) eta1_ptr_ + eta1_pitch_*tj);
        float* const hu_row  = (float*) ((char*) hu1_ptr_  +  hu1_pitch_*tj);
        float* const hv_row  = (float*) ((char*) hv1_ptr_  +  hv1_pitch_*tj);

        if (rk_order < 3) {

            const float C = 2.0f*r_*dt_/(R[0][j][i] + Hm);

            if  (step_ == 0) {
                //First step of RK2 ODE integrator

                eta_row[ti] =  R[0][j][i] + dt_*L1;
                hu_row[ti]  = (hu + dt_*L2) / (1.0f + C);
                hv_row[ti]  = (hv + dt_*L3) / (1.0f + C);
            }
            else if (step_ == 1) {
                //Second step of RK2 ODE integrator

                //First read Q^n
                const float eta_a = eta_row[ti];
                const float hu_a  =  hu_row[ti];
                const float hv_a  =  hv_row[ti];

                //Compute Q^n+1
                const float eta_b = 0.5f*(eta_a + (R[0][j][i] + dt_*L1));
                const float hu_b  = 0.5f*( hu_a + (hu + dt_*L2));
                const float hv_b  = 0.5f*( hv_a + (hv + dt_*L3));


                //Write to main memory
                eta_row[ti] = eta_b;
                hu_row[ti]  =  hu_b / (1.0f + 0.5f*C);
                hv_row[ti]  =  hv_b / (1.0f + 0.5f*C);

            }
        }


        else if (rk_order == 3) {
            // Third order Runge Kutta - only valid if r_ = 0.0 (no friction)

            if (step_ == 0) {
                //First step of RK3 ODE integrator
                // q^(1) = q^n + dt*L(q^n)

                eta_row[ti] =  R[0][j][i] + dt_*L1;
                hu_row[ti]  = (hu + dt_*L2);
                hv_row[ti]  = (hv + dt_*L3);

            } else if (step_ == 1) {
                // Second step of RK3 ODE integrator
                // Q^(2) = 3/4 Q^n + 1/4 ( Q^(1) + dt*L(Q^(1)) )
                // Q^n is here in h1, but will be used in next iteration as well --> write to h0

                // First read Q^n:
                const float eta_a = eta_row[ti];
                const float hu_a  =  hu_row[ti];
                const float hv_a  =  hv_row[ti];

                // Compute Q^(2):
                const float eta_b = 0.75f*eta_a + 0.25f*(R[0][j][i] + dt_*L1);
                const float hu_b  = 0.75f* hu_a + 0.25f*(hu + dt_*L2);
                const float hv_b  = 0.75f* hv_a + 0.25f*(hv + dt_*L3);

                // Write output to the input buffer:
                float* const eta_out_row = (float*) ((char*) eta0_ptr_ + eta0_pitch_*tj);
                float* const hu_out_row  = (float*) ((char*)  hu0_ptr_ +  hu0_pitch_*tj);
                float* const hv_out_row  = (float*) ((char*)  hv0_ptr_ +  hv0_pitch_*tj);
                eta_out_row[ti] = eta_b;
                hu_out_row[ti]  =  hu_b;
                hv_out_row[ti]  =  hv_b;

            } else if (step_ == 2) {
                // Third step of RK3 ODE integrator
                // Q^n+1 = 1/3 Q^n + 2/3 (Q^(2) + dt*L(Q^(2))

                // First read Q^n:
                const float eta_a = eta_row[ti];
                const float hu_a  =  hu_row[ti];
                const float hv_a  =  hv_row[ti];

                // Compute Q^n+1:
                const float eta_b = (eta_a + 2.0f*(R[0][j][i] + dt_*L1)) / 3.0f;
                const float hu_b  = ( hu_a + 2.0f*(hu + dt_*L2)) / 3.0f;
                const float hv_b  = ( hv_a + 2.0f*(hv + dt_*L3)) / 3.0f;

                //Write to main memory
                eta_row[ti] = eta_b;
                hu_row[ti]  =  hu_b;
                hv_row[ti]  =  hv_b;
            }
        }
    }



}

}

