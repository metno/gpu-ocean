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
#include "angle_texture.cu"


// KPSIMULATOR

//WARNING: Must match max_dt.cu and initBm_kernel.cu
//WARNING: This is error prone - as comparison with floating point numbers is not accurate
#define CDKLM_DRY_FLAG 1.0e-30f
#define CDKLM_DRY_EPS 1.0e-3f



/**
  * Decompose the north vector to x and y coordinates
  */
__device__
inline float2 getNorth(const int i, const int j,
                       const int nx, const int ny) {
    // Get the angle towards north from the y-axis
    const float s = i / (float) nx;
    const float t = j / (float) ny;
    const float angle = tex2D(angle_tex, s, t);
    
    // Decompose (code inspired from the cdklm_swe_2D kernel)
    return make_float2(sinf(angle), cosf(angle));
}

/**
  * Decompose the east vector to x and y coordinates
  */
__device__
inline float2 getEast(const int i, const int j,
                      const int nx, const int ny) {

    const float2 north = getNorth(i, j, nx, ny);
    return make_float2(north.y, -north.x);
}


__device__ float3 CDKLM16_F_func(const float3 Q, const float g) {
    float3 F;

    F.x = Q.x*Q.y;                        //h*u
    F.y = Q.x*Q.y*Q.y + 0.5f*g*Q.x*Q.x;   //h*u*u + 0.5f*g*h*h;
    F.z = Q.x*Q.y*Q.z;                    //h*u*v;

    return F;
}







/**
  * Note that the input vectors are (h, u, v), thus not the regular
  * (h, hu, hv). 
  * Note also that u and v are desingularized from the start.
  */
__device__ float3 CDKLM16_flux(float3 Qm, float3 Qp, const float g) {
    
    // Contribution from plus cell
    float3 Fp = make_float3(0.0f, 0.0f, 0.0f);
    float up = 0.0f;
    float cp = 0.0f;
    
    if (Qp.x > KPSIMULATOR_DEPTH_CUTOFF) {
        Fp = CDKLM16_F_func(Qp, g);
        up = Qp.y;         // u
        cp = sqrtf(g*Qp.x); // sqrt(g*h)
    }

    // Contribution from plus cell
    float3 Fm = make_float3(0.0f, 0.0f, 0.0f);
    float um = 0.0f;
    float cm = 0.0f;

    if (Qm.x > KPSIMULATOR_DEPTH_CUTOFF) {
        Fm = CDKLM16_F_func(Qm, g);
        um = Qm.y;         // u
        cm = sqrtf(g*Qm.x); // sqrt(g*h)
    }
    
    const float am = min(min(um-cm, up-cp), 0.0f); // largest negative wave speed
    const float ap = max(max(um+cm, up+cp), 0.0f); // largest positive wave speed

    // If symmetric Rieman fan, return zero flux
    if ( fabsf(ap - am) < KPSIMULATOR_FLUX_SLOPE_EPS ) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    float3 F;

    F.x = ((ap*Fm.x - am*Fp.x) + ap*am*(Qp.x-Qm.x))/(ap-am);
    F.y = ((ap*Fm.y - am*Fp.y) + ap*am*(Fp.x-Fm.x))/(ap-am);
    F.z = (Qm.y + Qp.y > 0) ? Fm.z : Fp.z; //Upwinding to be consistent

    return F;
}



/**
  * Adjusting the slope of K_x, found in Qx[3], to avoid negative values for h on the faces,
  * in the case of dry cells
  */
__device__
void adjustSlopes_x(const int bx, const int by, 
                    const int nx_, const int ny_, const float dx_, const float dy_,
                    float R[3][block_height+4][block_width+4],
                    float Qx[3][block_height+2][block_width+2], // used as if Qx[3][block_height][block_width + 2]
                    float Hi[block_height+3][block_width+3],
                    const float g_, 
                    const float f_, const float beta_, 
                    const int& bc_east_, const int& bc_west_) {
    
    // Need K_x (Qx[2]), coriolis parameter (f, beta), eta (R[0]), v (R[2]), H (Hi), g, dx

    
    const int j = threadIdx.y; // values in Qx
    const int l = j + 2; // values in R
    const int H_j = j + 1; // values in Hi
    
    for (int i=threadIdx.x; i<block_width+2; i+=blockDim.x) {
        // i referes to values in Qx
        const int k = i + 1; // values in R
        const int H_i = i; // values in Hi

        // Reconstruct h at east and west faces
        const float eta = R[0][l][k];
        
        float v   = R[2][l][k];
        // Fix west boundary for reconstruction of eta (corresponding to Kx)
        if ((bc_west_ == 1) && (bx + k < 2    )) { v = -v; }
        // Fix east boundary for reconstruction of eta (corresponding to Kx)
        if ((bc_east_ == 1) && (bx + k > nx_+2)) { v = -v; }
        
        // Coriolis in this cell
        const float2 north = getNorth(bx+k, by+l, nx_, ny_);
        const float coriolis_f = f_ + beta_ * ((bx + k + 0.5f)*dx_*north.x + (by + l + 0.5f)*dy_*north.y);
        
        const float dxfv = dx_*coriolis_f*v;
        
        const float H_west = 0.5f*(Hi[H_j][H_i  ] + Hi[H_j+1][H_i  ]);
        const float H_east = 0.5f*(Hi[H_j][H_i+1] + Hi[H_j+1][H_i+1]);
        
        const float h_west = eta + H_west - (Qx[2][j][i] + dxfv)/(2.0f*g_);
        const float h_east = eta + H_east + (Qx[2][j][i] + dxfv)/(2.0f*g_);
        
        // Adjust if negative water level
        Qx[2][j][i] = (h_west > 0) ? Qx[2][j][i] : -dxfv + 2.0f*g_*(eta + H_west);
        Qx[2][j][i] = (h_east > 0) ? Qx[2][j][i] : -dxfv - 2.0f*g_*(eta + H_east);
    }
}


/**
  * Adjusting the slope of L_y, found in Qx[3], to avoid negative values for h on the faces,
  * in the case of dry cells
  */
__device__
void adjustSlopes_y(const int bx, const int by, 
                    const int nx_, const int ny_, const float dx_, const float dy_,
                    float R[3][block_height+4][block_width+4],
                    float Qx[3][block_height+2][block_width+2], // used as if Qx[3][block_height+2][block_width]
                    float Hi[block_height+3][block_width+3],
                    const float g_, 
                    const float f_, const float beta_,
                    const int& bc_north_, const int& bc_south_) {
    
    // Need K_x (Qx[2]), coriolis parameter (f, beta), eta (R[0]), v (R[2]), H (Hi), g, dx

    
    const int i = threadIdx.x; // values in Qx
    const int k = i + 2; // values in R
    const int H_i = i + 1; // values in Hi
    
    for (int j=threadIdx.y; j<block_height+2; j+=blockDim.y) {
        // i referes to values in Qx
        const int l = j + 1; // values in R
        const int H_j = j; // values in Hi

        // Reconstruct h at east and west faces
        const float eta = R[0][l][k];
        
        float u   = R[1][l][k];
        // Fix south boundary for reconstruction of eta (corresponding to Ly)
        if ((bc_south_ == 1) && (by + l < 2    )) { u = -u; }
        // Fix north boundary for reconstruction of eta (corresponding to Ly)
        if ((bc_north_ == 1) && (by + l > ny_+2)) { u = -u; }
        
        // Coriolis in this cell
        const float2 north = getNorth(bx+k, by+l, nx_, ny_);
        const float coriolis_f = f_ + beta_ * ((bx + k + 0.5f)*dx_*north.x + (by + l + 0.5f)*dy_*north.y);

        const float dyfu = dy_*coriolis_f*u;
        
        const float H_south = 0.5f*(Hi[H_j  ][H_i] + Hi[H_j  ][H_i+1]);
        const float H_north = 0.5f*(Hi[H_j+1][H_i] + Hi[H_j+1][H_i+1]);
        
        const float h_south = eta + H_south - (Qx[2][j][i] - dyfu)/(2.0f*g_);
        const float h_north = eta + H_north + (Qx[2][j][i] - dyfu)/(2.0f*g_);
        
        // Adjust if negative water level
        Qx[2][j][i] = (h_south > 0) ? Qx[2][j][i] : dyfu + 2.0f*g_*(eta + H_south);
        Qx[2][j][i] = (h_north > 0) ? Qx[2][j][i] : dyfu - 2.0f*g_*(eta + H_north);
    }
}




__device__
float3 computeFFaceFlux(const int i, const int j, const int bx, const int nx_,
                float R[3][block_height+4][block_width+4],
                float Qx[3][block_height+2][block_width+2],
                float Hi[block_height+3][block_width+3],
                const float g_, const float coriolis_fm, const float coriolis_fp, const float dx_,
                const int& bc_east_, const int& bc_west_,
                const float2 north) {
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
    
    //Check if dry: if so return zero flux
    if (eta_bar_p == CDKLM_DRY_FLAG || eta_bar_m == CDKLM_DRY_FLAG) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    const float2 Rp = make_float2(up - 0.5f*Qx[0][j][i+1], vp - 0.5f*Qx[1][j][i+1]);
    const float2 Rm = make_float2(um + 0.5f*Qx[0][j][i  ], vm + 0.5f*Qx[1][j][i  ]);

    // H is RHx on the given face!
    const float H_face = 0.5f*( Hi[H_j][H_i] + Hi[H_j+1][H_i] );

    // Qx[2] is really dx*Kx
    const float Kx_p = Qx[2][j][i+1];
    const float Kx_m = Qx[2][j][i  ];
    
    // Fix west boundary for reconstruction of eta (corresponding to Kx)
    if ((bc_west_ == 1) && (bx + i + 2 == 2    )) { vm = -vm; }
    // Fix east boundary for reconstruction of eta (corresponding to Kx)
    if ((bc_east_ == 1) && (bx + i + 2 == nx_+2)) { vp = -vp; }
    
    //Reconstruct momentum along north
    const float vp_north = up*north.x + vp*north.y;
    const float vm_north = um*north.x + vm*north.y;
    
    // Reconstruct h
    const float hp = fmaxf(0.0f, eta_bar_p + H_face - (Kx_p + dx_*coriolis_fp*vp_north)/(2.0f*g_));
    const float hm = fmaxf(0.0f, eta_bar_m + H_face + (Kx_m + dx_*coriolis_fm*vm_north)/(2.0f*g_));

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
                const int& bc_north_, const int& bc_south_,
                const float2 east) {
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

    //Check if dry: if so return zero flux
    if (eta_bar_p == CDKLM_DRY_FLAG || eta_bar_m == CDKLM_DRY_FLAG) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    const float2 Rp = make_float2(up - 0.5f*Qy[0][j+1][i], vp - 0.5f*Qy[1][j+1][i]);
    const float2 Rm = make_float2(um + 0.5f*Qy[0][j  ][i], vm + 0.5f*Qy[1][j  ][i]);

    // H is RHx on the given face!
    const float H_face = 0.5f*( Hi[H_j][H_i] + Hi[H_j][H_i+1] );

    // Qy[2] is really dy*Ly
    const float Ly_p = Qy[2][j+1][i];
    const float Ly_m = Qy[2][j  ][i];

    // Fix south boundary for reconstruction of eta (corresponding to Ly)
    if ((bc_south_ == 1) && (by + j + 2 == 2    )) { um = -um; }
    // Fix north boundary for reconstruction of eta (corresponding to Ly)
    if ((bc_north_ == 1) && (by + j + 2 == ny_+2)) { up = -up; }
    
    // Reconstruct momentum along east
    const float up_east = up*east.x + vp*east.y;
    const float um_east = um*east.x + vm*east.y;
    
    // Reconstruct h
    const float hp = fmaxf(0.0f, eta_bar_p + H_face - ( Ly_p - dy_*coriolis_fp*up_east)/(2.0f*g_));
    const float hm = fmaxf(0.0f, eta_bar_m + H_face + ( Ly_m - dy_*coriolis_fm*um_east)/(2.0f*g_));

    // Our flux variables Q=(h, v, u)
    // Note that we swap u and v
    const float3 Qp = make_float3(hp, Rp.y, Rp.x);
    const float3 Qm = make_float3(hm, Rm.y, Rm.x);

    // Computed flux
    // Note that we swap back u and v
    const float3 flux = CDKLM16_flux(Qm, Qp, g_);
    return make_float3(flux.x, flux.z, flux.y);
}


__device__ 
void handleWallBC(
                const int& nx_, const int& ny_,
                const int& ti_, const int& tj_, 
                const int& tx_, const int& ty_, 
                const int& bc_north_, const int& bc_south_,
                const int& bc_east_, const int& bc_west_,
                float R[3][block_height+4][block_width+4]) {
    const int wall_bc = 1;

    const int i = tx_ + 2; //Skip local ghost cells, i.e., +2
    const int j = ty_ + 2;
        
    if (bc_north_ == wall_bc && tj_ == ny_+1) {
        R[0][j+1][i] =  R[0][j][i];
        R[1][j+1][i] =  R[1][j][i];
        R[2][j+1][i] = -R[2][j][i];

        R[0][j+2][i] =  R[0][j-1][i];
        R[1][j+2][i] =  R[1][j-1][i];
        R[2][j+2][i] = -R[2][j-1][i];
    }
    
    if (bc_south_ == wall_bc && tj_ == 2) {
        R[0][j-1][i] =  R[0][j][i];
        R[1][j-1][i] =  R[1][j][i];
        R[2][j-1][i] = -R[2][j][i];

        R[0][j-2][i] =  R[0][j+1][i];
        R[1][j-2][i] =  R[1][j+1][i];
        R[2][j-2][i] = -R[2][j+1][i];
    }
    
    if (bc_east_ == wall_bc && ti_ == nx_+1) {
        R[0][j][i+1] =  R[0][j][i];
        R[1][j][i+1] = -R[1][j][i];
        R[2][j][i+1] =  R[2][j][i];

        R[0][j][i+2] =  R[0][j][i-1];
        R[1][j][i+2] = -R[1][j][i-1];
        R[2][j][i+2] =  R[2][j][i-1];
    }
    
    if (bc_west_ == wall_bc && ti_ == 2) {
        R[0][j][i-1] =  R[0][j][i];
        R[1][j][i-1] = -R[1][j][i];
        R[2][j][i-1] =  R[2][j][i];

        R[0][j][i-2] =  R[0][j][i+1];
        R[1][j][i-2] = -R[1][j][i+1];
        R[2][j][i-2] =  R[2][j][i+1];
    }
}


/**
  * Uses a matrix stored as float 4
  * [x, y] * [u] = [x*u + y*v]
  * [z, w]   [v]   [z*u + w*v]
  * and multiply 
  */
__device__
inline float2 matMul(float4 M, float2 v) {
    return make_float2(M.x*v.x + M.y*v.y, M.z*v.x + M.w*v.y);
}


//texture<float, cudaTextureType2D> angle_tex;

extern "C" {
__global__ void cdklm_swe_2D(
        const int nx_, const int ny_,
        const float dx_, const float dy_, const float dt_,
        const float g_,

        const float theta_,

        const float f_, //< Coriolis coefficient (f_ - beta_*y0)
        const float beta_, //< Coriolis force f_ + beta_*y

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
        float land_value_,

        //Wind stress parameters
        const float wind_stress_t_,

        // Boundary conditions (1: wall, 2: periodic, 3: open boundary (flow relaxation scheme))
        // Note: these are packed north, east, south, west boolean bits into an int
        const int boundary_conditions_) {
            
    //const float land_value_ = 1.0e20;


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

    // Get the angle towards north and create the matrices for the basis transformation
    const float s = ti / (float) nx_;
    const float t = tj / (float) ny_;
    const float angle = tex2D(angle_tex, s, t);
    const float cos_a = cosf(angle);
    const float sin_a = sinf(angle);
    
    // B transforms from [x, y] to [n, e] (rotates by theta)
    // B = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    const float4 B = make_float4(cos_a, -sin_a, sin_a, cos_a);
    
    // BT transforms from [e, n] to [x, y] (rotates by -theta)
    // BT = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
    const float4 BT = make_float4(cos_a, sin_a, -sin_a, cos_a);
    
    // North and east vector in xy-coordinate system
    // Given x and y-aligned vectors, simply compute the dot product, 
    // i.e., 
    // hu_north = north.x*hu + north.y*hv
    // hu_east = east.x*hu + east.y*hv
    const float2 north = matMul(BT, make_float2(0.0, 1.0));
    const float2 east = make_float2(north.y, -north.x);
    
    //Up vector in east-north coordinate system
    // Given n and e-aligned vectors, simply compute the dot product,
    // i.e., 
    // hu = right.x*hu_north + right.y*hu_east
    // hv = up.x*hu_north + up.y*hu_east
    const float2 up = matMul(B, make_float2(0.0, 1.0));
    const float2 right = make_float2(up.y, -up.x);


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
    

    // Read Hi into shared memory
    // Read intersections on all non-ghost cells
    for(int j=ty; j < block_height+3; j+=blockDim.y) {
        // Skip ghost cells and
        const int l = clamp(by+j+1, 1, ny_+4);
        float* const Hi_row = (float*) ((char*) Hi_ptr_ + Hi_pitch_*l);
        for(int i=tx; i < block_width+3; i+=blockDim.x) {
            const int k = clamp(bx+i+1, 1, nx_+4);

            Hi[j][i] = Hi_row[k];
            
            if (fabsf(Hi[j][i] - land_value_) < CDKLM_DRY_EPS) {
                Hi[j][i] = CDKLM_DRY_FLAG;
            }
        }
    }
    __syncthreads();
    
    
    
    // Compute Coriolis terms needed for fluxes etc.
    // Global id should be including the 
    // FIXME! coriolis = f at (ti, tj) = (-0.5, -0.5), 
    //        meaning the corner of the domain *including* ghost cells.
    //        f will therefore be different in the interior based on the number of sponge cells, 
    //        which is not a good abstraction.
    //        Coriolis should be f at (ti, tj) = (?, ?) (because we don't know number of sponge cells here...)
    //        - havahol, 2019-11-13
    //beta * (i*dx, j*dy)*(north.x, north.y)
    const float coriolis_f_lower   = f_ + beta_ * ((ti+0.5f)*dx_*north.x + (tj-0.5f)*dy_*north.y);
    const float coriolis_f_central = f_ + beta_ * ((ti+0.5f)*dx_*north.x + (tj+0.5f)*dy_*north.y);
    const float coriolis_f_upper   = f_ + beta_ * ((ti+0.5f)*dx_*north.x + (tj+1.5f)*dy_*north.y);
    const float coriolis_f_left    = f_ + beta_ * ((ti-0.5f)*dx_*north.x + (tj+0.5f)*dy_*north.y);
    const float coriolis_f_right   = f_ + beta_ * ((ti+1.5f)*dx_*north.x + (tj+0.5f)*dy_*north.y);


    //Fix boundary conditions
    //This must match code in CDKLM16.py:callKernel(...)
    const int bc_north = (boundary_conditions_ >> 24) & 0xFF;
    const int bc_south = (boundary_conditions_ >> 16) & 0xFF;
    const int bc_east = (boundary_conditions_ >> 8) & 0xFF;
    const int bc_west = (boundary_conditions_ >> 0) & 0xFF;
    
    if (boundary_conditions_ > 0) {
        // These boundary conditions are dealt with inside shared memory
        handleWallBC(nx_, ny_,
                ti, tj,
                tx, ty,
                bc_north, bc_south,
                bc_east, bc_west,
                R);
    }

    __syncthreads();
    
    // Compensate for one layer of ghost cells
    float Hm = 0.25f*(Hi[ty+1][tx+1] + Hi[ty+2][tx+1] + Hi[ty+1][tx+2] + Hi[ty+2][tx+2]);

    //Create our "steady state" reconstruction variables (u, v)
    // K and L are never stored, but computed where needed.
    for (int j=ty; j<block_height+4; j+=blockDim.y) {
        const int l = clamp(by+j, 0, ny_+3);
        float* const Hm_row = (float*) ((char*) Hm_ptr_ + Hm_pitch_*l);
        for (int i=tx; i<block_width+4; i+=blockDim.x) {
            const int k = clamp(bx+i, 0, nx_+3);

            // h = eta + H
            const float local_Hm = Hm_row[k];
            //const float local_Hm = 0.25f*(Hi[j][i] + Hi[j+1][i] + Hi[j][i+1] + Hi[j+1][i+1]);
            const float h = R[0][j][i] + local_Hm;
            
            //Check if this cell is actually dry (or land)
            //NOTE: This requires that all four corners of a cell are dry to be considered dry cell
            if (fabsf(local_Hm - land_value_) <= CDKLM_DRY_EPS) {
                R[0][j][i] = CDKLM_DRY_FLAG;
                R[1][j][i] = 0.0f;
                R[2][j][i] = 0.0f;
            }
            // Check if the cell is almost dry
            else if (h < KPSIMULATOR_DESING_EPS) {
                
                if (h <= KPSIMULATOR_DEPTH_CUTOFF) {
                    R[0][j][i] = -local_Hm + KPSIMULATOR_DEPTH_CUTOFF;
                    R[1][j][i] = 0.0f;
                    R[2][j][i] = 0.0f;
                }
                else {                
                    // Desingularizing u and v
                    //R[0][j][i] = h - local_Hm;
                    R[1][j][i] = desingularize(h, R[1][j][i], KPSIMULATOR_DESING_EPS); 
                    R[2][j][i] = desingularize(h, R[2][j][i], KPSIMULATOR_DESING_EPS); 
                }
            }
            // Wet cells
            else {
                R[1][j][i] /= h;
                R[2][j][i] /= h;
            }

            
        }
    }
    __syncthreads();

    // Store desingulized hu and hv
    //Skip local ghost cells, i.e., +2
    float hu = 0.0f;
    float hv = 0.0f;
    if ((R[0][ty + 2][tx + 2] + Hm) > KPSIMULATOR_DEPTH_CUTOFF) {
        hu = R[1][ty + 2][tx + 2]*(R[0][ty + 2][tx + 2] + Hm);
        hv = R[2][ty + 2][tx + 2]*(R[0][ty + 2][tx + 2] + Hm);
    }




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

            const float left_u   = R[1][l][k-1];
            const float center_u = R[1][l][k  ];
            const float right_u  = R[1][l][k+1];
        
            float left_v   = R[2][l][k-1];
            float center_v = R[2][l][k  ];
            float right_v  = R[2][l][k+1];
            
            Qx[0][j][i] = minmodSlope(left_u, center_u, right_u, theta_);
            Qx[1][j][i] = minmodSlope(left_v, center_v, right_v, theta_);
            
            // Enforce wall boundary conditions for Kx:
            int global_thread_id_x = bx + i + 1; // index including ghost cells'
            // Western BC
            if (bc_west == 1) {
                if (global_thread_id_x < 3    ) { left_v   = -left_v;   }
                if (global_thread_id_x < 2    ) { center_v = -center_v; }
            }
            // Eastern BC
            if (bc_east == 1) {
                if (global_thread_id_x > nx_  ) { right_v  = -right_v;  }
                if (global_thread_id_x > nx_+1) { center_v = -center_v; }
            }
            
            // Get north vector for thread (bx + k, by +l)
            const float2 local_north = getNorth(bx+k, by+l, nx_, ny_);
            
            const float left_coriolis_f   = f_ + beta_ * ((bx + k - 0.5f)*dx_*local_north.x + (by + l + 0.5f)*dy_*local_north.y);
            const float center_coriolis_f = f_ + beta_ * ((bx + k + 0.5f)*dx_*local_north.x + (by + l + 0.5f)*dy_*local_north.y);
            const float right_coriolis_f  = f_ + beta_ * ((bx + k + 1.5f)*dx_*local_north.x + (by + l + 0.5f)*dy_*local_north.y);
            
            const float left_fv  = (local_north.x*left_u + local_north.y*left_v)*left_coriolis_f;
            const float center_fv = (local_north.x*center_u + local_north.y*center_v)*center_coriolis_f;
            const float right_fv  = (local_north.x*right_u + local_north.y*right_v)*right_coriolis_f;
            
            const float V_constant = dx_/(2.0f*g_);

            // Qx[2] = Kx, which we need to find differently than ux and vx
            const float backward = theta_*g_*(center_eta - left_eta   - V_constant*(center_fv + left_fv ) );
            const float central  =   0.5f*g_*(right_eta  - left_eta   - V_constant*(right_fv + 2*center_fv + left_fv) );
            const float forward  = theta_*g_*(right_eta  - center_eta - V_constant*(center_fv + right_fv) );

            // Qx[2] is really dx*Kx
            Qx[2][j][i] = minmodRaw(backward, central, forward);

        }
    }
    __syncthreads();
        
    // Adjust K_x slopes to avoid negative h = eta + H
    // Need K_x (Qx[2]), coriolis parameter (f, beta), eta (R[0]), v (R[2]), H (Hi), g, dx
    adjustSlopes_x(bx, by, nx_, ny_, dx_, dy_,
                   R, Qx, Hi,
                   g_, f_, beta_, 
                   bc_east, bc_west);
    __syncthreads();
    
    // Compute flux along x axis
    float3 flux_diff = (  
            computeFFaceFlux(
                tx+1, ty, bx, nx_, 
                R, Qx, Hi,
                g_, coriolis_f_central, coriolis_f_right, 
                dx_, 
                bc_north, bc_south, 
                north)
            - 
            computeFFaceFlux(
                tx , ty, bx, nx_, 
                R, Qx, Hi,
                g_, coriolis_f_left, coriolis_f_central, 
                dx_, 
                bc_north, bc_south, 
                north)) / dx_;
    __syncthreads();
    
    // Reconstruct eta_west, eta_east for use in bathymetry source term
    const float eta_west = R[0][ty+2][tx+2] - (Qx[2][ty][tx+1] + dx_*coriolis_f_central*R[2][ty+2][tx+2])/(2.0f*g_);
    const float eta_east = R[0][ty+2][tx+2] + (Qx[2][ty][tx+1] + dx_*coriolis_f_central*R[2][ty+2][tx+2])/(2.0f*g_);
    
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

            const float lower_v  = R[2][l-1][k];
            const float center_v = R[2][l  ][k];
            const float upper_v  = R[2][l+1][k];
            
            Qx[0][j][i] = minmodSlope(lower_u, center_u, upper_u, theta_);
            Qx[1][j][i] = minmodSlope(lower_v, center_v, upper_v, theta_);

            // Enforce wall boundary conditions for Ly
            int global_thread_id_y = by + j + 1; // index including ghost cells
            // southern BC
            if (bc_south == 1) {
                if (global_thread_id_y < 3    ) { lower_u  = -lower_u;  }
                if (global_thread_id_y < 2    ) { center_u = -center_u; }
            }
            // northern BC
            if (bc_north == 1) {
                if (global_thread_id_y > ny_  ) { upper_u  = -upper_u;  }
                if (global_thread_id_y > ny_+1) { center_u = -center_u; }
            }
            
            // Get north and east vectors for thread (bx + k, by +l)
            const float2 local_north = getNorth(bx+k, by+l, nx_, ny_);
            const float2 local_east = getEast(bx+k, by+l, nx_, ny_);
            
            const float lower_coriolis_f  = f_ + beta_ * ((bx + k + 0.5f)*dx_*local_north.x + (by + l - 0.5f)*dy_*local_north.y);
            const float center_coriolis_f = f_ + beta_ * ((bx + k + 0.5f)*dx_*local_north.x + (by + l + 0.5f)*dy_*local_north.y);
            const float upper_coriolis_f  = f_ + beta_ * ((bx + k + 0.5f)*dx_*local_north.x + (by + l + 1.5f)*dy_*local_north.y);

            const float lower_fu  = (local_east.x*lower_u  + local_east.y*lower_v )*lower_coriolis_f;
            const float center_fu = (local_east.x*center_u + local_east.y*center_v)*center_coriolis_f;
            const float upper_fu  = (local_east.x*upper_u  + local_east.y*upper_v )*upper_coriolis_f;

            const float U_constant = dy_/(2.0f*g_);

            const float backward = theta_*g_*(center_eta - lower_eta  + U_constant*(center_fu + lower_fu ) );
            const float central  =   0.5f*g_*(upper_eta  - lower_eta  + U_constant*(upper_fu + 2*center_fu + lower_fu) );
            const float forward  = theta_*g_*(upper_eta  - center_eta + U_constant*(center_fu + upper_fu) );

            // Qy[2] is really dy*Ly
            Qx[2][j][i] = minmodRaw(backward, central, forward);
        }
    }
    __syncthreads();

    // Adjust L_y slopes to avoid negative h = eta + H
    // Need L_x (Qx[2]), coriolis parameter (f, beta), eta (R[0]), u (R[1]), H (Hi), g, dx
    adjustSlopes_y(bx, by, nx_, ny_, dx_, dy_,
                   R, Qx, Hi,
                   g_, f_, beta_, 
                   bc_north, bc_south);
    __syncthreads();
    
    //Compute fluxes along the y axis
    flux_diff = flux_diff + 
        (computeGFaceFlux(
            tx, ty+1, by, ny_, 
            R, Qx, Hi, 
            g_, coriolis_f_central, coriolis_f_upper, 
            dy_, 
            bc_east, bc_west, 
            east)
        - 
        computeGFaceFlux(
            tx, ty, by, ny_, 
            R, Qx, Hi, 
            g_, coriolis_f_lower, coriolis_f_central, 
            dy_, 
            bc_east, bc_west, 
            east)) / dy_;
    __syncthreads();

    // Reconstruct eta_north, eta_south for use in bathymetry source term
    const float eta_south = R[0][ty+2][tx+2] - (Qx[2][ty+1][tx] - dy_*coriolis_f_central*R[1][ty+2][tx+2])/(2.0f*g_);
    const float eta_north = R[0][ty+2][tx+2] + (Qx[2][ty+1][tx] - dy_*coriolis_f_central*R[1][ty+2][tx+2])/(2.0f*g_);
    __syncthreads();
    
    //Sum fluxes and advance in time for all internal cells
    if (ti > 1 && ti < nx_+2 && tj > 1 && tj < ny_+2) {
        //Skip local ghost cells, i.e., +2
        const int i = tx + 2; 
        const int j = ty + 2;
        
        // Skip local ghost cells for Hi
        const int H_i = tx + 1;
        const int H_j = ty + 1;

        // Source terms (wind, coriolis, bathymetry)
        float st1 = 0.0f;
        float st2 = 0.0f;
        
        const float h = R[0][j][i] + Hm;
        //If wet cell
        if (h >= KPSIMULATOR_DEPTH_CUTOFF) {
            // If not land
            if (R[0][j][i] != CDKLM_DRY_FLAG) {
                // Wind
                const float X = windStressX(wind_stress_t_, ti+0.5, tj+0.5, nx_, ny_);
                const float Y = windStressY(wind_stress_t_, ti+0.5, tj+0.5, nx_, ny_);

                // Bottom topography source terms!
                // -g*(eta + H)*(-1)*dH/dx   * dx
                const float RHxp = 0.5f*( Hi[H_j  ][H_i+1] + Hi[H_j+1][H_i+1] );
                const float RHxm = 0.5f*( Hi[H_j  ][H_i  ] + Hi[H_j+1][H_i  ] );
                const float RHyp = 0.5f*( Hi[H_j+1][H_i  ] + Hi[H_j+1][H_i+1] );
                const float RHym = 0.5f*( Hi[H_j  ][H_i  ] + Hi[H_j  ][H_i+1] );
                
                float H_x = RHxp - RHxm;
                float H_y = RHyp - RHym;
                
                const float eta_sn = 0.5f*(eta_north + eta_south);
                const float eta_we = 0.5f*(eta_west  + eta_east);

                // TODO: We might want to use the mean of the reconstructed eta's at the faces here, instead of R[0]...
                //const float bathymetry1 = g_*(R[0][j][i] + Hm)*H_x;
                //const float bathymetry2 = g_*(R[0][j][i] + Hm)*H_y;
                const float bathymetry1 = g_*(eta_we + Hm)*H_x;
                const float bathymetry2 = g_*(eta_sn + Hm)*H_y;
                
                //Find north-going and east-going coriolis force
                const float hu_east =  coriolis_f_central*(hu*east.x + hv*east.y);
                const float hv_north = coriolis_f_central*(hu*north.x + hv*north.y);
                
                //Convert back to xy coordinate system
                const float hu_cor = right.x*hu_east + right.y*hv_north;
                const float hv_cor = up.x*hu_east + up.y*hv_north;

                // Total source terms
                st1 = X + hv_cor + bathymetry1/dx_;
                st2 = Y - hu_cor + bathymetry2/dy_;
            }
        }

        
        const float L1  = - flux_diff.x;
        const float L2  = - flux_diff.y + st1;
        const float L3  = - flux_diff.z + st2;

        float* const eta_row = (float*) ((char*) eta1_ptr_ + eta1_pitch_*tj);
        float* const hu_row  = (float*) ((char*) hu1_ptr_  +  hu1_pitch_*tj);
        float* const hv_row  = (float*) ((char*) hv1_ptr_  +  hv1_pitch_*tj);

        float updated_eta;
        float updated_hu;
        float updated_hv;
        
        if (rk_order < 3) {

#ifdef use_linear_friction
            const float C = 2.0f*r_*dt_/(R[0][j][i] + Hm);
#else
            float C = 0.0;
            if (r_ > 0.0) {
                if (h < KPSIMULATOR_DESING_EPS) {
                    const float u = desingularize(h, hu, KPSIMULATOR_DESING_EPS);
                    const float v = desingularize(h, hv, KPSIMULATOR_DESING_EPS);
                    C = dt_*r_*sqrt(u*u+v*v)/h;
                }
                else {
                    const float u = hu/h;
                    const float v = hv/h;
                    C = dt_*r_*sqrt(u*u+v*v)/h;
                }
            }
#endif
            
            if  (step_ == 0) {
                //First step of RK2 ODE integrator

                updated_eta =  R[0][j][i] + dt_*L1;
                updated_hu  = (hu + dt_*L2) / (1.0f + C);
                updated_hv  = (hv + dt_*L3) / (1.0f + C);
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
                updated_eta = eta_b;
                updated_hu  =  hu_b / (1.0f + 0.5f*C);
                updated_hv  =  hv_b / (1.0f + 0.5f*C);

            }
        }


        else if (rk_order == 3) {
            // Third order Runge Kutta - only valid if r_ = 0.0 (no friction)

            if (step_ == 0) {
                //First step of RK3 ODE integrator
                // q^(1) = q^n + dt*L(q^n)

                updated_eta =  R[0][j][i] + dt_*L1;
                updated_hu  = (hu + dt_*L2);
                updated_hv  = (hv + dt_*L3);

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
                updated_eta = eta_b;
                updated_hu  =  hu_b;
                updated_hv  =  hv_b;

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
                updated_eta = eta_b;
                updated_hu  =  hu_b;
                updated_hv  =  hv_b;
            }
        }
    

        const float updated_h = updated_eta + Hm;
        if ((updated_h <= KPSIMULATOR_DEPTH_CUTOFF) ) { 
            updated_eta = -Hm + KPSIMULATOR_DEPTH_CUTOFF;
            updated_hu  = 0.0f;
            updated_hv  = 0.0f;
        }

        if ( (rk_order == 3) && (step_ == 1) ) {
            float* const eta_out_row = (float*) ((char*) eta0_ptr_ + eta0_pitch_*tj);
            float* const hu_out_row  = (float*) ((char*)  hu0_ptr_ +  hu0_pitch_*tj);
            float* const hv_out_row  = (float*) ((char*)  hv0_ptr_ +  hv0_pitch_*tj);

            eta_out_row[ti] = fmaxf(-Hm + KPSIMULATOR_DEPTH_CUTOFF, updated_eta);
            hu_out_row[ti]  = updated_hu;
            hv_out_row[ti]  = updated_hv;
        } else {
            eta_row[ti] = fmaxf(-Hm + KPSIMULATOR_DEPTH_CUTOFF, updated_eta);
            hu_row[ti]  = updated_hu;
            hv_row[ti]  = updated_hv;
        }
    }
}

}

