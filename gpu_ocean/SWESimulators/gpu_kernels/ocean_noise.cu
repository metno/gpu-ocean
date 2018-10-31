/*
These OpenCL kernels generates random numbers for creating model error 
based on given covariance relations.

Copyright (C) 2018  SINTEF ICT

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


/**
  *  Generates two uniform random numbers based on the ANSIC Linear Congruential 
  *  Generator.
  */
__device__ float2 ansic_lcg(unsigned long long* seed_ptr) {
    unsigned long long seed = (*seed_ptr);
    double denum = 2147483648.0;
    unsigned long long modulo = 2147483647;

    seed = ((seed * 1103515245) + 12345) % modulo; //% 0x7fffffff;
    float u1 = seed / denum;

    seed = ((seed * 1103515245) + 12345) % modulo; //0x7fffffff;
    float u2 = seed / denum;

    (*seed_ptr) = seed;

    float2 out;
    out.x = u1;
    out.y = u2;

    return out;
    //return make_float2(u1, u2);
}

/**
  *  Generates two random numbers, drawn from a normal distribtion with mean 0 and
  *  variance 1. Based on the Box Muller transform.
  */
__device__ float2 boxMuller(unsigned long long* seed) {
    float2 u = ansic_lcg(seed);
    
    float r = sqrt(-2.0f*log(u.x));
    float n1 = r*cospi(2*u.y);
    float n2 = r*sinpi(2*u.y);

    float2 out;
    out.x = n1;
    out.y = n2;
    return out;
    
    //return make_float2(n1, n2);
}

/**
  * Kernel that generates uniform random numbers.
  */
extern "C" {
__global__ void uniformDistribution(
        // Size of data
        int seed_nx_, int seed_ny_,        
        int random_nx_,
        
        //Data
        unsigned long long* seed_ptr_, int seed_pitch_,
        float* random_ptr_, int random_pitch_
    ) {

    //Index of cell within domain
    const int ti = (blockDim.x * blockIdx.x) + threadIdx.x;
    const int tj = (blockDim.y * blockIdx.y) + threadIdx.y;

    // Each thread computes and writes two uniform numbers.

    if ((ti < seed_nx_) && (tj < seed_ny_)) {
    
        //Compute pointer to current row in the U array
        unsigned long long* const seed_row = (unsigned long long*) ((char*) seed_ptr_ + seed_pitch_*tj);
        float* const random_row = (float*) ((char*) random_ptr_ + random_pitch_*tj);
        
        unsigned long long seed = seed_row[ti];
        float2 u = ansic_lcg(&seed);

        seed_row[ti] = seed;

        if (2*ti + 1 < random_nx_) {
            random_row[2*ti    ] = u.x;
            random_row[2*ti + 1] = u.y;
        }
        else if (2*ti == random_nx_) {
            random_row[2*ti    ] = u.x;
        }
    }
}
} // extern "C"

/**
  * Kernel that generates normal distributed random numbers with mean 0 and variance 1.
  */
extern "C" {
__global__ void normalDistribution(
        // Size of data
        int seed_nx_, int seed_ny_,
        int random_nx_,               // random_ny_ is equal to seed_ny_
        
        //Data
        unsigned long long* seed_ptr_, int seed_pitch_, // size [seed_nx, seed_ny]
        float* random_ptr_, int random_pitch_           // size [random_nx, seed_ny]
    ) {
    
    //Index of cell within domain
    const int ti = (blockDim.x * blockIdx.x) + threadIdx.x;
    const int tj = (blockDim.y * blockIdx.y) + threadIdx.y;

    // Each thread computes and writes two uniform numbers.

    if ((ti < seed_nx_) && (tj < seed_ny_)) {
    
        //Compute pointer to current row in the U array
        unsigned long long* const seed_row = (unsigned long long*) ((char*) seed_ptr_ + seed_pitch_*tj);
        float* const random_row = (float*) ((char*) random_ptr_ + random_pitch_*tj);
        
        unsigned long long seed = seed_row[ti];
        float2 u = boxMuller(&seed);

        seed_row[ti] = seed;

        if (2*ti + 1 < random_nx_) {
            random_row[2*ti    ] = u.x;
            random_row[2*ti + 1] = u.y;
        }
        else if (2*ti == random_nx_) {
            random_row[2*ti    ] = u.x;
        }
    }
}
} // extern "C"

/**
  * Local function calculating the SOAR function given two grid locations
  */
__device__ float soar_covariance(int a_x, int a_y, int b_x, int b_y,
                      float dx, float dy, float soar_q0, float soar_L) {
    const float dist = sqrt( dx*dx*(a_x - b_x)*(a_x - b_x) +
                             dy*dy*(a_y - b_y)*(a_y - b_y) );
    return soar_q0*( 1.0f + dist/soar_L)*exp(-dist/soar_L);
}



/**
  * Kernel that generates a perturbation of the eta field on the coarse grid.
  * The perturbation is based on a SOAR covariance function using a cut-off value of 2.
  * The result is according to the use of periodic boundary conditions
  */
extern "C" {
__global__ void SOAR(
        // Size of data
        const int nx_, const int ny_,
        const float dx_, const float dy_,
                
        // Parameter for the SOAR function
        const float soar_q0_, const float soar_L_,
        
        // Periodic domain
        const int periodic_north_south_, const int periodic_east_west_,
        
        // random data (input) - if periodic BC: size [nx, ny]
        //                       else:           size [nx + 8, ny + 8]
        float* random_ptr_, const int random_pitch_,

        // Coarse grid data variable (output) - size [nx+4, ny+4]
        // Write to all cells
        float* coarse_ptr_, const int coarse_pitch_
    ) {

    //Index of cell within block
    const int tx = threadIdx.x; 
    const int ty = threadIdx.y;

    //Index of start of block within domain
    const int bx = blockDim.x * blockIdx.x;
    const int by = blockDim.y * blockIdx.y;

    //Index of cell within domain
    const int ti = bx + tx;
    const int tj = by + ty;

    const int cutoff = 2;

    // Local storage for xi (the random numbers)
    __shared__ float xi[block_height+4][block_width+4];

    // Read random numbers into local memory, using periodic BCs if needed:
    for (int j = ty; j < block_height+4; j += blockDim.y) {
        int global_j = 0;
        if (periodic_north_south_) {
            global_j = (by + j - cutoff - 2 + ny_) % ny_;
        } else {
            global_j = clamp(by + j, 0, ny_+8);
        }
        
        float* const random_row = (float*) ((char*) random_ptr_ + random_pitch_*global_j);
        
        for (int i = tx; i < block_width+4; i += blockDim.x) {
        
            int global_i = 0;
            
            if (periodic_east_west_) {
                global_i = (bx + i - cutoff - 2 + nx_) % nx_;
            } else {
                global_i = clamp(bx + i, 0, nx_+8);
            }
            
            xi[j][i] = random_row[global_i];
        }
    }

    __syncthreads();
    
    
    // Compute d_eta using the SOAR covariance function, and store in global memory
    // All reads are from local memory
    const int a_x = tx + cutoff;
    const int a_y = ty + cutoff;
    float Qxi = 0.0f;
    for (int b_y = ty; b_y < a_y + cutoff + 1; b_y++) {
        for (int b_x = tx; b_x < a_x + cutoff + 1; b_x++) {
            const float Q = soar_covariance(a_x, a_y, b_x, b_y,
                                            dx_, dy_, soar_q0_, soar_L_);
            Qxi += Q*xi[b_y][b_x];
        }
    }

    // Evaluate geostrophic balance and write eta, hu and hv to global memory
    if ((ti < nx_+4) && (tj < ny_+4)) {

        //Compute pointer to current row in the coarse array
        float* coarse_row = (float*) ((char*) coarse_ptr_ + coarse_pitch_*(tj));
        coarse_row[ti] = Qxi;
    }
}
} // extern "C"





/**
  * Kernel that adds a perturbation to the input fields eta, hu and hv.
  * The kernel assumes that the coarse grid is equal to the computational grid, and the 
  * values from the coarse grid can therefore be added to eta directly.
  * In order to avoid non-physical states and shock solutions, perturbations for hu and hv
  * are generated according to the geostrophic balance.
  */
extern "C" {
__global__ void geostrophicBalance(
        // Size of data
        const int nx_, const int ny_,
        const float dx_, const float dy_,
        const int ghost_cells_x_, const int ghost_cells_y_,

        // physical parameters
        const float g_, const float f_, const float beta_, const float y0_reference_cell_,
        
        // d_eta values (coarse grid) - size [nx + 4, ny + 4]
        float* coarse_ptr_, const int coarse_pitch_,
    
        // Ocean data variables - size [nx + 4, ny + 4]
        // Write to interior cells only,  [2:nx+2, 2:ny+2]
        float* eta_ptr_, const int eta_pitch_,
        float* hu_ptr_, const int hu_pitch_,
        float* hv_ptr_, const int hv_pitch_,

        // Ocean data parameter - size [nx + 5, ny + 5]
        float* Hi_ptr_, const int Hi_pitch_
    ) {

    //Index of cell within block
    const int tx = threadIdx.x; 
    const int ty = threadIdx.y;

    //Index of start of block within domain
    const int bx = blockDim.x * blockIdx.x + ghost_cells_x_; // Compansating for ghost cells
    const int by = blockDim.y * blockIdx.y + ghost_cells_y_; // Compensating for ghost cells

    //Index of cell within domain
    const int ti = bx + tx;
    const int tj = by + ty;

    // Shared memory for d_eta (also used for H)
    __shared__ float d_eta[block_height+2][block_width+2];


    // Use shared memory for d_eta to compute H_mid for given thread id
    for (int j = ty; j < block_height+1; j += blockDim.y) {
        const int global_j = clamp(by+j, 0, ny_+4);
        float* const Hi_row = (float*) ((char*) Hi_ptr_ + Hi_pitch_*(global_j));
        for (int i = tx; i < block_width+1; i += blockDim.x) {
            const int global_i = clamp(bx+i, 0, nx_+4);
            d_eta[j][i] = Hi_row[global_i];
        }
    }
    
    __syncthreads();
    
    const float H_mid = 0.25f*(d_eta[ty  ][tx] + d_eta[ty  ][tx+1] +
                               d_eta[ty+1][tx] + d_eta[ty+1][tx+1]   );
    
    __syncthreads();
    
    // Read d_eta from coarse_buffer into shared memory
    for (int j = ty; j < block_height+2; j += blockDim.y) {
        const int global_j = clamp(by+j-1, 0, ny_+3);
        float* const coarse_row = (float*) ((char*) coarse_ptr_ + coarse_pitch_*(global_j));
        for (int i = tx; i < block_width+2; i += blockDim.x) {
            const int global_i = clamp(bx+i-1, 0, nx_+3);
            d_eta[j][i] = coarse_row[global_i];
        }
    }
    
    __syncthreads();

    // Evaluate geostrophic balance and write eta, hu and hv to global memory
    if ( (ti > 1) && (tj > 1) && (ti < nx_+2) && (tj < ny_+2)) {

        //Compute pointer to current row in the U array
        float* const eta_row = (float*) ((char*) eta_ptr_ + eta_pitch_*(tj));
        float* const hu_row  = (float*) ((char*) hu_ptr_  + hu_pitch_*(tj));
        float* const hv_row = (float*) ((char*)  hv_ptr_ + hv_pitch_*(tj));
        
        const int eta_tx = tx+1;
        const int eta_ty = ty+1;

        const float coriolis = f_ + beta_*(tj - y0_reference_cell_)*dy_;

        // Total water depth in the given cell (H + eta + d_eta)
        const float h_mid = d_eta[eta_ty][eta_tx] + H_mid + eta_row[ti];

        // Slope of perturbation of eta
        const float eta_diff_x = (d_eta[eta_ty  ][eta_tx+1] - d_eta[eta_ty  ][eta_tx-1]) / (2.0f*dx_);
        const float eta_diff_y = (d_eta[eta_ty+1][eta_tx  ] - d_eta[eta_ty-1][eta_tx  ]) / (2.0f*dy_);

        // perturbation of hu and hv
        const float d_hu = -(g_/coriolis)*h_mid*eta_diff_y;
        const float d_hv =  (g_/coriolis)*h_mid*eta_diff_x;        

        if (true) {
            eta_row[ti] += d_eta[eta_ty][eta_tx];
             hu_row[ti] += d_hu;
             hv_row[ti] += d_hv;
        } else {
            eta_row[ti] = d_eta[eta_ty][eta_tx];
            hu_row[ti] = d_hu;
            hv_row[ti] = d_hv;
        }
    }
}
} // extern "C"




/**
  * Kernel that adds a perturbation to the input fields eta, hu and hv.
  * The kernel use a bicubic interpolation to transfer values from the coarse grid to the
  * computational grid. Since the derivatives of the eta field are known during the interpolation,
  * hu and hv are assigned their appropriate geostrophically balanced values directly.
  */
extern "C" {
__global__ void bicubicInterpolation(
        // Size of computational data
        const int nx_, const int ny_,
        const int ghost_cells_x_, const int ghost_cells_y_,
        const float dx_, const float dy_,
    
        // Size of coarse data
        const int coarse_nx_, const int coarse_ny_,
        const int coarse_ghost_cells_x_, const int coarse_ghost_cells_y_,
        const float coarse_dx_, const float coarse_dy_,
    
        // physical parameters
        const float g_, const float f_, const float beta_, const float y0_reference_cell_,
        
        // d_eta values (coarse grid) - size [nx + 4, ny + 4]
        float* coarse_ptr_, const int coarse_pitch_,
    
        // Ocean data variables - size [nx + 4, ny + 4]
        // Write to interior cells only,  [2:nx+2, 2:ny+2]
        float* eta_ptr_, const int eta_pitch_,
        float* hu_ptr_, const int hu_pitch_,
        float* hv_ptr_, const int hv_pitch_,

        // Ocean data parameter - size [nx + 5, ny + 5]
        float* Hi_ptr_, const int Hi_pitch_,
    ) {
    
    // Each thread is responsible for one grid point in the computational grid.
    
    //Index of cell within block
    const int tx = threadIdx.x; 
    const int ty = threadIdx.y;

    //Index of start of block within domain
    const int bx = blockDim.x * blockIdx.x + ghost_cells_x_; // Compansating for ghost cells
    const int by = blockDim.y * blockIdx.y + ghost_cells_y_; // Compensating for ghost cells

    //Index of cell within domain
    const int ti = bx + tx;
    const int tj = by + ty;

    // Shared memory for H and the coarse values
    __shared__ float shmem[block_height+2][block_width+2];


    // Use shared memory compute H_mid for given thread id
    for (int j = ty; j < block_height+1; j += blockDim.y) {
        const int global_j = clamp(by+j, 0, ny_+4);
        float* const Hi_row = (float*) ((char*) Hi_ptr_ + Hi_pitch_*(global_j));
        for (int i = tx; i < block_width+1; i += blockDim.x) {
            const int global_i = clamp(bx+i, 0, nx_+4);
            shmem[j][i] = Hi_row[global_i];
        }
    }
    
    __syncthreads();
    
    const float H_mid = 0.25f*(shmem[ty  ][tx] + shmem[ty  ][tx+1] +
                               shmem[ty+1][tx] + shmem[ty+1][tx+1]   );
    
    __syncthreads();
    
    
    // Find coarse index for thread (0,0). All threads need to know this in order to read
    // coarse data correctly into shmem.
    const int bx_x = (bx - ghost_cells_x_ + 0.5)*dx_;
    const int by_y = (by - ghost_cells_y_ + 0.5)*dy_;

    
    const int coarse_bx = (int)(floorf((bx_x/coarse_dx_) + coarse_ghost_cells_x_ - 1.5f));
    const int coarse_by = (int)(floorf((by_y/coarse_dy_) + coarse_ghost_cells_y_ - 1.5f));
    // We subtracted the one to account for the stencil size

    // Read d_eta from coarse_buffer into shared memory
    for (int j = ty; j < block_height+4; j += blockDim.y) {
        
        const int global_j = clamp(coarse_by+j, 0, coarse_ny_+3);
        float* const coarse_row = (float*) ((char*) coarse_ptr_ + coarse_pitch_*(global_j));
        
        for (int i = tx; i < block_width+4; i += blockDim.x) {
            
            const int global_i = clamp(coarse_bx+i, 0, coarse_nx_+3);
            shmem[j][i] = coarse_row[global_i];
        }
    }
    __syncthreads();    
        
    // Carry out bicubic interpolation and obtain eta, hu and hv 
    if ( (ti > 1) && (tj > 1) && (ti < nx_+2) && (tj < ny_+2)) {

        // Find coarse index for this thread
        const float x = (ti - ghost_cells_x_ + 0.5)*dx_;
        const float y = (tj - ghost_cells_y_ + 0.5)*dy_;
        
        // Location in the coarse grid:
        const int coarse_i = (int)(floorf((x/coarse_dx_) + coarse_ghost_cells_x_ - 0.5f));
        const int coarse_j = (int)(floorf((y/coarse_dy_) + coarse_ghost_cells_y_ - 0.5f));
        const float coarse_x = (coarse_i - coarse_ghost_cells_x_ + 0.5f)*coarse_dx_;
        const float coarse_y = (coarse_j - coarse_ghost_cells_y_ + 0.5f)*coarse_dy_;
        
        // Location in shmem:
        const int loc_i = coarse_i - coarse_bx + 1; // shmem ghost cell
        const int loc_j = coarse_j - coarse_by + 1; 
        
    
        // Read values for interpolation
        const float f00 = shmem[loc_j  ][loc_i  ];
        const float f01 = shmem[loc_j+1][loc_i  ];
        const float f10 = shmem[loc_j  ][loc_i+1];
        const float f11 = shmem[loc_j+1][loc_i+1];
        
        const float fx00 = (shmem[loc_j  ][loc_i+1] - shmem[loc_j  ][loc_i-1])/2.0f;
        const float fx01 = (shmem[loc_j+1][loc_i+1] - shmem[loc_j+1][loc_i-1])/2.0f;
        const float fx10 = (shmem[loc_j  ][loc_i+2] - shmem[loc_j  ][loc_i  ])/2.0f;
        const float fx11 = (shmem[loc_j+1][loc_i+2] - shmem[loc_j+1][loc_i  ])/2.0f;

        const float fy00 = (shmem[loc_j+1][loc_i  ] - shmem[loc_j-1][loc_i  ])/2.0f;
        const float fy01 = (shmem[loc_j+2][loc_i  ] - shmem[loc_j  ][loc_i  ])/2.0f;
        const float fy10 = (shmem[loc_j+1][loc_i+1] - shmem[loc_j-1][loc_i+1])/2.0f;
        const float fy11 = (shmem[loc_j+2][loc_i+1] - shmem[loc_j  ][loc_i+1])/2.0f;
        
        const float fy_10 = (shmem[loc_j+1][loc_i-1] - shmem[loc_j-1][loc_i-1])/2.0f;
        const float fy_11 = (shmem[loc_j+2][loc_i-1] - shmem[loc_j  ][loc_i-1])/2.0f;
        const float fy20  = (shmem[loc_j+1][loc_i+2] - shmem[loc_j-1][loc_i+2])/2.0f;
        const float fy21  = (shmem[loc_j+2][loc_i+2] - shmem[loc_j  ][loc_i+2])/2.0f;
    
        const float fxy00 = (fy10 - fy_10)/2.0f;
        const float fxy01 = (fy11 - fy_11)/2.0f;
        const float fxy10 = (fy20 -  fy00)/2.0f;
        const float fxy11 = (fy21 -  fy01)/2.0f;
    
        
        // Map (x,y) onto the unit square
        const float rel_x = (x - coarse_x)/coarse_dx_;
        const float rel_y = (y - coarse_y)/coarse_dy_;
        
        const float bi_linear_eta = f00*(1.0f-rel_x)*(1.0f-rel_y) + f10*rel_x*(1.0f-rel_y) + f01*(1.0f-rel_x)*rel_y + f11*rel_x*rel_y;
        
        
        
        //Compute pointer to current row in the U array
        float* const eta_row = (float*) ((char*) eta_ptr_ + eta_pitch_*(tj));
        float* const hu_row  = (float*) ((char*) hu_ptr_  + hu_pitch_*(tj));
        float* const hv_row = (float*) ((char*)  hv_ptr_ + hv_pitch_*(tj));

        eta_row[ti] += bi_linear_eta;
         hu_row[ti] += coarse_bx;
         hv_row[ti] += shmem[loc_j][loc_i];
    }
}
} // extern "C"
