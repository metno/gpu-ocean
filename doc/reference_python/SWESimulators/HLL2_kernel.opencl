/*
This OpenCL kernel implements the second order HLL flux

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


#define block_height 8
#define block_width 8



float3 F_func(const float3 Q, const float g) {
    float3 F;

    F.x = Q.y;                              //hu
    F.y = Q.y*Q.y / Q.x + 0.5f*g*Q.x*Q.x;   //hu*hu/h + 0.5f*g*h*h;
    F.z = Q.y*Q.z / Q.x;                    //hu*hv/h;

    return F;
}



float3 HLL_flux(const float3 Q_l, const float3 Q_r, const float g_) {    
    const float h_l = Q_l.x;
    const float h_r = Q_r.x;
    
    // Calculate velocities
    const float u_l = Q_l.y / h_l;
    const float u_r = Q_r.y / h_r;
    
    // Estimate the potential wave speeds
    const float c_l = sqrt(g_*h_l);
    const float c_r = sqrt(g_*h_r);
    
    // Compute h in the "star region", h^dagger
    const float h_dag = 0.5f * (h_l+h_r) - 0.25f * (u_r-u_l)*(h_l+h_r)/(c_l+c_r);
    
    const float q_l_tmp = sqrt(0.5f * ( (h_dag+h_l)*h_dag / (h_l*h_l) ) );
    const float q_r_tmp = sqrt(0.5f * ( (h_dag+h_r)*h_dag / (h_r*h_r) ) );
    
    const float q_l = (h_dag > h_l) ? q_l_tmp : 1.0f;
    const float q_r = (h_dag > h_r) ? q_r_tmp : 1.0f;
    
    // Compute wave speed estimates
    const float S_l = u_l - c_l*q_l;
    const float S_r = u_r + c_r*q_r;
    
    //Upwind selection
    if (S_l >= 0) {
        return F_func(Q_l, g_);
    }
    else if (S_r <= 0.0f) {
        return F_func(Q_r, g_);
    }
    //Or estimate flux in the star region
    else {
        const float3 F_l = F_func(Q_l, g_);
        const float3 F_r = F_func(Q_r, g_);
        const float3 flux = (S_r*F_l - S_l*F_r + S_r*S_l*(Q_r - Q_l)) / (S_r-S_l);
        return flux;
    }
}


/**
 * @return min(a, b, c), {a, b, c} > 0
 *         max(a, b, c), {a, b, c} < 0
 *         0           , otherwise
 */
float minmod(float a, float b, float c) {
	return 0.25f
		*copysign(1.0f, a)
		*(copysign(1.0f, a) + copysign(1.0f, b))
		*(copysign(1.0f, b) + copysign(1.0f, c))
		*min( min(fabs(a), fabs(b)), fabs(c) );
}

/**
  * Reconstructs a slope using the minmod limiter based on three 
  * consecutive values
  */
float reconstructSlope(float left, float center, float right, float theta) {
    const float backward = center - left;
    const float central = (right - left) * 0.5f;
    const float forward = right - center;
    return minmod(theta*backward, central, theta*forward);
}












__kernel void swe_2D(
        int nx_, int ny_,
        float dx_, float dy_, float dt_,
        float g_,
        
        float theta_,
        
        int step_,
        
        //Input h^n
        __global float* h0_ptr_, int h0_pitch_,
        __global float* hu0_ptr_, int hu0_pitch_,
        __global float* hv0_ptr_, int hv0_pitch_,
        
        //Output h^{n+1}
        __global float* h1_ptr_, int h1_pitch_,
        __global float* hu1_ptr_, int hu1_pitch_,
        __global float* hv1_ptr_, int hv1_pitch_) {
        
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
    __local float Qx[3][block_height][block_width+2];
    __local float Qy[3][block_height+2][block_width];
    __local float F[3][block_height][block_width+1];
    __local float G[3][block_height+1][block_width];
    
    
    
    
    
    //Read into shared memory
    for (int j=ty; j<block_height+4; j+=get_local_size(1)) {
        const int l = clamp(by + j, 0, ny_+3); // Out of bounds
        
        //Compute the pointer to current row in the arrays
        __global float* const h_row = (__global float*) ((__global char*) h0_ptr_ + h0_pitch_*l);
        __global float* const hu_row = (__global float*) ((__global char*) hu0_ptr_ + hu0_pitch_*l);
        __global float* const hv_row = (__global float*) ((__global char*) hv0_ptr_ + hv0_pitch_*l);
        
        for (int i=tx; i<block_width+4; i+=get_local_size(0)) {
            const int k = clamp(bx + i, 0, nx_+3); // Out of bounds
            
            Q[0][j][i] = h_row[k];
            Q[1][j][i] = hu_row[k];
            Q[2][j][i] = hv_row[k];
        }
    }
    __syncthreads();
    
    
    
    
    
    
    
    
    
    //Fix boundary conditions
    {
        const int i = tx + 2; //Skip local ghost cells, i.e., +2
        const int j = ty + 2;
        
        if (ti == 2) {
            Q[0][j][i-1] =  Q[0][j][i];
            Q[1][j][i-1] = -Q[1][j][i];
            Q[2][j][i-1] =  Q[2][j][i];
            
            Q[0][j][i-2] =  Q[0][j][i+1];
            Q[1][j][i-2] = -Q[1][j][i+1];
            Q[2][j][i-2] =  Q[2][j][i+1];
        }
        if (ti == nx_+1) {
            Q[0][j][i+1] =  Q[0][j][i];
            Q[1][j][i+1] = -Q[1][j][i];
            Q[2][j][i+1] =  Q[2][j][i];
            
            Q[0][j][i+2] =  Q[0][j][i-1];
            Q[1][j][i+2] = -Q[1][j][i-1];
            Q[2][j][i+2] =  Q[2][j][i-1];
        }
        if (tj == 2) {
            Q[0][j-1][i] =  Q[0][j][i];
            Q[1][j-1][i] =  Q[1][j][i];
            Q[2][j-1][i] = -Q[2][j][i];
            
            Q[0][j-2][i] =  Q[0][j+1][i];
            Q[1][j-2][i] =  Q[1][j+1][i];
            Q[2][j-2][i] = -Q[2][j+1][i];
        }
        if (tj == ny_+1) {
            Q[0][j+1][i] =  Q[0][j][i];
            Q[1][j+1][i] =  Q[1][j][i];
            Q[2][j+1][i] = -Q[2][j][i];
            
            Q[0][j+2][i] =  Q[0][j-1][i];
            Q[1][j+2][i] =  Q[1][j-1][i];
            Q[2][j+2][i] = -Q[2][j-1][i];
        }
    }
    __syncthreads();
    
    
    
    
    //Reconstruct slopes along x axis
    for (int j=ty; j<block_height; j+=get_local_size(1)) {
        const int l = j + 2; //Skip ghost cells
        for (int i=tx; i<block_width+2; i+=get_local_size(0)) {
            const int k = i + 1;
            for (int p=0; p<3; ++p) {
                Qx[p][j][i] = reconstructSlope(Q[p][l][k-1], Q[p][l][k], Q[p][l][k+1], theta_);
            }
        }
    }
    
    //Reconstruct slopes along y axis
    for (int j=ty; j<block_height+2; j+=get_local_size(1)) {
        const int l = j + 1;
        for (int i=tx; i<block_width; i+=get_local_size(0)) {            
            const int k = i + 2; //Skip ghost cells
            for (int p=0; p<3; ++p) {
                Qy[p][j][i] = reconstructSlope(Q[p][l-1][k], Q[p][l][k], Q[p][l+1][k], theta_);
            }
        }
    }
    __syncthreads();
    
    
    
    
    
    
    
    //Compute fluxes along the x axis
    for (int j=ty; j<block_height; j+=get_local_size(1)) {
        const int l = j + 2; //Skip ghost cells
        for (int i=tx; i<block_width+1; i+=get_local_size(0)) {
            const int k = i + 1;
            // Q at interface from the right and left
            const float3 Q_r = (float3)(Q[0][l][k+1] - 0.5f*Qx[0][j][i+1],
                                        Q[1][l][k+1] - 0.5f*Qx[1][j][i+1],
                                        Q[2][l][k+1] - 0.5f*Qx[2][j][i+1]);
            const float3 Q_l = (float3)(Q[0][l][k  ] + 0.5f*Qx[0][j][i  ],
                                        Q[1][l][k  ] + 0.5f*Qx[1][j][i  ],
                                        Q[2][l][k  ] + 0.5f*Qx[2][j][i  ]);
                                       
            // Computed flux
            const float3 flux = HLL_flux(Q_l, Q_r, g_);
            
            //Write to shared memory
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
            //NOte that hu and hv are swapped ("transposing" the domain)!
            const float3 Q_r = (float3)(Q[0][l+1][k] - 0.5f*Qy[0][j+1][i],
                                        Q[2][l+1][k] - 0.5f*Qy[2][j+1][i],
                                        Q[1][l+1][k] - 0.5f*Qy[1][j+1][i]);
            const float3 Q_l = (float3)(Q[0][l  ][k] + 0.5f*Qy[0][j  ][i],
                                        Q[2][l  ][k] + 0.5f*Qy[2][j  ][i],
                                        Q[1][l  ][k] + 0.5f*Qy[1][j  ][i]);
                                       
            // Computed flux
            const float3 flux = HLL_flux(Q_l, Q_r, g_);
            
            //Write to shared memory
            //Note that we here swap hu and hv back to the original
            G[0][j][i] = flux.x;
            G[1][j][i] = flux.z;
            G[2][j][i] = flux.y;
        }
    }
    __syncthreads();
    
    
    
    
    
    //Sum fluxes and advance in time for all internal cells
    if (ti > 1 && ti < nx_+2 && tj > 1 && tj < ny_+2) {
        const int i = tx + 2; //Skip local ghost cells, i.e., +2
        const int j = ty + 2;
        
        const float h1  = Q[0][j][i] + (F[0][ty][tx] - F[0][ty  ][tx+1]) * dt_ / dx_ 
                                     + (G[0][ty][tx] - G[0][ty+1][tx  ]) * dt_ / dy_;
        const float hu1 = Q[1][j][i] + (F[1][ty][tx] - F[1][ty  ][tx+1]) * dt_ / dx_ 
                                     + (G[1][ty][tx] - G[1][ty+1][tx  ]) * dt_ / dy_;
        const float hv1 = Q[2][j][i] + (F[2][ty][tx] - F[2][ty  ][tx+1]) * dt_ / dx_ 
                                     + (G[2][ty][tx] - G[2][ty+1][tx  ]) * dt_ / dy_;

        __global float* const h_row  = (__global float*) ((__global char*) h1_ptr_ + h1_pitch_*tj);
        __global float* const hu_row = (__global float*) ((__global char*) hu1_ptr_ + hu1_pitch_*tj);
        __global float* const hv_row = (__global float*) ((__global char*) hv1_ptr_ + hv1_pitch_*tj);
        
        if  (step_ == 0) {
            //First step of RK2 ODE integrator
            
            h_row[ti] = h1;
            hu_row[ti] = hu1;
            hv_row[ti] = hv1;
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
            hu_row[ti] = hu_b;
            hv_row[ti] = hv_b;
        }
    }
    
}