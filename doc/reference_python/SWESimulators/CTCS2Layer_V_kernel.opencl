/**
This OpenCL kernel implements part of the Centered in Time, Centered 
in Space (leapfrog) numerical scheme for the shallow water equations, 
described in 
L. P. Røed, "Documentation of simple ocean models for use in ensemble
predictions", Met no report 2012/3 and 2012/5 .

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

typedef __local float eta_shmem[block_height+1][block_width+2];
typedef __local float u_shmem[block_height+1][block_width+1];
typedef __local float v_shmem[block_height+2][block_width+2];
    
    
    
float windStressY(int wind_stress_type_,
                float dx_, float dy_, float dt_,
                float tau0_, float rho_, float alpha_, float xm_, float Rc_,
                float x0_, float y0_,
                float u0_, float v0_,
                float t_) {
    float Y = 0.0f;
    
    switch (wind_stress_type_) {
    case 2: //MOVING_CYCLONE:
        {
            const float x = (get_global_id(0)+0.5f)*dx_; 
            const float y = (get_global_id(1))*dy_;
            const float a = (x-x0_-u0_*(t_+dt_));
            const float aa = a*a;
            const float b = (y-y0_-v0_*(t_+dt_));
            const float bb = b*b;
            const float r = sqrt(aa+bb);
            const float c = 1.0f - r/Rc_;
            const float xi = c*c;
            
            Y = (tau0_/rho_) * (a/Rc_) * exp(-0.5f*xi);
        }
        break;
    }

    return Y;
}




/**
  * Kernel that evolves V one step in time.
  */
__kernel void computeVKernel(
        //Discretization parameters
        int nx_, int ny_,
        float dx_, float dy_, float dt_,
    
        //Physical parameters
        float g_, //< Gravitational constant
        float f_, //< Coriolis coefficient
        float r1_, //< Inter-layer friction coefficient
        float r2_, //< Bottom friction coefficient
    
        //Numerical diffusion
        float A_,
        
        //Density of each layer
        float rho1_,
        float rho2_,
    
        //Data for layer 1
        __global float* H1_ptr_, int H1_pitch_,
        __global float* eta1_1_ptr_, int eta1_1_pitch_, // eta^n
        __global float* U1_1_ptr_, int U1_1_pitch_,     // U^n
        __global float* V1_0_ptr_, int V1_0_pitch_,     // V^n-1, also output V^n+1
        __global float* V1_1_ptr_, int V1_1_pitch_,     // V^n
        
        //Data for layer 2
        __global float* H2_ptr_, int H2_pitch_,
        __global float* eta2_1_ptr_, int eta2_1_pitch_, 
        __global float* U2_1_ptr_, int U2_1_pitch_,     
        __global float* V2_0_ptr_, int V2_0_pitch_,     
        __global float* V2_1_ptr_, int V2_1_pitch_,     
    
        // Wind stress parameters
        int wind_stress_type_, 
        float tau0_, float alpha_, float xm_, float Rc_,
        float x0_, float y0_,
        float u0_, float v0_,
        float t_) {
                    
    eta_shmem H1_shared;
    eta_shmem eta1_shared;
    u_shmem U1_shared;
    v_shmem V1_shared;
    
    eta_shmem H2_shared;
    eta_shmem eta2_shared;
    u_shmem U2_shared;
    v_shmem V2_shared;

    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Start of block within domain
    const int bx = get_local_size(0) * get_group_id(0) + 1; //Skip global ghost cells
    const int by = get_local_size(1) * get_group_id(1) + 1; //Skip global ghost cells

    //Index of cell within domain
    const int ti = bx + tx;
    const int tj = by + ty;
    
    //Compute pointer to current row in the V array
    __global float* const V1_0_row = (__global float*) ((__global char*) V1_0_ptr_ + V1_0_pitch_*tj);
    __global float* const V2_0_row = (__global float*) ((__global char*) V2_0_ptr_ + V2_0_pitch_*tj);

    //Read current V
    float V1_0 = 0.0f;
    float V2_0 = 0.0f;
    if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_) {
        V1_0 = V1_0_row[ti];
        V2_0 = V2_0_row[ti];
    }

    //Read H and eta into shared memory: (nx+2)*(ny+1) cells
    for (int j=ty; j<block_height+1; j+=get_local_size(1)) {
        // "fake" global ghost cells by clamping
        const int l = clamp(by + j, 1, ny_);
        
        //Compute the pointer to current row in the H and eta arrays
        __global float* const H1_row = (__global float*) ((__global char*) H1_ptr_ + H1_pitch_*l);
        __global float* const H2_row = (__global float*) ((__global char*) H2_ptr_ + H2_pitch_*l);
        
        __global float* const eta1_1_row = (__global float*) ((__global char*) eta1_1_ptr_ + eta1_1_pitch_*l);
        __global float* const eta2_1_row = (__global float*) ((__global char*) eta2_1_ptr_ + eta2_1_pitch_*l);
        
        for (int i=tx; i<block_width+2; i+=get_local_size(0)) {
            // "fake" global ghost cells by clamping
            const int k = clamp(bx + i - 1, 1, nx_);
            
            H1_shared[j][i] = H1_row[k];
            H2_shared[j][i] = H2_row[k];
            
            eta1_shared[j][i] = eta1_1_row[k];
            eta2_shared[j][i] = eta2_1_row[k];
        }
    }

    //Read U into shared memory: (nx+1)*(ny+1) cells
    for (int j=ty; j<block_height+1; j+=get_local_size(1)) {
        // "fake" ghost cells by clamping
        const int l = clamp(by + j, 1, ny_);
        
        //Compute the pointer to current row in the U array
        __global float* const U1_1_row = (__global float*) ((__global char*) U1_1_ptr_ + U1_1_pitch_*l);
        __global float* const U2_1_row = (__global float*) ((__global char*) U2_1_ptr_ + U2_1_pitch_*l);
        
        for (int i=tx; i<block_width+1; i+=get_local_size(0)) {
            // Prevent out-of-bounds
            const int k = clamp(bx + i - 1, 0, nx_);
            
            U1_shared[j][i] = U1_1_row[k];
            U2_shared[j][i] = U2_1_row[k];
        }
    }
    

    //Read V into shared memory: (nx+2)*(ny+2) cells
    for (int j=ty; j<block_height+2; j+=get_local_size(1)) {
        // Prevent out-of-bounds
        const int l = clamp(by + j - 1, 0, ny_);
        
        //Compute the pointer to current row in the V array
        __global float* const V1_1_row = (__global float*) ((__global char*) V1_1_ptr_ + V1_1_pitch_*l);
        __global float* const V2_1_row = (__global float*) ((__global char*) V2_1_ptr_ + V2_1_pitch_*l);
        
        for (int i=tx; i<block_width+2; i+=get_local_size(0)) {
            // "fake" ghost cells by clamping
            const int k = clamp(bx + i - 1, 1, nx_);
            
            V1_shared[j][i] = V1_1_row[k];
            V2_shared[j][i] = V2_1_row[k];
        }
    }

    //Make sure all threads have read into shared mem
    barrier(CLK_LOCAL_MEM_FENCE);
    
    /**
      * Now get all our required variables as short-hands
      * here we use the notation of
      *  Var_00 as var_i,j
      *  Var_p0 as var_i+1,j
      *  Var_0m as var_i,j-1
      * etc
      */
    //Layer 1
    const float V1_00 = V1_shared[ty+1][tx+1]; //V at "center"
    const float V1_0p = V1_shared[ty+2][tx+1]; //V at "north"
    const float V1_0m = V1_shared[ty  ][tx+1]; //V at "south"
    const float V1_p0 = V1_shared[ty+1][tx+2]; //V at "east"
    const float V1_m0 = V1_shared[ty+1][tx  ]; //V at "west"

    const float U1_00 = U1_shared[ty  ][tx+1];
    const float U1_0p = U1_shared[ty+1][tx+1];
    const float U1_m0 = U1_shared[ty  ][tx  ];
    const float U1_mp = U1_shared[ty+1][tx  ];

    const float H1_m0 = H1_shared[ty  ][tx  ]; 
    const float H1_00 = H1_shared[ty  ][tx+1]; 
    const float H1_p0 = H1_shared[ty  ][tx+2];
    const float H1_mp = H1_shared[ty+1][tx  ];
    const float H1_0p = H1_shared[ty+1][tx+1];
    const float H1_pp = H1_shared[ty+1][tx+2];
    
    const float eta1_m0 = eta1_shared[ty  ][tx  ]; 
    const float eta1_00 = eta1_shared[ty  ][tx+1]; 
    const float eta1_p0 = eta1_shared[ty  ][tx+2];
    const float eta1_mp = eta1_shared[ty+1][tx  ]; 
    const float eta1_0p = eta1_shared[ty+1][tx+1]; 
    const float eta1_pp = eta1_shared[ty+1][tx+2];
    
    
    
    //Layer 2 (bottom)
    const float V2_00 = V2_shared[ty+1][tx+1];
    const float V2_0p = V2_shared[ty+2][tx+1];
    const float V2_0m = V2_shared[ty  ][tx+1];
    const float V2_p0 = V2_shared[ty+1][tx+2];
    const float V2_m0 = V2_shared[ty+1][tx  ];

    const float U2_00 = U2_shared[ty  ][tx+1];
    const float U2_0p = U2_shared[ty+1][tx+1];
    const float U2_m0 = U2_shared[ty  ][tx  ];
    const float U2_mp = U2_shared[ty+1][tx  ];

    const float H2_m0 = H2_shared[ty  ][tx  ]; 
    const float H2_00 = H2_shared[ty  ][tx+1]; 
    const float H2_p0 = H2_shared[ty  ][tx+2];
    const float H2_mp = H2_shared[ty+1][tx  ];
    const float H2_0p = H2_shared[ty+1][tx+1];
    const float H2_pp = H2_shared[ty+1][tx+2];
    
    const float eta2_m0 = eta2_shared[ty  ][tx  ]; 
    const float eta2_00 = eta2_shared[ty  ][tx+1]; 
    const float eta2_p0 = eta2_shared[ty  ][tx+2];
    const float eta2_mp = eta2_shared[ty+1][tx  ]; 
    const float eta2_0p = eta2_shared[ty+1][tx+1]; 
    const float eta2_pp = eta2_shared[ty+1][tx+2];
    
    
    //Reconstruct Eta_bar at the V position
    const float eta1_bar_m0 = 0.25f*(eta1_m0 + eta1_mp + eta1_00 + eta1_0p);
    const float eta1_bar_00 = 0.25f*(eta1_00 + eta1_0p + eta1_p0 + eta1_pp);
    
    const float eta2_bar_m0 = 0.25f*(eta2_m0 + eta2_mp + eta2_00 + eta2_0p);
    const float eta2_bar_00 = 0.25f*(eta2_00 + eta2_0p + eta2_p0 + eta2_pp);
    
    
    
    

    //Reconstruct H_bar and H_y (at the V position)
    const float H1_bar_m0 = 0.25f*(H1_m0 + H1_mp + H1_00 + H1_0p);
    const float H1_bar_00 = 0.25f*(H1_00 + H1_0p + H1_p0 + H1_pp);
    const float H1_y = 0.5f*(H1_00 + H1_0p);
    
    const float H2_bar_m0 = 0.25f*(H2_m0 + H2_mp + H2_00 + H2_0p);
    const float H2_bar_00 = 0.25f*(H2_00 + H2_0p + H2_p0 + H2_pp);
    const float H2_y = 0.5f*(H2_00 + H2_0p);
    
    
    
    //Compute layer thickness of top layer
    //FIXME: What is the actual depth of this top layer?
    const float h1_0p = H1_0p + eta1_0p;// - eta2_0p;
    const float h1_00 = H1_00 + eta1_00;// - eta2_00;
    const float h1_bar_00 = H1_bar_00 + eta1_bar_00;// - eta2_bar_00;
    const float h1_bar_m0 = H1_bar_m0 + eta1_bar_m0;// - eta2_bar_m0;
    
    const float h2_0p = H2_0p + eta2_0p;
    const float h2_00 = H2_00 + eta2_00;
    const float h2_bar_00 = H2_bar_00 + eta2_bar_00;
    const float h2_bar_m0 = H2_bar_m0 + eta2_bar_m0;
    
    
    
    //Compute pressure components
    //FIXME Are these the right pressure terms?
    const float epsilon = (rho2_ - rho1_)/rho2_;
    //const float P1_y = -0.5f*g_*(h1_0p + h1_00) * (h1_0p - h1_00 + H1_0p - H1_00) * (1.0f - epsilon);
    //const float P2_y = P1_y -0.5f*g_*(h2_0p + h2_00) * (h2_0p - h2_00 + H2_0p - H2_00);
    //const float P1_y = -0.5f*g_*(h1_0p + h1_00) * (eta1_0p - eta1_00) * (1.0f - epsilon);
    //const float P2_y = P1_y - 0.5f*g_*(h2_0p + h2_00) * (eta2_0p - eta2_00);
    //const float P1_y = -0.5f*g_*(h1_0p + h1_00) * (eta1_0p - eta1_00 + H1_0p - H1_00) * (1.0f - epsilon);
    //const float P2_y = P1_y -0.5f*g_*(h2_0p + h2_00) * (eta2_0p - eta2_00 + H2_0p - H2_00);
    const float P1_y = -0.5f*g_*(h1_0p + h1_00) * (eta1_0p - eta1_00 + h2_0p - h2_00) * (1.0f - epsilon);
    const float P2_y = -0.5f*g_*(h2_0p + h2_00) * (eta2_0p - eta2_00 + H2_0p - H2_00);
    
    

    //Reconstruct U at the V position
    const float U1_bar = 0.25f*(U1_m0 + U1_00 + U1_mp + U1_0p);
    const float U2_bar = 0.25f*(U2_m0 + U2_00 + U2_mp + U2_0p);
    
    
    

    //Calculate the friction coefficient
    //FIXME: Should this be h instead of H?
    const float C1 = 2.0f*r1_*dt_/H1_y;
    const float C2 = 2.0f*r2_*dt_/H2_y;
    
    
    
    //Calculate numerical diffusion / subgrid energy loss coefficient
    const float D = 2.0f*A_*dt_*(dx_*dx_ + dy_*dy_)/(dx_*dx_*dy_*dy_);

    
    
    
    //Calculate nonlinear effects
    const float N1_a = (V1_0p + V1_00)*(V1_0p + V1_00) / (h1_0p);
    const float N1_b = (V1_00 + V1_0m)*(V1_00 + V1_0m) / (h1_00);
    const float N1_c = (U1_0p + U1_00)*(V1_p0 + V1_00) / (h1_bar_00);
    const float N1_d = (U1_mp + U1_m0)*(V1_00 + V1_m0) / (h1_bar_m0);
    const float N1 = 0.25f*( N1_a - N1_b + (dy_/dx_)*(N1_c - N1_d) );
    
    const float N2_a = (V2_0p + V2_00)*(V2_0p + V2_00) / (h2_0p);
    const float N2_b = (V2_00 + V2_0m)*(V2_00 + V2_0m) / (h2_00);
    const float N2_c = (U2_0p + U2_00)*(V2_p0 + V2_00) / (h2_bar_00);
    const float N2_d = (U2_mp + U2_m0)*(V2_00 + V2_m0) / (h2_bar_m0);
    const float N2 = 0.25f*( N2_a - N2_b + (dy_/dx_)*(N2_c - N2_d) );
    
    
    
    
    
    //Calculate eddy viscosity term
    const float E1 = (V1_p0 - V1_0 + V1_m0)/(dx_*dx_) + (V1_0p - V1_0 + V1_0m)/(dy_*dy_);
    const float E2 = (V2_p0 - V2_0 + V2_m0)/(dx_*dx_) + (V2_0p - V2_0 + V2_0m)/(dy_*dy_);
    
    

    //Calculate the wind shear stress
    const float Y = windStressY(
        wind_stress_type_, 
        dx_, dy_, dt_,
        tau0_, rho1_, alpha_, xm_, Rc_,
        x0_, y0_,
        u0_, v0_,
        t_);
    
    //Compute the V at the next timestep
    float V1_2 = (V1_0 + 2.0f*dt_*(-f_*U1_bar + (N1 + P1_y)/dy_ + Y  + A_*E1) ) / (1.0f + C1 + D);
    float V2_2 = (V2_0 + 2.0f*dt_*(-f_*U2_bar + (N2 + P2_y)/dy_ + C1 + A_*E2) ) / (1.0f + C2 + D);

    //Write to main memory for internal cells
    if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_) {
        V1_0_row[ti] = V1_2;
        V2_0_row[ti] = V2_2;
    }
}










