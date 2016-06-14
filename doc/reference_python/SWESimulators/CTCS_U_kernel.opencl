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

float windStressX(int wind_stress_type_,
                float dx_, float dy_, float dt_,
                float tau0_, float rho_, float alpha_, float xm_, float Rc_,
                float x0_, float y0_,
                float u0_, float v0_,
                float t_) {
    
    float X = 0.0f;
    
    switch (wind_stress_type_) {
    case 0: //UNIFORM_ALONGSHORE
        {
            const float y = (get_global_id(1)+0.5f)*dy_;
            X = tau0_/rho_ * exp(-alpha_*y);
        }
        break;
    case 1: //BELL_SHAPED_ALONGSHORE
        if (t_ <= 48.0f*3600.0f) {
            const float a = alpha_*((get_global_id(0)+0.5f)*dx_-xm_);
            const float aa = a*a;
            const float y = (get_global_id(1)+0.5f)*dy_;
            X = tau0_/rho_ * exp(-aa) * exp(-alpha_*y);
        }
        break;
    case 2: //MOVING_CYCLONE
        {
            const float x = (get_global_id(0))*dx_;
            const float y = (get_global_id(1)+0.5f)*dy_;
            const float a = (x-x0_-u0_*(t_+dt_));
            const float aa = a*a;
            const float b = (y-y0_-v0_*(t_+dt_));
            const float bb = b*b;
            const float r = sqrt(aa+bb);
            const float c = 1.0f - r/Rc_;
            const float xi = c*c;
            
            X = -(tau0_/rho_) * (b/Rc_) * exp(-0.5f*xi);
        }
        break;
    }

    return X;
}


/**
  * Kernel that evolves U one step in time.
  */
__kernel void computeUKernel(
        //Discretization parameters
        int nx_, int ny_,
        float dx_, float dy_, float dt_,
    
        //Physical parameters
        float g_, //< Gravitational constant
        float f_, //< Coriolis coefficient
        float r_, //< Bottom friction coefficient
    
        //Numerical diffusion
        float A_,
    
        //Data
        __global float* H_ptr_, int H_pitch_,
        __global float* eta1_ptr_, int eta1_pitch_, // eta^n
        __global float* U0_ptr_, int U0_pitch_, // U^n-1, also output, U^n+1
        __global float* U1_ptr_, int U1_pitch_, // U^n
        __global float* V1_ptr_, int V1_pitch_, // V^n
    
        // Wind stress parameters
        int wind_stress_type_, 
        float tau0_, float rho_, float alpha_, float xm_, float Rc_,
        float x0_, float y0_,
        float u0_, float v0_,
        float t_) {
        
    __local float H_shared[block_height+2][block_width+1];
    __local float eta1_shared[block_height+2][block_width+1];
    __local float U1_shared[block_height+2][block_width+2];
    __local float V1_shared[block_height+1][block_width+1];

    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Start of block within domain
    const int bx = get_local_size(0) * get_group_id(0) + 1; //Skip global ghost cells
    const int by = get_local_size(1) * get_group_id(1) + 1; //Skip global ghost cells

    //Index of cell within domain
    const int ti = bx + tx;
    const int tj = by + ty;
    
    //Compute pointer to current row in the U array
    __global float* const U0_row = (__global float*) ((__global char*) U0_ptr_ + U0_pitch_*tj);

    //Read current U
    float U0 = 0.0f;
    if (ti > 0 && ti < nx_ && tj > 0 && tj < ny_+1) {
        U0 = U0_row[ti];
    }

    //Read H and eta into shared memory: (nx+1)*(ny+2) cells
    for (int j=ty; j<block_height+2; j+=get_local_size(1)) {
        // "fake" global ghost cells by clamping
        const int l = clamp(by + j - 1, 1, ny_);
        
        //Compute the pointer to current row in the H and eta arrays
        __global float* const H_row = (__global float*) ((__global char*) H_ptr_ + H_pitch_*l);
        __global float* const eta1_row = (__global float*) ((__global char*) eta1_ptr_ + eta1_pitch_*l);
        
        for (int i=tx; i<block_width+1; i+=get_local_size(0)) {
            // "fake" global ghost cells by clamping
            const int k = clamp(bx + i, 1, nx_);
            
            H_shared[j][i] = H_row[k];
            eta1_shared[j][i] = eta1_row[k];
        }
    }

    //Read U into shared memory: (nx+2)*(ny+2) cells
    for (int j=ty; j<block_height+2; j+=get_local_size(1)) {
        // "fake" ghost cells by clamping
        const int l = clamp(by + j - 1, 1, ny_);
        
        //Compute the pointer to current row in the U array
        __global float* const U1_row = (__global float*) ((__global char*) U1_ptr_ + U1_pitch_*l);
        
        for (int i=tx; i<block_width+2; i+=get_local_size(0)) {
            // Prevent out-of-bounds
            const int k = clamp(bx + i - 1, 0, nx_);
            
            U1_shared[j][i] = U1_row[k];
        }
    }
    

    //Read V into shared memory: (nx+1)*(ny+1) cells
    for (int j=ty; j<block_height+1; j+=get_local_size(1)) {
        // Prevent out-of-bounds
        const int l = clamp(by + j - 1, 0, ny_);
        
        //Compute the pointer to current row in the U array
        __global float* const V1_row = (__global float*) ((__global char*) V1_ptr_ + V1_pitch_*l);
        
        for (int i=tx; i<block_width+1; i+=get_local_size(0)) {
            // "fake" ghost cells by clamping
            const int k = clamp(bx + i, 1, nx_);
            
            V1_shared[j][i] = V1_row[k];
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
    const float U_00 = U1_shared[ty+1][tx+1]; //U at "center"
    const float U_0p = U1_shared[ty+2][tx+1]; //U at "north"
    const float U_0m = U1_shared[ty  ][tx+1]; //U at "south"
    const float U_p0 = U1_shared[ty+1][tx+2]; //U at "east"
    const float U_m0 = U1_shared[ty+1][tx  ]; //U at "west"
    
    const float V_00 = V1_shared[ty+1][tx  ];
    const float V_p0 = V1_shared[ty+1][tx+1];
    const float V_0m = V1_shared[ty  ][tx  ];
    const float V_pm = V1_shared[ty  ][tx+1];
    
    const float H_0m = H_shared[ty  ][tx  ]; 
    const float H_00 = H_shared[ty+1][tx  ]; 
    const float H_0p = H_shared[ty+2][tx  ];
    const float H_pm = H_shared[ty  ][tx+1];
    const float H_p0 = H_shared[ty+1][tx+1]; 
    const float H_pp = H_shared[ty+2][tx+1];
    
    const float eta_0m = eta1_shared[ty  ][tx  ]; 
    const float eta_00 = eta1_shared[ty+1][tx  ]; 
    const float eta_0p = eta1_shared[ty+2][tx  ];
    const float eta_pm = eta1_shared[ty  ][tx+1];
    const float eta_p0 = eta1_shared[ty+1][tx+1]; 
    const float eta_pp = eta1_shared[ty+2][tx+1];

    //Reconstruct H_bar and H_x (at the U position)
    const float H_bar_0m = 0.25f*(H_0m + H_pm + H_00 + H_p0);
    const float H_bar_00 = 0.25f*(H_00 + H_p0 + H_0p + H_pp);
    const float H_x = 0.5f*(H_00 + H_p0);
    
    //Reconstruct Eta_bar at the V position
    const float eta_bar_0m = 0.25f*(eta_0m + eta_pm + eta_00 + eta_p0);
    const float eta_bar_00 = 0.25f*(eta_00 + eta_p0 + eta_0p + eta_pp);

    //Reconstruct V at the U position
    const float V_bar = 0.25f*(V_0m + V_00 + V_pm + V_p0);

    //Calculate the friction coefficient
    const float C = 1.0 + 2*r_*dt_/H_x + 2*A_*dt_*(dx_*dx_ + dy_*dy_)/(dx_*dx_*dy_*dy_);

    //Calculate the pressure/gravitational effect
    //const float P_x = -g_*H_x*(eta_p0 - eta_00);
    //const float P_x_hat = -0.5f*g_*(eta_p0*eta_p0 - eta_00*eta_00);
    
    const float h_p0 = H_p0 + eta_p0;
    const float h_00 = H_00 + eta_00;
    const float P_x = -0.5f*g_*(h_p0 + h_00) * (h_p0 - h_00 + H_p0 - H_00);
    //const float P_x = -0.5f*g_*(h_p0*h_p0 - h_00*h_00 - 2.0f*(h_p0+h_00)*(H_p0-H_00));
    //const float P_x_hat = 0.0f;
    
    //Calculate nonlinear effects
    const float N_a = (U_p0 + U_00)*(U_p0 + U_00) / (H_p0 + eta_p0);
    const float N_b = (U_00 + U_m0)*(U_00 + U_m0) / (H_00 + eta_00);
    const float N_c = (U_0p + U_00)*(V_p0 + V_00) / (H_bar_00 + eta_bar_00);
    const float N_d = (U_00 + U_0m)*(V_pm + V_0m) / (H_bar_0m + eta_bar_0m);
    float N = 0.25f*( N_a - N_b + (dx_/dy_)*(N_c - N_d) );
    
    //Calculate eddy viscosity term
    float E = (U_p0 - U0 + U_m0)/(dx_*dx_) + (U_0p - U0 + U_0m)/(dy_*dy_);
    
    //Calculate the wind shear stress
    float X = windStressX(
        wind_stress_type_, 
        dx_, dy_, dt_,
        tau0_, rho_, alpha_, xm_, Rc_,
        x0_, y0_,
        u0_, v0_,
        t_);
    
    //Compute the V at the next timestep
    float U2 = (U0 + 2.0f*dt_*(f_*V_bar + (N + P_x)/dx_ + X + A_*E) ) / C;

    //Write to main memory for internal cells
    if (ti > 0 && ti < nx_ && tj > 0 && tj < ny_+1) {
        U0_row[ti] = U2;
    }
}