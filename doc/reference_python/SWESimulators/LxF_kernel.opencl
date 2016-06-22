/*
This OpenCL kernel implements the classical Lax-Friedrichs scheme
for the shallow water equations, with edge fluxes.

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



float3 F(const float3 Q, const float g) {
    float3 F;

    F.x = Q.y;                              //hu
    F.y = Q.y*Q.y / Q.x + 0.5f*g*Q.x*Q.x;   //hu*hu/h + 0.5f*g*h*h;
    F.z = Q.y*Q.z / Q.x;                    //hu*hv/h;

    return F;
}

float3 G(const float3 Q, const float g) {
    float3 G;

    G.x = Q.z;                              //hv
    G.y = Q.y*Q.z / Q.x;                    //hu*hv/h;
    G.z = Q.z*Q.z / Q.x + 0.5f*g*Q.x*Q.x;   //hv*hv/h + 0.5f*g*h*h;

    return G;
}

__kernel void swe_2D(
        int nx_, int ny_,
        float dx_, float dy_, float dt_,
        float g_,
        
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
    const int ti = get_global_id(0) + 1; //Skip global ghost cells, i.e., +1
    const int tj = get_global_id(1) + 1;
    
    __local float h[block_height+2][block_width+2];
    __local float hu[block_height+2][block_width+2];
    __local float hv[block_height+2][block_width+2];
    
    
    //Read into shared memory
    for (int j=ty; j<block_height+2; j+=get_local_size(1)) {
        const int l = clamp(by + j, 0, ny_+1); // Out of bounds
        
        //Compute the pointer to current row in the arrays
        __global float* const h_row = (__global float*) ((__global char*) h0_ptr_ + h0_pitch_*l);
        __global float* const hu_row = (__global float*) ((__global char*) hu0_ptr_ + hu0_pitch_*l);
        __global float* const hv_row = (__global float*) ((__global char*) hv0_ptr_ + hv0_pitch_*l);
        
        for (int i=tx; i<block_width+2; i+=get_local_size(0)) {
            const int k = clamp(bx + i, 0, nx_+1); // Out of bounds
            
            h[j][i] = h_row[k];
            hu[j][i] = hu_row[k];
            hv[j][i] = hv_row[k];
        }
    }

    //Make sure all threads have read into shared mem
    __syncthreads();
    
    const int i = tx + 1; //Skip local ghost cells, i.e., +1
    const int j = ty + 1;
    
    //Fix boundary conditions
    if (ti == 1) {
        h[j][i-1]  =   h[j][i];
        hu[j][i-1] = -hu[j][i];
        hv[j][i-1] =  hv[j][i];
    }
    if (ti == nx_) {
        h[j][i+1]  =   h[j][i];
        hu[j][i+1] = -hu[j][i];
        hv[j][i+1] =  hv[j][i];
    }
    if (tj == 1) {
        h[j-1][i]  =   h[j][i];
        hu[j-1][i] =  hu[j][i];
        hv[j-1][i] = -hv[j][i];
    }
    if (tj == ny_) {
        h[j+1][i]  =   h[j][i];
        hu[j+1][i] =  hu[j][i];
        hv[j+1][i] = -hv[j][i];
    }
    
    __syncthreads();
    
    
    //Compute for all internal cells
    if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_+1) {
    
        const float3 Q_east  = (float3)(h[j][i+1], hu[j][i+1], hv[j][i+1]);
        const float3 Q_west  = (float3)(h[j][i-1], hu[j][i-1], hv[j][i-1]);
        const float3 Q_north = (float3)(h[j+1][i], hu[j+1][i], hv[j+1][i]);
        const float3 Q_south = (float3)(h[j-1][i], hu[j-1][i], hv[j-1][i]);

        const float3 F_east = F(Q_east, g_);
        const float3 F_west = F(Q_west, g_);
        const float3 G_north = G(Q_north, g_);
        const float3 G_south = G(Q_south, g_);

        float3 Q1 = 0.25f*(Q_east + Q_west + Q_north + Q_south)
            - dt_/(2.0f*dx_)*(F_east - F_west)
            - dt_/(2.0f*dy_)*(G_north - G_south);

        __global float* const h_row  = (__global float*) ((__global char*) h1_ptr_ + h1_pitch_*tj);
        __global float* const hu_row = (__global float*) ((__global char*) hu1_ptr_ + hu1_pitch_*tj);
        __global float* const hv_row = (__global float*) ((__global char*) hv1_ptr_ + hv1_pitch_*tj);
        
        h_row[ti] = Q1.x;
        hu_row[ti] = Q1.y;
        hv_row[ti] = Q1.z;
    }
}