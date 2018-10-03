#ifndef COMMON_CL
#define COMMON_CL

#include "../config.h"
#include "../windStress_params.h"

#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#define CLK_LOCAL_MEM_FENCE
#endif

#define _180_OVER_PI 57.29578f
#define PI_OVER_180 0.01745329f

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



/**
  * Reads a block of data  with one ghost cell for the shallow water equations
  */
void readBlock1(__global float* h_ptr_, int h_pitch_,
                __global float* hu_ptr_, int hu_pitch_,
                __global float* hv_ptr_, int hv_pitch_,
                __local float Q[3][block_height+2][block_width+2], 
                const int nx_, const int ny_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of block within domain
    const int bx = get_local_size(0) * get_group_id(0);
    const int by = get_local_size(1) * get_group_id(1);
    
    //Read into shared memory
    for (int j=ty; j<block_height+2; j+=get_local_size(1)) {
        const int l = clamp(by + j, 0, ny_+1); // Out of bounds
        
        //Compute the pointer to current row in the arrays
        __global float* const h_row = (__global float*) ((__global char*) h_ptr_ + h_pitch_*l);
        __global float* const hu_row = (__global float*) ((__global char*) hu_ptr_ + hu_pitch_*l);
        __global float* const hv_row = (__global float*) ((__global char*) hv_ptr_ + hv_pitch_*l);
        
        for (int i=tx; i<block_width+2; i+=get_local_size(0)) {
            const int k = clamp(bx + i, 0, nx_+1); // Out of bounds
            
            Q[0][j][i] = h_row[k];
            Q[1][j][i] = hu_row[k];
            Q[2][j][i] = hv_row[k];
        }
    }
}





/**
  * Reads a block of data  with two ghost cells for the shallow water equations
  */
void readBlock2(__global float* h_ptr_, int h_pitch_,
                __global float* hu_ptr_, int hu_pitch_,
                __global float* hv_ptr_, int hv_pitch_,
                __local float Q[3][block_height+4][block_width+4], 
                const int nx_, const int ny_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of block within domain
    const int bx = get_local_size(0) * get_group_id(0);
    const int by = get_local_size(1) * get_group_id(1);
    
    //Read into shared memory
    for (int j=ty; j<block_height+4; j+=get_local_size(1)) {
        const int l = clamp(by + j, 0, ny_+3); // Out of bounds
        
        //Compute the pointer to current row in the arrays
        __global float* const h_row = (__global float*) ((__global char*) h_ptr_ + h_pitch_*l);
        __global float* const hu_row = (__global float*) ((__global char*) hu_ptr_ + hu_pitch_*l);
        __global float* const hv_row = (__global float*) ((__global char*) hv_ptr_ + hv_pitch_*l);
        
        for (int i=tx; i<block_width+4; i+=get_local_size(0)) {
            const int k = clamp(bx + i, 0, nx_+3); // Out of bounds
            
            Q[0][j][i] = h_row[k];
            Q[1][j][i] = hu_row[k];
            Q[2][j][i] = hv_row[k];
        }
    }
}

/**
  * Reads a block of data  with two ghost cells for the shallow water equations
  */
void readBlock2single(__global float* data_ptr_, int data_pitch_,
		      __local float shmem[block_height+4][block_width+4],
		      const int nx_, const int ny_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of block within domain
    const int bx = get_local_size(0) * get_group_id(0);
    const int by = get_local_size(1) * get_group_id(1);
    
    //Read into shared memory
    for (int j=ty; j<block_height+4; j+=get_local_size(1)) {
        const int l = clamp(by + j, 0, ny_+3); // Out of bounds
        
        //Compute the pointer to current row in the arrays
        __global float* const data_row = (__global float*) ((__global char*) data_ptr_ + data_pitch_*l);
        
        for (int i=tx; i<block_width+4; i+=get_local_size(0)) {
            const int k = clamp(bx + i, 0, nx_+3); // Out of bounds
	    shmem[j][i] = data_row[k];
        }
    }
}



/**
  * Writes a block of data to global memory for the shallow water equations.
  */
void writeBlock1(__global float* h_ptr_, int h_pitch_,
                 __global float* hu_ptr_, int hu_pitch_,
                 __global float* hv_ptr_, int hv_pitch_,
                 __local float Q[3][block_height+2][block_width+2],
                 const int nx_, const int ny_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of cell within domain
    const int ti = get_global_id(0) + 1; //Skip global ghost cells, i.e., +1
    const int tj = get_global_id(1) + 1;
    
    //Only write internal cells
    if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_+1) {
        const int i = tx + 1; //Skip local ghost cells, i.e., +1
        const int j = ty + 1;

        __global float* const h_row  = (__global float*) ((__global char*) h_ptr_ + h_pitch_*tj);
        __global float* const hu_row = (__global float*) ((__global char*) hu_ptr_ + hu_pitch_*tj);
        __global float* const hv_row = (__global float*) ((__global char*) hv_ptr_ + hv_pitch_*tj);
        
        h_row[ti]  = Q[0][j][i];
        hu_row[ti] = Q[1][j][i];
        hv_row[ti] = Q[2][j][i];
    }
}





/**
  * Writes a block of data to global memory for the shallow water equations.
  */
void writeBlock2(__global float* h_ptr_, int h_pitch_,
                 __global float* hu_ptr_, int hu_pitch_,
                 __global float* hv_ptr_, int hv_pitch_,
                 __local float Q[3][block_height+4][block_width+4], 
                 const int nx_, const int ny_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of cell within domain
    const int ti = get_global_id(0) + 2; //Skip global ghost cells, i.e., +2
    const int tj = get_global_id(1) + 2;
    
    //Only write internal cells
    if (ti > 1 && ti < nx_+2 && tj > 1 && tj < ny_+2) {
        const int i = tx + 2; //Skip local ghost cells, i.e., +2
        const int j = ty + 2;

        __global float* const h_row  = (__global float*) ((__global char*) h_ptr_ + h_pitch_*tj);
        __global float* const hu_row = (__global float*) ((__global char*) hu_ptr_ + hu_pitch_*tj);
        __global float* const hv_row = (__global float*) ((__global char*) hv_ptr_ + hv_pitch_*tj);
        
        h_row[ti]  = Q[0][j][i];
        hu_row[ti] = Q[1][j][i];
        hv_row[ti] = Q[2][j][i];
    }
}






/**
  * No flow boundary conditions for the shallow water equations
  * with one ghost cell in each direction
  */
void noFlowBoundary1(__local float Q[3][block_height+2][block_width+2], const int nx_, const int ny_) {
    //Global index
    const int ti = get_global_id(0) + 1; //Skip global ghost cells, i.e., +1
    const int tj = get_global_id(1) + 1;
    
    //Block-local indices
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    const int i = tx + 1; //Skip local ghost cells, i.e., +1
    const int j = ty + 1;
    
    //Fix boundary conditions
    if (ti == 1) {
        Q[0][j][i-1] =  Q[0][j][i];
        Q[1][j][i-1] = -Q[1][j][i];
        Q[2][j][i-1] =  Q[2][j][i];
    }
    if (ti == nx_) {
        Q[0][j][i+1] =  Q[0][j][i];
        Q[1][j][i+1] = -Q[1][j][i];
        Q[2][j][i+1] =  Q[2][j][i];
    }
    if (tj == 1) {
        Q[0][j-1][i] =  Q[0][j][i];
        Q[1][j-1][i] =  Q[1][j][i];
        Q[2][j-1][i] = -Q[2][j][i];
    }
    if (tj == ny_) {
        Q[0][j+1][i] =  Q[0][j][i];
        Q[1][j+1][i] =  Q[1][j][i];
        Q[2][j+1][i] = -Q[2][j][i];
    }
}




/**
  * No flow boundary conditions for the shallow water equations
  * with two ghost cells in each direction
  *
  * BC values are defined as follows: 
  * 1: Wall boundary condition
  * 2: Periodic boundary condition
  * 3: Open boundary (numerical sponge)
  */
void noFlowBoundary2Mix(__local float Q[3][block_height+4][block_width+4],
			const int nx_, const int ny_,
			const int bc_north_, const int bc_east_,
			const int bc_south_, const int bc_west_) {
    
    //Global index
    const int ti = get_global_id(0) + 2; //Skip global ghost cells, i.e., +2
    const int tj = get_global_id(1) + 2;
    
    //Block-local indices
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    const int i = tx + 2; //Skip local ghost cells, i.e., +2
    const int j = ty + 2;
    
    if (ti == 2 && bc_west_ == 1) {
	// Wall boundary on west
	Q[0][j][i-1] =  Q[0][j][i];
	Q[1][j][i-1] = -Q[1][j][i];
	Q[2][j][i-1] =  Q[2][j][i];
        
	Q[0][j][i-2] =  Q[0][j][i+1];
	Q[1][j][i-2] = -Q[1][j][i+1];
	Q[2][j][i-2] =  Q[2][j][i+1];
    }
    if (ti == nx_+1 && bc_east_ == 1) {
	// Wall boundary on east
	Q[0][j][i+1] =  Q[0][j][i];
	Q[1][j][i+1] = -Q[1][j][i];
	Q[2][j][i+1] =  Q[2][j][i];
        
	Q[0][j][i+2] =  Q[0][j][i-1];
	Q[1][j][i+2] = -Q[1][j][i-1];
	Q[2][j][i+2] =  Q[2][j][i-1];
    }
    if (tj == 2 && bc_south_ == 1) {
	// Wall boundary on south
	Q[0][j-1][i] =  Q[0][j][i];
	Q[1][j-1][i] =  Q[1][j][i];
	Q[2][j-1][i] = -Q[2][j][i];
        
	Q[0][j-2][i] =  Q[0][j+1][i];
	Q[1][j-2][i] =  Q[1][j+1][i];
	Q[2][j-2][i] = -Q[2][j+1][i];
    }
    if (tj == ny_+1 && bc_north_ == 1) {
	// Wall boundary on north
	Q[0][j+1][i] =  Q[0][j][i];
	Q[1][j+1][i] =  Q[1][j][i];
	Q[2][j+1][i] = -Q[2][j][i];
        
	Q[0][j+2][i] =  Q[0][j-1][i];
	Q[1][j+2][i] =  Q[1][j-1][i];
	Q[2][j+2][i] = -Q[2][j-1][i];
    }
}


/**
  * No flow boundary conditions for the shallow water equations
  * with two ghost cells in each direction
  */
void noFlowBoundary2(__local float Q[3][block_height+4][block_width+4], const int nx_, const int ny_, const int boundary_conditions_type_) {
    if (boundary_conditions_type_ == 2) {
	return;
    }
    int bc_north = 1;
    int bc_east = 1;
    int bc_south = 1;
    int bc_west = 1;
    if (boundary_conditions_type_ == 3) {
	bc_north = 2;
	bc_south = 2;
    }
    else if (boundary_conditions_type_ == 4) {
	bc_east = 2;
	bc_west = 2;
    }

    noFlowBoundary2Mix(Q, nx_, ny_, bc_north, bc_east, bc_south, bc_west);
}




/**
  * Evolves the solution in time along the x axis (dimensional splitting)
  */
void evolveF1(__local float Q[3][block_height+2][block_width+2],
              __local float F[3][block_height+1][block_width+1],
              const int nx_, const int ny_,
              const float dx_, const float dt_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of cell within domain
    const int ti = get_global_id(0) + 1; //Skip global ghost cells, i.e., +1
    const int tj = get_global_id(1) + 1;
    
    if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_+1) {
        const int i = tx + 1; //Skip local ghost cells, i.e., +1
        const int j = ty + 1;
        
        Q[0][j][i] = Q[0][j][i] + (F[0][ty][tx] - F[0][ty][tx+1]) * dt_ / dx_;
        Q[1][j][i] = Q[1][j][i] + (F[1][ty][tx] - F[1][ty][tx+1]) * dt_ / dx_;
        Q[2][j][i] = Q[2][j][i] + (F[2][ty][tx] - F[2][ty][tx+1]) * dt_ / dx_;
    }
}






/**
  * Evolves the solution in time along the x axis (dimensional splitting)
  */
void evolveF2(__local float Q[3][block_height+4][block_width+4],
              __local float F[3][block_height+1][block_width+1],
              const int nx_, const int ny_,
              const float dx_, const float dt_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of cell within domain
    const int ti = get_global_id(0) + 2; //Skip global ghost cells, i.e., +2
    const int tj = get_global_id(1) + 2;
    
    if (ti > 1 && ti < nx_+2 && tj > 1 && tj < ny_+2) {
        const int i = tx + 2; //Skip local ghost cells, i.e., +1
        const int j = ty + 2;
        
        Q[0][j][i] = Q[0][j][i] + (F[0][ty][tx] - F[0][ty][tx+1]) * dt_ / dx_;
        Q[1][j][i] = Q[1][j][i] + (F[1][ty][tx] - F[1][ty][tx+1]) * dt_ / dx_;
        Q[2][j][i] = Q[2][j][i] + (F[2][ty][tx] - F[2][ty][tx+1]) * dt_ / dx_;
    }
}






/**
  * Evolves the solution in time along the y axis (dimensional splitting)
  */
void evolveG1(__local float Q[3][block_height+2][block_width+2],
              __local float G[3][block_height+1][block_width+1],
              const int nx_, const int ny_,
              const float dy_, const float dt_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of cell within domain
    const int ti = get_global_id(0) + 1; //Skip global ghost cells, i.e., +1
    const int tj = get_global_id(1) + 1;
    
    if (ti > 0 && ti < nx_+1 && tj > 0 && tj < ny_+1) {
        const int i = tx + 1; //Skip local ghost cells, i.e., +1
        const int j = ty + 1;
        
        Q[0][j][i] = Q[0][j][i] + (G[0][ty][tx] - G[0][ty+1][tx]) * dt_ / dy_;
        Q[1][j][i] = Q[1][j][i] + (G[1][ty][tx] - G[1][ty+1][tx]) * dt_ / dy_;
        Q[2][j][i] = Q[2][j][i] + (G[2][ty][tx] - G[2][ty+1][tx]) * dt_ / dy_;
    }
}







/**
  * Evolves the solution in time along the y axis (dimensional splitting)
  */
void evolveG2(__local float Q[3][block_height+4][block_width+4],
              __local float G[3][block_height+1][block_width+1],
              const int nx_, const int ny_,
              const float dy_, const float dt_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of cell within domain
    const int ti = get_global_id(0) + 2; //Skip global ghost cells, i.e., +2
    const int tj = get_global_id(1) + 2;
    
    if (ti > 1 && ti < nx_+2 && tj > 1 && tj < ny_+2) {
        const int i = tx + 2; //Skip local ghost cells, i.e., +2
        const int j = ty + 2;
        
        Q[0][j][i] = Q[0][j][i] + (G[0][ty][tx] - G[0][ty+1][tx]) * dt_ / dy_;
        Q[1][j][i] = Q[1][j][i] + (G[1][ty][tx] - G[1][ty+1][tx]) * dt_ / dy_;
        Q[2][j][i] = Q[2][j][i] + (G[2][ty][tx] - G[2][ty+1][tx]) * dt_ / dy_;
    }
}










/**
  * Reconstructs a slope using the minmod limiter based on three 
  * consecutive values
  */
float minmodSlope(float left, float center, float right, float theta) {
    const float backward = (center - left) * theta;
    const float central = (right - left) * 0.5f;
    const float forward = (right - center) * theta;
    
	return 0.25f
		*copysign(1.0f, backward)
		*(copysign(1.0f, backward) + copysign(1.0f, central))
		*(copysign(1.0f, central) + copysign(1.0f, forward))
		*min( min(fabs(backward), fabs(central)), fabs(forward) );
}

float minmodRaw(float backward, float central, float forward) {

    return 0.25f
	*copysign(1.0f, backward)
	*(copysign(1.0f, backward) + copysign(1.0f, central))
	*(copysign(1.0f, central) + copysign(1.0f, forward))
	*min( min(fabs(backward), fabs(central)), fabs(forward) );
}


/**
  * Reconstructs a minmod slope for a whole block along x
  */
void minmodSlopeX(__local float  Q[3][block_height+4][block_width+4],
                  __local float Qx[3][block_height+2][block_width+2],
                  const float theta_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Reconstruct slopes along x axis
    for (int j=ty; j<block_height; j+=get_local_size(1)) {
        const int l = j + 2; //Skip ghost cells
        for (int i=tx; i<block_width+2; i+=get_local_size(0)) {
            const int k = i + 1;
            for (int p=0; p<3; ++p) {
                Qx[p][j][i] = 0.5f * minmodSlope(Q[p][l][k-1], Q[p][l][k], Q[p][l][k+1], theta_);
            }
        }
    }
}



/**
  * Reconstructs a minmod slope for a whole block along y
  */
void minmodSlopeY(__local float  Q[3][block_height+4][block_width+4],
                  __local float Qy[3][block_height+2][block_width+2],
                  const float theta_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    for (int j=ty; j<block_height+2; j+=get_local_size(1)) {
        const int l = j + 1;
        for (int i=tx; i<block_width; i+=get_local_size(0)) {            
            const int k = i + 2; //Skip ghost cells
            for (int p=0; p<3; ++p) {
                Qy[p][j][i] = 0.5f * minmodSlope(Q[p][l-1][k], Q[p][l][k], Q[p][l+1][k], theta_);
            }
        }
    }
}

/*
 * Compute x-component of wind vector.
 * @param wind_speed Wind speed in m/s.
 * @param wind_direction Wind direction in degrees (clockwise, 0 being wind blowing from north towards south).
 */
float wind_u(float wind_speed, float wind_direction) {
	return -wind_speed * sin(wind_direction * PI_OVER_180);
}

/*
 * Compute y-component of wind vector.
 * @param wind_speed Wind speed in m/s.
 * @param wind_direction Wind direction in degrees (clockwise, 0 being wind blowing from north towards south).
 */
float wind_v(float wind_speed, float wind_direction) {
	return -wind_speed * cos(wind_direction * PI_OVER_180);
}

float windStressX(float wind_stress_, float dx_, float dy_, float dt, float t_) {
    /*******************************************************************************
    *******************************************************************************
    *******************************************************************************
    WARNING: Returning 0.0
    *******************************************************************************
    *******************************************************************************
    *******************************************************************************/
    return 0.0f;
}

float windStressY(float wind_stress_t, float dx_, float dy_, float dt, float t_) {
    /*******************************************************************************
    *******************************************************************************
    *******************************************************************************
    WARNING: Returning 0.0
    *******************************************************************************
    *******************************************************************************
    *******************************************************************************/
    return 0.0f;
}





float3 F_func(const float3 Q, const float g) {
    float3 F;

    F.x = Q.y;                              //hu
    F.y = Q.y*Q.y / Q.x + 0.5f*g*Q.x*Q.x;   //hu*hu/h + 0.5f*g*h*h;
    F.z = Q.y*Q.z / Q.x;                    //hu*hv/h;

    return F;
}


/**
  * Central upwind flux function
  * Takes Q = [h, hu, hv] as input, not [w, hu, hv].
  */
float3 CentralUpwindFlux(const float3 Qm, float3 Qp, const float g) {
    const float3 Fp = F_func(Qp, g);
    const float up = Qp.y / Qp.x;   // hu / h
    const float cp = sqrt(g*Qp.x); // sqrt(g*h)

    const float3 Fm = F_func(Qm, g);
    const float um = Qm.y / Qm.x;   // hu / h
    const float cm = sqrt(g*Qm.x); // sqrt(g*h)
    
    const float am = min(min(um-cm, up-cp), 0.0f); // largest negative wave speed
    const float ap = max(max(um+cm, up+cp), 0.0f); // largest positive wave speed
    
    return ((ap*Fm - am*Fp) + ap*am*(Qp-Qm))/(ap-am);
}


float3 F_func_bottom(const float3 Q, const float h, const float u, const float g) {
    float3 F;

    F.x = Q.y;                       //hu
    F.y = Q.y*u + 0.5f*g*(h*h);      //hu*u + 0.5f*g*h*h;
    F.z = Q.z*u;                     //hv*u;

    return F;
}

/**
  * Central upwind flux function
  * Takes Q = [eta, hu, hv] as input
  */
float3 CentralUpwindFluxBottom(const float3 Qm, float3 Qp, const float H, const float g) {
    const float hp = Qp.x + H;  // h = eta + H
    const float up = Qp.y / (float) hp; // hu/h
    const float3 Fp = F_func_bottom(Qp, hp, up, g);
    const float cp = sqrt(g*hp); // sqrt(g*h)

    const float hm = Qm.x + H;
    const float um = Qm.y / (float) hm;   // hu / h
    const float3 Fm = F_func_bottom(Qm, hm, um, g);
    const float cm = sqrt(g*hm); // sqrt(g*h)
    
    const float am = min(min(um-cm, up-cp), 0.0f); // largest negative wave speed
    const float ap = max(max(um+cm, up+cp), 0.0f); // largest positive wave speed
    // Related to dry zones
    // The constant is a compiler constant in the CUDA code.
    const float KPSIMULATOR_FLUX_SLOPE_EPS = 1.0e-4f;
    if ( fabs(ap - am) < KPSIMULATOR_FLUX_SLOPE_EPS ) {
	return (float3)(0.0f, 0.0f, 0.0f);
    }
    
    return ((ap*Fm - am*Fp) + ap*am*(Qp-Qm))/(ap-am);
}


/**
  *  Source terms related to bathymetry  
  */
float bottomSourceTerm2(__local float   Q[3][block_height+4][block_width+4],
			__local float  Qx[3][block_height+2][block_width+2],
			__local float RHx[block_height+4][block_width+4],
			const float g, 
			const int p, const int q) {
    // Compansating for the smaller shmem for Qx relative to Q:
    const int pQx = p - 1;
    const int qQx = q - 2;
    
    const float hp = Q[0][q][p] + Qx[0][qQx][pQx];
    const float hm = Q[0][q][p] - Qx[0][qQx][pQx];
    // g (w - B)*B_x -> KP07 equations (3.15) and (3.16)
    // With eta: g (eta + H)*(-H_x)
    return -0.5f*g*(RHx[q][p+1] - RHx[q][p])*(hp + RHx[q][p+1] + hm + RHx[q][p]);
}

float bottomSourceTerm3(__local float   Q[3][block_height+4][block_width+4],
			__local float  Qy[3][block_height+2][block_width+2],
			__local float RHy[block_height+4][block_width+4],
			const float g, 
			const int p, const int q) {
    // Compansating for the smaller shmem for Qy relative to Q:
    const int pQy = p - 2;
    const int qQy = q - 1;
    
    const float hp = Q[0][q][p] + Qy[0][qQy][pQy];
    const float hm = Q[0][q][p] - Qy[0][qQy][pQy];
    return -0.5f*g*(RHy[q+1][p] - RHy[q][p])*(hp + RHy[q+1][p] + hm + RHy[q][p]);
}




#endif // COMMON_CL
