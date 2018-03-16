 /*
This OpenCL kernel implements part of the Forward Backward Linear 
numerical scheme for the shallow water equations, described in 
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

#include "../config.h"

#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#define CLK_LOCAL_MEM_FENCE
#endif



/**
  *  Generates two uniform random numbers based on the ANSIC Linear Congruential 
  *  Generator.
  */
float2 ansic_lcg(float* seed_ptr) {
    long seed = (*seed_ptr);

    seed = ((seed * 1103515245) + 12345) % 0x7fffffff;
    float u1 = seed / 2147483648.0f;

    seed = ((seed * 1103515245) + 12345) % 0x7fffffff;
    float u2 = seed / 2147483648.0f;

    (*seed_ptr) = seed;
    return (float2)(u1, u2);
}

/**
  *  Generates two random numbers, drawn from a normal distribtion with mean 0 and
  *  variance 1. Based on the Box Muller transform.
  */
float2 boxMuller(float* seed) {
    float2 u = ansic_lcg(seed);
    
    float r = sqrt(-2.0f*log(u.x));
    float n1 = r*cospi(2*u.y);
    float n2 = r*sinpi(2*u.y);
    
    return (float2)(n1, n2);
}

/**
  * Kernel that generates uniform random numbers.
  */
__kernel void uniformDistribtion(
        // Size of data
        int seed_nx_, int seed_ny_,
	int random_nx_, 
	
        //Data
        __global float* seed_ptr_, int seed_pitch_,
        __global float* random_ptr_, int random_pitch_
    ) {

    //Index of cell within domain
    const int ti = get_global_id(0); 
    const int tj = get_global_id(1);

    // Each thread computes and writes two uniform numbers.

    if ((ti < seed_nx_) && (tj < seed_ny_)) {
    
	//Compute pointer to current row in the U array
	__global float* const seed_row = (__global float*) ((__global char*) seed_ptr_ + seed_pitch_*tj);
	__global float* const random_row = (__global float*) ((__global char*) random_ptr_ + random_pitch_*tj);
	
	float seed = seed_row[ti];
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


