/*
This OpenCL kernel implements the Kurganov-Petrova numerical scheme 
for the shallow water equations, described in 
A. Kurganov & Guergana Petrova
A Second-Order Well-Balanced Positivity Preserving Central-Upwind
Scheme for the Saint-Venant System Communications in Mathematical
Sciences, 5 (2007), 133-160. 

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

#include "common.opencl"


// Use block_width as number of threads
#define NUM_THREADS block_width


/**
  * Reduces from elements number of values to a single dt. 
  */
__kernel void reduce_dt(__global float* data, int elements, float dt_scale, float dt_max)
{
    volatile __local float sdata[NUM_THREADS];
    
    const unsigned int tid = get_local_id(0);
    float dt = FLT_MAX;
    
    //Stride through global memory to find per-thread min
    for (unsigned int i=tid; i<elements; i+=NUM_THREADS) {
        dt = fmin(dt, data[i]);
    }
    
    //Save to shared memory
    sdata[tid] = dt;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //Local memory reduction to find min
    //First reduce to 64 elements with synchronization
    /********************************************************
      * WARNING: This assumes power of two number of threads
      *******************************************************/
    if (NUM_THREADS >= 512) {
        if (tid < 256) {
            sdata[tid] = fmin(sdata[tid], sdata[tid + 256]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (NUM_THREADS >= 256) {
        if (tid < 128) {
            sdata[tid] = fmin(sdata[tid], sdata[tid + 128]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (NUM_THREADS >= 128) {
        if (tid < 64) {
            sdata[tid] = fmin(sdata[tid], sdata[tid + 64]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#ifdef NVIDIA
    // Then use the 32-wide SIMD-execution to reduce from 64 to 1
    if (tid < 32) {
        if (NUM_THREADS >=  64) { 
            sdata[tid] = fmin(sdata[tid], sdata[tid + 32]);
        }
        if (NUM_THREADS >=  32) { 
            sdata[tid] = fmin(sdata[tid], sdata[tid + 16]);
        }
        if (NUM_THREADS >=  16) { 
            sdata[tid] = fmin(sdata[tid], sdata[tid + 8]);
        }
        if (NUM_THREADS >=  8) { 
            sdata[tid] = fmin(sdata[tid], sdata[tid + 4]);
        }
        if (NUM_THREADS >=  4) { 
            sdata[tid] = fmin(sdata[tid], sdata[tid + 2]);
        }
        if (NUM_THREADS >=  2) { 
            sdata[tid] = fmin(sdata[tid], sdata[tid + 1]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#else
    if (NUM_THREADS >=  64) {
        if (tid < 32) { 
            sdata[tid] = fmin(sdata[tid], sdata[tid + 32]); 
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (NUM_THREADS >=  32) { 
        if (tid < 16) {
            sdata[tid] = fmin(sdata[tid], sdata[tid + 16]); 
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (NUM_THREADS >=  16) { 
        if (tid < 8) {
            sdata[tid] = fmin(sdata[tid], sdata[tid + 8]);  
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (NUM_THREADS >=  8) { 
        if (tid < 4) {
            sdata[tid] = fmin(sdata[tid], sdata[tid + 4]);  
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (NUM_THREADS >=  4) { 
        if (tid < 2) {
            sdata[tid] = fmin(sdata[tid], sdata[tid + 2]);  
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (NUM_THREADS >=  2) { 
        if (tid < 1) {
            sdata[tid] = fmin(sdata[tid], sdata[tid + 1]);  
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
#endif
    //Write to global memory
    if (tid == 0) {
        dt = sdata[0];

        if (dt == FLT_MAX) {
            //If no water at all, and no sources, 
            //we really do not need to simulate, 
            //but using FLT_MAX will make things crash...
            dt = 1.0e-7f;
        }

        data[0] = fmin(dt * dt_scale, dt_max);
    }
}
