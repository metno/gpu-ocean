#include "../computeEta_types.h"
#include "../config.h"

#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#define CLK_LOCAL_MEM_FENCE
#endif

/**
  * Kernel that evolves eta one step in time.
  */
__kernel void computeEta(
		computeEta_args args,
        //Data
        __global float* U_ptr, int U_pitch,
        __global float* V_ptr, int V_pitch,
        __global float* eta_ptr, int eta_pitch) {

    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);

    //Index of block within domain
    const int bx = get_local_size(0) * get_group_id(0);
    const int by = get_local_size(1) * get_group_id(1);

    //Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    __local float U_shared[WGNY][WGNX+1];
    __local float V_shared[WGNY+1][WGNX];

    //Compute pointer to current row in the U array
    __global float* const eta_row = (__global float*) ((__global char*) eta_ptr + eta_pitch*tj);

    //Read current eta
    float eta_current = 0.0f;
    if (ti < args.nx && tj < args.ny) {
        eta_current = eta_row[ti];
    }

    //Read U into shared memory
    for (int j=ty; j<WGNY; j+=get_local_size(1)) {
        const unsigned int l = by + j;

        //Compute the pointer to current row in the V array
        __global float* const U_row = (__global float*) ((__global char*) U_ptr + U_pitch*l);

        for (int i=tx; i<WGNX+1; i+=get_local_size(0)) {
            const unsigned int k = bx + i;
            if (k < args.nx+1 && l < args.ny) {
                U_shared[j][i] = U_row[k];
            }
            else {
                U_shared[j][i] = 0.0f;
            }
        }
    }

    //Read V into shared memory
    for (int j=ty; j<WGNY+1; j+=get_local_size(1)) {
        const unsigned int l = by + j;
        //Compute the pointer to current row in the V array
        __global float* const V_row = (__global float*) ((__global char*) V_ptr + V_pitch*l);
        for (int i=tx; i<WGNX; i+=get_local_size(0)) {
            const unsigned int k = bx + i;
            if (k < args.nx && l < args.ny+1) {
                V_shared[j][i] = V_row[k];
            }
            else {
                V_shared[j][i] = 0.0f;
            }
        }
    }

    //Make sure all threads have read into shared mem
    barrier(CLK_LOCAL_MEM_FENCE);

    //Compute the eta at the next timestep
    float eta_next = eta_current - args.dt/args.dx * (U_shared[ty][tx+1] - U_shared[ty][tx])
                                 - args.dt/args.dy * (V_shared[ty+1][tx] - V_shared[ty][tx]);

    //Write to main memory
    if (ti < args.nx && tj < args.ny) {
        eta_row[ti] = eta_next;
    }
}
