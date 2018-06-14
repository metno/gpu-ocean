#include "../computeU_types.h"
#include "../config.h"
#include "common.opencl"

#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#define CLK_LOCAL_MEM_FENCE
#endif

/**
  * Kernel that evolves U one step in time.
  */
__kernel void computeU(
		computeU_args args,
		__global const wind_stress_params* ws,
        //Data
        __global float* H_ptr, int H_pitch,
        __global float* U_ptr, int U_pitch,
        __global float* V_ptr, int V_pitch,
        __global float* eta_ptr, int eta_pitch,
		float t) {

    __local float H_shared[WGNY][WGNX+1];
    __local float V_shared[WGNY+1][WGNX+1];
    __local float eta_shared[WGNY][WGNX+1];

    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);

    //Index of block within domain
    const int bx = get_local_size(0) * get_group_id(0);
    const int by = get_local_size(1) * get_group_id(1);

    //Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    //Compute pointer to row "tj" in the U array
    __global float* const U_row = (__global float*) ((__global char*) U_ptr + U_pitch*tj);

    //Read current U
    float U_current = 0.0f;
    if (ti < args.nx + 1 && tj < args.ny) {
        U_current = U_row[ti];
    }

    //Read H and eta into local memory
    for (int j=ty; j<WGNY; j+=get_local_size(1)) {
        const int l = by + j;

        //Compute the pointer to row "l" in the H and eta arrays
        __global float* const H_row = (__global float*) ((__global char*) H_ptr + H_pitch*l);
        __global float* const eta_row = (__global float*) ((__global char*) eta_ptr + eta_pitch*l);

        for (int i=tx; i<WGNX+1; i+=get_local_size(0)) {
            const int k = bx + i - 1;

            if (k >= 0 && k < args.nx && l < args.ny+1) {
                H_shared[j][i] = H_row[k];
                eta_shared[j][i] = eta_row[k];
            }
            else {
                H_shared[j][i] = 0.0f;
                eta_shared[j][i] = 0.0f;
            }
        }
    }

    //Read V into shared memory
    for (int j=ty; j<WGNY+1; j+=get_local_size(1)) {
        const int l = by + j;

        //Compute the pointer to current row in the V array
        __global float* const V_row = (__global float*) ((__global char*) V_ptr + V_pitch*l);

        for (int i=tx; i<WGNX+1; i+=get_local_size(0)) {
            const int k = bx + i - 1;

            if (k >= 0 && k < args.nx && l < args.ny+1) {
                V_shared[j][i] = V_row[k];
            }
            else {
                V_shared[j][i] = 0.0f;
            }
        }
    }

    //Make sure all threads have read into shared mem
    barrier(CLK_LOCAL_MEM_FENCE);

    //Reconstruct H at the U position
    float H_m = 0.5f*(H_shared[ty][tx] + H_shared[ty][tx+1]);

    //Reconstruct V at the U position
    float V_m = 0.0f;
    if (tj==0) {
        V_m = 0.5f*(V_shared[ty+1][tx] + V_shared[ty+1][tx+1]);
    }
    else if (tj==args.ny-1) {
        V_m = 0.5f*(V_shared[ty][tx] + V_shared[ty][tx+1]);
    }
    else {
        V_m = 0.25f*(V_shared[ty][tx] + V_shared[ty][tx+1]
                + V_shared[ty+1][tx] + V_shared[ty+1][tx+1]);
    }

    //Calculate the friction coefficient
    float B = H_m/(H_m + args.r*args.dt);

    //Calculate the gravitational effect
    float P = args.g*H_m*(eta_shared[ty][tx] - eta_shared[ty][tx+1])/args.dx;

    //Calculate the wind shear stress
    float X = windStressX(ws, args.dx, args.dy, args.dt, t);

    //Compute the U at the next timestep
    float U_next = B*(U_current + args.dt*(args.f*V_m + P + X) );

    //Write to main memory for internal cells
    if (ti < args.nx+1 && tj < args.ny) {
        //Closed boundaries
        if (ti == 0 || ti == args.nx) {
            U_next = 0.0f;
        }
        U_row[ti] = U_next;
    }
}

/**
  * Python-wrapper.
  */
__kernel void computeUKernel(
        //Discretization parameters
        int nx, int ny,
        float dx, float dy, float dt,

        //Physical parameters
        float g, //< Gravitational constant
        float f, //< Coriolis coefficient
        float r, //< Bottom friction coefficient

        //Data
        __global float* H_ptr, int H_pitch,
        __global float* U_ptr, int U_pitch,
        __global float* V_ptr, int V_pitch,
        __global float* eta_ptr, int eta_pitch,

        __global const wind_stress_params* ws,
        float t) {

	computeU_args args;
	args.nx = nx;
	args.ny = ny;
	args.dt = dt;
	args.dx = dx;
	args.dy = dy;
	args.r = r;
	args.f = f;
	args.g = g;

    computeU(args,
        ws,
        H_ptr, H_pitch,
        U_ptr, U_pitch,
        V_ptr, V_pitch,
        eta_ptr, eta_pitch,
        t);

}
