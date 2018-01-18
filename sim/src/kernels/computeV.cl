#include "../computeV_types.h"
#include "../windStress_types.h"
//#include "../config.h"
#include "common.opencl"

#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#define CLK_LOCAL_MEM_FENCE
#endif

/**
  * Kernel that evolves V one step in time.
  */
__kernel void computeV(
		computeV_args args,
		windStress_args ws,
        //Data
        __global float* H_ptr, int H_pitch,
        __global float* U_ptr, int U_pitch,
        __global float* V_ptr, int V_pitch,
        __global float* eta_ptr, int eta_pitch,
		float t) {

    __local float H_shared[WGNY+1][WGNX];
    __local float U_shared[WGNY+1][WGNX+1];
    __local float eta_shared[WGNY+1][WGNX];

    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);

    //Index of block within domain
    const int bx = get_local_size(0) * get_group_id(0);
    const int by = get_local_size(1) * get_group_id(1);

    //Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    //Compute pointer to current row in the U array
    __global float* const V_row = (__global float*) ((__global char*) V_ptr + V_pitch*tj);

    //Read current V
    float V_current = 0.0f;
    if (ti < args.nx && tj < args.ny+1) {
        V_current = V_row[ti];
    }

    //Read H and eta into shared memory
    for (int j=ty; j<WGNY+1; j+=get_local_size(1)) {
        const int l = by + j - 1;

        //Compute the pointer to current row in the H and eta arrays
        __global float* const H_row = (__global float*) ((__global char*) H_ptr + H_pitch*l);
        __global float* const eta_row = (__global float*) ((__global char*) eta_ptr + eta_pitch*l);

        for (int i=tx; i<WGNY; i+=get_local_size(0)) {
            const int k = bx + i;
            if (k < args.nx && l >= 0 && l < args.ny+1) {
                H_shared[j][i] = H_row[k];
                eta_shared[j][i] = eta_row[k];
            }
            else {
                H_shared[j][i] = 0.0f;
                eta_shared[j][i] = 0.0f;
            }
        }
    }

    //Read U into shared memory
    for (int j=ty; j<WGNY+1; j+=get_local_size(1)) {
        const int l = by + j - 1;

        //Compute the pointer to current row in the V array
        __global float* const U_row = (__global float*) ((__global char*) U_ptr + U_pitch*l);

        for (int i=tx; i<WGNX+1; i+=get_local_size(0)) {
            const int k = bx + i;
            if (k < args.nx+1 && l >= 0 && l < args.ny) {
                U_shared[j][i] = U_row[k];
            }
            else {
                U_shared[j][i] = 0.0f;
            }
        }
    }

    //Make sure all threads have read into shared mem
    barrier(CLK_LOCAL_MEM_FENCE);

    //Reconstruct H at the V position
    float H_m = 0.5f*(H_shared[ty][tx] + H_shared[ty+1][tx]);

    //Reconstruct U at the V position
    float U_m;
    if (ti==0) {
        U_m = 0.5f*(U_shared[ty][tx+1] + U_shared[ty+1][tx+1]);
    }
    else if (ti==args.nx-1) {
        U_m = 0.5f*(U_shared[ty][tx] + U_shared[ty+1][tx]);
    }
    else {
        U_m = 0.25f*(U_shared[ty][tx] + U_shared[ty][tx+1]
                + U_shared[ty+1][tx] + U_shared[ty+1][tx+1]);
    }

    //Calculate the friction coefficient
    float B = H_m/(H_m + args.r*args.dt);

    //Calculate the gravitational effect
    float P = args.g*H_m*(eta_shared[ty][tx] - eta_shared[ty+1][tx])/args.dy;

    //Calculate the wind shear stress
    float Y = windStressY(
    	ws.wind_stress_type,
		args.dx, args.dy, args.dt,
		ws.tau0, ws.rho, ws.alpha, ws.xm, ws.Rc,
		ws.x0, ws.y0,
		ws.u0, ws.v0,
		t);

    //Compute the V at the next timestep
    float V_next = B*(V_current + args.dt*(-args.f*U_m + P + Y) );

    //Write to main memory
    if (ti < args.nx && tj < args.ny+1) {
        //Closed boundaries
        if (tj == 0) {
            V_next = 0.0f;
        }
        else if (tj == args.ny) {
            V_next = 0.0f;
        }

        V_row[ti] = V_next;
    }
}

/**
  * Python-wrapper.
  */
__kernel void computeVKernel(
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

        // Wind stress parameters
        int wind_stress_type,
        float tau0, float rho, float alpha, float xm, float Rc,
        float x0, float y0,
        float u0, float v0,
        float t) {

	computeV_args args;
	args.nx = nx;
	args.ny = ny;
	args.dt = dt;
	args.dx = dx;
	args.dy = dy;
	args.r = r;
	args.f = f;
	args.g = g;

	windStress_args ws;
	ws.wind_stress_type = wind_stress_type;
	ws.tau0 = tau0;
	ws.rho = rho;
	ws.alpha = alpha;
	ws.xm = xm;
	ws.Rc = Rc;
	ws.x0 = x0;
	ws.y0 = y0;
	ws.u0 = u0;
	ws.v0 = v0;

    computeV(args,
        ws,
        H_ptr, H_pitch,
        U_ptr, U_pitch,
        V_ptr, V_pitch,
        eta_ptr, eta_pitch,
        t);

}
