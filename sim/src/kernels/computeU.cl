#include "../computeU_types.h"
#include "../config.h"

#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define local
#define CLK_LOCAL_MEM_FENCE
#endif

/*
Computes U at the eastern cell edge (and at the western cell edge if this cell is next to a western ghost cell).
Input:
  - args.nx, args.ny: Internal grid size (nx, ny), i.e. not including grid points at the outside of ghost cells.
  - eta: Values defined at the center of each cell, including ghost cells. Size: (nx + 1, ny + 1).
  - V: Values defined at the center of the southern and northern edge of each grid cell, excluding the western and eastern
       ghost cells. Size: (nx - 1, ny + 2).
  - Hr_u: Values defined at the center of the western and eastern edge of each grid cell, excluding the western edge of the western ghost cells
          and the eastern edge of the eastern ghost cells. Size: (nx, ny + 1)
Output:
  - U: Values defined at the center of the western and eastern edge of each grid cell, excluding the southern and northern
       ghost cells. Size: (nx + 2, ny - 1).
*/
__kernel void computeU (
    __global const float *eta,
    __global const float *V,
    __global const float *Hr_u,
    __global float *U,
    computeU_args args)
{
    const int nx = args.nx;
    const int ny = args.ny;
    const float dt = args.dt;
    const float dx = args.dx;
    const float R = args.R;
    const float F = args.F;
    const float g = args.g;

    // work item within group
    const unsigned int lx = get_local_id(0);
    const unsigned int ly = get_local_id(1);

   	// work item within domain
    const unsigned int gx = get_global_id(0);
    const unsigned int gy = get_global_id(1);

    // local and global id (linearized index)
    const unsigned int lid = lx + get_local_size(0) * ly;
    /// XXX: Check sizes in gmem. We may want to change them to allow easier indexing
    const unsigned int gid = gx+1 + (nx+2) * gy; //U (+1 because we want the eastern interface)
    const unsigned int eta_gid = gx + (nx+1) * gy;
    const unsigned int v_gid = gx + (nx-1) * gy;
    const unsigned int hr_u_gid = gx + nx * gy;

    // allocate local work-group memory for Hr_u, eta, and V
    local float Hr_u_local[WGNX * WGNY];
    local float eta_local[(WGNX + 1) * WGNY];
    local float V_local[(WGNX + 1) * (WGNY + 1)];

    // copy Hr_u from global to local memory
    Hr_u_local[lid] = Hr_u[hr_u_gid];

    // copy eta from global to local memory
    eta_local[lx + ly * (WGNX + 1)] = eta[eta_gid];
    if(lx == WGNX-1) {
    	eta_local[lx + ly * (WGNX + 1) + 1] = eta[eta_gid + 1];
    }

    // copy V from global to local memory
    V_local[lx + ly * (WGNX + 1)] = V[v_gid];
    if(lx == WGNX-1) {
    	V_local[lx + ly * (WGNX + 1) + 1] = V[v_gid + 1];
    }
    if(ly == WGNY-1) {
        V_local[lx + ly * (WGNX + 1) + (WGNX + 1)] = V[v_gid + (nx - 1)];
    }
    if(lx == WGNX-1 && ly == WGNY-1) { // upper-right corner
    	V_local[lx + ly * (WGNX + 1) + (WGNX + 1) + 1] = V[v_gid + (nx - 1) + 1];
    }

    // ensure all work-items have copied their values to local memory before proceeding
    // assuming CLK_GLOBAL_MEM_FENCE is not necessary since the read happens before the write in each work-item
    barrier(CLK_LOCAL_MEM_FENCE);

    // reconstructing V at U-positions
    float Vr;
    if (gy == 0 || gy == ny-2) {
        Vr = 0.5f * (V_local[lx + ly * (WGNX + 1)] + V_local[lx + ly * (WGNX + 1) + 1]);
    } else {
    	Vr = 0.25f * (V_local[lx + ly * (WGNX + 1)] + V_local[lx + ly * (WGNX + 1) + (WGNX + 1)] +
    			V_local[lx + ly * (WGNX + 1) + 1] + V_local[lx + ly * (WGNX + 1) + (WGNX + 1) + 1]);
    }

    const float B = Hr_u_local[lid] / (Hr_u_local[lid] + R * dt);
    const float P = g * Hr_u_local[lid] *
    		(eta_local[lx + ly * (WGNX + 1)] - eta_local[lx + ly * (WGNX + 1) + 1]) / dx;

    if (gx < nx+2 && gy < ny-1) {
    	//U[gid] = gid;
    	//U[gid] = 0.0f;
    	U[gid] = B * (U[gid] + dt * (F * Vr + P));
    	/*if (gy == 0 || gy == ny-2) {
    		U[gid] = 13.0f;
    	}*/
    }
}
