#include "../computeV_types.h"
#include "../config.h"

#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define local
#define CLK_LOCAL_MEM_FENCE
#endif

/*
Computes V at the northern cell edge (and at the southern cell edge if this cell is next to a southern ghost cell).
Input:
  - args.nx, args.ny: Internal grid size (nx, ny), i.e. not including grid points at the outside of ghost cells.
  - eta: Values defined at the center of each cell, including ghost cells. Size: (nx + 1, ny + 1).
  - U: Values defined at the center of the western and eastern edge of each grid cell, excluding the southern and northern
     ghost cells. Size: (nx + 2, ny - 1).
  - Hr_v: Values defined at the center of the southern and northern edge of each grid cell, excluding the southern edge of the southern ghost cells
          and the northern edge of the northern ghost cells. Size: (nx + 1, ny)
Output:
  - V: Values defined at the center of the southern and northern edge of each grid cell, excluding the western and eastern
       ghost cells. Size: (nx - 1, ny + 2).
*/
__kernel void computeV (
    __global const float *eta,
    __global const float *U,
    __global float *V,
    __global float *Hr_v,
    computeV_args args)
{
    const int nx = args.nx;
    const int ny = args.ny;
    const float dt = args.dt;
    const float dy = args.dy;
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
    const unsigned int gid = gx + (nx-1) * (gy+1); //V (+1 because we want the northern interface)
	const unsigned int eta_gid = gx + (nx+1) * gy;
	const unsigned int u_gid = gx + (nx+2) * gy;
	const unsigned int hr_v_gid = gx + (nx+1) * (gy+1); //(+1 because we want the northern interface)

    // allocate local work-group memory for Hr_v, eta, and U
    local float Hr_v_local[WGNX * WGNY];
    local float eta_local[WGNX * (WGNY + 1)];
    local float U_local[(WGNX + 1) * (WGNY + 1)];

    // copy Hr_v from global to local memory
	Hr_v_local[lid] = Hr_v[hr_v_gid];

	// copy eta from global to local memory
	eta_local[lx + ly * WGNX] = eta[eta_gid];
	if(ly == WGNY-1) {
		eta_local[lx + ly * WGNX + WGNX] = eta[eta_gid + (nx+1)];
	}

	// copy U from global to local memory
	U_local[lx + ly * (WGNX + 1)] = U[u_gid];
	if(lx == WGNX-1) {
		U_local[lx + ly * (WGNX + 1) + 1] = U[u_gid+1];
	}
	if(ly == WGNY-1) {
		U_local[lx + ly * (WGNX + 1) + (WGNX + 1)] = U[u_gid + (nx + 2)];
	}
	if(lx == WGNX-1 && ly == WGNY-1) { // upper-right corner
		U_local[lx + ly * (WGNX + 1) + (WGNX + 1) + 1] = U[u_gid + (nx + 2) + 1];
	}
	
	// ensure all work-items have copied their values to local memory before proceeding
	// assuming CLK_GLOBAL_MEM_FENCE is not necessary since the read happens before the write in each work-item
    barrier(CLK_LOCAL_MEM_FENCE);

    // reconstructing U at V-positions
    float Ur;
    if (gx == 0) {
        Ur = 0.5f * (U_local[lx+1 + ly * (WGNX + 1)] + U_local[lx+1 + ly * (WGNX + 1) + (WGNX + 1)]); //(+1 we never use the outer U-values)
    } else if (gx == nx-2) {
        Ur = 0.5f * (U_local[lx-1 + ly * (WGNX + 1)] + U_local[lx-1 + ly * (WGNX + 1) + (WGNX + 1)]); //(-1 we never use the outer U-values)
    } else {
    	Ur = 0.25f * (U_local[lx + ly * (WGNX + 1)] + U_local[lx + ly * (WGNX + 1)+1] +
    			U_local[lx + ly * (WGNX + 1) + (WGNX + 1)] + U_local[lx + ly * (WGNX + 1) + 1 + (WGNX + 1)]);
    }

    const float B = Hr_v_local[lid] / (Hr_v_local[lid] + R * dt);
    const float P = g * Hr_v_local[lid] *
    		(eta_local[lx + ly * WGNX] - eta_local[lx + ly * WGNX + WGNX]) / dy;

    if (gx < nx-1 && gy < ny+2) {
    	V[gid] = B * (V[gid] + dt * (-F * Ur + P));
    }
}
