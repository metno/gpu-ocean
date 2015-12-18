#include "../computeEta_types.h"
#include "../config.h"

#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define local
#define CLK_LOCAL_MEM_FENCE
#endif

/*
Computes Eta at the center of the (non-ghost) cell. Eta is the sea surface deviation away from the equilibrium depth (H).
Input:
  - args.nx, args.ny: Internal grid size (nx, ny), i.e. not including grid points at the outside of ghost cells.
  - U: Depth averaged velocity in the x-direction. Defined at western and eastern cell edges, excluding those of the southern
       and northern ghost cells. Size: (nx + 2, ny - 1).
  - V: Depth averaged velocity in the y-direction. Defined at southern and northern cell edges, excluding those of the western
       and eastern ghost cells. Size: (nx - 1, ny + 2).
Input+output:
  - eta: Values defined at the center of each cell, including ghost cells. Size: (nx + 1, ny + 1).
*/
__kernel void computeEta (
    __global const float *U,
    __global const float *V,
    __global float *eta,
    computeEta_args args)
{
    const int nx = args.nx;
    const int ny = args.ny;

    // work item within group
	const unsigned int lx = get_local_id(0);
	const unsigned int ly = get_local_id(1);

	// work item within domain
	const unsigned int gx = get_global_id(0);
	const unsigned int gy = get_global_id(1);

	// local and global id (linearized index)
	const unsigned int lid = lx + get_local_size(0) * ly;
	const unsigned int gid = gx + (nx+1) * gy;

    // allocate local work-group memory for U and V
    local float U_local[(WGNX + 1) * (WGNY + 1)];
    local float V_local[(WGNX + 1) * (WGNY + 1)];

    // copy U from global to local memory
    U_local[lid] = U[gid];
	if(lx == WGNX-1) {
		U_local[lid+1] = U[gid+1];
	}
	if(ly == WGNY-1) {
		U_local[lid + (WGNX + 1)] = U[gid + (nx + 1)];
	}

    // copy V from global to local memory
	V_local[lid] = V[gid];
	if(lx == WGNX-1) {
		V_local[lid+1] = V[gid+1];
	}
	if(ly == WGNY-1) {
		V_local[lid + (WGNX + 1)] = V[gid + (nx + 1)];
	}

    // ensure all work-items have copied their values to local memory before proceeding
    barrier(CLK_LOCAL_MEM_FENCE); // assuming CLK_GLOBAL_MEM_FENCE is not necessary since the read happens before the write in each work-item

    if (gx < nx+1 && gy < ny+1) {
    	//eta[gid] = gid;
    	eta[gid] = eta[gid] - args.dt / args.dx * (U_local[lid + 1] - U_local[lid])
                        	  - args.dt / args.dy * (V_local[lid] - V_local[lid + (WGNX + 1)]);
    }
}
