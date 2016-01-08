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
	/// XXX: Check sizes in gmem. We may want to change them to allow easier indexing
	const unsigned int gid = gx + (nx+1) * gy; // eta
	const unsigned int u_gid = gx + (nx+2) * gy;
	const unsigned int v_gid = gx + (nx-1) * gy;

    // allocate local work-group memory for U and V
	local float eta_local[WGNX * WGNY];
    local float U_local[(WGNX + 1) * WGNY];
    local float V_local[WGNX * (WGNY + 1)];

    // copy eta from global to local memory
	eta_local[lx + ly * WGNX] = eta[gid];

    // copy U and V from global to local memory
    U_local[lx + ly * (WGNX + 1)] = U[u_gid];
    V_local[lx + ly * WGNX] = V[v_gid];
	if(lx == WGNX-1) {
		U_local[lx + ly * (WGNX + 1) + 1] = U[u_gid+1];
	}
	if(ly == WGNY-1) {
		V_local[lx + ly * WGNX + WGNX] = V[v_gid + (nx - 1)];
	}

	// ensure all work-items have copied their values to local memory before proceeding
	// assuming CLK_GLOBAL_MEM_FENCE is not necessary since the read happens before the write in each work-item
    barrier(CLK_LOCAL_MEM_FENCE);

    //DEBUG
    /*int n, e, s, w;

    if (lx < WGNX && ly < WGNY && lx > 0 && ly > 0) {
    	n = eta_local[lx + ly * (WGNX + 1) + (WGNX + 1)]; // (lx+1) and (ly+1) due to apron
    	e = eta_local[lx + ly * (WGNX + 1) + 1];
    	s = eta_local[lx + ly * (WGNX + 1) - (WGNX + 1)];
    	w = eta_local[lx + ly * (WGNX + 1) - 1];
    }*/

    if (gx < nx+1 && gy < ny-1) {//ny+1) { <--- WHY??!?!1
    	//eta[gid] = gid;
    	//eta[gid] = 0.0f;
    	//if(n || e || s || w)
    		//eta[gid] = 1.0f;
    	eta[gid] = eta[gid] - args.dt / args.dx * (U_local[lx + ly * (WGNX + 1) + 1] - U_local[lx + ly * (WGNX + 1)])
                        	  - args.dt / args.dy * (V_local[lx + ly * WGNX + WGNX] - V_local[lx + ly * WGNX]);
    }
}
