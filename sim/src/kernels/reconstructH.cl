#include "../reconstructH_types.h"
#include "../config.h"

#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define local
#define CLK_LOCAL_MEM_FENCE
#endif

/*
Reconstructs the equilibrium depth, H, at the western and southern cell edge.
Input:
  - args.nx, args.ny: Internal grid size (nx, ny), i.e. not including grid points at the outside of ghost cells.
  - H: Values defined at the center of each cell, including ghost cells. Size: (nx + 1, ny + 1).
Output:
  - Hr_u: Values defined at the center of the western and eastern edge of each grid cell, excluding the western edge of the western ghost cells
          and the eastern edge of the eastern ghost cells. Size: (nx, ny + 1)
  - Hr_v: Values defined at the center of the southern and northern edge of each grid cell, excluding the southern edge of the southern ghost cells
          and the northern edge of the northern ghost cells. Size: (nx + 1, ny)
*/
__kernel void ReconstructH (
    __global const float *H,
    __global float *Hr_u,
    __global float *Hr_v,
    ReconstructH_args args)
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
	const int gid = gx + gy * (nx + 1); // H
	const unsigned int hr_u_gid = gx+1 + nx * gy; //(+1 because we want the eastern interface)
	const unsigned int hr_v_gid = gx + (nx + 1) * (gy+1); //(+1 because we want the northern interface)

	// allocate local work-group memory for U and V
	local float H_local[(WGNX + 1) * (WGNY + 1)];

	// copy H from global to local memory
	H_local[lx + ly * (WGNX + 1)] = H[gid];
	if(lx == WGNX-1) {
		H_local[lx + ly * (WGNX + 1) + 1] = H[gid+1];
	}
	if(ly == WGNY-1) {
		H_local[lx + ly * (WGNX + 1) + (WGNX + 1)] = H[gid + (nx + 1)];
	}

    // ensure all work-items have copied their values to local memory before proceeding
    barrier(CLK_LOCAL_MEM_FENCE); // assuming CLK_GLOBAL_MEM_FENCE is not necessary since the read happens before the write in each work-item

    if (gx < nx || gy < ny+1) {
		Hr_u[hr_u_gid] = 0.5 * (H_local[lx + ly * (WGNX+1) + 1] + H_local[lx + ly * (WGNX+1)]);
    }
	if (gx < nx+1 || gy < ny) {
        Hr_v[hr_v_gid] = 0.5 * (H_local[lx + ly * (WGNX+1) + (WGNX+1)] + H_local[lx + ly * (WGNX+1)]);
    }
}
