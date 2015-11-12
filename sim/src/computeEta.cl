#include "computeEta_types.h"
#include "config.h"

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

    // global work-item indices
    const int gx = get_global_id(0); // range: [0, nx - 2]
    const int gy = get_global_id(1); // range: [0, ny - 2]
    const int gid = gx + gy * (nx - 1); // range: [0, (nx - 2) + (ny - 2) * (nx + 1)]

    // local work-item indices and sizes
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lnx = get_local_size(0); // assert(lnx == WGNX)
    const int lny = get_local_size(1); // assert(lny == WGNY)

    // U_local indices and sizes
    const int lunx = lnx + 1;
    const int luid_west = lx + ly * lunx;
    const int luid_east = lx + 1 + ly * lunx;
    const int gunx = nx + 2;
    const int guid_west = (gx + 1) + gy * gunx;
    const int guid_east = (gx + 2) + gy * gunx;

    // V_local indices and sizes
    const int lvnx = lnx;
    const int lvid_south = lx + ly * lvnx;
    const int lvid_north = lx + (ly + 1) * lvnx;
    const int gvnx = nx - 1;
    const int gvid_south = gx + (gy + 1) * gvnx;
    const int gvid_north = gx + (gy + 2) * gvnx;

    // allocate local work-group memory for U and V
    local float U_local[(WGNX + 1) * WGNY];
    local float V_local[WGNX * (WGNY + 1)];

    // copy U from global to local memory
    U_local[luid_east] = U[guid_east];
    if (lx == 0) // only if we're at the west side of the work-group
        U_local[luid_west] = U[guid_west];

    // copy V from global to local memory
    V_local[lvid_north] = V[gvid_north];
    if (ly == 0) // only if we're at the south side of the work-group
        V_local[lvid_south] = V[gvid_south];

    // ensure all work-items have copied their values to local memory before proceeding
    barrier(CLK_LOCAL_MEM_FENCE); // assuming CLK_GLOBAL_MEM_FENCE is not necessary since the read happens before the write in each work-item

    // compute eta
    const int geid = (gx + 1) + (gy + 1) * (nx + 1);
    eta[geid] = eta[geid] - args.dt / args.dx * (U_local[luid_east] - U_local[luid_west])
                          - args.dt / args.dy * (V_local[lvid_south] - V_local[lvid_north]);
}
