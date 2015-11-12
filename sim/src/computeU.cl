#include "computeU_types.h"
#include "config.h"

/*
Computes U at the east cell edge (and at the west cell edge if this cell is next to a western ghost cell).
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
/*
    const int nx = args.nx;
    const int ny = args.ny;

    // global indices ++
    const int gx = get_global_id(0); // range: [0, nx - 2]
    const int gy = get_global_id(1); // range: [0, ny - 2]
    const int gid = gx + gy * (nx - 1); // range: [0, (nx - 2) + (ny - 2) * (nx - 1)]

    // local indices ++
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lnx = get_local_size(0); // assert(lnx == WGNX)
    const int lny = get_local_size(1); // assert(lny == WGNY)

//    // local indices ++ in H_local (extended to accommodate neighbors to the west and south)
//    const int lex = lx + 1;
//    const int ley = ly + 1;
//    const int lenx = lnx + 1;
//    const int leny = lny + 1;
//    const int leid = lex + ley * lenx;
//    const int leid_east = lex - 1 + ley * lenx;
//    const int leid_south = lex + (ley - 1) * lenx;

    // allocate local work-group memory for Hr_u, eta, and V
    local float Hr_u_local[(WGNX + 1) * WGNY];
    local float eta_local[(WGNX + 2) * WGNY]; // ### Assuming we need eta on the western and eastern outsides as well ... CHECK!
    local float eta_local[(WGNX + 1) * (WGNY + 1)];

    // copy Hr_u from global to local memory ... TBD
    Hr_u_local[lhid_east] = Hr_u[ghid_east];
    if (lx == 0) // only if we're at the west side of the work-group
        Hr_u_local[lhid_west] = Hr_u[ghid_west];

    // copy eta from global to local memory ... TBD

    // copy V from global to local memory ... TBD


    // ensure all work-items have copied their values to local memory before proceeding
    barrier(CLK_LOCAL_MEM_FENCE); // assuming CLK_GLOBAL_MEM_FENCE is not necessary since the read happens before the write in each work-item


    // reconstruct V ... TBD

    // compute U
    U[]



    // reconstruct using basic linear interpolation ...
    // ... at the western cell edge
    if (gx > 0) // only if we're not a western ghost cell
        Hr_u[gx - 1 + gy * nx] = 0.5 * (H_local[leid_east] + H_local[leid]);
    // ... at the southern cell edge
    if (gy > 0) // only if we're not a southern ghost cell
        Hr_v[gx + (gy - 1) * (nx + 1)] = 0.5 * (H_local[leid_south] + H_local[leid]);
*/
}
