#include "reconstructH_types.h"


/*
Reconstructs the equilibrium depth, H, at the western and southern cell edge.
Input:
  - args.nx, args.ny: Internal grid dimension (nx, ny), i.e. not including grid points at the outside of ghost cells.
  - H: Values defined at the center of each cell, including ghost cells. Dimension: (nx + 1, ny + 1).
Output:
  - Hr_u: Values defined at the center of the western and eastern edge of each grid cell, excluding the western edge of the western ghost cells
          and the eastern edge of the eastern ghost cells. Dimension: (nx, ny + 1)
  - Hr_v: Values defined at the center of the southern and northern edge of each grid cell, excluding the southern edge of the southern ghost cells
          and the northern edge of the northern ghost cells. Dimension: (nx + 1, ny)
*/
__kernel void ReconstructH (
    __global const float *H,
    __global float *Hr_u,
    __global float *Hr_v,
    __local float *H_local, // dimension: (wgnx + 1, wgny + 1) where wgnx and wgny are the work-group size in each dimension
    ReconstructH_args args)
{
    const int nx = args.nx;
    const int ny = args.ny;

    const int gnx = get_global_size(0);
    const int gny = get_global_size(1);
    // assert(gnx == nx + 1)
    // assert(gny == ny + 1)

    // global indices ++
    const int gx = get_global_id(0); // range: [0, nx]
    const int gy = get_global_id(1); // range: [0, ny]
    const int gid = gx + gy * (nx + 1); // range: [0, nx + ny * (nx + 1)]

    // local indices ++
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lnx = get_local_size(0);
    const int lny = get_local_size(1);
    const int lid = lx + ly * lnx;

    // local indices ++ in H_local (extended to accommodate neighbors to the west and south)
    const int lex = lx + 1;
    const int ley = ly + 1;
    const int lenx = lnx + 1;
    const int leny = lny + 1;
    const int leid = lex + ley * lenx;
    const int leid_east = lex - 1 + ley * lenx;
    const int leid_south = lex + (ley - 1) * lenx;

    // copy H from global to local memory
    H_local[leid] = H[gid]; // copy to this cell
    if ((lx == 0) && (gx > 0)) // only if we're at the west side of the work-group, but not if we're at the west side of the global domain
        H_local[ley * lenx] = H[(gx - 1) + gy * (nx + 1)]; // copy to neighbor cell to the west
    if ((ly == 0) && (gy > 0)) // only if we're at the south side of the work-group, but not if we're at the south side of the global domain
        H_local[lex] = H[gx + (gy - 1) * (nx + 1)]; // copy to neighbor cell to the south

    // ensure all work-items have copied their values to local memory before proceeding
    barrier(CLK_LOCAL_MEM_FENCE); // assuming CLK_GLOBAL_MEM_FENCE is not necessary since the read happens before the write in each work-item

    // reconstruct using basic linear interpolation ...
    // ... at the western cell edge
    if (gx > 0) // only if we're not a western ghost cell
        Hr_u[gx - 1 + gy * nx] = 0.5 * (H_local[leid_east] + H_local[leid]);
    // ... at the southern cell edge
    if (gy > 0) // only if we're not a southern ghost cell
        Hr_v[gx + (gy - 1) * (nx + 1)] = 0.5 * (H_local[leid_south] + H_local[leid]);
}
