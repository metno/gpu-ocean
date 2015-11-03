#include "reconstructH_types.h"


/*
Reconstructs the equilibrium depth, H, at cell edges.
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
    ReconstructH_args args)
{
    const int nx = args.nx;
    const int ny = args.ny;

    int gx = get_global_id(0);
    int gy = get_global_id(1);

    if (gx < nx)
        Hr_u[gx + gy * nx] = 0.5 * (H[gx + gy * (nx + 1)] + H[gx + 1 + gy * (nx + 1)]);
    if (gy < ny)
        Hr_v[gx + gy * (nx + 1)] = 0.5 * (H[gx + gy * (nx + 1)] + H[gx + (gy + 1) * (nx + 1)]);
}
