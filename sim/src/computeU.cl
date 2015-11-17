#include "computeU_types.h"
#include "config.h"

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

    // global indices ++
    const int gx = get_global_id(0); // range: [0, nx - 2]
    const int gy = get_global_id(1); // range: [0, ny - 2]
    if (gx > nx - 2 || gy > ny - 2)
        return; // quit if we're in the padding area
    const int gnx = get_global_size(0); // assert(gnx == nx - 1)
    const int gid = gx + gy * gnx; // range: [0, (nx - 2) + (ny - 2) * (nx - 1)]

    // local indices ++
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lnx = get_local_size(0); // assert(lnx == WGNX)
    const int lny = get_local_size(1); // assert(lny == WGNY)

    // global indices for U
    const int guid_west = gx + 1 + gy * (nx + 2);
    const int guid_east = gx + 2 + gy * (nx + 2);

    // local and global indices for Hr_u
    const int lhid_west = lx +     ly * (lnx + 1);
    const int lhid_east = lx + 1 + ly * (lnx + 1);
    const int ghid_west = gx +     gy * (gnx + 1);
    const int ghid_east = gx + 1 + gy * (gnx + 1);

    // local and global indices for eta
    const int leid_west = lx     + ly * (lnx + 2);
    const int leid      = lx + 1 + ly * (lnx + 2);
    const int leid_east = lx + 2 + ly * (lnx + 2);
    const int geid_west = gx     + (gy + 1) * (gnx + 2);
    const int geid      = gx + 1 + (gy + 1) * (gnx + 2);
    const int geid_east = gx + 2 + (gy + 1) * (gnx + 2);

    // local and global indices for V
    const int lvid_south     = lx + 1 +       ly * (lnx + 2);
    const int lvid_north     = lx + 1 + (ly + 1) * (lnx + 2);
    const int lvid_southeast = lx + 2 +       ly * (lnx + 2);
    const int lvid_northeast = lx + 2 + (ly + 1) * (lnx + 2);
    const int gvid_south     = gx +     (gy + 1) * gnx;
    const int gvid_north     = gx +     (gy + 2) * gnx;
    const int gvid_southeast = gx + 1 + (gy + 1) * gnx;
    const int gvid_northeast = gx + 1 + (gy + 2) * gnx;

    // allocate local work-group memory for Hr_u, eta, and V
    local float Hr_u_local[(WGNX + 1) * WGNY];
    local float eta_local[(WGNX + 2) * WGNY];
    local float V_local[(WGNX + 2) * (WGNY + 1)];

    // copy Hr_u from global to local memory
    Hr_u_local[lhid_east] = Hr_u[ghid_east];
    if (lx == 0) // if we're at the west side of the work-group
        Hr_u_local[lhid_west] = Hr_u[ghid_west];

    // copy eta from global to local memory
    eta_local[leid] = eta[geid];
    if (lx == 0) // if we're at the west side of the work-group
        eta_local[leid_west] = eta[geid_west];
    if (lx == lnx - 1) // if we're at the east side of the work-group
        eta_local[leid_east] = eta[geid_east];

    // copy V from global to local memory
    if (gy == 0) { // if we're next to the southern ghost cell
        if (gx == 0) // if we're next to the western ghost cell
            V_local[lvid_south] = V[gvid_south];
        if (gx < gnx - 1) // if we're not next to the eastern ghost cell
            V_local[lvid_southeast] = V[gvid_southeast];
    } else {
        if (gx == 0) // if we're next to the western ghost cell
            V_local[lvid_north] = V[gvid_north];
        if (gx < gnx - 1) // if we're not next to the eastern ghost cell
            V_local[lvid_northeast] = V[gvid_northeast];
    }

    // ensure all work-items have copied their values to local memory before proceeding
    barrier(CLK_LOCAL_MEM_FENCE); // assuming CLK_GLOBAL_MEM_FENCE is not necessary since the read happens before the write in each work-item

    if (gx == 0) { // if we're next to the western ghost cell
        // compute U on the western cell edge
        const float Vr_west = 0.5f * (V_local[lvid_south] + V_local[lvid_north]); // linear interpolation
        const float B_west = 1.0f + R * dt / Hr_u_local[lhid_west];
        const float P_west = g * Hr_u_local[lhid_west] * (eta_local[leid] - eta_local[leid_west]) / dx;
        U[guid_west] = 1.0f / B_west * (U[guid_west] + dt * (F * Vr_west - P_west));
    }

    // compute U on the eastern cell edge
    float Vr_east;
    if (gx == gnx - 1) { // linear interpolation if we're next to the eastern ghost cell
        Vr_east = 0.5f * (V_local[lvid_south] + V_local[lvid_north]);
    } else { // otherwise bilinear interpolation
        Vr_east = 0.25f * (V_local[lvid_south] + V_local[lvid_north] + V_local[lvid_southeast] + V_local[lvid_northeast]);
    }
    const float B_east = 1.0f + R * dt / Hr_u_local[lhid_east];
    const float P_east = g * Hr_u_local[lhid_east] * (eta_local[leid_east] - eta_local[leid]) / dx;
    U[guid_east] = 1.0f / B_east * (U[guid_east] + dt * (F * Vr_east - P_east));
}
