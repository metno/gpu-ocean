#include "../computeV_types.h"
#include "../config.h"

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

	// global indices ++
    // assert(get_global_size(0) >= nx - 1)
    // assert(get_global_size(1) >= ny - 1)
    const int gx = get_global_id(0); // range: [0, nx - 2 + padding]
    const int gy = get_global_id(1); // range: [0, ny - 2 + padding]
    if (gx > nx - 2 || gy > ny - 2)
        return; // quit if we're in the padding area
    const int gnx = nx + 2;
    const int gny = ny - 1;

    // local indices ++
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int lnx = get_local_size(0); // assert(lnx == WGNX)
    const int lny = get_local_size(1); // assert(lny == WGNY)

    // global indices for V
    const int gvid_south = gx + (gy + 1) * (nx - 1);
    const int gvid_north = gx + (gy + 2) * (nx - 1);

    // local and global indices for Hr_v
    const int lhid_south = lx +       ly * lnx;
    const int lhid_north = lx + (ly + 1) * lnx;
    const int ghid_south = gx +       gy * (gnx - 1);
    const int ghid_north = gx + (gy + 1) * (gnx - 1);

    // local and global indices for eta
    const int leid_south = lx     + ly * lnx;
    const int leid       = lx     + (ly + 1) * lnx;
    const int leid_north = lx     + (ly + 2) * lnx;
    const int geid_south = gx     + (gy + 1) * (gnx - 1);
    const int geid       = gx     + (gy + 2) * (gnx - 1);
    const int geid_north = gx     + (gy + 3) * (gnx - 1);

    // local and global indices for U
    const int luid_west      = lx +     (ly + 1) * (lnx + 1);
    const int luid_east      = lx + 1 + (ly + 1) * (lnx + 1);
    const int luid_northwest = lx +     (ly + 2) * (lnx + 1);
    const int luid_northeast = lx + 1 + (ly + 2) * (lnx + 1);
    const int guid_west      = gx +           gy * gnx;
    const int guid_east      = gx + 1 +       gy * gnx;
    const int guid_northwest = gx + 1 + (gy + 2) * gnx;
    const int guid_northeast = gx + 2 + (gy + 2) * gnx;
    
    // allocate local work-group memory for Hr_v, eta, and U
    local float Hr_v_local[WGNX * (WGNY + 1)];
    local float eta_local[WGNX * (WGNY + 2)];
    local float U_local[(WGNX + 1) * (WGNY + 2)];
    
    // copy Hr_v from global to local memory
    Hr_v_local[lhid_north] = Hr_v[ghid_north];
    if (ly == 0) // if we're at the south side of the work-group
        Hr_v_local[lhid_south] = Hr_v[ghid_south];

    // copy eta from global to local memory
    eta_local[leid] = eta[geid];
    if (ly == 0) // if we're at the south side of the work-group
        eta_local[leid_south] = eta[geid_south];
    if (ly == lny - 1) // if we're at the north side of the work-group
        eta_local[leid_north] = eta[geid_north];

    // copy U from global to local memory
    if (gx == 0) { // if we're next to the western ghost cell
        if (gy == 0) // if we're next to the southern ghost cell
            U_local[luid_west] = U[guid_west];
        if (gy < gny) // if we're not next to the northern ghost cell
            U_local[luid_northwest] = U[guid_northwest];
    } else {
        if (gy == 0) // if we're next to the southern ghost cell
            U_local[luid_east] = U[guid_east];
        if (gy < gny) // if we're not next to the northern ghost cell
            U_local[luid_northeast] = U[guid_northeast];
    }
	
    // ensure all work-items have copied their values to local memory before proceeding
    barrier(CLK_LOCAL_MEM_FENCE); // assuming CLK_GLOBAL_MEM_FENCE is not necessary since the read happens before the write in each work-item
    
    if (gy == 0) { // if we're next to the southern ghost cell
        // compute V on the southern cell edge
        const float Ur_south = 0.5f * (U_local[luid_west] + U_local[luid_east]); // linear interpolation
        const float B_south = 1.0f + R * dt / Hr_v_local[lhid_south];
        const float P_south = g * Hr_v_local[lhid_south] * (eta_local[leid] - eta_local[leid_south]) / dy;
        V[gvid_south] = 1.0f / B_south * (V[gvid_south] + dt * (F * Ur_south - P_south));
    }

    // compute V on the northern cell edge
    float Ur_north;
    if (gy == gny) { // linear interpolation if we're next to the northern ghost cell
        Ur_north = 0.5f * (U_local[luid_west] + U_local[luid_east]);
    } else { // otherwise bilinear interpolation
        Ur_north = 0.25f * (U_local[luid_west] + U_local[luid_east] + U_local[luid_northwest] + U_local[luid_northeast]);
    }
    const float B_north = 1.0f + R * dt / Hr_v_local[lhid_north];
    const float P_north = g * Hr_v_local[lhid_north] * (eta_local[leid_north] - eta_local[leid]) / dy;
    //V[gvid_north] = 1.0f / B_north * (V[gvid_north] + dt * (F * Ur_north - P_north));
    V[gvid_north] = 0.0f;
}
