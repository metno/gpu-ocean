#include "computeV_types.h"
#include "config.h"

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

	int gx = get_global_id(0);
	int gy = get_global_id(1);
	
	int gid = gx + gy*nx;
	
	int lx = get_local_id(0);
    int ly = get_local_id(1);
    
    int lnx = get_local_size(0); // assert(lnx == WGNX)
    int lny = get_local_size(1); // assert(lny == WGNY)
    
    int lid = lx + ly*lnx;
    
    local float Hr_v_local[WGNX * (WGNY + 1)];
    local float eta_local[WGNX * (WGNY + 1)];
    local float U_local[(WGNX + 1) * (WGNY + 1)];
    
    // CONT HERE!
    Hr_v_local[lid] = Hr_v[gid];
    eta_local[lid] = eta[gid];
    
	for (int j=ly; j<lny+1; j+=lny) {
		const unsigned int l = gy + j;
		for (int i=lx; i<lnx+1; i+=lnx) {
			const unsigned int k = gx + i;
            if (k < nx+1 && l < ny) { //!< no reading outside the domain
				U_local[i + j*lnx] = U[k + l*nx];
			} else {
				U_local[i + j*lnx] = 0.0f;
			}
		}
	}
	
    barrier(CLK_LOCAL_MEM_FENCE); //!< make sure all values are read to local memory before doing any computations
    
    float B_v = 1.0f + R*dt / Hr_v_local[lid];

    float P_v = g * Hr_v_local[lid] * (eta_local[lid] - eta_local[lid+1]) / dy;

    // U (hat) reconstructed
    float Ur;	
	if (gx == 0) {
        Ur = 0.5f * (U_local[lid+1] + U_local[lx+1 + (ly+1)*lnx]);
	} else if (gx == nx-1) {
        Ur = 0.5f * (U_local[lid] + U_local[lx + (ly+1)*lnx]);
	} else {
        Ur = 0.25f * (U_local[lid] + U_local[lid+1] +
						U_local[lx + (ly+1)*lnx] + U_local[lx+1 + (ly+1)*lnx]);
	}

    // Do not compute V in the ghost cells
    if (gx < nx && gy < ny+1) //!< no writing outside the domain
    	V[gid] = 1.0f / B_v * (V[gid] + dt * (F*Ur - P_v));
}
