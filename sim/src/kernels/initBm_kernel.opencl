/*
This software is part of GPU Ocean. 

Copyright (C) 2018 SINTEF Digital
Copyright (C) 2018 Norwegian Meteorological Institute

This OpenCL kernel initializes bottom topography in cell centers,
based on bottom topography given on cell intersections.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "common.opencl"

/**
 * Initializing bottom bathymetry on cell mid-points (Bm) based on 
 * given bathymetry on cell intersections (Bi).
 * Grid (including ghost cells) of nx by ny cells, and 
 * nx+1 by ny+1 intersections.
 */
__kernel void initBm(const int nx_, const int ny_,
	    __global float* Bi_ptr_, int Bi_pitch_,
	    __global float* Bm_ptr_, int Bm_pitch_ ) {

    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    
    //Index of block within domain
    const int bx = get_local_size(0) * get_group_id(0);
    const int by = get_local_size(1) * get_group_id(1);

    //Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);
    
    //Shared memory variables
    __local float Bi[block_height+1][block_width+1];
    
    // Read Bi into shared memory
    for (int j=ty; j<block_height+1; j+=get_local_size(1)) {
	const int glob_j = by + j;
	if (glob_j < ny_+1) {
	    __global float* const Bi_row = (__global float*) ((__global char*) Bi_ptr_ + Bi_pitch_*glob_j);
	    for (int i=tx; i<block_width+1; i+=get_local_size(0)) {
		const int glob_i = bx + i;
		if (glob_i < nx_+1) {
		    Bi[j][i] = Bi_row[glob_i];
		}
	    }
	}
    }

    // Sync so that all threads have written to shared memory
    barrier(CLK_LOCAL_MEM_FENCE);

    // Calculate Bm and write to main memory
    if (ti < nx_ && tj < ny_) {
	__global float* Bm_row = (__global float*) ((__global char*) Bm_ptr_ + Bm_pitch_*tj);
	Bm_row[ti] = 0.25f*(Bi[ty][tx] + Bi[ty+1][tx] + Bi[ty][tx+1] + Bi[ty+1][tx+1]);
    }
}

/**
 *  Kernel for changing water elevation to water depth (e.g. for CDKLM16)
 *  (nx, ny) is the size of domain **including** ghost cells.
 */
__kernel void waterElevationToDepth(const int nx_, const int ny_,
				    __global float* h_ptr_, int h_pitch_,
				    __global float* Bm_ptr_, int Bm_pitch_ ) {
    
    //Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    if (ti < nx_ && tj < ny_) {
	__global float* const h_row = (__global float*) ((__global char*) h_ptr_ + h_pitch_*tj);
	__global float* const Bm_row = (__global float*) ((__global char*) Bm_ptr_ + Bm_pitch_*tj);
	h_row[ti] -= Bm_row[ti];
    }
    
}


/**
 *  Kernel for changing water elevation to water depth (e.g. for CDKLM16)
 *  (nx, ny) is the size of domain **including** ghost cells.
 */
__kernel void waterDepthToElevation(const int nx_, const int ny_,
				    __global float* w_ptr_, int w_pitch_,
				    __global float* h_ptr_, int h_pitch_,
				    __global float* Bm_ptr_, int Bm_pitch_ ) {
    
    //Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    if (ti < nx_ && tj < ny_) {
	__global float* const h_row = (__global float*) ((__global char*) h_ptr_ + h_pitch_*tj);
	__global float* const Bm_row = (__global float*) ((__global char*) Bm_ptr_ + Bm_pitch_*tj);
	__global float* const w_row = (__global float*) ((__global char*) w_ptr_ + w_pitch_*tj);
	
	w_row[ti] = h_row[ti] + Bm_row[ti];
    }
    
}
