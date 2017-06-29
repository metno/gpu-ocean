/*
These OpenCL kernels implement boundary conditions for 
the Coriolis well balanced reconstruction scheme, as given by the publication of Chertock, Dudzinski, Kurganov and Lukacova-Medvidova (CDFLM) in 2016.
Copyright (C) 2016  SINTEF ICT

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


 // Fix north-south boundary before east-west (to get the corners right)
__kernel void boundaryKernel_NS(
	// Discretization parameters
        int nx_, int ny_,
	
        // Data
        __global float* h_ptr_, int h_pitch_,
        __global float* u_ptr_, int u_pitch_,
	__global float* v_ptr_, int v_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    int opposite_row_index = ny_ + tj;
    if ( tj > ny_ + 2) {
	opposite_row_index = tj - ny_;
    }
    
    // Set ghost cells equal to inner neighbour's value
    if ((tj < 3 || tj >  ny_+2)
	&& tj > -1  && tj < ny_+6 && ti > -1 && ti < nx_+6 ) {
	__global float* ghost_row_h = (__global float*) ((__global char*) h_ptr_ + h_pitch_*tj);
	__global float* opposite_row_h = (__global float*) ((__global char*) h_ptr_ + h_pitch_*opposite_row_index);

	__global float* ghost_row_u = (__global float*) ((__global char*) u_ptr_ + u_pitch_*tj);
	__global float* opposite_row_u = (__global float*) ((__global char*) u_ptr_ + u_pitch_*opposite_row_index);

	__global float* ghost_row_v = (__global float*) ((__global char*) v_ptr_ + v_pitch_*tj);
	__global float* opposite_row_v = (__global float*) ((__global char*) v_ptr_ + v_pitch_*opposite_row_index);

	ghost_row_h[ti] = opposite_row_h[ti];
	ghost_row_u[ti] = opposite_row_u[ti];
	ghost_row_v[ti] = opposite_row_v[ti];

    }
}


// Fix north-south boundary before east-west (to get the corners right)
__kernel void boundaryKernel_EW(
	// Discretization parameters
        int nx_, int ny_,
	
	// Data
        __global float* h_ptr_, int h_pitch_,
        __global float* u_ptr_, int u_pitch_,
	__global float* v_ptr_, int v_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    int opposite_col_index = nx_ + ti;
    if ( ti > nx_+2 ) {
	opposite_col_index = ti - nx_;
    }
    
    // Set ghost cells equal to inner neighbour's value
    if ( (ti > -1) && (ti < nx_+6) &&
	 (tj > -1) && (tj < ny_+6)    ) {

	if ( (ti < 3) || (ti > nx_+2) ) {
	    __global float* h_row = (__global float*) ((__global char*) h_ptr_ + h_pitch_*tj);
	    __global float* u_row = (__global float*) ((__global char*) u_ptr_ + u_pitch_*tj);
	    __global float* v_row = (__global float*) ((__global char*) v_ptr_ + v_pitch_*tj);
	    
	    h_row[ti] = h_row[opposite_col_index];
	    u_row[ti] = u_row[opposite_col_index];
	    v_row[ti] = v_row[opposite_col_index];
	}
    }
}
