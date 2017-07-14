/*
General kernels for periodic boundary conditions.

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



/*
 *  These kernels assumes that values are defined in cell centers, and that the halo is symmetric in both north and south directions
 * 
 * Fix north-south boundary before east-west (to get the corners right)
 */

// Rename to periodicBoundary_NS
__kernel void boundaryKernel_NS(
	// Discretization parameters
        int nx_, int ny_,
	int halo_x, int halo_y,
	
        // Data
        __global float* h_ptr_, int h_pitch_,
        __global float* u_ptr_, int u_pitch_,
	__global float* v_ptr_, int v_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    int opposite_row_index = tj + ny_;
    if ( tj > ny_ + halo_y - 1) {
	opposite_row_index = tj - ny_;
    }
    
    // Set ghost cells equal to inner neighbour's value
    if ((tj < halo_y || tj >  ny_+halo_y-1)
	&& tj > -1  && tj < ny_+(2*halo_y) && ti > -1 && ti < nx_+(2*halo_x) ) {
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
	int halo_x, int halo_y,

	// Data
        __global float* h_ptr_, int h_pitch_,
        __global float* u_ptr_, int u_pitch_,
	__global float* v_ptr_, int v_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    int opposite_col_index = ti + nx_;
    if ( ti > nx_ + halo_x - 1 ) {
	opposite_col_index = ti - nx_;
    }
    
    // Set ghost cells equal to inner neighbour's value
    if ( (ti > -1) && (ti < nx_+(2*halo_x)) &&
	 (tj > -1) && (tj < ny_+(2*halo_y))    ) {

	if ( (ti < halo_x) || (ti > nx_+halo_x-1) ) {
	    __global float* h_row = (__global float*) ((__global char*) h_ptr_ + h_pitch_*tj);
	    __global float* u_row = (__global float*) ((__global char*) u_ptr_ + u_pitch_*tj);
	    __global float* v_row = (__global float*) ((__global char*) v_ptr_ + v_pitch_*tj);
	    
	    h_row[ti] = h_row[opposite_col_index];
	    u_row[ti] = u_row[opposite_col_index];
	    v_row[ti] = v_row[opposite_col_index];
	}
    }
}


/*
 *  These kernels handles periodic boundary conditions for values defined on cell intersections, and assumes that the halo consist of the same number of ghost cells on each periodic boundary.
 * 
 * The values at the actual boundary is defined by the input values on the western and southern boundaries.
 * 
 * Fix north-south boundary before east-west (to get the corners right)
 */

__kernel void periodic_boundary_intersections_NS(
	// Discretization parameters
        int nx_, int ny_,
	int halo_x, int halo_y,
	
        // Data
        __global float* data_ptr_, int data_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    int opposite_row_index = tj + ny_;
    if ( tj > ny_ + halo_y - 1) {
	opposite_row_index = tj - ny_;
    }
    
    // Set ghost cells equal to inner opposite's value
    if ((tj < halo_y || tj >  ny_+halo_y-1)
	&& tj > -1  && tj < ny_+(2*halo_y)+1 && ti > -1 && ti < nx_+(2*halo_x)+1 ) {
	__global float* ghost_row = (__global float*) ((__global char*) data_ptr_ + data_pitch_*tj);
	__global float* opposite_row = (__global float*) ((__global char*) data_ptr_ + data_pitch_*opposite_row_index);

	ghost_row[ti] = opposite_row[ti];
    }
}

// Fix north-south boundary before east-west (to get the corners right)
__kernel void periodic_boundary_intersections_EW(
	// Discretization parameters
        int nx_, int ny_,
	int halo_x, int halo_y,

	// Data
	__global float* data_ptr_, int data_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    int opposite_col_index = ti + nx_;
    if ( ti > nx_ + halo_x - 1 ) {
	opposite_col_index = ti - nx_;
    }
    
    // Set ghost cells equal to inner neighbour's value
    if ( (ti > -1) && (ti < nx_+2*halo_x + 1) &&
	 (tj > -1) && (tj < ny_+2*halo_y + 1)    ) {

	if ( (ti < halo_x) || (ti > nx_+halo_x-1) ) {
	    __global float* data_row = (__global float*) ((__global char*) data_ptr_ + data_pitch_*tj);
	    
	    data_row[ti] = data_row[opposite_col_index];
	}
    }
}
