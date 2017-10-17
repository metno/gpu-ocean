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

__kernel void periodicBoundary_NS(
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
__kernel void periodicBoundary_EW(
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
 *  These kernels handles periodic boundary conditions for values defined on cell intersections, and assumes that the halo consists of the same number of ghost cells on each periodic boundary.
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


/*
 *  These kernels handles wall boundary conditions for values defined on cell intersections, and assumes that the halo consists of the same number of ghost cells on each periodic boundary.
 * 
 */

__kernel void closed_boundary_intersections_EW(
	// Discretization parameters
        int nx_, int ny_,
	int halo_x_, int halo_y_,
	
        // Data
        __global float* data_ptr_, int data_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    
    if ( ti == 0 && tj < ny_ + (2*halo_x_) + 1) {
	__global float* data_row = (__global float*) ((__global char*) data_ptr_ + data_pitch_*tj);
	// Western boundary:
	for (int i = 0; i < halo_x_; ++i) {
	    data_row[i] = data_row[2*halo_x_ - i];
	}
	// Eastern boundary:
	for (int i = 0; i < halo_x_; ++i) {
	    data_row[nx_ + 2*halo_x_ - i] = data_row[nx_ + i];
	}
    }
}

__kernel void closed_boundary_intersections_NS(
	// Discretization parameters
        int nx_, int ny_,
	int halo_x_, int halo_y_,

	// Data
	__global float* data_ptr_, int data_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    if ( tj == 0 && ti < ny_ + (2*halo_y_) +1) {
	// Southern boundary:
	for (int j = 0; j < halo_y_; ++j) {
	    const int inner_index = 2*halo_y_ - j;
	    __global float* ghost_row = (__global float*) ((__global char*) data_ptr_ + data_pitch_*j);
	    __global float* inner_row = (__global float*) ((__global char*) data_ptr_ + data_pitch_*inner_index);
	    ghost_row[ti] = inner_row[ti];
	}
	// Northern boundary:
	for (int j = 0; j < halo_y_; ++j) {
	    const int ghost_index = ny_ + 2*halo_y_ - j;
	    const int inner_index = ny_ + j;
	    __global float* ghost_row = (__global float*) ((__global char*) data_ptr_ + data_pitch_*ghost_index);
	    __global float* inner_row = (__global float*) ((__global char*) data_ptr_ + data_pitch_*inner_index);
	    ghost_row[ti] = inner_row[ti];
	}
	
    }
}



/*
 *  These kernels implement numerical sponge.
 * 
 *  Assume that the outermost row has the desired values already
 */

__kernel void numericalSponge_NS(
	// Discretization parameters
	int boundary_condition_north_, int boundary_condition_south_,
	int nx_, int ny_,
	int halo_x_, int halo_y_,
	int sponge_cells_north_,
	int sponge_cells_south_,
	
        // Data
        __global float* h_ptr_, int h_pitch_,
        __global float* u_ptr_, int u_pitch_,
	__global float* v_ptr_, int v_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    // Extrapolate on south side:
    // Keep outer edge as is!
    if ((tj < sponge_cells_south_) && (tj > 0)
	&& (ti > 0) && (ti < nx_ + 2*halo_x_-1) ) {

	// Get base value
	__global float* inner_row_h = (__global float*) ((__global char*) h_ptr_ + h_pitch_*sponge_cells_south_);
	__global float* inner_row_u = (__global float*) ((__global char*) u_ptr_ + u_pitch_*sponge_cells_south_);
	__global float* inner_row_v = (__global float*) ((__global char*) v_ptr_ + v_pitch_*sponge_cells_south_);
	float inner_value_h = inner_row_h[ti];
	float inner_value_u = inner_row_u[ti];
	float inner_value_v = inner_row_v[ti];

	// Get target value
	__global float* outer_row_h = (__global float*) ((__global char*) h_ptr_ + h_pitch_*0);
	__global float* outer_row_u = (__global float*) ((__global char*) u_ptr_ + u_pitch_*0);
	__global float* outer_row_v = (__global float*) ((__global char*) v_ptr_ + v_pitch_*0);
	float outer_value_h = outer_row_h[ti];
	float outer_value_u = outer_row_u[ti];
	float outer_value_v = outer_row_v[ti];

	// Find target cell
	__global float* target_row_h = (__global float*) ((__global char*) h_ptr_ + h_pitch_*tj);
	__global float* target_row_u = (__global float*) ((__global char*) u_ptr_ + u_pitch_*tj);
	__global float* target_row_v = (__global float*) ((__global char*) v_ptr_ + v_pitch_*tj);
	
	// Interpolate:
	float ratio = ((float)(tj - 0))/(sponge_cells_south_ - 0);
	target_row_h[ti] = outer_value_h + ratio*(inner_value_h - outer_value_h);
	target_row_u[ti] = outer_value_u + ratio*(inner_value_u - outer_value_u);
	target_row_v[ti] = outer_value_v + ratio*(inner_value_v - outer_value_v);
    }
        
}

    
__kernel void numericalSponge_EW(
	// Discretization parameters
        int nx_, int ny_,
	int halo_x_, int halo_y_,
	int sponge_cells_west_,
	int sponge_cells_east_,
	
        // Data
        __global float* h_ptr_, int h_pitch_,
        __global float* u_ptr_, int u_pitch_,
	__global float* v_ptr_, int v_pitch_) {
    // Intentionally left emtpy
}
