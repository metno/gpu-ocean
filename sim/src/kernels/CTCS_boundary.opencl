/*
These OpenCL kernels implement boundary conditions for 
the Centered in Time, Centered in Space(leapfrog)
numerical scheme for the shallow water equations, described in 
L. P. Røed, "Documentation of simple ocean models for use in ensemble
predictions", Met no report 2012/3 and 2012/5 .

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

// Boundary conditions are defined as
// 1: Closed wall
// 2: Periodic
// 3: Open (Flow Relaxation Scheme)
// 4: Open (Linear interpolation)

 // Fix north-south boundary before east-west (to get the corners right)
__kernel void boundaryEtaKernel_NS(
	// Discretization parameters
        int nx_, int ny_,
        int halo_x_, int halo_y_,
	int bc_north_, int bc_south_,
	
        // Data
        __global float* eta_ptr_, int eta_pitch_) {
    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    int opposite_row_index = ny_;
    //if (tj == ny_+1) {
    if ( (tj == ny_+1 && bc_north_ == 2) || (tj == 0 && bc_south_ == 1) ) {
	opposite_row_index = 1;
    }
    
    // Set ghost cells equal to inner neighbour's value
    if (((tj == 0     && bc_south_ < 3)  ||
	 (tj == ny_+1 && bc_north_ < 3)) &&
	ti > 0 && ti < nx_+1) {
	__global float* ghost_row = (__global float*) ((__global char*) eta_ptr_ + eta_pitch_*tj);
	__global float* opposite_row = (__global float*) ((__global char*) eta_ptr_ + eta_pitch_*opposite_row_index);
	ghost_row[ti] = opposite_row[ti];
    }
    // TODO: USE HALO PARAMS
}

// Fix north-south boundary before east-west (to get the corners right)
__kernel void boundaryEtaKernel_EW(
	// Discretization parameters
        int nx_, int ny_,
        int halo_x_, int halo_y_,
	int bc_east_, int bc_west_,

        // Data
        __global float* eta_ptr_, int eta_pitch_) {
    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    int opposite_col_index = nx_;
    if ( (ti == nx_+1 && bc_east_ == 2) || (ti == 0 && bc_west_ == 1) ) {
	opposite_col_index = 1;
    }
    
    // Set ghost cells equal to inner neighbour's value
    if (((ti == 0     && bc_west_ < 3)  ||
	 (ti == nx_+1 && bc_east_ < 3)) &&
	tj > -1 && tj < ny_+2) {
	__global float* eta_row = (__global float*) ((__global char*) eta_ptr_ + eta_pitch_*tj);
	eta_row[ti] = eta_row[opposite_col_index];
    }
    // TODO: USE HALO PARAMS
}

// NS need to be called before EW!
__kernel void boundaryUKernel_NS(
        // Discretization parameters
        int nx_, int ny_,
        int nx_halo_, int ny_halo_,
	int bc_north_, int bc_south_,

        // Data
        __global float* U_ptr_, int U_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    // Check if thread is in the domain:
    if (ti <= nx_+2 && tj <= ny_+1) {
	// The thread's row:
	__global float* u_row = (__global float*) ((__global char*) U_ptr_ + U_pitch_*tj);

	 int opposite_row_index = ny_;
	 if ( (tj == ny_+1 && bc_north_ == 2) || (tj == 0 && bc_south_ == 1) ) {
	     opposite_row_index = 1;
	 }
	
	 if ( ((tj == 0     && bc_south_ < 3)  ||
	       (tj == ny_+1 && bc_north_ < 3)) &&
	      ti > 0 && ti < nx_+1 ) {
	    __global float* u_opposite_row = (__global float*) ((__global char*) U_ptr_ + U_pitch_*opposite_row_index);
	    u_row[ti] = u_opposite_row[ti];
	}
    } 
}

__kernel void boundaryUKernel_EW(
        // Discretization parameters
        int nx_, int ny_,
        int nx_halo_, int ny_halo_,
	int bc_east_, int bc_west_,
	
        // Data
        __global float* U_ptr_, int U_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    // Check if thread is in the domain:
    if (ti <= nx_+2 && tj <= ny_+1) {	
	__global float* u_row = (__global float*) ((__global char*) U_ptr_ + U_pitch_*tj);

	if ( (ti < 2 && bc_west_ == 1) || (ti > nx_ && bc_east_ == 1) ) {
	    u_row[ti] = 0;
	}
	else if (bc_west_ == 2) { // bc_east is then automatically also 2
	    // Periodic
	    int opposite_col_index = nx_+1;
	    if (ti == nx_+2) {
		opposite_col_index = 1;
	    }
	    
	    // We should have computed both u_row[1] and u_row[nx_+1],
	    // and these two should already have the same values.
	    if ( ti == 0 || ti == nx_+2) {
		u_row[ti] = u_row[opposite_col_index];
	    }
	}
    }
}



// NS need to be called before EW!
__kernel void boundaryVKernel_NS(
        // Discretization parameters
        int nx_, int ny_,
        int nx_halo_, int ny_halo_,
	int bc_north_, int bc_south_,

        // Data
        __global float* V_ptr_, int V_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    // Check if thread is in the domain:
    if (ti <= nx_+1 && tj <= ny_+2) {	
	__global float* v_row = (__global float*) ((__global char*) V_ptr_ + V_pitch_*tj);

	
	if ( (tj < 2 && bc_south_ == 1 ) || (tj > ny_ && bc_north_ == 1) ) {
	    v_row[ti] = 0;
	}
	else if (bc_north_ == 2) { // implicit bc_south_ == 2
	    // Periodic
	    int opposite_row_index = ny_;
	    if (tj == ny_+2) {
		opposite_row_index = 1;
	    }
	    
	    if ( (tj == 0 || tj == ny_+2) && ti > 0 && ti < nx_+1 ) {
		__global float* v_opposite_row = (__global float*) ((__global char*) V_ptr_ + V_pitch_*opposite_row_index);
		v_row[ti] = v_opposite_row[ti];
	    }
	}
    }
}

__kernel void boundaryVKernel_EW(
        // Discretization parameters
        int nx_, int ny_,
        int nx_halo_, int ny_halo_,
	int bc_east_, int bc_west_,

        // Data
        __global float* V_ptr_, int V_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    // Check if thread is in the domain:
    if (ti <= nx_+1 && tj <= ny_+2) {	
	__global float* v_row = (__global float*) ((__global char*) V_ptr_ + V_pitch_*tj);

	 int opposite_col_index = nx_;
	 if ( (ti == nx_+1 && bc_east_ == 2) || (ti == 0 && bc_west_ == 1) ) {
	     opposite_col_index = 1;
	 }
	
	// We should have computed both u_row[1] and u_row[nx_+1],
	// and these two should already have the same values.
	 if ( (ti == 0 && bc_west_ < 3) || (ti == nx_+1 && bc_east_ < 3) )   {

	     v_row[ti] = v_row[opposite_col_index];
	}
    }
}


__kernel void boundary_linearInterpol_NS(
	// Discretization parameters
        int nx_, int ny_,
        int nx_halo_, int ny_halo_,
	int staggered_x_, int staggered_y_, // U->(1,0), V->(0,1), eta->(0,0)
	
	// Boundary condition parameters
	int sponge_cells_north_, int sponge_cells_south_,
	int bc_north_, int bc_south_,
		
        // Data
        __global float* data_ptr_, int data_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    if (( ((bc_south_ == 4) &&
	   (tj < sponge_cells_south_) && (tj > 0)) ||
	  ((bc_north_ == 4) &&
	   (tj > ny_ + 2*ny_halo_ + staggered_y_ - sponge_cells_north_ - 1) &&
	   (tj < ny_ + 2*ny_halo_ + staggered_y_ - 1)) ) &&
	(ti > 0) && (ti < nx_ + 2*ny_halo_ + staggered_x_ -1))
	{
	    // Identify inner and outer row
	    int inner_row = sponge_cells_south_;
	    int outer_row = 0;
	    if (tj > sponge_cells_south_) {
		inner_row = ny_ + 2*ny_halo_ + staggered_y_ - sponge_cells_north_ - 1;
		outer_row = ny_ + 2*ny_halo_ + staggered_y_ - 1;
	    }

	    // Get outer value
	    __global float* outer_row_ptr = (__global float*) ((__global char*) data_ptr_ + data_pitch_*outer_row);
	    float outer_value = outer_row_ptr[ti];

	    // Get inner value
	    __global float* inner_row_ptr = (__global float*) ((__global char*) data_ptr_ + data_pitch_*inner_row);
	    float inner_value = inner_row_ptr[ti];

	    // Find target cell
	    __global float* target_row_ptr = (__global float*) ((__global char*) data_ptr_ + data_pitch_*tj);

	    // Interpolate:
	    float ratio = ((float)(tj - outer_row))/(inner_row - outer_row);
	    target_row_ptr[ti] = outer_value + ratio*(inner_value - outer_value);
	}
}



__kernel void boundary_linearInterpol_EW(
	// Discretization parameters
        int nx_, int ny_,
        int nx_halo_, int ny_halo_,
	int staggered_x_, int staggered_y_, // U->(1,0), V->(0,1), eta->(0,0)
	
	// Boundary condition parameters
	int sponge_cells_east_, int sponge_cells_west_,
	int bc_east_, int bc_west_,
		
        // Data
        __global float* data_ptr_, int data_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    if (( ((bc_west_ == 4) &&
	   (ti < sponge_cells_west_) && (ti > 0)) ||
	  ((bc_east_ == 4) &&
	   (ti > nx_ + 2*nx_halo_ + staggered_x_ - sponge_cells_east_ - 1) &&
	   (ti < nx_ + 2*nx_halo_ + staggered_x_ - 1)) ) &&
	(tj > 0) && (tj < ny_ + 2*ny_halo_ + staggered_y_-1))
	{

	    // Identify inner and outer row
	    int inner_col = sponge_cells_west_;
	    int outer_col = 0;
	    if (ti > sponge_cells_west_) {
		inner_col = nx_ + 2*nx_halo_ + staggered_x_ - sponge_cells_east_ - 1;
		outer_col = nx_ + 2*nx_halo_ + staggered_x_ - 1;
	    }

	    // Get row:
	    __global float* data_row = (__global float*) ((__global char*) data_ptr_ + data_pitch_*tj);

	    // Get inner value
	    float inner_value = data_row[inner_col];

	    // Get outer value
	    float outer_value = data_row[outer_col];

	    // Interpolate:
	    float ratio = ((float)(ti - outer_col))/(inner_col - outer_col);
	    data_row[ti] = outer_value + ratio*(inner_value - outer_value);
	}
}
