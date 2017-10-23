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
