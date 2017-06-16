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


// Fix north-south boundary before east-west (to get the corners right)
__kernel void closedBoundaryEtaKernel_NS(
	// Discretization parameters
        int nx_, int ny_,
        int halo_x_, int halo_y_,

        // Data
        __global float* eta_ptr_, int eta_pitch_) {
    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    int inner_factor = 1;
    if (tj == ny_+1) {
	inner_factor = -1;
    }
    
    // Set ghost cells equal to inner neighbour's value
    if ((tj == 0 || tj == ny_+1) && ti > 0 && ti < nx_+1) {
	__global float* outer_eta_row = (__global float*) ((__global char*) eta_ptr_ + eta_pitch_*tj);
	__global float* inner_eta_row = (__global float*) ((__global char*) eta_ptr_ + eta_pitch_*(tj+inner_factor));
	outer_eta_row[ti] = inner_eta_row[ti];
    }
    // TODO: USE HALO PARAMS
}

// Fix north-south boundary before east-west (to get the corners right)
__kernel void closedBoundaryEtaKernel_EW(
	// Discretization parameters
        int nx_, int ny_,
        int halo_x_, int halo_y_,

        // Data
        __global float* eta_ptr_, int eta_pitch_) {
    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    int inner_factor = 1;
    if (ti == nx_+1) {
	inner_factor = -1;
    }
    
    // Set ghost cells equal to inner neighbour's value
    if ((ti == 0 || ti == nx_+1) && tj > -1 && tj < ny_+2) {
	__global float* eta_row = (__global float*) ((__global char*) eta_ptr_ + eta_pitch_*tj);
	eta_row[ti] = eta_row[ti+inner_factor];
    }
    // TODO: USE HALO PARAMS
}

// Set east and west boundary and ghost cell to zero
// Set north and south ghost cell equal to inner neighbour.
__kernel void closedBoundaryUKernel(
        // Discretization parameters
        int nx_, int ny_,
        int halo_x_, int halo_y_,

        // Data
        __global float* U_ptr_, int U_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    // Check if thread is in the domain:
    if (ti <= nx_+2 && tj <= ny_+1) {	
	__global float* u_row = (__global float*) ((__global char*) U_ptr_ + U_pitch_*tj);

	// Check if east or west boundary
	if ( ti < 2 || ti > nx_ ) {
	    u_row[ti] = 0;
	}
	else if (tj == 0) {
	    __global float* u_inner_row = (__global float*) ((__global char*) U_ptr_ + U_pitch_*(tj+1));
	    u_row[ti] = u_inner_row[ti];
	}
	else if (tj == ny_+1) {
	    __global float* u_inner_row = (__global float*) ((__global char*) U_ptr_ + U_pitch_*(tj-1));
	    u_row[ti] = u_inner_row[ti];
	}
    } 
    
    // TODO: USE HALO PARAMS
}


// Set north and south  boundary and ghost cell to zero
// Set east and west ghost cell equal to inner neighbour.
__kernel void closedBoundaryVKernel(
        // Discretization parameters
        int nx_, int ny_,
        int halo_x_, int halo_y_,

        // Data
        __global float* V_ptr_, int V_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    // Check if thread is in the domain:
    if (ti <= nx_+1 && tj <= ny_+2) {	
	__global float* v_row = (__global float*) ((__global char*) V_ptr_ + V_pitch_*tj);

	// Check if north or south boundary
	if ( tj < 2 || tj > ny_ ) {
	    v_row[ti] = 0;
	}
	else if (ti == 0) {
	    v_row[ti] = v_row[ti+1];
	}
	else if (ti == nx_+1) {
	    v_row[ti] = v_row[ti-1];
	}
    } 
    
    // TODO: USE HALO PARAMS
}










 // Fix north-south boundary before east-west (to get the corners right)
__kernel void periodicBoundaryEtaKernel_NS(
	// Discretization parameters
        int nx_, int ny_,
        int halo_x_, int halo_y_,

        // Data
        __global float* eta_ptr_, int eta_pitch_) {
    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    int opposite_row_index = ny_;
    if (tj == ny_+1) {
	opposite_row_index = 1;
    }
    
    // Set ghost cells equal to inner neighbour's value
    if ((tj == 0 || tj == ny_+1) && ti > 0 && ti < nx_+1) {
	__global float* ghost_row = (__global float*) ((__global char*) eta_ptr_ + eta_pitch_*tj);
	__global float* opposite_row = (__global float*) ((__global char*) eta_ptr_ + eta_pitch_*opposite_row_index);
	ghost_row[ti] = opposite_row[ti];
    }
    // TODO: USE HALO PARAMS
}

// Fix north-south boundary before east-west (to get the corners right)
__kernel void periodicBoundaryEtaKernel_EW(
	// Discretization parameters
        int nx_, int ny_,
        int halo_x_, int halo_y_,

        // Data
        __global float* eta_ptr_, int eta_pitch_) {
    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    int opposite_col_index = nx_;
    if (ti == nx_+1) {
	opposite_col_index = 1;
    }
    
    // Set ghost cells equal to inner neighbour's value
    if ((ti == 0 || ti == nx_+1) && tj > -1 && tj < ny_+2) {
	__global float* eta_row = (__global float*) ((__global char*) eta_ptr_ + eta_pitch_*tj);
	eta_row[ti] = eta_row[opposite_col_index];
    }
    // TODO: USE HALO PARAMS
}

// NS need to be called before EW!
__kernel void periodicBoundaryUKernel_NS(
        // Discretization parameters
        int nx_, int ny_,
        int nx_halo_, int ny_halo_,

        // Data
        __global float* U_ptr_, int U_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    // Check if thread is in the domain:
    if (ti <= nx_+2 && tj <= ny_+1) {	
	__global float* u_row = (__global float*) ((__global char*) U_ptr_ + U_pitch_*tj);

	 int opposite_row_index = ny_;
	 if (tj == ny_+1) {
	     opposite_row_index = 1;
	 }
	
	 if ( (tj == 0 || tj == ny_+1) && ti > 0 && ti < nx_+1 ) {
	    __global float* u_opposite_row = (__global float*) ((__global char*) U_ptr_ + U_pitch_*opposite_row_index);
	    u_row[ti] = u_opposite_row[ti];
	}
    } 
}

__kernel void periodicBoundaryUKernel_EW(
        // Discretization parameters
        int nx_, int ny_,
        int nx_halo_, int ny_halo_,

        // Data
        __global float* U_ptr_, int U_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    // Check if thread is in the domain:
    if (ti <= nx_+2 && tj <= ny_+1) {	
	__global float* u_row = (__global float*) ((__global char*) U_ptr_ + U_pitch_*tj);

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



// NS need to be called before EW!
__kernel void periodicBoundaryVKernel_NS(
        // Discretization parameters
        int nx_, int ny_,
        int nx_halo_, int ny_halo_,

        // Data
        __global float* V_ptr_, int V_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    // Check if thread is in the domain:
    if (ti <= nx_+1 && tj <= ny_+2) {	
	__global float* v_row = (__global float*) ((__global char*) V_ptr_ + V_pitch_*tj);

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

__kernel void periodicBoundaryVKernel_EW(
        // Discretization parameters
        int nx_, int ny_,
        int nx_halo_, int ny_halo_,

        // Data
        __global float* V_ptr_, int V_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    // Check if thread is in the domain:
    if (ti <= nx_+1 && tj <= ny_+2) {	
	__global float* v_row = (__global float*) ((__global char*) V_ptr_ + V_pitch_*tj);

	 int opposite_col_index = nx_;
	 if (ti == nx_+1) {
	     opposite_col_index = 1;
	 }
	
	// We should have computed both u_row[1] and u_row[nx_+1],
	// and these two should already have the same values.
	 if ( ti == 0 || ti == nx_+1) {
	     v_row[ti] = v_row[opposite_col_index];
	}
    }
}
