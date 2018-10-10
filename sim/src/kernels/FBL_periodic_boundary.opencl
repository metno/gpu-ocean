/*
This OpenCL kernel implements part of the Forward Backward Linear 
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


__kernel void closedBoundaryUKernel(
        // Discretization parameters
        int nx_, int ny_,
        int bc_north_, int bc_east_, int bc_south_, int bc_west_,

        // Data
        __global float* U_ptr_, int U_pitch_) {

   // Note that U values on the western boundary are sat to zero
    // by the step-kernel, and U values on all other boundaries are 
    // never written by the step-kernel. 
    // It should be sufficient to only call this kernel only once, 
    // at the beginning of the entire simulation.

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    //Compute pointer to current row in the U array
    __global float* const U_row = (__global float*) ((__global char*) U_ptr_ + U_pitch_*tj);	
    
    if ( ( ((ti == 0  ) && (bc_west_ == 1)) || 
           ((ti == nx_) && (bc_east_ == 1))    ) && tj < ny_ + 2) {
        U_row[ti] = 0.0f;
    }
    
    // We set U = 0 outside of the boundary as well, on north and south.
    if ( ( ((tj == 0    ) && (bc_south_ == 1)) ||
           ((tj == ny_+1) && (bc_north_ == 1))    ) && ti < nx_+1) {
        U_row[ti] = 0.0f;
    }
}

__kernel void periodicBoundaryUKernel_NS(
        // Discretization parameters
        int nx_, int ny_,

        // Data
        __global float* U_ptr_, int U_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    int opposite_row_index = ny_;
    if (tj == ny_+1) {
        opposite_row_index = 1;
    }
    
    // Set ghost cells equal to inner neighbour's value
    if (((tj == 0) || (tj == ny_ + 1)) && (ti > 0) && (ti < nx_))  {
        __global float* ghost_row = (__global float*) ((__global char*) U_ptr_ + U_pitch_*tj);
        __global float* opposite_row = (__global float*) ((__global char*) U_ptr_ + U_pitch_*opposite_row_index);
        ghost_row[ti] = opposite_row[ti];
    }
}

__kernel void periodicBoundaryUKernel_EW(
        // Discretization parameters
        int nx_, int ny_,

        // Data
        __global float* U_ptr_, int U_pitch_) {
    // U has no ghost cells in x-directions, but the values 
    // *on* the boundary need to match. 
    // The compute_U_kernel fixes the western (ti = 0) boundary,
    // and the eastern boundary (ti = nx) needs to be sat the same.
    
    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    // Check if thread is in the domain:
    if ((ti == nx_) && (tj <  ny_+2)) {
        __global float* U_row = (__global float*) ((__global char*) U_ptr_ + U_pitch_*tj);
        U_row[ti] = U_row[0];
    }
}




__kernel void closedBoundaryVKernel(
        // Discretization parameters
        int nx_, int ny_,
        int bc_north_, int bc_east_, int bc_south_, int bc_west_,

        // Data
        __global float* V_ptr_, int V_pitch_) {

    // Note that V values on the southern boundary are sat to zero
    // by the step-kernel, and V values on all other boundaries are 
    // never written by the step-kernel. 
    // It should be sufficient to only call this kernel only once, 
    // at the beginning of the entire simulation.

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    //Compute pointer to current row in the V array
    __global float* const V_row = (__global float*) ((__global char*) V_ptr_ + V_pitch_*tj);	
    
    if ( ( ((tj < 2  ) && (bc_south_ == 1)) || 
           (((tj == ny_+1) || (tj == ny_+2)) && (bc_north_ == 1)) )
         && ti < nx_ + 2) {
        V_row[ti] = 0.0f;
    }
    
    // We set V = 0 outside of the east and west boundary as well
    if ( ( ((ti == 0    ) && (bc_west_ == 1)) ||
           ((ti == nx_+1) && (bc_east_ == 1))    ) && tj < ny_+3) {
        V_row[ti] = 0.0f;
    }   
}


__kernel void periodicBoundaryVKernel_NS(
        // Discretization parameters
        int nx_, int ny_,

        // Data
        __global float* V_ptr_, int V_pitch_) {

    // One row of ghost values must be updated with the opposite 
    // interior cells' values.
    // The northern boundary must be given the value from the southern boundary

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    int opposite_row_index = ny_;
    if (tj == ny_+ 1) {
        opposite_row_index = 1;
    }
    if (tj == ny_+2) {
        opposite_row_index = 2;
    }
    
    // Set ghost cells equal to inner neighbour's value
    if ( ((tj == 0) || (tj == ny_ + 1) || (tj == ny_ + 2))
          && (ti > 0) && (ti < nx_+1) )  {
        __global float* ghost_row = (__global float*) ((__global char*) V_ptr_ + V_pitch_*tj);
        __global float* opposite_row = (__global float*) ((__global char*) V_ptr_ + V_pitch_*opposite_row_index);
        ghost_row[ti] = opposite_row[ti]; 
    }
}
    
__kernel void periodicBoundaryVKernel_EW(
        // Discretization parameters
        int nx_, int ny_,

        // Data
        __global float* V_ptr_, int V_pitch_) {
    // Straight forward with one ghost column on each side

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    int opposite_col_index = nx_;
    if (ti == nx_+1) {
        opposite_col_index = 1;
    }
    
    // Check if thread is in the domain:
    if ( ((ti == 0) || (ti == nx_+1)) && (tj <  ny_+3) ) {
        __global float* V_row = (__global float*) ((__global char*) V_ptr_ + V_pitch_*tj);
        V_row[ti] = V_row[opposite_col_index];
    }
}


__kernel void closedBoundaryEtaKernel(
        // Discretization parameters
        int nx_, int ny_,
        int bc_north_, int bc_east_, int bc_south_, int bc_west_,

        // Data
        __global float* eta_ptr_, int eta_pitch_) {
    // All eta values living outside of a closed boundary should
    // be ignored by the step-kernel. Anyway, we but them to zero to
    // make sure they are well defined, but this kernel should not need to
    // be called between time-steps.
    
     // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    //Compute pointer to current row in the eta array
    __global float* const eta_row = (__global float*) ((__global char*) eta_ptr_ + eta_pitch_*tj); 
    
    if ( (  ((tj == 0  ) && (bc_south_ == 1)) || 
            ((tj == ny_+1)  && (bc_north_ == 1)) )
         && ti < nx_ + 2) {
        eta_row[ti] = 0.0f;
    }
    
    // We set U = 0 outside of the east and west boundary as well
    if ( ( ((ti == 0    ) && (bc_west_ == 1)) ||
           ((ti == nx_+1) && (bc_east_ == 1))    )
         && tj < ny_+2) {
        eta_row[ti] = 0.0f;
    }
}

__kernel void periodicBoundaryEtaKernel_NS(
        // Discretization parameters
        int nx_, int ny_,

        // Data
        __global float* eta_ptr_, int eta_pitch_) {
    
    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);
    
    int opposite_row_index = ny_;
    if (tj == ny_+1) {
        opposite_row_index = 1;
    }
    
    // Set northern ghost cells
    if ( ((tj == 0) || (tj == ny_+1)) &&
          (ti > 0) && (ti < nx_+1) ) {
        __global float* ghost_row = (__global float*) ((__global char*) eta_ptr_ + eta_pitch_*tj);
        __global float* opposite_row = (__global float*) ((__global char*) eta_ptr_ + eta_pitch_*opposite_row_index);
        ghost_row[ti] = opposite_row[ti]; 
    }
}

__kernel void periodicBoundaryEtaKernel_EW(
        // Discretization parameters
        int nx_, int ny_,

        // Data
        __global float* eta_ptr_, int eta_pitch_) {
    
    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);
    
    __global float* const eta_row = (__global float*) ((__global char*) eta_ptr_ + eta_pitch_*tj);

    int opposite_col_index = nx_;
    if (ti == nx_+1) {
        opposite_col_index = 1;
    }
    
    // Set northern ghost cells
    if ( ((ti == 0) || (ti == nx_+1)) &&
          (tj < ny_+2) ) {
        eta_row[ti] = eta_row[opposite_col_index];
    }
}

