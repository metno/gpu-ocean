/*
This software is part of GPU Ocean. 

Copyright (C) 2018 SINTEF Digital
Copyright (C) 2018 Norwegian Meteorological Institute

This OpenCL kernel implements part of the Forward Backward Linear 
numerical scheme for the shallow water equations, described in 
L. P. Røed, "Documentation of simple ocean models for use in ensemble
predictions", Met no report 2012/3 and 2012/5 .

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
        int nx_halo_, int ny_halo_,

        // Data
        __global float* U_ptr_, int U_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    //Compute pointer to current row in the U array
    __global float* const U_row = (__global float*) ((__global char*) U_ptr_ + U_pitch_*tj);	
    
    if ( (ti ==0 || ti == nx_) && tj < ny_halo_) {
        U_row[ti] = 0.0f;
    }    
}

__kernel void periodicBoundaryUKernel(
        // Discretization parameters
        int nx_, int ny_,
        int nx_halo_, int ny_halo_,

        // Data
        __global float* U_ptr_, int U_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    // Set periodic boundary
    // Compute pointers to rows "tj" of U arrays
    if ( ti == 0 && tj < ny_halo_) {
        __global float* const U_row = (__global float*) ((__global char*) U_ptr_ + U_pitch_*tj);
        U_row[0] = U_row[nx_];
    }
}

__kernel void updateGhostCellsUKernel(
        // Discretization parameters
        int nx_, int ny_,
        int nx_halo_, int ny_halo_,

        // Data
        __global float* U_ptr_, int U_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);
    
    // Set ghost cells on upper domain
    if (tj == ny_ && ti < nx_+1 && ny_halo_ > ny_) {
        __global float* const U_ghost = (__global float*) ((__global char*) U_ptr_ + U_pitch_*ny_);
        __global float* const U_lower = (__global float*) ((__global char*) U_ptr_ + U_pitch_*0);

        U_ghost[ti] = U_lower[ti];
    }
      
}


__kernel void periodicBoundaryVKernel(
        // Discretization parameters
        int nx_, int ny_,
        int nx_halo_, int ny_halo_,

        // Data
        __global float* V_ptr_, int V_pitch_) {


    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    // Set periodic boundary
    if (tj == 0 && ti < nx_halo_) {
        __global float* const V_top_row = (__global float*) ((__global char*) V_ptr_ + V_pitch_*ny_);
        __global float* const V_lower_boundary = (__global float*)((__global char*) V_ptr_ + V_pitch_*0);
        V_lower_boundary[ti] = V_top_row[ti];
    }
}

__kernel void closedBoundaryVKernel(
        // Discretization parameters
        int nx_, int ny_,
        int nx_halo_, int ny_halo_,

        // Data
        __global float* V_ptr_, int V_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

        //Compute pointer to current row in the V array
    __global float* const V_row = (__global float*) ((__global char*) V_ptr_ + V_pitch_*tj);	
    
    if ( (tj ==0 || tj == ny_) && ti < nx_halo_) {
        V_row[ti] = 0.0f;
    }    
}


    
__kernel void updateGhostCellsVKernel(
        // Discretization parameters
        int nx_, int ny_,
        int nx_halo_, int ny_halo_,

        // Data
        __global float* V_ptr_, int V_pitch_) {


    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    // Set ghost cells on east domain
    if (ti == nx_ && tj < ny_+1 && nx_halo_ > nx_) {
        __global float* const V_row = (__global float*) ((__global char*) V_ptr_ + V_pitch_*tj);
        V_row[nx_] = V_row[0];
    }
}


__kernel void periodicBoundaryEtaKernel(
        // Discretization parameters
        int nx_, int ny_,
        int nx_halo_, int ny_halo_,

        // Data
        __global float* eta_ptr_, int eta_pitch_) {
    
    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);
    
    __global float* const eta_row = (__global float*) ((__global char*) eta_ptr_ + eta_pitch_*tj);

    
    // Set northern ghost cells
    if (tj == ny_ && ti < nx_ && ny_halo_ > ny_) {
        // eta_row is eta_north
        __global float* const eta_bottom = (__global float*) ((__global char*) eta_ptr_ + eta_pitch_*0);
        eta_row[ti] = eta_bottom[ti];
    }

    // Set eastern ghost cells
    if (ti == nx_ && tj < ny_ && nx_halo_ > nx_) {
        eta_row[ti] = eta_row[0];
    }
}
