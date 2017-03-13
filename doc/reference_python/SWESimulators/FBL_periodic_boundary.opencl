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


__kernel void periodicBoundaryUKernel(
        // Discretization parameters
	int nx_, int ny_,

	// Data
	__global float* U_ptr_, int U_pitch_) {

    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    

    // Compute point to rows "tj" of U arrays
    __global float* const U_row = (__global float*) ((__global char*) U_ptr_ + U_pitch_*tj);
    if ( ti == 0 && tj < ny_) {
	U_row[0] = U_row[nx_-1];
    }
    if ( ti == nx_ && tj < ny_ ) {
	U_row[nx_] = U_row[1];
    }
      
}


__kernel void periodicBoundaryVKernel(
        // Discretization parameters
	int nx_, int ny_,

	// Data
	__global float* V_ptr_, int V_pitch_) {


    // Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1);

    if (tj == 0 && ti < nx_) {
	__global float* const V_top_row = (__global float*) ((__global char*) V_ptr_ + V_pitch_*(ny_ - 1));
	__global float* const V_lower_boundary = (__global float*)((__global char*) V_ptr_ + V_pitch_*0);
	V_lower_boundary[ti] = V_top_row[ti];
	//V_top_row[ti] = 1;
    }

    if (tj == ny_ && ti < nx_) {
	__global float* const V_upper_boundary = (__global float*)((__global char*) V_ptr_ + V_pitch_*ny_);
	__global float* const V_bottom_row = (__global float*)((__global char*) V_ptr_ + V_pitch_*1);
	V_upper_boundary[ti] = V_bottom_row[ti];

    }

}


