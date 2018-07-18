/*
This OpenCL kernel implements a selection of drift trajectory algorithms.

Copyright (C) 2018  SINTEF ICT

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

//#include "../config.h"

#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#define CLK_LOCAL_MEM_FENCE
#endif


/**
  * Kernel that evolves drifter positions along u and v.
  */
__kernel void passiveDrifterKernel(
        //Discretization parameters
        int nx_, int ny_,
        float dx_, float dy_, float dt_,

	float x_zero_reference_cell_, // the cell column representing x0 (x0 at western face)
	float y_zero_reference_cell_, // the cell row representing y0 (y0 at southern face)
	
	// Data
        __global float* eta_ptr_, int eta_pitch_,
        __global float* hu_ptr_, int hu_pitch_,
        __global float* hv_ptr_, int hv_pitch_,
	// H should be read from buffer, but for now we use a constant value
	//__global float* H_ptr_, int H_pitch_,
	float H_,

	int periodic_north_south_,
	int periodic_east_west_,
	
	int num_drifters_,
	__global float* drifters_positions_, int drifters_pitch_,
	float sensitivity_) {
    //Index of thread within block
    const int tx = get_local_id(0);
    const int ty = get_local_id(1); // Should be 0
    
    //Index of block within domain
    const int bx = get_local_size(0) * get_group_id(0);
    const int by = get_local_size(1) * get_group_id(1); // Should be 0
    
    //Index of cell within domain
    const int ti = get_global_id(0);
    const int tj = get_global_id(1); // Should be 0

    if (ti < num_drifters_ + 1) {
	// Obtain pointer to our particle:
	__global float* drifter = (__global float*) ((__global char*) drifters_positions_ + drifters_pitch_*ti);
	float drifter_pos_x = drifter[0];
	float drifter_pos_y = drifter[1];

	// Find cell ID for the cell in which our particle is
	int cell_id_x = (int)(ceil(drifter_pos_x/dx_) + x_zero_reference_cell_);
	int cell_id_y = (int)(ceil(drifter_pos_y/dy_) + y_zero_reference_cell_);

	// Read the water velocity from global memory
	__global float* eta_row = (__global float*) ((__global char*) eta_ptr_ + eta_pitch_*cell_id_y);
	float h = H_ + eta_row[cell_id_x];

	__global float* hu_row = (__global float*) ((__global char*) hu_ptr_ + hu_pitch_*cell_id_y);
	float u = hu_row[cell_id_x]/h;

	__global float* hv_row = (__global float*) ((__global char*) hv_ptr_ + hv_pitch_*cell_id_y);
	float v = hv_row[cell_id_x]/h;

	// Move drifter
	drifter_pos_x += sensitivity_*u*dt_;
	drifter_pos_y += sensitivity_*v*dt_;

	// Ensure boundary conditions
	if (periodic_east_west_ && (drifter_pos_x < 0)) {
	    drifter_pos_x += + nx_*dx_;
	}
	if (periodic_east_west_ && (drifter_pos_x > nx_*dx_)) {
	    drifter_pos_x -= nx_*dx_;
	}
	if (periodic_north_south_ && (drifter_pos_y < 0)) {
	    drifter_pos_y += ny_*dy_;
	}
	if (periodic_north_south_ && (drifter_pos_y > nx_*dx_)) {
	    drifter_pos_y -= ny_*dy_;
	}

	// Write to global memory
	drifter[0] = drifter_pos_x;
	drifter[1] = drifter_pos_y;
    }
}


__kernel void enforceBoundaryConditions(
        //domain parameters
	float domain_size_x_, float domain_size_y_,

	int periodic_north_south_,
	int periodic_east_west_,
	
	int num_drifters_,
	__global float* drifters_positions_, int drifters_pitch_) {
    
    //Index of drifter
    const int ti = get_global_id(0);
    const int tj = get_global_id(1); // Should be 0

    if (ti < num_drifters_ + 1) {
	// Obtain pointer to our particle:
	__global float* drifter = (__global float*) ((__global char*) drifters_positions_ + drifters_pitch_*ti);
	float drifter_pos_x = drifter[0];
	float drifter_pos_y = drifter[1];

	// Ensure boundary conditions
	if (periodic_east_west_ && (drifter_pos_x < 0)) {
	    drifter_pos_x += + domain_size_x_;
	}
	if (periodic_east_west_ && (drifter_pos_x > domain_size_x_)) {
	    drifter_pos_x -= domain_size_x_;
	}
	if (periodic_north_south_ && (drifter_pos_y < 0)) {
	    drifter_pos_y += domain_size_y_;
	}
	if (periodic_north_south_ && (drifter_pos_y > domain_size_y_)) {
	    drifter_pos_y -= domain_size_y_;
	}

	// Write to global memory
	drifter[0] = drifter_pos_x;
	drifter[1] = drifter_pos_y;
    }
}

