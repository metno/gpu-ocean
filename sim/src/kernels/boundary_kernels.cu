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

#include "common.cu"



/*
 *  These kernels assumes that values are defined in cell centers, and that the halo is symmetric in both north and south directions
 * 
 * Fix north-south boundary before east-west (to get the corners right)
 */

__global__ void periodicBoundary_NS(
	// Discretization parameters
        int nx_, int ny_,
	int halo_x, int halo_y,
	
        // Data
        __device__ float* h_ptr_, int h_pitch_,
        __device__ float* u_ptr_, int u_pitch_,
	__device__ float* v_ptr_, int v_pitch_) {

    // Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;

    int opposite_row_index = tj + ny_;
    if ( tj > ny_ + halo_y - 1) {
	opposite_row_index = tj - ny_;
    }
    
    // Set ghost cells equal to inner neighbour's value
    if ((tj < halo_y || tj >  ny_+halo_y-1)
	&& tj > -1  && tj < ny_+(2*halo_y) && ti > -1 && ti < nx_+(2*halo_x) ) {
	__device__ float* ghost_row_h = (__device__ float*) ((__device__ char*) h_ptr_ + h_pitch_*tj);
	__device__ float* opposite_row_h = (__device__ float*) ((__device__ char*) h_ptr_ + h_pitch_*opposite_row_index);

	__device__ float* ghost_row_u = (__device__ float*) ((__device__ char*) u_ptr_ + u_pitch_*tj);
	__device__ float* opposite_row_u = (__device__ float*) ((__device__ char*) u_ptr_ + u_pitch_*opposite_row_index);

	__device__ float* ghost_row_v = (__device__ float*) ((__device__ char*) v_ptr_ + v_pitch_*tj);
	__device__ float* opposite_row_v = (__device__ float*) ((__device__ char*) v_ptr_ + v_pitch_*opposite_row_index);

	ghost_row_h[ti] = opposite_row_h[ti];
	ghost_row_u[ti] = opposite_row_u[ti];
	ghost_row_v[ti] = opposite_row_v[ti];

    }
}


// Fix north-south boundary before east-west (to get the corners right)
__global__ void periodicBoundary_EW(
	// Discretization parameters
        int nx_, int ny_,
	int halo_x, int halo_y,

	// Data
        __device__ float* h_ptr_, int h_pitch_,
        __device__ float* u_ptr_, int u_pitch_,
	__device__ float* v_ptr_, int v_pitch_) {

    // Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;

    int opposite_col_index = ti + nx_;
    if ( ti > nx_ + halo_x - 1 ) {
	opposite_col_index = ti - nx_;
    }
    
    // Set ghost cells equal to inner neighbour's value
    if ( (ti > -1) && (ti < nx_+(2*halo_x)) &&
	 (tj > -1) && (tj < ny_+(2*halo_y))    ) {

	if ( (ti < halo_x) || (ti > nx_+halo_x-1) ) {
	    __device__ float* h_row = (__device__ float*) ((__device__ char*) h_ptr_ + h_pitch_*tj);
	    __device__ float* u_row = (__device__ float*) ((__device__ char*) u_ptr_ + u_pitch_*tj);
	    __device__ float* v_row = (__device__ float*) ((__device__ char*) v_ptr_ + v_pitch_*tj);
	    
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

__global__ void periodic_boundary_intersections_NS(
	// Discretization parameters
        int nx_, int ny_,
	int halo_x, int halo_y,
	
        // Data
        __device__ float* data_ptr_, int data_pitch_) {

    // Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;

    int opposite_row_index = tj + ny_;
    if ( tj > ny_ + halo_y - 1) {
	opposite_row_index = tj - ny_;
    }
    
    // Set ghost cells equal to inner opposite's value
    if ((tj < halo_y || tj >  ny_+halo_y-1)
	&& tj > -1  && tj < ny_+(2*halo_y)+1 && ti > -1 && ti < nx_+(2*halo_x)+1 ) {
	__device__ float* ghost_row = (__device__ float*) ((__device__ char*) data_ptr_ + data_pitch_*tj);
	__device__ float* opposite_row = (__device__ float*) ((__device__ char*) data_ptr_ + data_pitch_*opposite_row_index);

	ghost_row[ti] = opposite_row[ti];
    }
}

// Fix north-south boundary before east-west (to get the corners right)
__global__ void periodic_boundary_intersections_EW(
	// Discretization parameters
        int nx_, int ny_,
	int halo_x, int halo_y,

	// Data
	__device__ float* data_ptr_, int data_pitch_) {

    // Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;

    int opposite_col_index = ti + nx_;
    if ( ti > nx_ + halo_x - 1 ) {
	opposite_col_index = ti - nx_;
    }
    
    // Set ghost cells equal to inner neighbour's value
    if ( (ti > -1) && (ti < nx_+2*halo_x + 1) &&
	 (tj > -1) && (tj < ny_+2*halo_y + 1)    ) {

	if ( (ti < halo_x) || (ti > nx_+halo_x-1) ) {
	    __device__ float* data_row = (__device__ float*) ((__device__ char*) data_ptr_ + data_pitch_*tj);
	    
	    data_row[ti] = data_row[opposite_col_index];
	}
    }
}


/*
 *  These kernels handles wall boundary conditions for values defined on cell intersections, and assumes that the halo consists of the same number of ghost cells on each periodic boundary.
 * 
 */

__global__ void closed_boundary_intersections_EW(
	// Discretization parameters
        int nx_, int ny_,
	int halo_x_, int halo_y_,
	
        // Data
        __device__ float* data_ptr_, int data_pitch_) {

    // Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;

    
    if ( ti == 0 && tj < ny_ + (2*halo_x_) + 1) {
	__device__ float* data_row = (__device__ float*) ((__device__ char*) data_ptr_ + data_pitch_*tj);
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

__global__ void closed_boundary_intersections_NS(
	// Discretization parameters
        int nx_, int ny_,
	int halo_x_, int halo_y_,

	// Data
	__device__ float* data_ptr_, int data_pitch_) {

    // Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;

    if ( tj == 0 && ti < ny_ + (2*halo_y_) +1) {
	// Southern boundary:
	for (int j = 0; j < halo_y_; ++j) {
	    const int inner_index = 2*halo_y_ - j;
	    __device__ float* ghost_row = (__device__ float*) ((__device__ char*) data_ptr_ + data_pitch_*j);
	    __device__ float* inner_row = (__device__ float*) ((__device__ char*) data_ptr_ + data_pitch_*inner_index);
	    ghost_row[ti] = inner_row[ti];
	}
	// Northern boundary:
	for (int j = 0; j < halo_y_; ++j) {
	    const int ghost_index = ny_ + 2*halo_y_ - j;
	    const int inner_index = ny_ + j;
	    __device__ float* ghost_row = (__device__ float*) ((__device__ char*) data_ptr_ + data_pitch_*ghost_index);
	    __device__ float* inner_row = (__device__ float*) ((__device__ char*) data_ptr_ + data_pitch_*inner_index);
	    ghost_row[ti] = inner_row[ti];
	}
	
    }
}



/*
 *  These kernels implement open boundary conditions as linear interpolation.
 * 
 *  Assume that the outermost row has the desired values already
 * 
 *  The computational domain is (nx, ny), meaning that the data is defined on a 
 *  domain (nx + 2*halo_x_, ny + 2*halo_y). The domain of interest is however smaller, and 
 *  bounded by the sponge_cells.
 *  With sponge in both south and norht, the sponge area is defined as follows:
 *  In the south: (0, sponge_cells_south-1) 
 *  In the north: (ny + 2*halo_x - sponge_cells_north, ny + 2*halo_y-1)
 *
 *  This is true regardless if the boundary on the opposite side is a sponge or not.
 */

__global__ void linearInterpolation_NS(
	// Discretization parameters
	int boundary_condition_north_, int boundary_condition_south_,
	int nx_, int ny_,
	int halo_x_, int halo_y_,
	int sponge_cells_north_,
	int sponge_cells_south_,
	
        // Data
        __device__ float* h_ptr_, int h_pitch_,
        __device__ float* u_ptr_, int u_pitch_,
	__device__ float* v_ptr_, int v_pitch_) {

    // Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;

    // Extrapolate on northern and southern boundary:
    // Keep outer edge as is!
    if (( ((boundary_condition_south_ == 4)
	   &&(tj < sponge_cells_south_) && (tj > 0)) ||
	  ((boundary_condition_north_ == 4)
	   &&(tj > ny_ + 2*halo_y_ - 1 - sponge_cells_north_) && (tj < ny_ + 2*halo_y_ -1)) )
	&& (ti > 0) && (ti < nx_ + 2*halo_x_-1) ) {

	// Identify inner and outer row
	int inner_row = sponge_cells_south_;
	int outer_row = 0;
	if (tj > sponge_cells_south_) {
	    inner_row = ny_ + 2*halo_y_ - 1 - sponge_cells_north_;
	    outer_row = ny_ + 2*halo_y_ - 1;
	}
	
	// Get inner value
	__device__ float* inner_row_h = (__device__ float*) ((__device__ char*) h_ptr_ + h_pitch_*inner_row);
	__device__ float* inner_row_u = (__device__ float*) ((__device__ char*) u_ptr_ + u_pitch_*inner_row);
	__device__ float* inner_row_v = (__device__ float*) ((__device__ char*) v_ptr_ + v_pitch_*inner_row);
	float inner_value_h = inner_row_h[ti];
	float inner_value_u = inner_row_u[ti];
	float inner_value_v = inner_row_v[ti];

	// Get outer value
	__device__ float* outer_row_h = (__device__ float*) ((__device__ char*) h_ptr_ + h_pitch_*outer_row);
	__device__ float* outer_row_u = (__device__ float*) ((__device__ char*) u_ptr_ + u_pitch_*outer_row);
	__device__ float* outer_row_v = (__device__ float*) ((__device__ char*) v_ptr_ + v_pitch_*outer_row);
	float outer_value_h = outer_row_h[ti];
	float outer_value_u = outer_row_u[ti];
	float outer_value_v = outer_row_v[ti];

	// Find target cell
	__device__ float* target_row_h = (__device__ float*) ((__device__ char*) h_ptr_ + h_pitch_*tj);
	__device__ float* target_row_u = (__device__ float*) ((__device__ char*) u_ptr_ + u_pitch_*tj);
	__device__ float* target_row_v = (__device__ float*) ((__device__ char*) v_ptr_ + v_pitch_*tj);
	
	// Interpolate:
	float ratio = ((float)(tj - outer_row))/(inner_row - outer_row);
	target_row_h[ti] = outer_value_h + ratio*(inner_value_h - outer_value_h);
	target_row_u[ti] = outer_value_u + ratio*(inner_value_u - outer_value_u);
	target_row_v[ti] = outer_value_v + ratio*(inner_value_v - outer_value_v);
    }
}

    
__global__ void linearInterpolation_EW(
	// Discretization parameters
	int boundary_condition_east_, int boundary_condition_west_,
        int nx_, int ny_,
	int halo_x_, int halo_y_,
	int sponge_cells_east_,
	int sponge_cells_west_,
	
        // Data
        __device__ float* h_ptr_, int h_pitch_,
        __device__ float* u_ptr_, int u_pitch_,
	__device__ float* v_ptr_, int v_pitch_) {
    
    // Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;

    // Extrapolate on northern and southern boundary:
    // Keep outer edge as is!
    if (( ((boundary_condition_west_ == 4)
	   &&(ti < sponge_cells_west_) && (ti > 0)) ||
	  ((boundary_condition_east_ == 4)
	   &&(ti > nx_ + 2*halo_x_ - 1 - sponge_cells_east_) && (ti < nx_ + 2*halo_x_ -1)) )
	&& (tj > 0) && (tj < ny_ + 2*halo_y_-1) ) {

	// Identify inner and outer row
	int inner_col = sponge_cells_west_;
	int outer_col = 0;
	if (ti > sponge_cells_west_) {
	    inner_col = nx_ + 2*halo_x_ - 1 - sponge_cells_east_;
	    outer_col = nx_ + 2*halo_x_ - 1;
	}

	// Get rows
	__device__ float* h_row = (__device__ float*) ((__device__ char*) h_ptr_ + h_pitch_*tj);
	__device__ float* u_row = (__device__ float*) ((__device__ char*) u_ptr_ + u_pitch_*tj);
	__device__ float* v_row = (__device__ float*) ((__device__ char*) v_ptr_ + v_pitch_*tj);
	
	// Get inner value
	float inner_value_h = h_row[inner_col];
	float inner_value_u = u_row[inner_col];
	float inner_value_v = v_row[inner_col];

	// Get outer value
	float outer_value_h = h_row[outer_col];
	float outer_value_u = u_row[outer_col];
	float outer_value_v = v_row[outer_col];

	// Interpolate:
	float ratio = ((float)(ti - outer_col))/(inner_col - outer_col);
	h_row[ti] = outer_value_h + ratio*(inner_value_h - outer_value_h);
	u_row[ti] = outer_value_u + ratio*(inner_value_u - outer_value_u);
	v_row[ti] = outer_value_v + ratio*(inner_value_v - outer_value_v);
    }
}


/*
 *  These kernels implement open boundary conditions using the Flow Relaxation Scheme
 * 
 *  Assume that the outermost row has the desired values representing the exterior solution.
 * 
 *  The computational domain is (nx, ny), meaning that the data is defined on a 
 *  domain (nx + 2*halo_x_, ny + 2*halo_y). The domain of interest is however smaller, and 
 *  bounded by the sponge_cells.
 *  With sponge in both south and norht, the sponge area is defined as follows:
 *  In the south: (0, sponge_cells_south-1) 
 *  In the north: (ny + 2*halo_x - sponge_cells_north, ny + 2*halo_y-1)
 *
 *  This is true regardless if the boundary on the opposite side is a sponge or not.
 */
__global__ void flowRelaxationScheme_NS(
	// Discretization parameters
	int boundary_condition_north_, int boundary_condition_south_,
	int nx_, int ny_,
	int halo_x_, int halo_y_,
	int sponge_cells_north_,
	int sponge_cells_south_,
	
        // Data
        __device__ float* h_ptr_, int h_pitch_,
        __device__ float* u_ptr_, int u_pitch_,
	__device__ float* v_ptr_, int v_pitch_) {

    // Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;

    // Extrapolate on northern and southern boundary:
    // Keep outer edge as is!
    if (( ((boundary_condition_south_ == 3)
	   &&(tj < sponge_cells_south_) && (tj > 0)) ||
	  ((boundary_condition_north_ == 3)
	   &&(tj > ny_ + 2*halo_y_ - 1 - sponge_cells_north_) && (tj < ny_ + 2*halo_y_ -1)) )
	&& (ti > 0) && (ti < nx_ + 2*halo_x_-1) ) {

	int exterior_row = 0;
	int j = tj;
	if (tj > sponge_cells_south_) {
	    exterior_row = ny_ + 2*halo_y_ - 1;
	    j = (ny_ + 2*halo_y_ -1) - tj;
	}
	//int current_col = ti;
	float alpha = 1.0f - tanh((j-1.0f)/2.0f);
	
	
	// Get exterior value
	__device__ float* exterior_row_h = (__device__ float*) ((__device__ char*) h_ptr_ + h_pitch_*exterior_row);
	__device__ float* exterior_row_u = (__device__ float*) ((__device__ char*) u_ptr_ + u_pitch_*exterior_row);
	__device__ float* exterior_row_v = (__device__ float*) ((__device__ char*) v_ptr_ + v_pitch_*exterior_row);
	float exterior_value_h = exterior_row_h[ti];
	float exterior_value_u = exterior_row_u[ti];
	float exterior_value_v = exterior_row_v[ti];

	// Find target cell
	__device__ float* target_row_h = (__device__ float*) ((__device__ char*) h_ptr_ + h_pitch_*tj);
	__device__ float* target_row_u = (__device__ float*) ((__device__ char*) u_ptr_ + u_pitch_*tj);
	__device__ float* target_row_v = (__device__ float*) ((__device__ char*) v_ptr_ + v_pitch_*tj);
	
	// Interpolate:
	target_row_h[ti] = (1.0f-alpha)*target_row_h[ti] + alpha*exterior_value_h;
	target_row_u[ti] = (1.0f-alpha)*target_row_u[ti] + alpha*exterior_value_u;	
	target_row_v[ti] = (1.0f-alpha)*target_row_v[ti] + alpha*exterior_value_v;
	
    }
}


__global__ void flowRelaxationScheme_EW(
	// Discretization parameters
	int boundary_condition_east_, int boundary_condition_west_,
        int nx_, int ny_,
	int halo_x_, int halo_y_,
	int sponge_cells_east_,
	int sponge_cells_west_,
	
        // Data
        __device__ float* h_ptr_, int h_pitch_,
        __device__ float* u_ptr_, int u_pitch_,
	__device__ float* v_ptr_, int v_pitch_) {
    
    // Index of cell within domain
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;

    // Extrapolate on northern and southern boundary:
    // Keep outer edge as is!
    if (( ((boundary_condition_west_ == 3)
	   &&(ti < sponge_cells_west_) && (ti > 0)) ||
	  ((boundary_condition_east_ == 3)
	   &&(ti > nx_ + 2*halo_x_ - 1 - sponge_cells_east_) && (ti < nx_ + 2*halo_x_ -1)) )
	&& (tj > 0) && (tj < ny_ + 2*halo_y_-1) ) {

	int exterior_col = 0;
	int j = ti;
	if (ti > sponge_cells_west_) {
	    exterior_col = nx_ + 2*halo_x_ - 1;
	    j = (nx_ + 2*halo_x_ -1) - ti;
	}
	//int current_col = ti;
	float alpha = 1.0f - tanh((j-1.0f)/2.0f);
	
	// Get rows
	__device__ float* h_row = (__device__ float*) ((__device__ char*) h_ptr_ + h_pitch_*tj);
	__device__ float* u_row = (__device__ float*) ((__device__ char*) u_ptr_ + u_pitch_*tj);
	__device__ float* v_row = (__device__ float*) ((__device__ char*) v_ptr_ + v_pitch_*tj);

	// Get exterior value
	float exterior_value_h = h_row[exterior_col];
	float exterior_value_u = u_row[exterior_col];
	float exterior_value_v = v_row[exterior_col];

	// Interpolate:
	h_row[ti] = (1.0f-alpha)*h_row[ti] + alpha*exterior_value_h;
	u_row[ti] = (1.0f-alpha)*u_row[ti] + alpha*exterior_value_u;
	v_row[ti] = (1.0f-alpha)*v_row[ti] + alpha*exterior_value_v;
    }
}
