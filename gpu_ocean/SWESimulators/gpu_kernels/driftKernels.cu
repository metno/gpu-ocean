/*
This software is part of GPU Ocean. 

Copyright (C) 2018 SINTEF Digital
Copyright (C) 2018 Norwegian Meteorological Institute

This CUDA kernel implements a selection of drift trajectory algorithms.

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



/**
  * Kernel that evolves drifter positions along u and v.
  */
  
//Code relating to wind-data

texture<float, cudaTextureType2D> wind_X_current;
texture<float, cudaTextureType2D> wind_X_next;

texture<float, cudaTextureType2D> wind_Y_current;
texture<float, cudaTextureType2D> wind_Y_next;

__device__ float windX(float wind_t_, float ti_, float tj_, int nx_, int ny_) {
    
    //Normalize coordinates (to [0, 1])
    const float s = ti_ / float(nx_);
    const float t = tj_ / float(ny_);
    
    //Look up current and next timestep (using bilinear texture interpolation)
    float current = tex2D(wind_X_current, s, t);
    float next = tex2D(wind_X_next, s, t);
    
    //Interpolate in time
    return wind_t_*next + (1.0f - wind_t_)*current;
}

__device__ float windY(float wind_t_, float ti_, float tj_, int nx_, int ny_) {
    
    //Normalize coordinates (to [0, 1])
    const float s = ti_ / float(nx_);
    const float t = tj_ / float(ny_);
    
    //Look up current and next timestep (using bilinear texture interpolation)
    float current = tex2D(wind_Y_current, s, t);
    float next = tex2D(wind_Y_next, s, t);
    
    //Interpolate in time
    return wind_t_*next + (1.0f - wind_t_)*current;
}



__device__ float currentVelocityU(
        float* eta_ptr_, const int eta_pitch_,
        float* hu_ptr_, const int hu_pitch_,
        float* Hm_ptr_, const int Hm_pitch_,
        int cell_id_x, int cell_id_y) {
    
    // Read the water velocity from global memory
    float* const eta_row_y = (float*) ((char*) eta_ptr_ + eta_pitch_*cell_id_y);
    float* const Hm_row_y = (float*) ((char*) Hm_ptr_ + Hm_pitch_*cell_id_y);
    float const h = Hm_row_y[cell_id_x] + eta_row_y[cell_id_x];

    float* const hu_row = (float*) ((char*) hu_ptr_ + hu_pitch_*cell_id_y);
    
    float const u = hu_row[cell_id_x]/h;
    
    return u;
}

__device__ float currentVelocityV(
        float* eta_ptr_, const int eta_pitch_,
        float* hv_ptr_, const int hv_pitch_, 
        float* Hm_ptr_, const int Hm_pitch_,
        int cell_id_x, int cell_id_y) {
    
    // Read the water velocity from global memory
    float* const eta_row_y = (float*) ((char*) eta_ptr_ + eta_pitch_*cell_id_y);
    float* const Hm_row_y = (float*) ((char*) Hm_ptr_ + Hm_pitch_*cell_id_y);
    float const h = Hm_row_y[cell_id_x] + eta_row_y[cell_id_x];

    float* const hv_row = (float*) ((char*) hv_ptr_ + hv_pitch_*cell_id_y);
    
    float const v = hv_row[cell_id_x]/h;
    
    return v;
}

extern "C" {
__global__ void passiveDrifterKernel(
        //Discretization parameters
        const int nx_, const int ny_,
        const float dx_, const float dy_, const float dt_,

        const int x_zero_reference_cell_, // the cell column representing x0 (x0 at western face)
        const int y_zero_reference_cell_, // the cell row representing y0 (y0 at southern face)
        
        // Data
        float* eta_ptr_, const int eta_pitch_,
        float* hu_ptr_, const int hu_pitch_,
        float* hv_ptr_, const int hv_pitch_,
        // H should be read from buffer, but for now we use a constant value
        float* Hm_ptr_, const int Hm_pitch_,

        const int periodic_north_south_,
        const int periodic_east_west_,
        
        const int num_drifters_,
        float* drifters_positions_, const int drifters_pitch_,
        const float sensitivity_,
        const float wind_t_, 
        const float wind_drift_factor_) 
        {

    //Index of thread within block (only needed in one dim)
    const int tx = threadIdx.x;
        
    //Index of block within domain (only needed in one dim)
    const int bx = blockDim.x * blockIdx.x;
        
    //Index of cell within domain (only needed in one dim)
    const int ti = bx + tx;
    
    if (ti < num_drifters_ + 1) {
        // Obtain pointer to our particle:
        float* drifter = (float*) ((char*) drifters_positions_ + drifters_pitch_*ti);
        float drifter_pos_x = drifter[0];
        float drifter_pos_y = drifter[1];
        
        // Find cell ID for the cell in which our particle is
        int const cell_id_x = (int)(ceil(drifter_pos_x/dx_) + x_zero_reference_cell_);
        int const cell_id_y = (int)(ceil(drifter_pos_y/dy_) + y_zero_reference_cell_);
        
        float const frac_x = drifter_pos_x / dx_ - floor(drifter_pos_x / dx_);
        float const frac_y = drifter_pos_y / dy_ - floor(drifter_pos_y / dy_);
        
        int cell_id_x0;
        int cell_id_x1;
        float x_factor;
        
        if (frac_x < 0.5) {
            cell_id_x0 = cell_id_x - 1;
            cell_id_x1 = cell_id_x;
            x_factor = 0.5 + frac_x;
            }
        else {
            cell_id_x0 = cell_id_x;
            cell_id_x1 = cell_id_x + 1;
            x_factor = frac_x - 0.5;
            }
        
        int cell_id_y0;
        int cell_id_y1;
        float y_factor;
        
        if (frac_y < 0.5) {
            cell_id_y0 = cell_id_y - 1;
            cell_id_y1 = cell_id_y;
            y_factor = 0.5 + frac_y;
            }
        else {
            cell_id_y0 = cell_id_y;
            cell_id_y1 = cell_id_y + 1;
            y_factor = frac_y - 0.5;
            }

        float u_x0y0 = currentVelocityU(eta_ptr_, eta_pitch_,hu_ptr_, hu_pitch_,Hm_ptr_, Hm_pitch_,cell_id_x0, cell_id_y0);
        float u_x1y0 = currentVelocityU(eta_ptr_, eta_pitch_,hu_ptr_, hu_pitch_,Hm_ptr_, Hm_pitch_,cell_id_x1, cell_id_y0);
        float u_x0y1 = currentVelocityU(eta_ptr_, eta_pitch_,hu_ptr_, hu_pitch_,Hm_ptr_, Hm_pitch_,cell_id_x0, cell_id_y1);
        float u_x1y1 = currentVelocityU(eta_ptr_, eta_pitch_,hu_ptr_, hu_pitch_,Hm_ptr_, Hm_pitch_,cell_id_x1, cell_id_y1);
        
        float v_x0y0 = currentVelocityV(eta_ptr_, eta_pitch_,hv_ptr_, hv_pitch_,Hm_ptr_, Hm_pitch_,cell_id_x0, cell_id_y0);
        float v_x1y0 = currentVelocityV(eta_ptr_, eta_pitch_,hv_ptr_, hv_pitch_,Hm_ptr_, Hm_pitch_,cell_id_x1, cell_id_y0);
        float v_x0y1 = currentVelocityV(eta_ptr_, eta_pitch_,hv_ptr_, hv_pitch_,Hm_ptr_, Hm_pitch_,cell_id_x0, cell_id_y1);
        float v_x1y1 = currentVelocityV(eta_ptr_, eta_pitch_,hv_ptr_, hv_pitch_,Hm_ptr_, Hm_pitch_,cell_id_x1, cell_id_y1);

        if (wind_drift_factor_) {
            u_x0y0 = u_x0y0 + windX(wind_t_, cell_id_x0+0.5, cell_id_y0+0.5, nx_, ny_) * wind_drift_factor_;
            u_x1y0 = u_x1y0 + windX(wind_t_, cell_id_x1+0.5, cell_id_y0+0.5, nx_, ny_) * wind_drift_factor_;
            u_x0y1 = u_x0y1 + windX(wind_t_, cell_id_x0+0.5, cell_id_y1+0.5, nx_, ny_) * wind_drift_factor_;
            u_x1y1 = u_x1y1 + windX(wind_t_, cell_id_x1+0.5, cell_id_y1+0.5, nx_, ny_) * wind_drift_factor_;
            
            v_x0y0 = v_x0y0 + windY(wind_t_, cell_id_x0+0.5, cell_id_y0+0.5, nx_, ny_) * wind_drift_factor_;
            v_x1y0 = v_x1y0 + windY(wind_t_, cell_id_x1+0.5, cell_id_y0+0.5, nx_, ny_) * wind_drift_factor_;
            v_x0y1 = v_x0y1 + windY(wind_t_, cell_id_x0+0.5, cell_id_y1+0.5, nx_, ny_) * wind_drift_factor_;
            v_x1y1 = v_x1y1 + windY(wind_t_, cell_id_x1+0.5, cell_id_y1+0.5, nx_, ny_) * wind_drift_factor_;
            }
        
        float const u_y0 = (1-x_factor)*u_x0y0 + x_factor * u_x1y0; 
        float const u_y1 = (1-x_factor)*u_x0y1 + x_factor * u_x1y1; 
        
        float const v_y0 = (1-x_factor)*v_x0y0 + x_factor * v_x1y0; 
        float const v_y1 = (1-x_factor)*v_x0y1 + x_factor * v_x1y1;
        
        float const u = (1-y_factor)*u_y0 + y_factor *u_y1;
        float const v = (1-y_factor)*v_y0 + y_factor *v_y1;
        
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
        if (periodic_north_south_ && (drifter_pos_y > ny_*dy_)) {
            drifter_pos_y -= ny_*dy_;
        }

        // Write to global memory
        drifter[0] = drifter_pos_x;
        drifter[1] = drifter_pos_y;
    }
}
} // extern "C"
    

extern "C" {
__global__ void enforceBoundaryConditions(
        //domain parameters
        float domain_size_x_, float domain_size_y_,

        int periodic_north_south_,
        int periodic_east_west_,
        
        int num_drifters_,
        float* drifters_positions_, int drifters_pitch_) {
    
    //Index of drifter (only needed in one dimension)
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ti < num_drifters_ + 1) {
        // Obtain pointer to our particle:
        float* drifter = (float*) ((char*) drifters_positions_ + drifters_pitch_*ti);
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
} // extern "C"
