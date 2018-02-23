
/**
 * 2nd order Runge-Kutta time integration
 */
__kernel void RungeKutta(
                         int nx_, int ny_,
                         // Input [h, hu, hv] at time n
                         __global float* U1_ptr_, int U1_pitch_,
                         __global float* U2_ptr_, int U2_pitch_,
                         __global float* U3_ptr_, int U3_pitch_,
                         
                         // flux in/out of each cell
                         __global float* R1_ptr_, int R1_pitch_,
                         __global float* R2_ptr_, int R2_pitch_,
                         __global float* R3_ptr_, int R3_pitch_,
                         
                         // Input/output [h, hu, hv] at time n+1
                         __global float* Q1_ptr_, int Q1_pitch_,
                         __global float* Q2_ptr_, int Q2_pitch_,
                         __global float* Q3_ptr_, int Q3_pitch_,
                         
                         //Bottom topography at midpoints
                         __global float* Hm_ptr_, int Hm_pitch_,
                        
                        float r_, //< Bottom friction coefficient
                         
                         __global float* dt_ptr_, //< Size of timestep to take
                        
                         int step_ //< first or second step in RK2
                         ) {
                             
    //Index of cell within domain
    const int ti = get_global_id(0) + 2; //Skip global ghost cells, i.e., +2
    const int tj = get_global_id(1) + 2;
    
    //Don't process outside our domain
    if (ti <= 1 || ti >= nx_+2 || tj <= 1 || tj >= ny_+2) {
        return;
    }
	
    __global float* const U1_row = (__global float*) ((__global char*) U1_ptr_ + U1_pitch_*tj);
    __global float* const U2_row = (__global float*) ((__global char*) U2_ptr_ + U2_pitch_*tj);
    __global float* const U3_row = (__global float*) ((__global char*) U3_ptr_ + U3_pitch_*tj);
    
	const float U1 = U1_row[ti];
	const float U2 = U2_row[ti];
	const float U3 = U3_row[ti];

    __global float* const R1_row = (__global float*) ((__global char*) R1_ptr_ + R1_pitch_*tj);
    __global float* const R2_row = (__global float*) ((__global char*) R2_ptr_ + R2_pitch_*tj);
    __global float* const R3_row = (__global float*) ((__global char*) R3_ptr_ + R3_pitch_*tj);
    
	const float R1 = R1_row[ti];
	const float R2 = R2_row[ti];
	const float R3 = R3_row[ti];
    
    __global float* const Q1_row = (__global float*) ((__global char*) Q1_ptr_ + Q1_pitch_*tj);
    __global float* const Q2_row = (__global float*) ((__global char*) Q2_ptr_ + Q2_pitch_*tj);
    __global float* const Q3_row = (__global float*) ((__global char*) Q3_ptr_ + Q3_pitch_*tj);
    
    const float dt = dt_ptr_[0];
    
    // Read H in mid-cell:
    __global float* const Hm_row  = (__global float*) ((__global char*) Hm_ptr_ + Hm_pitch_*tj);
    const float Hm = Hm_row[ti];

    //Friction coefficient
    const float C = 2.0f*r_*dt/(U1+Hm);
                    
    //First step of RK2 ODE integrator
    if  (step_ == 0) {
        Q1_row[ti] =  U1 + dt*R1;
        Q2_row[ti] = (U2 + dt*R2) / (1.0f + C);
        Q3_row[ti] = (U3 + dt*R3) / (1.0f + C);
    }
    //Second step of RK2 ODE integrator
    else if (step_ == 1) {
        float Q1 = Q1_row[ti];
        float Q2 = Q2_row[ti];
        float Q3 = Q3_row[ti];
        
        //Compute Q^n+1
        Q1_row[ti] = 0.5f*(Q1 + (U1 + dt*R1));
        Q2_row[ti] = 0.5f*(Q2 + (U2 + dt*R2)) / (1.0f + 0.5f*C);
        Q3_row[ti] = 0.5f*(Q3 + (U3 + dt*R3)) / (1.0f + 0.5f*C);
    }
}








