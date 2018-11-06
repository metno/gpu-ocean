# -*- coding: utf-8 -*-

"""
This python class implements an the
Implicit Equal-Weight Particle Filter, for use on
simplified ocean models.

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
"""


from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time
import gc
import pycuda.driver as cuda
from scipy.special import lambertw


from SWESimulators import Common, OceanStateNoise

class IEWPFOcean:
    
    def __init__(self, ensemble, debug=False, show_errors=False,
                 block_width=16, block_height=16):
        
        self.gpu_ctx = ensemble.gpu_ctx
        self.master_stream = cuda.Stream()
        
        self.debug = debug
        self.show_errors = show_errors
        
        # Store information needed internally in the class
        self.dx = np.float32(ensemble.dx) 
        self.dy = np.float32(ensemble.dy)
        self.dt = np.float32(ensemble.dt)
        self.nx = np.int32(ensemble.nx)
        self.ny = np.int32(ensemble.ny)
        self.soar_q0 = np.float32(ensemble.small_scale_perturbation_amplitude)
        self.soar_L  = np.float32(ensemble.particles[0].small_scale_model_error.soar_L)
        self.f = np.float32(ensemble.f)
        self.g = np.float32(ensemble.g)
        self.const_H = np.float32(np.max(ensemble.base_H))

        self.boundaryConditions = ensemble.boundaryConditions
        
        self.geoBalanceConst = np.float32(self.g*self.const_H/(2.0*self.f))

        self.Nx = np.int32(self.nx*self.ny*3)  # state dimension
        self.numParticles = np.int32(ensemble.getNumParticles())
        self.numDrifters  = np.int32(ensemble.getNumDrifters())
        
        
        # The underlying assumptions are:
        # 1) that the equilibrium depth is constant:
        assert(np.max(ensemble.base_H) == np.min(ensemble.base_H))
        # 2) that both boundaries are periodic:
        assert(self.boundaryConditions.isPeriodicNorthSouth() and \
               self.boundaryConditions.isPeriodicEastWest())
        # 3) that the Coriolis force is constant for the entire domain:
        assert (ensemble.beta == 0)
        # 4) that dx and dy are the same
        assert (self.dx == self.dy)
        
        
        # Do not store the ensemble!!!
        
        # Create constant matrix S and copy to the GPU
        self.S_host, self.S_device = None, None
        self.S_host = self._createS(ensemble)
        self.S_device = Common.CUDAArray2D(self.master_stream, 2, 2, 0, 0, self.S_host)
        
        # Create constant localized SVD matrix and copy to the GPU
        self.localSVD_host, self.localSVD_device = None, None
        self.localSVD_host = self._generateLocaleSVDforP(ensemble)
        self.localSVD_device = Common.CUDAArray2D(self.master_stream, 49, 49, 0, 0, self.localSVD_host)
    
        
        # Allocate extra memory needed for reduction kernel.
        # Currently: one single GPU buffer with 1x1 elements
        self.reduction_buffer = None
        reduction_buffer_host = np.zeros((1,1), dtype=np.float32)
        self.reduction_buffer = Common.CUDAArray2D(self.master_stream, 1, 1, 0, 0, reduction_buffer_host)
        
        # Generate kernels
        self.reduction_kernels = self.gpu_ctx.get_kernel("reductions.cu", \
                                                         defines={})
        self.iewpf_kernels = self.gpu_ctx.get_kernel("iewpf_kernels.cu", \
                                                     defines={'block_width': block_width, 'block_height': block_height})
        
        
        # Get CUDA functions and define data types for prepared_{async_}call()
        self.squareSumKernel = self.reduction_kernels.get_function("squareSum")
        self.squareSumKernel.prepare("iiPP")
        
        self.setBufferToZeroKernel = self.iewpf_kernels.get_function("setBufferToZero")
        self.setBufferToZeroKernel.prepare("iiPi")
        
        self.halfTheKalmanGainKernel = self.iewpf_kernels.get_function("halfTheKalmanGain")
        self.halfTheKalmanGainKernel.prepare("iiffffiifffPi")
        
        self.localSVDOnGlobalXiKernel = self.iewpf_kernels.get_function("localSVDOnGlobalXi")
        self.localSVDOnGlobalXiKernel.prepare("iiiiPiPi")
        
        
        #Compute kernel launch parameters
        self.local_size_reductions  = (128, 1, 1)
        self.global_size_reductions = (1,   1)
        
        self.local_size_Kalman  = (7, 7, 1)
        self.global_size_Kalman = (1, 1)
        
        self.local_size_SVD  = (7, 7, 1)
        self.global_size_SVD = (1, 1)
        
        self.local_size_domain = (block_width, block_height, 1)
        self.global_size_domain = ( \
                                   int(np.ceil(self.nx / float(self.local_size_domain[0]))), \
                                   int(np.ceil(self.ny / float(self.local_size_domain[1]))) \
                                  ) 
    
       
    
    
    def __del__(self):
        self.cleanUp()
        
        
    def cleanUp(self):
        # All allocated data needs to be freed from here
        if self.S_device is not None:
            self.S_device.release()
        if self.localSVD_device is not None:
            self.localSVD_device.release()
        if self.reduction_buffer is not None:
            self.reduction_buffer.release()
        self.gpu_ctx = None
    
    
    
    
    ### MAIN IEWPF METHOD
    def iewpf(self, ensemble, infoPlots=None, it=None):
        """
        The complete IEWPF algorithm implemented on the GPU.
        
        Retrieves innovations and target weights from the ensemble, and updates 
        each particle according to the IEWPF method.
        """
        # Step -1: Deterministic step
        t = ensemble.step_truth(self.dt, stochastic=True)
        t = ensemble.step_particles(self.dt, stochastic=False)
        
        # Step 0: Obtain innovations
        observed_drifter_positions = ensemble.observeTrueDrifters()
        innovations = ensemble.getInnovations()
        w_rest = -np.log(1.0/ensemble.getNumParticles())*np.ones(ensemble.getNumParticles())
        
        # save plot before
        if infoPlots is not None:
            self._keepPlot(ensemble, infoPlots, it, 1)
        
        # Step 1: Find maximum weight
        target_weight = self.obtainTargetWeight(ensemble, innovations)
        
        for p in range(ensemble.getNumParticles()):
                        
            # Loop step 1: Pull particles towards observation by adding a Kalman gain term
            #     Also, we find phi within this function
            phi = self.addKalmanGain(ensemble.particles[p], observed_drifter_positions, innovations[p])
            
            # Loop step 2: Sample xi \sim N(0, P), and get gamma in the process
            gamma = self.sampleFromP(ensemble.particles[p], observed_drifter_positions)
            
            # Loop step 3: Solve implicit equation
            alpha = self.solveImplicitEquation(phi, gamma, target_weight, w_rest[p], particle_id=p)
            
            # Loop steps 4:Add scaled sample from P to the state vector
            ensemble.particles[p].small_scale_model_error.perturbSim(ensemble.particles[p],\
                                                                     update_random_field=False, \
                                                                     perturbation_scale=alpha)   
            
            # TODO
            #ensemble.particles[p].drifters.setDrifterPositions(newPos)
        
        # save plot after
        if infoPlots is not None:
            self._keepPlot(ensemble, infoPlots, it, 3)
    
    
    
    def iewpf_timer(self, ensemble, infoPlots=None, it=None):
        """
        The complete IEWPF algorithm implemented on the GPU.
        
        Retrieves innovations and target weights from the ensemble, and updates 
        each particle according to the IEWPF method.
        """
        # Step -1: Deterministic step
        print ("----------")
        start_pre_loop = cuda.Event()
        start_pre_loop.record(self.master_stream)
                
        t = ensemble.step_truth(self.dt, stochastic=True)
        t = ensemble.step_particles(self.dt, stochastic=False)

        deterministic_step_event = cuda.Event()
        deterministic_step_event.record(self.master_stream)
        deterministic_step_event.synchronize()
        gpu_elapsed = deterministic_step_event.time_since(start_pre_loop)*1.0e-3
        print ("Deterministic timestep took: " + str(gpu_elapsed))
        
        # Step 0: Obtain innovations
        observed_drifter_positions = ensemble.observeTrueDrifters()
        
        observe_drifters_event = cuda.Event()
        observe_drifters_event.record(self.master_stream)
        observe_drifters_event.synchronize()
        gpu_elapsed = observe_drifters_event.time_since(deterministic_step_event)*1.0e-3
        print("Observing drifters took:     " + str(gpu_elapsed))
        
        
        innovations = ensemble.getInnovations()
        innovations_event = cuda.Event()
        innovations_event.record(self.master_stream)
        innovations_event.synchronize()
        gpu_elapsed = innovations_event.time_since(observe_drifters_event)*1.0e-3
        print("innovations_event took:      " + str(gpu_elapsed))
        
        w_rest = -np.log(1.0/ensemble.getNumParticles())*np.ones(ensemble.getNumParticles())

        # Step 1: Find maximum weight
        target_weight = self.obtainTargetWeight(ensemble, innovations)
        
        dummy = self.download_reduction_buffer()
        target_weight_event = cuda.Event()
        target_weight_event.record(self.master_stream)
        target_weight_event.synchronize()
        gpu_elapsed = target_weight_event.time_since(innovations_event)*1.0e-3
        print("Finding target weight took:  " + str(gpu_elapsed))
        
        for p in range(ensemble.getNumParticles()):
            print ("----------")
            print ("Starting particle " + str(p))
            start_loop = cuda.Event()
            kalman_event = cuda.Event()
            p_event = cuda.Event()
            add_scaled_event = cuda.Event()
            
            start_loop.record(self.master_stream)
            
            # Loop step 1: Pull particles towards observation by adding a Kalman gain term
            #     Also, we find phi within this function
            phi = self.addKalmanGain(ensemble.particles[p], observed_drifter_positions, innovations[p])
            
            kalman_event.record(self.master_stream)
            kalman_event.synchronize()
            gpu_elapsed = kalman_event.time_since(start_loop)*1.0e-3
            print ("Kalman gain took:   " + str(gpu_elapsed))
            
            
            # Loop step 2: Sample xi \sim N(0, P), and get gamma in the process
            gamma = self.sampleFromP(ensemble.particles[p], observed_drifter_positions)
            
            p_event.record(self.master_stream)
            p_event.synchronize()
            gpu_elapsed = p_event.time_since(kalman_event)*1.0e-3
            print ("Sample from P took: " + str(gpu_elapsed) )
            
            # Loop step 3: Solve implicit equation
            alpha = self.solveImplicitEquation(phi, gamma, target_weight, w_rest[p], particle_id=p)
            
            
            
            # Loop steps 4:Add scaled sample from P to the state vector
            ensemble.particles[p].small_scale_model_error.perturbSim(ensemble.particles[p],\
                                                                     update_random_field=False, \
                                                                     perturbation_scale=alpha)   
            
            add_scaled_event.record(self.master_stream)
            add_scaled_event.synchronize()
            gpu_elapsed = add_scaled_event.time_since(p_event)*1.0e-3
            print ("Add scaled xi took: " + str(gpu_elapsed) )
            
            print ("Done particle " + str(p))
            print ("----------")
            # TODO
            #ensemble.particles[p].drifters.setDrifterPositions(newPos)
            #print "IEWPF done for particle: ", p
        # save plot after
        if infoPlots is not None:
            self._keepPlot(ensemble, infoPlots, it, 3)
    
    ###------------------------    
    ### GPU Methods
    ###------------------------
    # Functions needed for the GPU implementation of IEWPF
    
    def obtainGamma(self, sim):
        """
        Gamma = sum(xi^2), where xi \sim N(0,I)
        Calling a kernel that sums the square of all elements in the random buffer of the
        small scale model error of the provided simulator.
        """
        self.squareSumKernel.prepared_async_call(self.global_size_reductions,
                                                 self.local_size_reductions, 
                                                 self.master_stream,
                                                 self.nx, self.ny,
                                                 sim.small_scale_model_error.random_numbers.data.gpudata,
                                                 self.reduction_buffer.data.gpudata)
        return self.download_reduction_buffer()[0,0]
        
            
    def setNoiseBufferToZero(self, sim):
        self.setBufferToZeroKernel.prepared_async_call(self.global_size_domain,
                                                      self.local_size_domain, 
                                                      self.master_stream,
                                                      sim.nx, sim.ny,
                                                      sim.small_scale_model_error.random_numbers.data.gpudata,
                                                      sim.small_scale_model_error.random_numbers.pitch)
        
    def addKalmanGain(self, sim, all_observed_drifter_positions, innovation):
        
        # Reset the random numbers buffer for the given sim to zero:
        self.setNoiseBufferToZero(sim)
        
        # Find phi as we go: phi = d^T S d
        phi = 0.0
        
        # Loop over drifters to get half the Kalman gain for each innovation
        for drifter in range(self.numDrifters):
            local_innovation = innovation[drifter,:]
            observed_drifter_position = all_observed_drifter_positions[drifter,:]
            
            cell_id_x = np.int32(int(np.floor(observed_drifter_position[0]/sim.dx)))
            cell_id_y = np.int32(int(np.floor(observed_drifter_position[1]/sim.dy)))

            # 1) Solve linear problem
            e = np.dot(self.S_host, local_innovation)
                        
            self.halfTheKalmanGainKernel.prepared_async_call(self.global_size_Kalman,
                                                             self.local_size_Kalman,
                                                             self.master_stream,
                                                             self.nx, self.ny, self.dx, self.dy,
                                                             self.soar_q0, self.soar_L,
                                                             cell_id_x, cell_id_y,
                                                             self.geoBalanceConst,
                                                             np.float32(e[0,0]), np.float32(e[0,1]),
                                                             sim.small_scale_model_error.random_numbers.data.gpudata,
                                                             sim.small_scale_model_error.random_numbers.pitch)
            
            phi += local_innovation[0]*e[0,0] + local_innovation[1]*e[0,1]
            
        # The final step of the Kalman gain is to obtain geostrophic balance on the obtained field.
        sim.small_scale_model_error.perturbSim(sim, update_random_field=False)
    
        return phi
        # end of addKalmanGain
        #----------------------------------
    
    
    def sampleFromP(self, sim, all_observed_drifter_positions):
        
        # Sample from N(0,I)
        sim.small_scale_model_error.generateNormalDistribution()
        
        # Obtain gamma
        sim.gpu_stream.synchronize()
        gamma = self.obtainGamma(sim)
        sim.gpu_stream.synchronize()
            
        for drifter in range(self.numDrifters):
            #print "\nhei from drifter ", drifter
            observed_drifter_position = all_observed_drifter_positions[drifter,:]
            
            cell_id_x = int(np.floor(observed_drifter_position[0]/sim.dx))
            cell_id_y = int(np.floor(observed_drifter_position[1]/sim.dy))
            #print "cell id: ", (cell_id_x, cell_id_y)
        
            self.applyLocalSVDOnGlobalXi(sim, cell_id_x, cell_id_y)
                
        return gamma
    
    def applyLocalSVDOnGlobalXi(self, sim, drifter_cell_id_x, drifter_cell_id_y):
        # Assuming that the random numbers buffer for the given sim is filled with N(0,I) numbers
        self.localSVDOnGlobalXiKernel.prepared_async_call(self.global_size_SVD,
                                                          self.local_size_SVD,
                                                          self.master_stream,
                                                          self.nx, self.ny,
                                                          np.int32(drifter_cell_id_x),
                                                          np.int32(drifter_cell_id_y),
                                                          self.localSVD_device.data.gpudata,
                                                          self.localSVD_device.pitch, 
                                                              sim.small_scale_model_error.random_numbers.data.gpudata,
                                                             sim.small_scale_model_error.random_numbers.pitch)
    
    
    
    ###------------------------    
    ### CPU Methods
    ###------------------------
    # Parts of the efficient IEWPF method that is solved on the CPU
    
    # As we have S = (HQH^T + R)^-1, we can do step 1 of the IEWPF algorithm
    def obtainTargetWeight(self, ensemble, d, w_rest=None):
        if w_rest is None:
            w_rest = -np.log(1.0/ensemble.getNumParticles())*np.ones(ensemble.getNumParticles())
            
        Ne = ensemble.getNumParticles()
        c = np.zeros(Ne)
        for particle in range(Ne):
            # Obtain db = d^T S d
            db = 0.0
            for drifter in range(ensemble.driftersPerOceanModel):
                e = np.dot(self.S_host, d[particle,drifter,:])
                db += np.dot(e, d[particle, drifter, :])
            c[particle] = w_rest[particle] + 0.5*db
            if self.debug: print( "c[" + str(particle) + "]: ", c[particle])
            if self.debug: print ("exp(-c[" + str(particle) + "]: ", np.exp(-c[particle]))
        return np.min(c)

    
    ### Solving the implicit equation on the CPU:
    
    def _old_implicitEquation(self, alpha, gamma, Nx, a):
        return (alpha-1.0)*gamma - Nx*np.log(alpha) + a
    
    def _implicitEquation(self, alpha, gamma, Nx, target_weight, c):
        """
        This is the equation that we now should have solved by using the lambert W function
        """
        return np.log(alpha*alpha*Nx/gamma) - (alpha*alpha*Nx/gamma) - ((target_weight - c)/Nx) + 1
        
    def solveImplicitEquation(self, phi, gamma, 
                               target_weight, w_rest, particle_id=None):
        """
        Solving the scalar implicit equation using the Lambert W function, and 
        updating the buffers eta_a, hu_a, hv_a as:
        x_a = x_a + alpha*xi
        """
        if self.debug:
            print ("gamma: ", gamma)
            print ("Nx: ", self.Nx)
            print ("w_rest: ", w_rest)
            print ("target_weight: ", target_weight)
            print ("phi: ", phi)

        # 6) Find c
        c = phi - w_rest
        if self.debug: 
            print ("c = phi - w_rest: ", c)

        # 7) Solving the Lambert W function
        lambert_W_arg = -np.exp((target_weight - c)/self.Nx  - 1)

        alpha_min1 = np.sqrt(-(gamma/self.Nx)*np.real(lambertw(lambert_W_arg, k=-1)))
        alpha_zero = np.sqrt(-(gamma/self.Nx)*np.real(lambertw(lambert_W_arg)))
        
        if self.debug: 
            print ("Check a against the Lambert W requirement: ")
            print ("-e^-1 < z < 0 : ", -1.0/np.exp(1), " < ", lambert_W_arg, " < ", 0, " = ", \
                    (-1.0/np.exp(1) < lambert_W_arg, lambert_W_arg < 0))
            print ("Obtained (alpha k=-1, alpha k=0): ", (alpha_min1, alpha_zero))
            print ("The two branches from Lambert W: ", (lambertw(lambert_W_arg), lambertw(lambert_W_arg, k=-1)))
            print ("The two reals from Lambert W: ", (np.real(lambertw(lambert_W_arg)), np.real(lambertw(lambert_W_arg, k=-1))))

        alpha = alpha_zero
        if lambert_W_arg > (-1.0/np.exp(1)) :
            alpha_u = np.random.rand()
            if alpha_u < 0.5:
                alpha = alpha_min1
                if self.debug: print ("Drew alpha from -1-branch")
        elif self.show_errors:
            print ("!!!!!!!!!!!!")
            print ("BAD BAD ARGUMENT TO LAMBERT W")
            print ("Particle ID: ", particle_id)
            print( "Obtained (alpha k=0, alpha k=-1): ", (alpha_zero, alpha_min1))
            print ("The requirement is lamber_W_arg > (-1.0/exp(1)): " + str(lambert_W_arg) + " > " + str(-1.0/np.exp(1.0)))
            print ("gamma: ", gamma)
            print ("Nx: ", self.Nx)
            print ("w_rest: ", w_rest)
            print ("target_weight: ", target_weight)
            print ("phi: ", phi)
            print ("The two branches from Lambert W: ", (lambertw(lambert_W_arg), lambertw(lambert_W_arg, k=-1)))
            print ("Checking implicit equation with alpha (k=0, k=-1): ", \
            (self._implicitEquation(alpha_zero, gamma, self.Nx, target_weight, c), \
             self._implicitEquation(alpha_min1, gamma, self.Nx, target_weight, c)))
            print( "!!!!!!!!!!!!")
        
        
        if self.debug: 
            print ("--------------------------------------")
            print ("Obtained (lambert_ans k=0, lambert_ans k=-1): ", (lambertw(lambert_W_arg), lambertw(lambert_W_arg, k=-1)))
            print ("Obtained (alpha k=0, alpha k=-1): ", (alpha_zero, alpha_min1))
            print ("Checking implicit equation with alpha (k=0, k=-1): ", \
            (self._implicitEquation(alpha_zero, gamma, self.Nx, target_weight, c), \
             self._implicitEquation(alpha_min1, gamma, self.Nx, target_weight, c)))
            print ("Selected alpha: ", alpha)
            print ("\n")

        return alpha
        
        
    def _createS(self, ensemble):
        """
        Create the 2x2 matrix S = (HQH^T + R)^-1

        Constant as long as
         - one drifter only,
         - H(x,y) = const, and
         - double periodic boundary conditions
        """

        # Local storage for x and y correlations:
        x_corr = np.zeros((7,7))
        y_corr = np.zeros((7,7))
        tmp_x = np.zeros((7,7))
        tmp_y = np.zeros((7,7))

        # Mid_coordinates:
        mid_i, mid_j = 3, 3

        # Fill the buffers with U_{GB}^T H^T
        x_corr[mid_j+1, mid_i] = -self.geoBalanceConst/self.dy
        x_corr[mid_j-1, mid_i] =  self.geoBalanceConst/self.dy
        y_corr[mid_j, mid_i+1] =  self.geoBalanceConst/self.dx
        y_corr[mid_j, mid_i-1] = -self.geoBalanceConst/self.dx
        if self.debug: self.showMatrices(x_corr, y_corr, "$U_{GB}^T  H^T$")
    
        # Apply the SOAR function to fill x and y with 7x5 and 5x7 respectively
        # First for x:
        for j,i in [mid_j+1, mid_i], [mid_j-1, mid_i]:
            for b in range(j-2, j+3):
                for a in range(i-2, i+3):
                    tmp_x[b, a] += x_corr[j,i]*self._SOAR_Q_CPU(a, b, i, j)
        # Then for y:
        for j,i in [mid_j, mid_i+1], [mid_j, mid_i-1]:
            for b in range(j-2, j+3):
                for a in range(i-2, i+3):
                    tmp_y[b, a] += y_corr[j,i]*self._SOAR_Q_CPU(a, b, i, j)
        if self.debug: self.showMatrices(tmp_x, tmp_y, "$Q_{SOAR} U_{GB}^T H^T$")   
        
        # Apply the SOARfunction again to fill the points needed to find drift in (mid_i, mid_j)
        # For both x and y:
        # This means that we only need to evaluate Q_{SOAR} Q_{SOAR} U_{GB}^T H^T at four points
        for j,i in [mid_j+1, mid_i], [mid_j-1, mid_i], [mid_j, mid_i-1], [mid_j, mid_i+1]:
            x_corr[j,i] = 0
            y_corr[j,i] = 0
            for b in range(j-2, j+3):
                for a in range(i-2, i+3):
                    SOAR_Q_res = self._SOAR_Q_CPU(a, b, i, j)
                    x_corr[j,i] += tmp_x[b, a]*SOAR_Q_res
                    y_corr[j,i] += tmp_y[b, a]*SOAR_Q_res
            if self.debug: print ("(j, i ,x_corr[j,i], y_corr[j,i]): ", (j, i ,x_corr[j,i], y_corr[j,i]))
        if self.debug: self.showMatrices(x_corr, y_corr, "$Q_{SOAR} Q_{SOAR} U_{GB}^T H^T$")

        # geostrophic balance:
        x_hu = -self.geoBalanceConst*(x_corr[mid_j+1, mid_i  ] - x_corr[mid_j-1, mid_i  ])/self.dy
        x_hv =  self.geoBalanceConst*(x_corr[mid_j  , mid_i+1] - x_corr[mid_j  , mid_i-1])/self.dx
        y_hu = -self.geoBalanceConst*(y_corr[mid_j+1, mid_i  ] - y_corr[mid_j-1, mid_i  ])/self.dy
        y_hv =  self.geoBalanceConst*(y_corr[mid_j  , mid_i+1] - y_corr[mid_j  , mid_i-1])/self.dx 

        # Structure the information as a  
        HQHT = np.matrix([[x_hu, y_hu],[x_hv, y_hv]])    
        if self.debug: print ("HQHT\n", HQHT)
        if self.debug: print ("ensemble.observation_cov\n", ensemble.observation_cov)
        S_inv = HQHT + ensemble.observation_cov
        if self.debug: print ("S_inv\n", S_inv)
        S = np.linalg.inv(S_inv)
        if self.debug: print( "S\n", S)
        return S.astype(np.float32, order='C')
        
    
    ###---------------------------
    ### Download GPU buffers
    ###---------------------------
    
    def download_S(self):
        return self.S_device.download(self.master_stream)
    
    def download_localSVD(self):
        return self.localSVD_device.download(self.master_stream)
    
    def download_reduction_buffer(self):
        return self.reduction_buffer.download(self.master_stream)
    
        
    def showMatrices(self, x, y, title, z = None):
        num_cols = 2
        if z is not None:
            num_cols = 3
        fig = plt.figure(figsize=(num_cols*2,2))
        plt.subplot(1,num_cols,1)
        plt.imshow(x.copy(), origin="lower", interpolation="None")
        plt.xlabel('(%.2E, %.2E)' % (np.min(x), np.max(x)))
        plt.subplot(1,num_cols,2)
        plt.imshow(y.copy(), origin="lower", interpolation="None")
        plt.xlabel('(%.2E, %.2E)' % (np.min(y), np.max(y)))
        if z is not None:
            plt.subplot(1, num_cols, 3)
            plt.imshow(z.copy(), origin="lower", interpolation="None")
            plt.xlabel('(%.2E, %.2E)' % (np.min(z), np.max(z)))
        plt.suptitle(title)

    ###-----------------------
    ### IEWPF CPU functions
    ###-----------------------
        
    def _SOAR_Q_CPU(self, a_x, a_y, b_x, b_y):
        """
        CPU implementation of a SOAR covariance function between grid points
        (a_x, a_y) and (b_x, b_y) with periodic boundaries
        """
        dist_x = min((a_x - b_x)**2, (a_x - (b_x + self.nx))**2, (a_x - (b_x - self.nx))**2)
        dist_y = min((a_y - b_y)**2, (a_y - (b_y + self.ny))**2, (a_y - (b_y - self.ny))**2)
        
        dist = np.sqrt( self.dx*self.dx*dist_x  +  self.dy*self.dy*dist_y)
        
        return self.soar_q0*(1.0 + dist/self.soar_L)*np.exp(-dist/self.soar_L)



    def _createCutoffSOARMatrixQ(self, ensemble, nx=None, ny=None, cutoff=2):
        
        if nx is None:
            nx = ensemble.nx
        if ny is None:
            ny = ensemble.ny
        
        Q = np.zeros((ny*nx, ny*nx))
        for a_y in range(ny):
            for a_x in range(nx):
                j = a_y*nx + a_x
                for b_y in range(a_y-cutoff, a_y+cutoff+1):
                    if b_y < 0:    
                         b_y = b_y + ny
                    if b_y > ny-1: 
                        b_y = b_y - ny
                    for b_x in range(a_x-cutoff, a_x+cutoff+1):
                        if b_x < 0:
                            b_x = b_x + nx
                        if b_x > nx-1: 
                            b_x = b_x - nx
                        i = b_y*nx + b_x
                        
                        # In the SOAR function, we use the ensemble nx and ny, to correctly 
                        # account for periodic boundaries.
                        Q[j, i] = self._SOAR_Q_CPU(a_x, a_y, b_x, b_y)
        return Q


    def _createUGBmatrix(self, ensemble, nx=None, ny=None):
    
        if nx is None:
            nx = ensemble.nx
        if ny is None:
            ny = ensemble.ny
        
        I = np.eye(nx*ny)
        A_hu = np.zeros((ny*nx, ny*nx))
        A_hv = np.zeros((ny*nx, ny*nx))
        for a_y in range(ny):
            for a_x in range(nx):
                j = a_y*nx + a_x
                
                # geo balance for hu:
                i = (a_y+1)*nx + a_x
                if a_y == ny-1:
                    i = 0*nx + a_x
                A_hu[j,i] = 1.0
                i = (a_y-1)*nx + a_x
                if a_y == 0:
                    i = (ny-1)*nx + a_x
                A_hu[j,i] = -1.0

                # geo balance for hv:
                i = a_y*nx + a_x + 1
                if a_x == nx-1:
                    i = a_y*nx + 0
                A_hv[j,i] = 1.0

                i = a_y*nx + a_x - 1
                if a_x == 0:
                    i = a_y*nx + nx - 1
                A_hv[j,i] = -1.0

        A_hu *= -self.geoBalanceConst/self.dy
        A_hv *=  self.geoBalanceConst/self.dx
            
        return np.bmat([[I], [A_hu], [A_hv]])

    def _createMatrixH(self, nx, ny, pos_x, pos_y):
        H = np.zeros((2, 3*nx*ny))
        index = pos_y*nx + pos_x
        H[0, 1*nx*ny + index] = 1
        H[1, 2*nx*ny + index] = 1
        return H

    def _generateLocaleSVDforP(self, ensemble):
        """
        Generates the local square root of the SVD-block needed for P^1/2.

        Finding:   U*Sigma*V^H = I - Q*U_GB^T*H^T*S*H*U_GB*Q
        Returning: U*sqrt(Sigma)
        """

        # Since the structure of the SVD-block is the same for any drifter position, we build the block
        # on a 7x7 domain with the observation in the middle cell
        local_nx = 7
        local_ny = 7
        pos_x = 3
        pos_y = 3

        # Create the matrices needed
        H      = self._createMatrixH(local_nx, local_ny, pos_x, pos_y)
        Q_soar = self._createCutoffSOARMatrixQ(ensemble, nx=local_nx, ny=local_ny)
        U_GB   = self._createUGBmatrix(ensemble, nx=local_nx, ny=local_ny)

        UQ = np.dot(U_GB, Q_soar)
        HUQ = np.dot(H, UQ)
        SHUQ = np.dot(self.S_host, HUQ)
        HTSHUQ = np.dot(H.transpose(), SHUQ)
        UTHTSHUQ = np.dot(U_GB.transpose(), HTSHUQ)
        QUTHTSHUQ = np.dot(Q_soar, UTHTSHUQ)

        svd_input = np.eye(local_nx*local_nx) - QUTHTSHUQ

        u, s, vh = np.linalg.svd(svd_input, full_matrices=True)

        if self.debug:
            SVD_prod = np.dot(u, np.dot(np.diag(s), vh))
            fig = plt.figure(figsize=(4,4))
            plt.imshow(SVD_prod, interpolation="None")
            plt.title("SVD_prod")
            plt.colorbar()

            fig = plt.figure(figsize=(4,4))
            plt.imshow(SVD_prod - np.eye(49), interpolation="None")
            plt.title("SVD_prod - I")
            plt.colorbar()

            fig = plt.figure(figsize=(4,4))
            plt.imshow(u, interpolation="None")
            plt.title("u")
            plt.colorbar()


        return np.dot(u, np.diag(np.sqrt(s))).astype(np.float32, order='C')
    
    
  
    
    
    def iewpf_CPU(self, ensemble, infoPlots=None, it=None):
        """
        The complete IEWPF algorithm implemented on the CPU.
        
        Retrieves innovations and target weights from the ensemble, and updates 
        each particle according to the IEWPF method.
        """
        # Step -1: Deterministic step
        t = ensemble.step_truth(self.dt, stochastic=True)
        t = ensemble.step_particles(self.dt, stochastic=False)


        # Step 0: Obtain innovations
        observed_drifter_position = ensemble.observeTrueDrifters()
        innovations = ensemble.getInnovations()
        w_rest = -np.log(1.0/ensemble.getNumParticles())*np.ones(ensemble.getNumParticles())

        # save plot halfway
        if infoPlots is not None:
            self._keepPlot(ensemble, infoPlots, it, 1)

        # Step 1: Find maximum weight
        target_weight = self.obtainTargetWeight(ensemble, innovations)
        #print "WWWWWWWWWWWWWWW"
        #print "Target weight: ", target_weight
        #print "-log(target_weight): ", -np.log(target_weight)
        #print "exp(-target_weight): ", np.exp(-target_weight)
        #print "1/Ne: ", 1.0/ensemble.getNumParticles()
        #print "WWWWWWWWWWWWWWW"


        for p in range(ensemble.getNumParticles()):
            #iewpfOcean.debug = p==0
            
            # Loop step 1: Pull particles towards observation by adding a Kalman gain term
            eta_a, hu_a, hv_a, phi = self.applyKalmanGain_CPU(ensemble.particles[p], \
                                                              observed_drifter_position,
                                                              innovations[p], target_weight)
            
            

            # Loop step 2: Sample xi \sim N(0, P)
            p_eta, p_hu, p_hv, gamma = self.drawFromP_CPU(ensemble.particles[p], 
                                                          observed_drifter_position)
            xi = [p_eta, p_hu, p_hv] 

            

            # Loop steps 3 and 4: Solve implicit equation and add scaled sample from P to the state vector
            self.applyScaledPSample_CPU(ensemble.particles[p], eta_a, hu_a, hv_a, \
                                        phi, xi, gamma, 
                                        target_weight, w_rest[p])

            # CPU step: Fix boundaries and upload the resulting state to the GPU
            eta_a = self._expand_to_periodic_boundaries(eta_a, 2)
            hu_a  = self._expand_to_periodic_boundaries(hu_a,  2)
            hv_a  = self._expand_to_periodic_boundaries(hv_a,  2)
            ensemble.particles[p].upload(eta_a, hu_a, hv_a)

            # TODO
            #ensemble.particles[p].drifters.setDrifterPositions(newPos)
            #print "IEWPF (CPU) done for particle: ", p
        # save plot after
        if infoPlots is not None:
            self._keepPlot(ensemble, infoPlots, it, 3)
    
    
    def iewpf_traditional_CPU(self, ensemble, infoPlots=None, it=None):
        """
        The complete IEWPF algorithm (as described by the SotA-18 paper) implemented on the CPU.
        
        Retrieves innovations and target weights from the ensemble, and updates 
        each particle according to the IEWPF method.
        """
        # Step -1: Deterministic step
        t = ensemble.step_truth(self.dt, stochastic=True)
        t = ensemble.step_particles(self.dt, stochastic=False)


        # Step 0: Obtain innovations
        observed_drifter_position = ensemble.observeTrueDrifters()
        innovations = ensemble.getInnovations()
        w_rest = -np.log(1.0/ensemble.getNumParticles())*np.ones(ensemble.getNumParticles())

        # save plot halfway
        if infoPlots is not None:
            self._keepPlot(ensemble, infoPlots, it, 1)

        # Step 1: Find maximum weight
        target_weight = self.obtainTargetWeight(ensemble, innovations)
        #print "WWWWWWWWWWWWWWW"
        #print "Target weight: ", target_weight
        #print "-log(target_weight): ", -np.log(target_weight)
        #print "exp(-target_weight): ", np.exp(-target_weight)
        #print "1/Ne: ", 1.0/ensemble.getNumParticles()
        #print "WWWWWWWWWWWWWWW"


        for p in range(ensemble.getNumParticles()):
            #iewpfOcean.debug = p==0

            # Step 2: Sample xi Ìƒ N(0, P)
            p_eta, p_hu, p_hv, gamma = self.drawFromP_CPU(ensemble.particles[p], 
                                                          observed_drifter_position)
            xi = [p_eta, p_hu, p_hv] 

            # Step 3: Pull particles towards observation by adding a Kalman gain term
            eta_a, hu_a, hv_a, phi = self.applyKalmanGain_CPU(ensemble.particles[p], \
                                                              observed_drifter_position,
                                                              innovations[p], target_weight)

            # Step 4: Solve implicit equation and add scaled sample from P
            self.applyScaledPSample_CPU(ensemble.particles[p], eta_a, hu_a, hv_a, \
                                        phi, xi, gamma, 
                                        target_weight, w_rest[p])

            eta_a = self._expand_to_periodic_boundaries(eta_a, 2)
            hu_a  = self._expand_to_periodic_boundaries(hu_a,  2)
            hv_a  = self._expand_to_periodic_boundaries(hv_a,  2)
            ensemble.particles[p].upload(eta_a, hu_a, hv_a)

            # TODO
            #ensemble.particles[p].drifters.setDrifterPositions(newPos)

        # save plot after
        if infoPlots is not None:
            self._keepPlot(ensemble, infoPlots, it, 3)
    
    
    def _expand_to_periodic_boundaries(self, interior, ghostcells):
        if ghostcells == 0:
            return interior
        (ny, nx) = interior.shape

        nx_halo = nx + 2*ghostcells
        ny_halo = ny + 2*ghostcells
        newBuf = np.zeros((ny_halo, nx_halo))
        newBuf[ghostcells:-ghostcells, ghostcells:-ghostcells] = interior 
        for g in range(ghostcells):
            newBuf[g, :] = newBuf[ny_halo - 2*ghostcells + g, :]
            #newBuf[ny_halo - 2*ghostcells + g, :] *=0
            newBuf[ny_halo - 1 - g, :] = newBuf[2*ghostcells - 1 - g, :]
            #newBuf[2*ghostcells - 1 - g, :] *=0
        for g in range(ghostcells):
            newBuf[:, g] = newBuf[:, nx_halo - 2*ghostcells + g]
            newBuf[:, nx_halo - 1 - g] = newBuf[:, 2*ghostcells - 1 - g]
        return newBuf
    
    def _keepPlot(self, ensemble, infoPlots, it, stage):
        title = "it=" + str(it) + " before IEWPF"
        if stage == 2:
            title = "it=" + str(it) + " during IEWPF (with deterministic step)"
        elif stage == 3:
            title = "it=" + str(it) + " after IEWPF"
        infoFig = ensemble.plotDistanceInfo(title=title, printInfo=False)
        plt.close(infoFig)
        infoPlots.append(infoFig)

        
    def _apply_periodic_boundary(self, index, dim_size):
        if index < 0:
            return index + dim_size
        elif index >= dim_size:
            return index - dim_size
        return index
    
    def _apply_local_SVD_to_global_xi_CPU(self, global_xi, pos_x, pos_y):
        """
        Despite the bad name, this is a good function!

        It takes as input:
         - the global_xi stored in a (ny, nx) buffer
         - the drifter cell position (pos_x, pos_y)
        
        It find the local sqrt (SVD) as U*sqrt(Sigma) in a (49, 49) buffer in self.
        

        The global_xi buffer is modified so that xi = U*sqrt(Sigma)*xi

        Note that we have to make a copy of xi so that we don't read already updated values.

        The function assumes periodic boundary conditions in both dimensions.
        """


        # Copy the result (representing the multiplication with I)
        read_global_xi = global_xi.copy()

        # Read the non-zero structure from tildeP to tildeP_block
        for loc_y_j in range(7):
            global_y_j = pos_y - 3 + loc_y_j
            global_y_j = self._apply_periodic_boundary(global_y_j, self.ny)
            for loc_x_j in range(7):
                global_x_j = pos_x - 3 + loc_x_j
                global_x_j = self._apply_periodic_boundary(global_x_j, self.nx)

                global_j = global_y_j*self.nx + global_x_j
                local_j = loc_y_j*7 + loc_x_j

                #loc_vec[local_j] = glob_vec[global_j]

                xi_j = 0.0
                for loc_y_i in range(7):
                    global_y_i = pos_y - 3 + loc_y_i
                    global_y_i = self._apply_periodic_boundary(global_y_i, self.ny)
                    for loc_x_i in range(7):
                        global_x_i = pos_x - 3 + loc_x_i
                        global_x_i = self._apply_periodic_boundary(global_x_i, self.nx)

                        global_i = global_y_i*self.nx + global_x_i
                        local_i = loc_y_i*7 + loc_x_i

                        xi_j += self.localSVD_host[local_j, local_i]*read_global_xi[global_y_i, global_x_i]

                global_xi[global_y_j, global_x_j] = xi_j

    def drawFromP_CPU(self, sim, observed_drifter_pos):

        # 1) Draw \tilde{\xi} \sim N(0, I)
        sim.small_scale_model_error.generateNormalDistributionCPU()
        if self.debug: print ("noise shape: ", sim.small_scale_model_error.random_numbers_host.shape)

        # Comment in these lines to see how the SVD structure affect the random numbers
        #sim.small_scale_model_error.random_numbers_host *= 0
        #sim.small_scale_model_error.random_numbers_host += 1
        original = None
        if self.debug: original = sim.small_scale_model_error.random_numbers_host.copy()

        # 1.5) Find gamma, which is needed by step 3
        gamma = np.sum(sim.small_scale_model_error.random_numbers_host **2)
        if self.debug: print ("Gamma obtained from standard gaussian: ", gamma)
        if self.debug: self.showMatrices(sim.small_scale_model_error.random_numbers_host, original,\
                                         "Std normal dist numbers")

        # 2) For each drifter, apply the local sqrt SVD-term
        for drifter in range(sim.drifters.getNumDrifters()):

            # 2.1) Find the cell index for the noise.random_numbers_host buffer.
            # This buffer has no ghost cells.
            cell_id_x = int(np.floor(observed_drifter_pos[drifter,0]/sim.dx))
            cell_id_y = int(np.floor(observed_drifter_pos[drifter,1]/sim.dy))

            # 2.2) Apply the local sqrt(SVD)-term
            self._apply_local_SVD_to_global_xi_CPU(sim.small_scale_model_error.random_numbers_host, \
                                                   cell_id_x, cell_id_y)
            if self.debug: self.showMatrices(sim.small_scale_model_error.random_numbers_host, \
                                             original - sim.small_scale_model_error.random_numbers_host, \
                                             "Std normal dist numbers after SVD from drifter " + str(drifter))

        # 3 and 4) Apply SOAR and geostrophic balance
        H_mid = sim.downloadBathymetry()[0]
        p_eta, p_hu, p_hv = sim.small_scale_model_error._obtainOceanPerturbations_CPU(H_mid, sim.f, sim.coriolis_beta, sim.g)

        if self.debug:
            self.showMatrices(p_eta[1:-1, 1:-1], p_hu, "Sample from P", p_hv)
            draw_from_q_maker = OceanStateNoise.OceanStateNoise(sim.gpu_ctx, sim.gpu_stream, \
                                                                sim.nx, sim.ny, sim.dx, sim.dy, \
                                                                sim.boundary_conditions, \
                                                                sim.small_scale_model_error.staggered, \
                                                                sim.small_scale_model_error.soar_q0, \
                                                                sim.small_scale_model_error.soar_L)
            draw_from_q_maker.random_numbers_host = original.copy()
            q_eta, q_hu, q_hv = draw_from_q_maker._obtainOceanPerturbations_CPU(H_mid, sim.f, sim.coriolis_beta, sim.g)
            self.showMatrices(q_eta[1:-1, 1:-1], q_hu, "Equivalent sample from Q", q_hv)
            self.showMatrices(p_eta[1:-1, 1:-1] - q_eta[1:-1, 1:-1], p_hu - q_hu, "diff sample from P and sample from Q", p_hv - q_hv)


        #gamma_from_p = np.sum(p_eta[1:-1, 1:-1]**2) + np.sum(p_hu**2) + np.sum(p_hv**2)
        #if debug: print "Gamma obtained from P^1/2 xi: ", gamma_from_p

        return p_eta[1:-1, 1:-1], p_hu, p_hv, gamma
    
    
    
    def applyKalmanGain_CPU(self, sim, \
                        all_observed_drifter_positions, innovation, target_weight, \
                        returnKalmanGainTerm=False):
        """
        Creating a Kalman gain type field, K = QH^T S d
        Returning two different values: x_a = x + K, and also phi = d^T S d
        Return eta_a, hu_a, hv_a, phi
        """
        # Following the 3rd step of the IEWPF algorithm
        if self.debug: print ("(nx, ny, dx, dy): ", (sim.nx, sim.ny, sim.dx, sim.dy))
        if self.debug: print ("all_observed_drifter_positions: ", all_observed_drifter_positions)
        if self.debug: print ("innovation:  ", innovation)
        if self.debug: print ("target_weight: ", target_weight)

        # 0.1) Allocate buffers
        total_K_eta = np.zeros((sim.ny, sim.nx))
        total_K_hu  = np.zeros((sim.ny, sim.nx))
        total_K_hv  = np.zeros((sim.ny, sim.nx))
        phi = 0.0

        # Assume drifters to be far appart, and make a Kalman gain factor for each drifter
        for drifter in range(sim.drifters.getNumDrifters()):

            local_innovation = innovation[drifter,:]
            observed_drifter_position = all_observed_drifter_positions[drifter,:]


            # 0.1) Find the cell index assuming no ghost cells
            cell_id_x = int(np.floor(observed_drifter_position[0]/sim.dx))
            cell_id_y = int(np.floor(observed_drifter_position[1]/sim.dy))
            if self.debug: print ("(cell_id_x, cell_id_y): ", (cell_id_x, cell_id_y))

            # 1) Solve linear problem
            e = np.dot(self.S_host, local_innovation)
            if self.debug: print("e: ", e)

            # 2) K = QH^T e = U_GB Q^{1/2} Q^{1/2} U_GB^T  H^T e
            #    Obtain the Kalman gain
            # 2.1) U_GB^T H^T e
            # 2.1.1) H^T: The transpose of the observation operator now maps the velocity at the 
            #        drifter position to the complete state vector:
            #        H^T [hu(posx, posy), hv(posx, posy)] = [zeros_eta, 0 0 hu(posx, posy) 0 0, 0 0 hv(posx, posy) 0 0]

            # 2.1.2) U_GB^T: map out to laplacian stencil
            local_huhv = np.zeros(4) # representing [north, east, south, west] positions from 
            north_east_south_west_index = [[4,3,0], [3,4,1], [2,3,2], [3,2,3]] #[[y-index-eta, x-index-eta, index-local_eta_soar]]
            # the x-component of the innovation spreads to north and south
            local_huhv[0] = -e[0,0]*self.geoBalanceConst/sim.dy # north 
            local_huhv[2] =  e[0,0]*self.geoBalanceConst/sim.dy # south
            # the y-component of the innovation spreads to east and west
            local_huhv[1] =  e[0,1]*self.geoBalanceConst/sim.dx # east
            local_huhv[3] = -e[0,1]*self.geoBalanceConst/sim.dx # west

            # 2.1.3) Q^{1/2}:
            local_eta = np.zeros((7,7))
            for j,i,soar_res_index in north_east_south_west_index:
                if self.debug: print (j,i), soar_res_index
                for b in range(j-2, j+3):
                    for a in range(i-2, i+3):
                        local_eta[b,a] += local_huhv[soar_res_index]*self._SOAR_Q_CPU(a, b, i, j)
            if self.debug: self.showMatrices(local_eta, local_eta, "local $\eta$ from global $\eta$")

            # 2.2) Apply U_GB Q^{1/2} to the result
            # 2.2.1)  Easiest way: map local_eta to a global K_eta_tmp buffer
            K_eta_tmp = np.zeros((sim.ny, sim.nx))
            if self.debug: print( "K_eta_tmp.shape",  K_eta_tmp.shape)
            for j in range(7):
                j_global = (cell_id_y-3+j+sim.ny)%sim.ny 
                for i in range(7):
                    i_global = (cell_id_x-3+i+sim.nx)%sim.nx 
                    K_eta_tmp[j_global, i_global] += local_eta[j,i]
                    #K_eta_tmp[j_global, i_global] += 10000000*local_eta[j,i]
            if self.debug: self.showMatrices(K_eta_tmp, local_eta, "global K_eta from local K_eta, halfway in the calc.")

            # 2.2.2) Use K_eta_tmp as the noise.random_numbers_host
            sim.small_scale_model_error.random_numbers_host = K_eta_tmp

            # 2.2.3) Apply soar + geo-balance
            H_mid = sim.downloadBathymetry()[0]
            K_eta , K_hu, K_hv = sim.small_scale_model_error._obtainOceanPerturbations_CPU(H_mid, sim.f, sim.coriolis_beta, sim.g)
            if self.debug: self.showMatrices(K_eta[1:-1, 1:-1], K_hu, "Kalman gain from drifter " + str(drifter), K_hv)

            total_K_eta += K_eta[1:-1, 1:-1]
            total_K_hu  += K_hu
            total_K_hv  += K_hv
            if self.debug: self.showMatrices(total_K_eta, total_K_hu, "Total Kalman gain after drifter " + str(drifter), total_K_hv)
            #showMatrices(total_K_eta, total_K_hu, "Total Kalman gain after drifter " + str(drifter), total_K_hv)


            # 3) Obtain phi = d^T * e
            phi += local_innovation[0]*e[0,0] + local_innovation[1]*e[0,1]
            if self.debug: print( "phi after drifter " + str(drifter) + ": ", phi)
        if self.debug: print( "phi: ", phi)

        if returnKalmanGainTerm:
            return total_K_eta, total_K_hu, total_K_hv, phi

        # 4) Obtain x_a
        eta_a, hu_a, hv_a = sim.download(interior_domain_only=True)
        if self.debug: self.showMatrices(eta_a, hu_a, "$M(x)$", hv_a)
        eta_a += total_K_eta
        hu_a += total_K_hu
        hv_a += total_K_hv
        if self.debug: print( "Shapes of x_a: ", eta_a.shape, hu_a.shape, hv_a.shape)
        if self.debug: self.showMatrices(eta_a, hu_a, "$x_a = M(x) + K$", hv_a)
        
        return eta_a, hu_a, hv_a, phi
            
            
    def applyScaledPSample_CPU(self, sim, eta_a, hu_a, hv_a, \
                           phi, xi, gamma, 
                           target_weight, w_rest, particle_id=None):
        """
        Solving the scalar implicit equation using the Lambert W function, and 
        updating the buffers eta_a, hu_a, hv_a as:
        x_a = x_a + alpha*xi
        """


        # 5) obtain gamma
        if self.debug: print ("Shapes of xi: ", xi[0].shape, xi[1].shape, xi[2].shape)
        #gamma = 0.0
        #for field in range(3):
        #    for j in range(ny):
        #        for i in range(nx):
        #            gamma += xi[field][j,i]*xi[field][j,i]
        if self.debug: print ("gamma: ", gamma)
        if self.debug: print ("Nx: ", self.Nx)
        if self.debug: print ("w_rest: ", w_rest)
        if self.debug: print ("target_weight: ", target_weight)
        if self.debug: print ("phi: ", phi)

        # 6) Find a
        a = phi - w_rest + target_weight
        if self.debug: print( "a = phi - w_rest + target_weight: ", a)
            
        c = phi - w_rest
        if self.debug: 
            print ("c = phi - w_rest: ", c)

        # 7) Solving the Lambert W function
        #alpha = 10000
        lambert_W_arg = -(gamma/self.Nx)*np.exp(a/self.Nx)*np.exp(-gamma/self.Nx)
        alpha_min1 = -(self.Nx/gamma)*np.real(lambertw(lambert_W_arg, k=-1))
        alpha_zero = -(self.Nx/gamma)*np.real(lambertw(lambert_W_arg))
        if self.debug: print ("Check a against the Lambert W requirement: ", a, " < ", - self.Nx + gamma - self.Nx*np.log(gamma/self.Nx), " = ", a <  - self.Nx + gamma - self.Nx*np.log(gamma/self.Nx))
        if self.debug: print ("-e^-1 < z < 0 : ", -1.0/np.exp(1), " < ", lambert_W_arg, " < ", 0, " = ", \
            (-1.0/np.exp(1) < lambert_W_arg, lambert_W_arg < 0))
        if self.debug: print ("Obtained (alpha k=-1, alpha k=0): ", (alpha_min1, alpha_zero))
        if self.debug: print ("The two branches from Lambert W: ", (lambertw(lambert_W_arg), lambertw(lambert_W_arg, k=-1)))
        if self.debug: print ("The two branches from Lambert W: ", (np.real(lambertw(lambert_W_arg)), np.real(lambertw(lambert_W_arg, k=-1))))

        alpha = alpha_zero
        if lambert_W_arg > (-1.0/np.exp(1)) :
            alpha_u = np.random.rand()
            if alpha_u < 0.5:
                alpha = alpha_min1
                if self.debug: print( "Drew alpha from -1-branch")
        elif self.show_errors:
            print ("!!!!!!!!!!!!")
            print ("BAD BAD ARGUMENT TO LAMBERT W")
            print ("Particle ID: ", particle_id)
            print ("Obtained (alpha k=0, alpha k=-1): ", (alpha_zero, alpha_min1))
            print ("The requirement is lamber_W_arg > (-1.0/exp(1)): " + str(lambert_W_arg) + " > " + str(-1.0/np.exp(1.0)))
            print ("gamma: ", gamma)
            print ("Nx: ", self.Nx)
            print ("w_rest: ", w_rest)
            print ("target_weight: ", target_weight)
            print ("phi: ", phi)
            print ("!!!!!!!!!!!!")
        #if self.debug: print "Drawing random number alpha_u: ", alpha_u
        #oldDebug = self.debug
        #self.debug = True
        if self.debug: print ("--------------------------------------")
        if self.debug: print ("Obtained (lambert_ans k=0, lambert_ans k=-1): ", (lambertw(lambert_W_arg), lambertw(lambert_W_arg, k=-1)))
        if self.debug: print ("Obtained (alpha k=0, alpha k=-1): ", (alpha_zero, alpha_min1))
        if self.debug: print ("Checking implicit equation with alpha (k=0, k=-1): ", \
            (self._implicitEquation(alpha_zero, gamma, self.Nx, a, c), self._implicitEquation(alpha_min1, gamma, self.Nx, a, c)))
        #if self.debug: print "Chose alpha = ", alpha
        #if self.debug: print "The implicit equation looked like: \n\t" + \
        #    "(alpha - 1)"+str(gamma)+" - " + str(self.Nx) + "log(alpha) + " + str(a) + " = 0"
        #if self.debug: print "Parameters: (gamma, Nx, aj)", (gamma, self.Nx, a)

        alpha = np.sqrt(alpha)
        if self.debug: print ("alpha = np.sqrt(alpha) ->  ", alpha)
        if self.debug: print( "--------------------------------------")
        #debug = oldDebug

        # 7.2) alpha*xi
        if self.debug: self.showMatrices(alpha*xi[0], alpha*xi[1], "alpha * xi", alpha*xi[2])

        # 8) Final nudge!
        eta_a += alpha*xi[0]
        hu_a  += alpha*xi[1]
        hv_a  += alpha*xi[2]
        if self.debug: self.showMatrices(eta_a, hu_a, "final x = x_a + alpha*xi", hv_a)

