import pyopencl
import os
import numpy as np

"""
Static function which reads a text file and creates an OpenCL kernel from that
"""
def get_kernel(cl_ctx, kernel_filename, block_width, block_height):
    #Create define string
    define_string = "#define block_width " + str(block_width) + "\n"
    define_string += "#define block_height " + str(block_height) + "\n"
    #print ("define_string:\n" + define_string)

    #Read the proper program
    # Kernels reside in gpu-ocean/sim/src/kernels
    module_path = os.path.dirname(os.path.realpath(__file__))
    fullpath = os.path.join(module_path, "../../sim/src/kernels", kernel_filename)
    
    options = ['-I', "../sim/src/kernels", '-I', "../../sim/src/kernels"]
    with open(fullpath, "r") as kernel_file:
        kernel_string = define_string + kernel_file.read()
        kernel = pyopencl.Program(cl_ctx, kernel_string).build(options)
        
    return kernel
    
    
        
        
        
        
        
        
        

"""
Class that holds data 
"""
class OpenCLArray2D:
    """
    Uploads initial data to the CL device
    """
    def __init__(self, cl_ctx, nx, ny, halo_x, halo_y, data, \
                 asymHalo=None):
        host_data = self.convert_to_float32(data)

        self.nx = nx
        self.ny = ny
        self.nx_halo = nx + 2*halo_x
        self.ny_halo = ny + 2*halo_y
        if (asymHalo is not None and len(asymHalo) == 4):
            # asymHalo = [halo_north, halo_east, halo_south, halo_west]
            self.nx_halo = nx + asymHalo[1] + asymHalo[3]
            self.ny_halo = ny + asymHalo[0] + asymHalo[2]
            
        assert(host_data.shape[1] == self.nx_halo), str(host_data.shape[1]) + " vs " + str(self.nx_halo)
        assert(host_data.shape[0] == self.ny_halo), str(host_data.shape[0]) + " vs " + str(self.ny_halo)
        
        assert(data.shape == (self.ny_halo, self.nx_halo))

        #Upload data to the device
        mf = pyopencl.mem_flags
        self.data = pyopencl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host_data)
        self.holds_data = True
        
        self.bytes_per_float = host_data.itemsize
        assert(self.bytes_per_float == 4)
        self.pitch = np.int32((self.nx_halo)*self.bytes_per_float)
        
        
    """
    Enables downloading data from CL device to Python
    """
    def download(self, cl_queue):
        if not self.holds_data:
            print "Throwing exception!!!!!"
            raise RuntimeError('OpenCL buffer has been freed')
        
        #Allocate data on the host for result
        host_data = np.empty((self.ny_halo, self.nx_halo), dtype=np.float32, order='C')
        
        #Copy data from device to host
        pyopencl.enqueue_copy(cl_queue, host_data, self.data)
        
        #Return
        return host_data

    """
    Frees the allocated memory buffers on the GPU 
    """
    def release(self):
        if self.holds_data:
            self.data.release()
            self.holds_data = False

    """
    Converts to C-style float 32 array suitable for the GPU/OpenCL
    """
    @staticmethod
    def convert_to_float32(data):
        if (not np.issubdtype(data.dtype, np.float32) or np.isfortran(data)):
            print "Converting H0"
            return data.astype(np.float32, order='C')
        else:
            return data



    
"""
A class representing an Arakawa A type (unstaggered, logically Cartesian) grid
"""
class SWEDataArakawaA:
    """
    Uploads initial data to the CL device
    """
    def __init__(self, cl_ctx, nx, ny, halo_x, halo_y, h0, hu0, hv0):
        self.h0  = OpenCLArray2D(cl_ctx, nx, ny, halo_x, halo_y, h0)
        self.hu0 = OpenCLArray2D(cl_ctx, nx, ny, halo_x, halo_y, hu0)
        self.hv0 = OpenCLArray2D(cl_ctx, nx, ny, halo_x, halo_y, hv0)
        
        self.h1  = OpenCLArray2D(cl_ctx, nx, ny, halo_x, halo_y, h0)
        self.hu1 = OpenCLArray2D(cl_ctx, nx, ny, halo_x, halo_y, hu0)
        self.hv1 = OpenCLArray2D(cl_ctx, nx, ny, halo_x, halo_y, hv0)

    """
    Swaps the variables after a timestep has been completed
    """
    def swap(self):
        self.h1,  self.h0  = self.h0,  self.h1
        self.hu1, self.hu0 = self.hu0, self.hu1
        self.hv1, self.hv0 = self.hv0, self.hv1
        
    """
    Enables downloading data from CL device to Python
    """
    def download(self, cl_queue):
        h_cpu  = self.h0.download(cl_queue)
        hu_cpu = self.hu0.download(cl_queue)
        hv_cpu = self.hv0.download(cl_queue)
        
        return h_cpu, hu_cpu, hv_cpu

    """
    Frees the allocated memory buffers on the GPU 
    """
    def release(self):
        self.h0.release()
        self.hu0.release()
        self.hv0.release()
        self.h1.release()
        self.hu1.release()
        self.hv1.release()
        
        
        
        
"""
A class representing an Arakawa C type (staggered, u fluxes on east/west faces, v fluxes on north/south faces) grid
We use h as cell centers
"""
class SWEDataArakawaC:
    """
    Uploads initial data to the CL device
    asymHalo needs to be on the form [north, east, south, west]
    """
    def __init__(self, cl_ctx, nx, ny, halo_x, halo_y, h0, hu0, hv0, \
                 asymHalo=None):
        #FIXME: This at least works for 0 and 1 ghost cells, but not convinced it generalizes
        assert(halo_x <= 1 and halo_y <= 1)
        # FIXME: asymHalo has not been tested for other values either.
        asymHaloU = asymHalo
        asymHaloV = asymHalo
        if (asymHalo is not None):
            #print(asymHalo)
            assert(max(asymHalo) <= 1)
            asymHaloU = [asymHalo[0], 0, asymHalo[2], 0]
            asymHaloV = [0, asymHalo[1], 0, asymHalo[3]]

            
        self.h0   = OpenCLArray2D(cl_ctx, nx, ny, halo_x, halo_y, h0, asymHalo)
        self.hu0  = OpenCLArray2D(cl_ctx, nx+1, ny, halo_x, halo_y, hu0, asymHaloU)
        self.hv0  = OpenCLArray2D(cl_ctx, nx, ny+1, halo_x, halo_y, hv0, asymHaloV)
        
        self.h1   = OpenCLArray2D(cl_ctx, nx, ny, halo_x, halo_y, h0, asymHalo)
        self.hu1  = OpenCLArray2D(cl_ctx, nx+1, ny, halo_x, halo_y, hu0, asymHaloU)
        self.hv1  = OpenCLArray2D(cl_ctx, nx, ny+1, halo_x, halo_y, hv0, asymHaloV)
                   
        
    """
    Swaps the variables after a timestep has been completed
    """
    def swap(self):
        #h is assumed to be constant (bottom topography really)
        self.h1,  self.h0  = self.h0, self.h1
        self.hu1, self.hu0 = self.hu0, self.hu1
        self.hv1, self.hv0 = self.hv0, self.hv1
        
    """
    Enables downloading data from CL device to Python
    """
    def download(self, cl_queue):
        h_cpu  = self.h0.download(cl_queue)
        hu_cpu = self.hu0.download(cl_queue)
        hv_cpu = self.hv0.download(cl_queue)
        
        return h_cpu, hu_cpu, hv_cpu

    """
    Frees the allocated memory buffers on the GPU 
    """
    def release(self):
        self.h0.release()
        self.hu0.release()
        self.hv0.release()
        self.h1.release()
        self.hu1.release()
        self.hv1.release()
        
    

"""
Class which represents different wind stresses
"""
class WindStressParams:

    """
    wind_type: TYpe of wind stress, 0=Uniform along shore, 1=bell shaped along shore, 2=moving cyclone
    wind_tau0: Amplitude of wind stress (Pa)
    wind_rho: Density of sea water (1025.0 kg / m^3)
    wind_alpha: Offshore e-folding length (1/(10*dx) = 5e-6 m^-1)
    wind_xm: Maximum wind stress for bell shaped wind stress
    wind_Rc: Distance to max wind stress from center of cyclone (10dx = 200 000 m)
    wind_x0: Initial x position of moving cyclone (dx*(nx/2) - u0*3600.0*48.0)
    wind_y0: Initial y position of moving cyclone (dy*(ny/2) - v0*3600.0*48.0)
    wind_u0: Translation speed along x for moving cyclone (30.0/sqrt(5.0))
    wind_v0: Translation speed along y for moving cyclone (-0.5*u0)
    """
    def __init__(self, 
                 type=99, # "no wind" \
                 tau0=0, rho=0, alpha=0, xm=0, Rc=0, \
                 x0=0, y0=0, \
                 u0=0, v0=0):
        self.type = np.int32(type)
        self.tau0 = np.float32(tau0)
        self.rho = np.float32(rho)
        self.alpha = np.float32(alpha)
        self.xm = np.float32(xm)
        self.Rc = np.float32(Rc)
        self.x0 = np.float32(x0)
        self.y0 = np.float32(y0)
        self.u0 = np.float32(u0)
        self.v0 = np.float32(v0)
                 

"""
Class that represents different forms for boundary conidtions
"""
class BoundaryConditions:
    """
    There is one parameter for each of the cartesian boundaries.
    Values can be set as follows:
    1 = Wall
    2 = Periodic (requires same for opposite boundary as well)
    3 = Numerical Sponge
    """
    def __init__(self, \
                 north=1, east=1, south=1, west=1, \
                 spongeCells=[0,0,0,0]):
        self.north = np.int32(north)
        self.east  = np.int32(east)
        self.south = np.int32(south)
        self.west  = np.int32(west)
        self.spongeCells = np.int32(spongeCells)
            
        # Checking that periodic boundaries are periodic
        assert not ((self.north == 2 or self.south == 2) and  \
                    (self.north != self.south)), \
                    'The given periodic boundary conditions are not periodically (north/south)'
        assert not ((self.east == 2 or self.west == 2) and \
                    (self.east != self.west)), \
                    'The given periodic boundary conditions are not periodically (east/west)'

    def isDefault(self):
        return (self.north == 1 and \
                self.east == 1 and \
                self.south == 1 and \
                self.east == 1)

    def isSponge(self):
        return (self.north == 3 or \
                self.east == 3 or \
                self.south == 3 or \
                self.west == 3)
    
    def _toString(self, cond):
        if cond == 1:
            return "Wall"
        elif cond == 2:
            return "Periodic"
        elif cond == 3:
            return "Numerical Sponge"
        else:
            return "Invalid :|"
        
    def __str__(self):
        msg = "north: "   + self._toString(self.north) + \
              ", east: "  + self._toString(self.east)  + \
              ", south: " + self._toString(self.south) + \
              ", west: "  + self._toString(self.west)
        msg = msg + ", spongeCells: " + str(self.spongeCells)
        return msg
    
"""
Class that checks boundary conditions and calls the required kernels for Arakawa A type grids.
"""
class BoundaryConditionsArakawaA:
    def __init__(self, cl_ctx, nx, ny, \
                 halo_x, halo_y, \
                 boundary_conditions, \
                 block_width = 16, block_height = 16):

        self.cl_ctx = cl_ctx
        self.boundary_conditions = boundary_conditions
        
        self.nx = np.int32(nx) 
        self.ny = np.int32(ny) 
        self.halo_x = np.int32(halo_x)
        self.halo_y = np.int32(halo_y)
        #print("boundary (ny, nx: ", (self.ny, self.nx))
        #print("boundary (halo_y, halo_x): ", (self.halo_y, self.halo_x))
        #print("numerical sponge cells (n,e,s,w): ", self.boundary_conditions.spongeCells)
        
        # Load kernel for periodic boundary
        self.boundaryKernels = get_kernel(self.cl_ctx,\
            "boundary_kernels.opencl", block_width, block_height)
       
        # Set kernel launch parameters
        self.local_size = (block_width, block_height)
        self.global_size = ( \
                             int(np.ceil((self.nx + 2*self.halo_x + 1)/float(self.local_size[0])) * self.local_size[0]), \
                             int(np.ceil((self.ny + 2*self.halo_y + 1)/float(self.local_size[1])) * self.local_size[1]) )


        
    def boundaryCondition(self, cl_queue, h, u, v):
                 
        if self.boundary_conditions.north == 2:
            self.periodic_boundary_NS(cl_queue, h, u, v)
        elif self.boundary_conditions.north == 3 or \
             self.boundary_conditions.south == 3:
            self.numerical_sponge_NS(cl_queue, h, u, v)
            
        if self.boundary_conditions.east == 2:
            self.periodic_boundary_EW(cl_queue, h, u, v)
        elif self.boundary_conditions.east == 3 or \
             self.boundary_conditions.west == 3:
            self.numerical_sponge_EW(cl_queue, h, u, v)
            
             
    def periodic_boundary_NS(self, cl_queue, h, u, v):
        self.boundaryKernels.periodicBoundary_NS( \
            cl_queue, self.global_size, self.local_size, \
            self.nx, self.ny, \
            self.halo_x, self.halo_y, \
            h.data, h.pitch, \
            u.data, u.pitch, \
            v.data, v.pitch)
        

    def periodic_boundary_EW(self, cl_queue, h, v, u):
        self.boundaryKernels.periodicBoundary_EW( \
            cl_queue, self.global_size, self.local_size, \
            self.nx, self.ny, \
            self.halo_x, self.halo_y, \
            h.data, h.pitch, \
            u.data, u.pitch, \
            v.data, v.pitch)


    def numerical_sponge_NS(self, cl_queue, h, u, v):
        self.boundaryKernels.numericalSponge_NS( \
            cl_queue, self.global_size, self.local_size, \
            self.boundary_conditions.north, self.boundary_conditions.south, \
            self.nx, self.ny, \
            self.halo_x, self.halo_y, \
            self.boundary_conditions.spongeCells[0], \
            self.boundary_conditions.spongeCells[2], \
            h.data, h.pitch, \
            u.data, u.pitch, \
            v.data, v.pitch)                                   

    def numerical_sponge_EW(self, cl_queue, h, u, v):
        self.boundaryKernels.numericalSponge_EW( \
            cl_queue, self.global_size, self.local_size, \
            self.boundary_conditions.east, self.boundary_conditions.west, \
            self.nx, self.ny, \
            self.halo_x, self.halo_y, \
            self.boundary_conditions.spongeCells[1], \
            self.boundary_conditions.spongeCells[3], \
            h.data, h.pitch, \
            u.data, u.pitch, \
            v.data, v.pitch)

    
                 
"""
Class for holding bathymetry defined on cell intersections (cell corners) and reconstructed on 
cell mid-points.
"""
class Bathymetry:
    
    def __init__(self, cl_ctx, cl_queue, nx, ny, halo_x, halo_y, Bi_host, \
                 boundary_conditions=BoundaryConditions(), \
                 block_width=16, block_height=16):
        self.cl_queue = cl_queue
        self.cl_ctx = cl_ctx
        # Convert scalar data to int32
        self.nx = np.int32(nx)
        self.ny = np.int32(ny)
        self.halo_x = np.int32(halo_x)
        self.halo_y = np.int32(halo_y)
        self.halo_nx = np.int32(nx + 2*halo_x)
        self.halo_ny = np.int32(ny + 2*halo_y)
        self.boundary_conditions = boundary_conditions
             
        # Check that Bi has the size corresponding to number of cell intersections
        BiShapeY, BiShapeX = Bi_host.shape
        assert(BiShapeX == nx+1+2*halo_x and BiShapeY == ny+1+2*halo_y), \
                "Wrong size of bottom bathymetry, should be defined on cell intersections, not cell centers. " + \
                str((BiShapeX, BiShapeY)) + " vs " + str((nx+1+2*halo_x, ny+1+2*halo_y))
        
        # Upload Bi to device
        self.Bi = OpenCLArray2D(cl_ctx, nx+1, ny+1, halo_x, halo_y, Bi_host)

        # Define OpenCL parameters
        self.local_size = (block_width, block_height) 
        self.global_size = ( \
                       int(np.ceil( (self.halo_nx+1) / float(self.local_size[0])) * self.local_size[0]), \
                       int(np.ceil( (self.halo_ny+1) / float(self.local_size[1])) * self.local_size[1]) \
        ) 
        
        # Check boundary conditions and make Bi periodic if necessary
        self.bi_boundary_kernel = get_kernel(self.cl_ctx, "boundary_kernels.opencl", block_width, block_height)
        self._boundaryConditions()
        
        # Allocate Bm
        Bm_host = np.zeros((self.halo_ny, self.halo_nx), dtype=np.float32, order='C')
        self.Bm = OpenCLArray2D(self.cl_ctx, nx, ny, halo_x, halo_y, Bm_host)
        
        # Load kernel for finding Bm from Bi
        self.initBm_kernel = get_kernel(self.cl_ctx, "initBm_kernel.opencl", block_width, block_height)
      
        
        # Call kernel
        self.initBm_kernel.initBm(self.cl_queue, self.global_size, self.local_size, \
                                   self.halo_nx, self.halo_ny, \
                                   self.Bi.data, self.Bi.pitch, \
                                   self.Bm.data, self.Bm.pitch)

                 
    def download(self, cl_queue):
        Bm_cpu = self.Bm.download(cl_queue)
        Bi_cpu = self.Bi.download(cl_queue)

        return Bi_cpu, Bm_cpu

    """
    Frees the allocated memory buffers on the GPU 
    """
    def release(self):
        self.Bm.release()
        self.Bi.release()
        
    # Transforming water elevation into water depth
    def waterElevationToDepth(self, h):
        
        assert ((h.ny_halo, h.nx_halo) == (self.halo_ny, self.halo_nx)), \
            "h0 not the correct shape: " + str(h0.shape) + ", but should be " + str((self.halo_ny, self.halo_nx))

        # Call kernel        
        self.initBm_kernel.waterElevationToDepth(self.cl_queue, self.global_size, self.local_size, \
                                   self.halo_nx, self.halo_ny, \
                                   h.data, h.pitch, \
                                   self.Bm.data, self.Bm.pitch)
        
    # Transforming water depth into water elevation
    def waterDepthToElevation(self, w, h):

        assert ((h.ny_halo, h.nx_halo) == (self.halo_ny, self.halo_nx)), \
            "h0 not the correct shape: " + str(h0.shape) + ", but should be " + str((self.halo_ny, self.halo_nx))
        assert ((w.ny_halo, w.nx_halo) == (self.halo_ny, self.halo_nx)), \
            "w not the correct shape: " + str(w.shape) + ", but should be " + str((self.halo_ny, self.halo_nx))
        # Call kernel        
        self.initBm_kernel.waterDepthToElevation(self.cl_queue, self.global_size, self.local_size, \
                                   self.halo_nx, self.halo_ny, \
                                   w.data, w.pitch, \
                                   h.data, h.pitch, \
                                   self.Bm.data, self.Bm.pitch)
        
    def _boundaryConditions(self):
        # North-south:
        if (self.boundary_conditions.north == 2) and (self.boundary_conditions.south == 2):
            self.bi_boundary_kernel.periodic_boundary_intersections_NS( \
                self.cl_queue, self.global_size, self.local_size, \
                self.nx, self.ny, self.halo_x, self.halo_y, \
                self.Bi.data, self.Bi.pitch)
        else:
            self.bi_boundary_kernel.closed_boundary_intersections_NS( \
                self.cl_queue, self.global_size, self.local_size, \
                self.nx, self.ny, self.halo_x, self.halo_y, \
                self.Bi.data, self.Bi.pitch)

        # East-west:
        if (self.boundary_conditions.east == 2) and (self.boundary_conditions.west == 2):
            self.bi_boundary_kernel.periodic_boundary_intersections_EW( \
                self.cl_queue, self.global_size, self.local_size, \
                self.nx, self.ny, self.halo_x, self.halo_y, \
                self.Bi.data, self.Bi.pitch)
        else:
            self.bi_boundary_kernel.closed_boundary_intersections_EW( \
                self.cl_queue, self.global_size, self.local_size, \
                self.nx, self.ny, self.halo_x, self.halo_y, \
                self.Bi.data, self.Bi.pitch)
                 
                 
                 
                 
