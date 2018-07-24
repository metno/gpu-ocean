from pycuda.compiler import SourceModule
import pycuda.gpuarray
import pycuda.driver as cuda
import os
import numpy as np

import warnings
import functools
import WindStress

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated.
    Reference: https://stackoverflow.com/questions/2536307/how-do-i-deprecate-python-functions
    """
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        # MLS: Seems wrong to mess with standard filter settings in this context.
        # Code is nevertheless not removed, as this could be relevant at a later time.
        #warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        #warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func

def get_kernel(kernel_filename, block_width, block_height):
    """
    Static function which reads a text file and creates an CUDA kernel from that
    """
    #Create define string
    define_string = "#define block_width " + str(block_width) + "\n"
    define_string += "#define block_height " + str(block_height) + "\n"
    #print ("define_string:\n" + define_string)

    #Read the proper program
    # Kernels reside in gpu-ocean/sim/src/kernels
    module_path = os.path.dirname(os.path.realpath(__file__))
    fullpath = os.path.join(module_path, "../../sim/src/kernels", kernel_filename)
    
    options = [os.path.join(module_path, "../../sim/src/kernels")]
    with open(fullpath, "r") as kernel_file:
        kernel_string = define_string + kernel_file.read()
        kernel = SourceModule(kernel_string, include_dirs=options)
        
    return kernel
    
        
        
        
        
        
        
class CUDAArray2D:
    """
    Class that holds data 
    """
    
    def __init__(self, nx, ny, halo_x, halo_y, data, \
                 asymHalo=None, double_precision=False, integers=False):
        """
        Uploads initial data to the CUDA device
        """
        self.double_precision = double_precision
        self.integers = integers
        host_data = None
        if self.double_precision:
            host_data = data
        else:
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
        self.data = pycuda.gpuarray.to_gpu_async(host_data)
        self.holds_data = True
        
        self.bytes_per_float = host_data.itemsize
        assert(self.bytes_per_float == 4 or
              (self.double_precision and self.bytes_per_float == 8))
        self.pitch = np.int32((self.nx_halo)*self.bytes_per_float)
        
        
    def upload(self, gpu_stream, data):
        """
        Filling the allocated buffer with new data
        """
        if not self.holds_data:
            raise RuntimeError('The buffer has been freed before upload is called')
        
        # Make sure that the input is of correct size:
        if self.double_precision:
            host_data = data
        else:
            host_data = self.convert_to_float32(data)
            
        assert(host_data.shape[1] == self.nx_halo), str(host_data.shape[1]) + " vs " + str(self.nx_halo)
        assert(host_data.shape[0] == self.ny_halo), str(host_data.shape[0]) + " vs " + str(self.ny_halo)
        assert(host_data.shape == (self.ny_halo, self.nx_halo))
        
        assert(host_data.itemsize == self.bytes_per_float), "Host data itemsize is " + str(host_data.itemsize) + ", but should have been " + str(self.bytes_per_float)
        
        # Okay, everything is fine, now upload:
        self.data.set_async(host_data, stream=gpu_stream)
        
    
    def copyBuffer(self, gpu_stream, buffer):
        """
        Copying the given device buffer into the already allocated memory
        """
        if not self.holds_data:
            raise RuntimeError('The buffer has been freed before copying buffer')
        
        if not buffer.holds_data:
            raise RuntimeError('The provided buffer is either not allocated, or has been freed before copying buffer')
        
        # Make sure that the input is of correct size:
        assert(buffer.nx_halo == self.nx_halo), str(buffer.nx_halo) + " vs " + str(self.nx_halo)
        assert(buffer.ny_halo == self.ny_halo), str(buffer.ny_halo) + " vs " + str(self.ny_halo)
        
        assert(buffer.bytes_per_float == self.bytes_per_float), "Provided buffer itemsize is " + str(buffer.bytes_per_float) + ", but should have been " + str(self.bytes_per_float)
        
        # Okay, everything is fine - issue device-to-device-copy:
        total_num_bytes = self.bytes_per_float*self.nx_halo*self.ny_halo
        cuda.memcpy_dtod_async(self.data.ptr(), buffer.data.ptr(), total_num_bytes, stream=gpu_stream)
        
        
        
    def download(self, gpu_stream):
        """
        Enables downloading data from CUDA device to Python
        """
        if not self.holds_data:
            raise RuntimeError('CUDA buffer has been freed')
        
        #Copy data from device to host
        host_data = self.data.get(stream=gpu_stream)
        
        #Return
        return host_data

    
    def release(self):
        """
        Frees the allocated memory buffers on the GPU 
        """
        if self.holds_data:
            del self.data
            self.holds_data = False

    
    @staticmethod
    def convert_to_float32(data):
        """
        Converts to C-style float 32 array suitable for the GPU/CUDA
        """
        if (not np.issubdtype(data.dtype, np.float32) or np.isfortran(data)):
            return data.astype(np.float32, order='C')
        else:
            return data

    
class SWEDataArakawaA:
    """
    A class representing an Arakawa A type (unstaggered, logically Cartesian) grid
    """

    def __init__(self, nx, ny, halo_x, halo_y, h0, hu0, hv0):
        """
        Uploads initial data to the CUDA device
        """
        self.h0  = CUDAArray2D(nx, ny, halo_x, halo_y, h0)
        self.hu0 = CUDAArray2D(nx, ny, halo_x, halo_y, hu0)
        self.hv0 = CUDAArray2D(nx, ny, halo_x, halo_y, hv0)
        
        self.h1  = CUDAArray2D(nx, ny, halo_x, halo_y, h0)
        self.hu1 = CUDAArray2D(nx, ny, halo_x, halo_y, hu0)
        self.hv1 = CUDAArray2D(nx, ny, halo_x, halo_y, hv0)

    def swap(self):
        """
        Swaps the variables after a timestep has been completed
        """
        self.h1,  self.h0  = self.h0,  self.h1
        self.hu1, self.hu0 = self.hu0, self.hu1
        self.hv1, self.hv0 = self.hv0, self.hv1
        
    def download(self, gpu_stream):
        """
        Enables downloading data from CUDA device to Python
        """
        h_cpu  = self.h0.download(gpu_stream)
        hu_cpu = self.hu0.download(gpu_stream)
        hv_cpu = self.hv0.download(gpu_stream)
        
        return h_cpu, hu_cpu, hv_cpu
    

    def downloadPrevTimestep(self, gpu_stream):
        """
        Enables downloading data from the additional buffer of CUDA device to Python
        """
        h_cpu  = self.h1.download(gpu_stream)
        hu_cpu = self.hu1.download(gpu_stream)
        hv_cpu = self.hv1.download(gpu_stream)
        
        return h_cpu, hu_cpu, hv_cpu

    def release(self):
        """
        Frees the allocated memory buffers on the GPU 
        """
        self.h0.release()
        self.hu0.release()
        self.hv0.release()
        self.h1.release()
        self.hu1.release()
        self.hv1.release()
        
        
        
        
class SWEDataArakawaC:
    """
    A class representing an Arakawa C type (staggered, u fluxes on east/west faces, v fluxes on north/south faces) grid
    We use h as cell centers
    """
    def __init__(self, nx, ny, halo_x, halo_y, h0, hu0, hv0, \
                 asymHalo=None):
        """
        Uploads initial data to the CUDA device
        asymHalo needs to be on the form [north, east, south, west]
        """
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

        #print "SWEDataArakawaC"
        #print "(h0.shape, (nx, ny), asymHalo,  (halo_x, halo_y)): ", (h0.shape, (nx, ny), asymHalo,  (halo_x, halo_y))
        #print "(hu0.shape, (nx, ny), asymHalo, (halo_x, halo_y)): ", (hu0.shape, (nx+1, ny), asymHaloU,  (halo_x, halo_y))
        #print "(hv0.shape, (nx, ny), asymHalo,  (halo_x, halo_y)): ", (hv0.shape, (nx, ny+1), asymHaloV, (halo_x, halo_y))

        self.h0   = CUDAArray2D(nx, ny, halo_x, halo_y, h0, asymHalo)
        self.hu0  = CUDAArray2D(nx+1, ny, halo_x, halo_y, hu0, asymHaloU)
        self.hv0  = CUDAArray2D(nx, ny+1, halo_x, halo_y, hv0, asymHaloV)
        
        self.h1   = CUDAArray2D(nx, ny, halo_x, halo_y, h0, asymHalo)
        self.hu1  = CUDAArray2D(nx+1, ny, halo_x, halo_y, hu0, asymHaloU)
        self.hv1  = CUDAArray2D(nx, ny+1, halo_x, halo_y, hv0, asymHaloV)
                   
        
    def swap(self):
        """
        Swaps the variables after a timestep has been completed
        """
        #h is assumed to be constant (bottom topography really)
        self.h1,  self.h0  = self.h0, self.h1
        self.hu1, self.hu0 = self.hu0, self.hu1
        self.hv1, self.hv0 = self.hv0, self.hv1
        
    def download(self, gpu_stream):
        """
        Enables downloading data from CUDA device to Python
        """
        h_cpu  = self.h0.download(gpu_stream)
        hu_cpu = self.hu0.download(gpu_stream)
        hv_cpu = self.hv0.download(gpu_stream)
        
        return h_cpu, hu_cpu, hv_cpu

    def downloadPrevTimestep(self, gpu_stream):
        """
        Enables downloading data from the additional buffer of CUDA device to Python
        """
        h_cpu  = self.h1.download(gpu_stream)
        hu_cpu = self.hu1.download(gpu_stream)
        hv_cpu = self.hv1.download(gpu_stream)
        
        return h_cpu, hu_cpu, hv_cpu
    
    def release(self):
        """
        Frees the allocated memory buffers on the GPU 
        """
        self.h0.release()
        self.hu0.release()
        self.hv0.release()
        self.h1.release()
        self.hu1.release()
        self.hv1.release()

@deprecated
def WindStressParams(type=99, # "no wind" \
                 tau0=0, rho=0, alpha=0, xm=0, Rc=0, \
                 x0=0, y0=0, \
                 u0=0, v0=0, \
                 wind_speed=0, wind_direction=0):
    """
    Backward compatibility function to avoid rewriting old code and notebooks.
    
    SHOULD NOT BE USED IN NEW CODE! Make WindStress object directly instead.
    """
    
    type_ = np.int32(type)
    tau0_ = np.float32(tau0)
    rho_ = np.float32(rho)
    rho_air_ = np.float32(1.3) # new parameter
    alpha_ = np.float32(alpha)
    xm_ = np.float32(xm)
    Rc_ = np.float32(Rc)
    x0_ = np.float32(x0)
    y0_ = np.float32(y0)
    u0_ = np.float32(u0)
    v0_ = np.float32(v0)
    wind_speed_ = np.float32(wind_speed)
    wind_direction_ = np.float32(wind_direction)
    
    if type == 0:
        wind_stress = WindStress.UniformAlongShoreWindStress( \
            tau0=tau0_, rho=rho_, alpha=alpha_)
    elif type == 1:
        wind_stress = WindStress.BellShapedAlongShoreWindStress( \
            xm=xm_, tau0=tau0_, rho=rho_, alpha=alpha_)
    elif type == 2:
        wind_stress = WindStress.MovingCycloneWindStress( \
            Rc=Rc_, x0=x0_, y0=y0_, u0=u0_, v0=v0_)
    elif type == 50:
        wind_stress = WindStress.GenericUniformWindStress( \
            rho_air=rho_air_, wind_speed=wind_speed_, wind_direction=wind_direction_)
    elif type == 99:
        wind_stress = WindStress.NoWindStress()
    else:
        raise RuntimeError('Invalid wind stress type!')
    
    return wind_stress

class BoundaryConditions:
    """
    Class that represents different forms for boundary conidtions
    """
    def __init__(self, \
                 north=1, east=1, south=1, west=1, \
                 spongeCells=[0,0,0,0]):
        """
        There is one parameter for each of the cartesian boundaries.
        Values can be set as follows:
        1 = Wall
        2 = Periodic (requires same for opposite boundary as well)
        3 = Open Boundary with Flow Relaxation Scheme
        4 = Open linear interpolation
        Options 3 and 4 are of sponge type (requiring extra computational domain)
        """
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

    def get(self):
        return [self.north, self.east, self.south, self.west]
    
    def getSponge(self):
        return self.spongeCells
    
    def isDefault(self):
        return (self.north == 1 and \
                self.east == 1 and \
                self.south == 1 and \
                self.east == 1)

    def isSponge(self):
        return (self.north == 3 or self.north == 4 or \
                self.east  == 3 or self.east  == 4 or \
                self.south == 3 or self.south == 4 or \
                self.west  == 3 or self.west  == 4)
    
    def isPeriodicNorthSouth(self):
        return (self.north == 2 and self.south == 2)
    
    def isPeriodicEastWest(self):
        return (self.east == 2 and self.west == 2)
    
    
    def _toString(self, cond):
        if cond == 1:
            return "Wall"
        elif cond == 2:
            return "Periodic"
        elif cond == 3:
            return "Flow Relaxation Scheme"
        elif cond == 4:
            return "Open Linear Interpolation"
        else:
            return "Invalid :|"
        
    def __str__(self):
        msg = "north: "   + self._toString(self.north) + \
              ", east: "  + self._toString(self.east)  + \
              ", south: " + self._toString(self.south) + \
              ", west: "  + self._toString(self.west)
        msg = msg + ", spongeCells: " + str(self.spongeCells)
        return msg
    
class BoundaryConditionsArakawaA:
    """
    Class that checks boundary conditions and calls the required kernels for Arakawa A type grids.
    """

    def __init__(self, nx, ny, \
                 halo_x, halo_y, \
                 boundary_conditions, \
                 block_width = 16, block_height = 16):

        self.boundary_conditions = boundary_conditions
        
        self.nx = np.int32(nx) 
        self.ny = np.int32(ny) 
        self.halo_x = np.int32(halo_x)
        self.halo_y = np.int32(halo_y)
        #print("boundary (ny, nx: ", (self.ny, self.nx))
        #print("boundary (halo_y, halo_x): ", (self.halo_y, self.halo_x))
        #print("numerical sponge cells (n,e,s,w): ", self.boundary_conditions.spongeCells)
        
        # Load CUDA module for periodic boundary
        self.boundaryKernels = get_kernel("boundary_kernels.cu", block_width, block_height)

        # Get CUDA functions
        self.periodicBoundary_NS = self.boundaryKernels.get_function("periodicBoundary_NS")
        self.periodicBoundary_NS.prepare("iiiiPiPiPi")
        self.periodic_boundary_EW = self.boundaryKernels.get_function("periodic_boundary_EW")
        self.periodic_boundary_EW.prepare("iiiiPiPiPi")
        self.linear_interpolation_NS = self.boundaryKernels.get_function("linear_interpolation_NS")
        self.linear_interpolation_NS.prepare("iiiiiiiiPiPiPi")
        self.linear_interpolation_EW = self.boundaryKernels.get_function("linear_interpolation_EW")
        self.linear_interpolation_EW.prepare("iiiiiiiiPiPiPi")
        self.flowRelaxationScheme_NS = self.boundaryKernels.get_function("flowRelaxationScheme_NS")
        self.flowRelaxationScheme_NS.prepare("iiiiiiiiPiPiPi")
        self.flowRelaxationScheme_EW = self.boundaryKernels.get_function("flowRelaxationScheme_EW")
        self.flowRelaxationScheme_EW.prepare("iiiiiiiiPiPiPi")
       
        # Set kernel launch parameters
        self.local_size = (block_width, block_height, 1)
        self.global_size = ( \
                             int(np.ceil((self.nx + 2*self.halo_x + 1)/float(self.local_size[0])) * self.local_size[0]), \
                             int(np.ceil((self.ny + 2*self.halo_y + 1)/float(self.local_size[1])) * self.local_size[1]) )


        
    def boundaryCondition(self, gpu_stream, h, u, v):
        if self.boundary_conditions.north == 2:
            self.periodic_boundary_NS(gpu_stream, h, u, v)
        else:
            if (self.boundary_conditions.north == 3 or \
                self.boundary_conditions.south == 3):
                self.flow_relaxation_NS(gpu_stream, h, u, v)
            if (self.boundary_conditions.north == 4 or \
                self.boundary_conditions.south == 4):
                self.linear_interpolation_NS(gpu_stream, h, u, v)
            
            
        if self.boundary_conditions.east == 2:
            self.periodic_boundary_EW(gpu_stream, h, u, v)
        else:
            if (self.boundary_conditions.east == 3 or \
                self.boundary_conditions.west == 3):
                self.flow_relaxation_EW(gpu_stream, h, u, v)
            if (self.boundary_conditions.east == 4 or \
                self.boundary_conditions.west == 4):
                self.linear_interpolation_EW(gpu_stream, h, u, v)
             
    def periodic_boundary_NS(self, gpu_stream, h, u, v):
        self.periodicBoundary_NS.prepared_async_call( \
            self.global_size, self.local_size, gpu_stream, \
            self.nx, self.ny, \
            self.halo_x, self.halo_y, \
            h.data, h.pitch, \
            u.data, u.pitch, \
            v.data, v.pitch)
        

    def periodic_boundary_EW(self, gpu_stream, h, v, u):
        self.periodicBoundary_EW.prepared_async_call( \
            self.global_size, self.local_size, gpu_stream, \
            self.nx, self.ny, \
            self.halo_x, self.halo_y, \
            h.data, h.pitch, \
            u.data, u.pitch, \
            v.data, v.pitch)


    def linear_interpolation_NS(self, gpu_stream, h, u, v):
        self.linearInterpolation_NS.prepared_async_call( \
            self.global_size, self.local_size, gpu_stream, \
            self.boundary_conditions.north, self.boundary_conditions.south, \
            self.nx, self.ny, \
            self.halo_x, self.halo_y, \
            self.boundary_conditions.spongeCells[0], \
            self.boundary_conditions.spongeCells[2], \
            h.data, h.pitch, \
            u.data, u.pitch, \
            v.data, v.pitch)                                   

    def linear_interpolation_EW(self, gpu_stream, h, u, v):
        self.linearInterpolation_EW.prepared_async_call( \
            self.global_size, self.local_size, gpu_stream, \
            self.boundary_conditions.east, self.boundary_conditions.west, \
            self.nx, self.ny, \
            self.halo_x, self.halo_y, \
            self.boundary_conditions.spongeCells[1], \
            self.boundary_conditions.spongeCells[3], \
            h.data, h.pitch, \
            u.data, u.pitch, \
            v.data, v.pitch)

    def flow_relaxation_NS(self, gpu_stream, h, u, v):
        self.flowRelaxationScheme_NS.prepared_async_call( \
            self.global_size, self.local_size, gpu_stream, \
            self.boundary_conditions.north, self.boundary_conditions.south, \
            self.nx, self.ny, \
            self.halo_x, self.halo_y, \
            self.boundary_conditions.spongeCells[0], \
            self.boundary_conditions.spongeCells[2], \
            h.data, h.pitch, \
            u.data, u.pitch, \
            v.data, v.pitch)   

    def flow_relaxation_EW(self, gpu_stream, h, u, v):
        self.flowRelaxationScheme_EW.prepared_async_call( \
            self.global_size, self.local_size, gpu_stream, \
            self.boundary_conditions.east, self.boundary_conditions.west, \
            self.nx, self.ny, \
            self.halo_x, self.halo_y, \
            self.boundary_conditions.spongeCells[1], \
            self.boundary_conditions.spongeCells[3], \
            h.data, h.pitch, \
            u.data, u.pitch, \
            v.data, v.pitch)


        
class Bathymetry:
    """
    Class for holding bathymetry defined on cell intersections (cell corners) and reconstructed on 
    cell mid-points.
    """
    
    def __init__(self, gpu_stream, nx, ny, halo_x, halo_y, Bi_host, \
                 boundary_conditions=BoundaryConditions(), \
                 block_width=16, block_height=16):
        # Convert scalar data to int32
        self.gpu_stream = gpu_stream
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
        self.Bi = CUDAArray2D(nx+1, ny+1, halo_x, halo_y, Bi_host)

        # Define OpenCL parameters
        self.local_size = (block_width, block_height) 
        self.global_size = ( \
                       int(np.ceil( (self.halo_nx+1) / float(self.local_size[0])) * self.local_size[0]), \
                       int(np.ceil( (self.halo_ny+1) / float(self.local_size[1])) * self.local_size[1]) \
        ) 
        
        # Check boundary conditions and make Bi periodic if necessary
        # Load CUDA module for periodic boundary
        self.boundaryKernels = get_kernel("boundary_kernels.cu", block_width, block_height)

        # Get CUDA functions
        self.periodic_boundary_intersections_NS = self.boundaryKernels.get_function("periodic_boundary_intersections_NS")
        self.periodic_boundary_intersections_NS.prepare("iiiiPi")
        self.periodic_boundary_intersections_EW = self.boundaryKernels.get_function("periodic_boundary_intersections_EW")
        self.periodic_boundary_intersections_EW.prepare("iiiiPi")
        self.closed_boundary_intersections_NS = self.boundaryKernels.get_function("closed_boundary_intersections_NS")
        self.closed_boundary_intersections_NS.prepare("iiiiPi")
        self.closed_boundary_intersections_EW = self.boundaryKernels.get_function("closed_boundary_intersections_EW")
        self.closed_boundary_intersections_EW.prepare("iiiiPi")

        self._boundaryConditions()
        
        # Allocate Bm
        Bm_host = np.zeros((self.halo_ny, self.halo_nx), dtype=np.float32, order='C')
        self.Bm = CUDAArray2D(nx, ny, halo_x, halo_y, Bm_host)

        # Load kernel for finding Bm from Bi
        self.initBm_kernel = get_kernel("initBm_kernel.cu", block_width, block_height)

        # Get CUDA functions
        self.initBm = self.initBm_kernel.get_function("initBm")
        self.initBm.prepare("iiPiPi")
        self.waterElevationToDepth = self.initBm_kernel.get_function("waterElevationToDepth")
        self.waterElevationToDepth.prepare("iiPiPi")

        # Call kernel
        self.initBm.prepared_async_call(self.global_size, self.local_size, self.gpu_stream, \
                                   self.halo_nx, self.halo_ny, \
                                   self.Bi.data, self.Bi.pitch, \
                                   self.Bm.data, self.Bm.pitch)

                 
    def download(self, gpu_stream):
        Bm_cpu = self.Bm.download(gpu_stream)
        Bi_cpu = self.Bi.download(gpu_stream)

        return Bi_cpu, Bm_cpu

    def release(self):
        """
        Frees the allocated memory buffers on the GPU 
        """
        self.Bm.release()
        self.Bi.release()
        
    # Transforming water elevation into water depth
    def waterElevationToDepth(self, h):
        
        assert ((h.ny_halo, h.nx_halo) == (self.halo_ny, self.halo_nx)), \
            "h0 not the correct shape: " + str(h0.shape) + ", but should be " + str((self.halo_ny, self.halo_nx))

        # Call kernel        
        self.waterElevationToDepth.prepared_async_call(self.global_size, self.local_size, self.gpu_stream, \
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
        self.waterDepthToElevation.prepared_async_call(self.global_size, self.local_size, self.gpu_stream, \
                                   self.halo_nx, self.halo_ny, \
                                   w.data, w.pitch, \
                                   h.data, h.pitch, \
                                   self.Bm.data, self.Bm.pitch)
        
    def _boundaryConditions(self):
        # North-south:
        if (self.boundary_conditions.north == 2) and (self.boundary_conditions.south == 2):
            self.periodic_boundary_intersections_NS.prepared_async_call( \
                self.global_size, self.local_size, self.gpu_stream, \
                self.nx, self.ny, self.halo_x, self.halo_y, \
                self.Bi.data, self.Bi.pitch)
        else:
            self.closed_boundary_intersections_NS.prepared_async_call( \
                self.global_size, self.local_size, self.gpu_stream, \
                self.nx, self.ny, self.halo_x, self.halo_y, \
                self.Bi.data, self.Bi.pitch)

        # East-west:
        if (self.boundary_conditions.east == 2) and (self.boundary_conditions.west == 2):
            self.periodic_boundary_intersections_EW.prepared_async_call( \
                self.global_size, self.local_size, self.gpu_stream, \
                self.nx, self.ny, self.halo_x, self.halo_y, \
                self.Bi.data, self.Bi.pitch)
        else:
            self.closed_boundary_intersections_EW.prepared_async_call( \
                self.global_size, self.local_size, self.gpu_stream, \
                self.nx, self.ny, self.halo_x, self.halo_y, \
                self.Bi.data, self.Bi.pitch)
                 
                 
                 
                 
