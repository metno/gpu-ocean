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

    #Read the proper program
    module_path = os.path.dirname(os.path.realpath(__file__))
    fullpath = os.path.join(module_path, kernel_filename)
    options = ['-I', "'" + module_path + "'"]
    with open(fullpath, "r") as kernel_file:
        kernel_string = define_string + kernel_file.read()
        kernel = pyopencl.Program(cl_ctx, kernel_string).build(options)
        
    return kernel
    
    
"""
Converts to C-style float 32 array suitable for the GPU
"""
def convert_to_float32(data):
    if (not np.issubdtype(data.dtype, np.float32) or np.isfortran(data)):
        print "Converting H0"
        return data.astype(np.float32, order='C')
    else:
        return data
        
        
        
class ClData:
    def __init__(self, cl_ctx, data=None):
        if (data == None):
            self.data = pyopencl.buffer()
        else:
            data
        

"""
Class that holds data 
"""
class OpenCLArray2D:
    """
    Uploads initial data to the CL device
    """
    def __init__(self, cl_ctx, nx, ny, halo_x, halo_y, data):
        host_data = convert_to_float32(data)
        
        self.nx = nx
        self.ny = ny
        self.nx_halo = nx + 2*halo_x
        self.ny_halo = ny + 2*halo_y
        assert(host_data.shape[1] == self.nx_halo)
        assert(host_data.shape[0] == self.ny_halo)
        
        assert(data.shape == (self.ny_halo, self.nx_halo))

        #Upload data to the device
        mf = pyopencl.mem_flags
        self.data = pyopencl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host_data)
        
        self.bytes_per_float = host_data.itemsize
        assert(self.bytes_per_float == 4)
        self.pitch = np.int32((self.nx_halo)*self.bytes_per_float)
        
        
    """
    Enables downloading data from CL device to Python
    """
    def download(self, cl_queue):
        #Allocate data on the host for result
        host_data = np.empty((self.ny_halo, self.nx_halo), dtype=np.float32, order='C')
        
        #Copy data from device to host
        pyopencl.enqueue_copy(cl_queue, host_data, self.data)
        
        #Return
        return host_data


        
        
        
"""
A class representing an Akrawa A type (unstaggered, logically Cartesian) grid
"""
class SWEDataArkawaA:
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
A class representing an Akrawa A type (unstaggered, logically Cartesian) grid
"""
class SWEDataArkawaA:
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
A class representing an Akrawa C type (staggered, u fluxes on east/west faces, v fluxes on north/south faces) grid
We use h as cell centers
"""
class SWEDataArkawaC:
    """
    Uploads initial data to the CL device
    """
    def __init__(self, cl_ctx, nx, ny, halo_x, halo_y, h0, hu0, hv0):
        #FIXME: This at least works for 0 and 1 ghost cells, but not convinced it generalizes
        assert(halo_x <= 1 && halo_y <= 1)
        
        self.h0   = OpenCLArray2D(cl_ctx, nx, ny, halo_x, halo_y, h0)
        self.hu0  = OpenCLArray2D(cl_ctx, nx+1, ny, 0, halo_y, hu0)
        self.hv0  = OpenCLArray2D(cl_ctx, nx, ny+1, halo_x, 0, hv0)
        
        self.h1   = OpenCLArray2D(cl_ctx, nx, ny, halo_x, halo_y, h0)
        self.hu1  = OpenCLArray2D(cl_ctx, nx+1, ny, 0, halo_y, hu0)
        self.hv1  = OpenCLArray2D(cl_ctx, nx, ny+1, halo_x, 0, hv0)

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
        
        


