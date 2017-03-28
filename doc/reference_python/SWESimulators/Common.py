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
    module_path = os.path.dirname(os.path.realpath(__file__))
    fullpath = os.path.join(module_path, kernel_filename)
    options = ['-I', "'" + module_path + "'"]
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
        self.hu0  = OpenCLArray2D(cl_ctx, nx+1, ny, 0, halo_y, hu0, asymHaloU)
        self.hv0  = OpenCLArray2D(cl_ctx, nx, ny+1, halo_x, 0, hv0, asymHaloV)
        
        self.h1   = OpenCLArray2D(cl_ctx, nx, ny, halo_x, halo_y, h0, asymHalo)
        self.hu1  = OpenCLArray2D(cl_ctx, nx+1, ny, 0, halo_y, hu0, asymHaloU)
        self.hv1  = OpenCLArray2D(cl_ctx, nx, ny+1, halo_x, 0, hv0, asymHaloV)
                   
        
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
A class representing asymmetric Arakawa C type grid (see above)
"""
class SWEDataAsymArakawaC:
    """
    Uploads initial data to the CL device
    """
    def __init__(self, cl_ctx, nx, ny, \
                 halo_north, halo_west, halo_south, halo_east, \
                 h0, hu0, hv0):
        #FIXME: This at least works for 0 and 1 ghost cells, but not convinced it generalizes
        assert(halo_x <= 1 and halo_y <= 1)

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
                 north=1, east=1, south=1, west=1):
        self.north = north
        self.east  = east
        self.south = south
        self.west  = west

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
            
            
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
