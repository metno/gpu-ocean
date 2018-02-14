import unittest
import time
import numpy as np
import sys
import gc
import pyopencl

from testUtils import *

sys.path.insert(0, '../')
from SWESimulators import Common

#reload(GlobalParticles)

class OpenCLArray2DTest(unittest.TestCase):

    def setUp(self):

        #Set which CL device to use, and disable kernel caching
        self.cl_ctx = make_cl_ctx()
                    
        # Make some host data which we can play with
        self.nx = 3
        self.ny = 5
        self.nx_halo = 1
        self.ny_halo = 2
        self.dataShape = (self.ny + 2*self.ny_halo, self.nx + 2*self.nx_halo)
        
        self.buf1 = np.zeros(self.dataShape, dtype=np.float32, order='C')
        self.buf2 = np.zeros(self.dataShape)
        self.buf3 = np.zeros(self.dataShape, dtype=np.float32, order='C')
        for j in range(self.dataShape[0]):
            for i in range(self.dataShape[1]):
                self.buf1[j,i] = i*100 + j
                self.buf2[j,i] = self.buf1[j,i]
                self.buf3[j,i] = j*1000 - i
                
        self.explicit_free = False
        
        self.device_name = self.cl_ctx.devices[0].name
        self.cl_queue = pyopencl.CommandQueue(self.cl_ctx)
        self.tests_failed = True

        self.clarray = Common.OpenCLArray2D(self.cl_ctx, \
                                            self.nx, self.ny, self.nx_halo, self.ny_halo, \
                                            self.buf1)
        
        
    def tearDown(self):
        if self.tests_failed:
            print "\nDevice name: " + self.device_name
        if not self.explicit_free:
            self.clarray.release()
                                            

    ### START TESTS ###

    def test_init(self):
        self.assertEqual(self.clarray.nx,  self.nx)
        self.assertEqual(self.clarray.ny, self.ny)
        self.assertEqual(self.clarray.nx_halo, self.nx + 2*self.nx_halo)
        self.assertEqual(self.clarray.ny_halo, self.ny + 2*self.ny_halo)
        
        self.assertTrue(self.clarray.holds_data)
        self.assertEqual(self.clarray.bytes_per_float, 4)
        self.assertEqual(self.clarray.pitch, 4*(self.nx + 2*self.nx_halo))
        self.tests_failed = False

    def test_release(self):
        self.explicit_free = True
        self.clarray.release()
        self.assertFalse(self.clarray.holds_data)

        with self.assertRaises(RuntimeError):
            self.clarray.download(self.cl_queue)

        with self.assertRaises(RuntimeError):
            self.clarray.upload(self.cl_queue, self.buf3)
        
        self.tests_failed = False

    def test_download(self):
        
        host_data = self.clarray.download(self.cl_queue)
        self.assertEqual(host_data.tolist(), self.buf1.tolist())
        self.tests_failed = False

    def tests_upload(self):
        self.clarray.upload(self.cl_queue, self.buf3)
        host_data = self.clarray.download(self.cl_queue)
        self.assertEqual(host_data.tolist(), self.buf3.tolist())
        self.tests_failed = False
