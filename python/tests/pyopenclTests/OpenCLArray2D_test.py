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
        self.dbuf1 = np.zeros(self.dataShape)
        self.buf3 = np.zeros(self.dataShape, dtype=np.float32, order='C')
        self.dbuf3= np.zeros(self.dataShape)
        for j in range(self.dataShape[0]):
            for i in range(self.dataShape[1]):
                self.buf1[j,i] = i*100 + j
                self.dbuf1[j,i] = self.buf1[j,i]
                self.buf3[j,i] = j*1000 - i
                self.dbuf3[j,i] = self.buf3[j,i]
                
        self.explicit_free = False
        
        self.device_name = self.cl_ctx.devices[0].name
        self.cl_queue = pyopencl.CommandQueue(self.cl_ctx)
        self.tests_failed = True

        self.clarray = Common.OpenCLArray2D(self.cl_ctx, \
                                            self.nx, self.ny, \
                                            self.nx_halo, self.ny_halo, \
                                            self.buf1)

        self.double_clarray = None

        
    def tearDown(self):
        if self.tests_failed:
            print("Device name: " + self.device_name)
        if not self.explicit_free:
            self.clarray.release()
        if self.double_clarray is not None:
            self.double_clarray.release()

    ### Utils ###
    def init_double(self):
        self.double_clarray = Common.OpenCLArray2D(self.cl_ctx, \
                                                   self.nx, self.ny, \
                                                   self.nx_halo, self.ny_halo, \
                                                   self.dbuf1, \
                                                   double_precision=True)
            
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
        #self.explicit_free = True
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

    def test_upload(self):
        self.clarray.upload(self.cl_queue, self.buf3)
        host_data = self.clarray.download(self.cl_queue)
        self.assertEqual(host_data.tolist(), self.buf3.tolist())
        self.tests_failed = False

    def test_copy_buffer(self):
        clarray2 = Common.OpenCLArray2D(self.cl_ctx, \
                                        self.nx, self.ny, self.nx_halo, self.ny_halo, \
                                        self.buf3)

        host_data_pre_copy = self.clarray.download(self.cl_queue)
        self.assertEqual(host_data_pre_copy.tolist(), self.buf1.tolist())
        
        self.clarray.copyBuffer(self.cl_queue, clarray2)
        host_data_post_copy = self.clarray.download(self.cl_queue)
        self.assertEqual(host_data_post_copy.tolist(), self.buf3.tolist())
        
        self.tests_failed = False
        
    # Double precision
    def test_double_init(self):
        self.init_double()

        self.assertEqual(self.double_clarray.nx,  self.nx)
        self.assertEqual(self.double_clarray.ny, self.ny)
        self.assertEqual(self.double_clarray.nx_halo, self.nx + 2*self.nx_halo)
        self.assertEqual(self.double_clarray.ny_halo, self.ny + 2*self.ny_halo)
        
        self.assertTrue(self.double_clarray.holds_data)
        self.assertEqual(self.double_clarray.bytes_per_float, 8)
        self.assertEqual(self.double_clarray.pitch, 8*(self.nx + 2*self.nx_halo))
        self.tests_failed = False

    def test_double_release(self):
        self.init_double()
        
        self.double_clarray.release()
        self.assertFalse(self.double_clarray.holds_data)

        with self.assertRaises(RuntimeError):
            self.double_clarray.download(self.cl_queue)

        with self.assertRaises(RuntimeError):
            self.double_clarray.upload(self.cl_queue, self.dbuf3)
        
        self.tests_failed = False
    

    def test_double_download(self):
        self.init_double()
        
        host_data = self.double_clarray.download(self.cl_queue)
        self.assertEqual(host_data.tolist(), self.dbuf1.tolist())
        self.tests_failed = False

    def test_double_upload(self):
        self.init_double()

        self.double_clarray.upload(self.cl_queue, self.dbuf3)
        host_data = self.double_clarray.download(self.cl_queue)
        self.assertEqual(host_data.tolist(), self.dbuf3.tolist())
        self.tests_failed = False

    def test_double_copy_buffer(self):
        self.init_double()
        
        double_clarray2 = Common.OpenCLArray2D(self.cl_ctx, \
                                               self.nx, self.ny, \
                                               self.nx_halo, self.ny_halo, \
                                               self.dbuf3, \
                                               double_precision=True)

        host_data_pre_copy = self.double_clarray.download(self.cl_queue)
        self.assertEqual(host_data_pre_copy.tolist(), self.dbuf1.tolist())
        
        self.double_clarray.copyBuffer(self.cl_queue, double_clarray2)
        host_data_post_copy = self.double_clarray.download(self.cl_queue)
        self.assertEqual(host_data_post_copy.tolist(), self.dbuf3.tolist())
        
        self.tests_failed = False

    def test_cross_precision_copy_buffer(self):
        self.init_double()
        
        single_clarray2 = Common.OpenCLArray2D(self.cl_ctx, \
                                               self.nx, self.ny, \
                                               self.nx_halo, self.ny_halo, \
                                               self.buf3)

        host_data_pre_copy = self.double_clarray.download(self.cl_queue)
        self.assertEqual(host_data_pre_copy.tolist(), self.dbuf1.tolist())
        
        with self.assertRaises(AssertionError):
            self.double_clarray.copyBuffer(self.cl_queue, single_clarray2)
        
        self.tests_failed = False

    

        
