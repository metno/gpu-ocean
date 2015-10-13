#include "oclutils.h"
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "boost/format.hpp"

#undef NDEBUG
//#define NDEBUG

#define EXECFULL
#define EXECNOOP

using namespace std;

static void errorCallback(const char *errInfo, const void *privateInfo, size_t cb, void *userData)
{
    cout << "errorCallback:" << errInfo << endl;
}

/**
 * Creates two NxN input matrices A and B with random values.
 * @param size Input: The size of N
 * @param a Output: The A matrix with random values
 * @param b Output: The B matrix with random values
 */
static void createInputMatrices(size_t size, vector<float> &a, vector<float> &b)
{
    srand48(time(0));
    a.resize(size * size);
    b.resize(size * size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            a[j * size + i] = (lrand48() / float(RAND_MAX)) * 100;
            b[j * size + i] = (lrand48() / float(RAND_MAX)) * 100;
        }
    }
}

/**
 * Public function for multiplying two NxN matrices.
 * Throws std::runtime_error if something goes wrong.
 * @param size Input: The size of N
 * @param execOnCpu Input: Whether to execute the kernel on the CPU
 */
void matmul(size_t size, bool execOnCpu)
{
    if (size <= 0)
        throw runtime_error((boost::format("size (%1%) <= 0") % size).str());

    cl_int error = CL_SUCCESS;

    // --- BEGIN get device and create context ---------------------------
    // get platforms
    vector<cl::Platform> platforms;
    OpenCLUtils::getPlatforms(&platforms);
    if (platforms.empty())
        throw runtime_error("No OpenCL platform found");

    // get device
    vector<cl::Device> devices;
    const cl_device_type deviceType = execOnCpu ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU;
    cl_uint pfmIndex = 0;
    for (pfmIndex = 0; pfmIndex < platforms.size(); ++pfmIndex) {
        platforms[pfmIndex].getDevices(deviceType, &devices);
        if (!devices.empty())
            break; // found at last one relevant device on this platform
    }
    if (pfmIndex == platforms.size())
        throw runtime_error("No relevant OpenCL devices found on any platform");

    const int devIndex = 0; // for now, use first device
    const string deviceName = OpenCLUtils::getDeviceName(devices[devIndex]);
    const string platformName = OpenCLUtils::getPlatformName(platforms[pfmIndex]);

    // create context
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[pfmIndex])(),
        0
    };
    cl::Context context(devices, contextProperties, errorCallback, 0, &error);
    CL_CHECK(error);
    // --- END get device and create context ---------------------------


    // --- BEGIN initialize kernels
    vector<pair<string, string> > sources;
#ifdef EXECFULL
    sources.push_back(make_pair("MatMul", "matmul.cl"));
#endif
#ifdef EXECNOOP
    sources.push_back(make_pair("MatMulNoop", "matmul_noop.cl"));
#endif
    OpenCLUtils::initKernels(context, devices, sources, (boost::format("-D MATRIX_SIZE=%d") % size).str());
    // --- END initialize kernels


    // --- BEGIN set up kernel arguments ---------------------------

    // generate input matrices with random values
    vector<float> ha;
    vector<float> hb;
    createInputMatrices(size, ha, hb);

    // create buffers for input matrices
    cl::Buffer a = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size * size, ha.data(), &error);
    CL_CHECK(error);
    cl::Buffer b = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size * size, hb.data(), &error);
    CL_CHECK(error);

#ifdef EXECFULL
    // create buffer for output matrix
    cl::Buffer ab_full(context, CL_MEM_WRITE_ONLY, sizeof(float) * size * size, 0, &error);
    CL_CHECK(error);

    // set kernel args
    OpenCLUtils::getKernel("MatMul").setArg<cl::Buffer>(0, a);
    OpenCLUtils::getKernel("MatMul").setArg<cl::Buffer>(1, b);
    OpenCLUtils::getKernel("MatMul").setArg<cl::Buffer>(2, ab_full);
#endif
#ifdef EXECNOOP
    // create buffer for output matrix
    cl::Buffer ab_noop(context, CL_MEM_WRITE_ONLY, sizeof(float) * size * size, 0, &error);
    CL_CHECK(error);

    // set kernel args
    OpenCLUtils::getKernel("MatMulNoop").setArg<cl::Buffer>(0, a);
    OpenCLUtils::getKernel("MatMulNoop").setArg<cl::Buffer>(1, b);
    OpenCLUtils::getKernel("MatMulNoop").setArg<cl::Buffer>(2, ab_noop);
#endif
    // --- END set up kernel arguments ---------------------------

    // create command queue
    cl::CommandQueue queue(context, devices[devIndex], CL_QUEUE_PROFILING_ENABLE, &error);
    CL_CHECK(error);


    // --- BEGIN execute kernels ---------------------------
    cl::Event event;

#ifdef EXECFULL
    // execute full multiplication
    CL_CHECK(queue.enqueueNDRangeKernel(
                 OpenCLUtils::getKernel("MatMul"), cl::NullRange, cl::NDRange(size, size), cl::NullRange, 0, &event));
    CL_CHECK(event.wait());
    // examine contents of ab_full ... TBD
    const float msecs_full = OpenCLUtils::elapsedMilliseconds(event);
#else
    const float msecs_full = -1;
#endif

#ifdef EXECNOOP
    // execute noop (only copying memory between host and device)
    CL_CHECK(queue.enqueueNDRangeKernel(
                 OpenCLUtils::getKernel("MatMulNoop"), cl::NullRange, cl::NDRange(size, size), cl::NullRange, 0, &event));
    CL_CHECK(event.wait());
    // examine contents of ab_noop ... TBD
    const float msecs_noop = OpenCLUtils::elapsedMilliseconds(event);
#else
    const float msecs_noop = -1;
#endif
    // --- END execute kernels ---------------------------

#ifndef NDEBUG
    cout << (boost::format("matrix size: %4d x %4d;   msecs: %12.6f;   noop msecs: %12.6f   (platform: %s; device: %s)")
            % size % size % msecs_full % msecs_noop % platformName % deviceName) << endl;
#endif
}
