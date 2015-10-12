#include "oclutils.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "boost/format.hpp"

#undef NDEBUG
#define NDEBUG

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
 * Returns the execution time from an event.
 * @param event Input: The event object
 * @return Elapsed execution time in milliseconds
 */
static float elapsedMilliseconds(const cl::Event &event)
{
    return (event.getProfilingInfo<CL_PROFILING_COMMAND_END>()
            - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * .000001; // note: _START and _END values are both in nanoseconds
}

/**
 * Prints the build log of a program on stderr.
 * @param program Input: The program object
 * @param devices Input: The device objects
 */
static void printProgramBuildLog(const cl::Program &program, vector<cl::Device> &devices)
{
    string log;
    log.resize(2048);
    cerr << "build program failure detected:\n";
    for (int i = 0; i < devices.size(); ++i) {
        cerr << "============ build log for device " << i << ": ============\n";
        cl_int error;
        const string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[i], &error);
        CL_CHECK(error);
        cerr << log << endl;
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


    // --- BEGIN set up input structures ---------------------------

    // generate matrices with random values
    vector<float> ha;
    vector<float> hb;
    createInputMatrices(size, ha, hb);

    // create buffers
    cl::Buffer a = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size * size, ha.data(), &error);
    CL_CHECK(error);
    cl::Buffer b = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size * size, hb.data(), &error);
    CL_CHECK(error);

    // --- END set up input structures ---------------------------


    // --- BEGIN set up kernels and output structures ---------------------------

#ifdef EXECFULL
    // create program 1 (full multiplication) from source
    cl::Program program_full(context, OpenCLUtils::loadSingleKernel("matmul.cl"), &error);
    CL_CHECK(error);

    // compile program 1
    error = program_full.build(devices, (boost::format("-D MATRIX_SIZE=%d") % size).str().c_str(), 0, 0);
    if (error == CL_BUILD_PROGRAM_FAILURE)
        printProgramBuildLog(program_full, devices);
    CL_CHECK(error);

    // create kernel 1
    cl::Kernel kernel_full(program_full, "MatMul", &error);
    CL_CHECK(error);

    // create buffer for output matrix
    cl::Buffer ab_full(context, CL_MEM_WRITE_ONLY, sizeof(float) * size * size, 0, &error);
    CL_CHECK(error);

    // set kernel args
    kernel_full.setArg<cl::Buffer>(0, a);
    kernel_full.setArg<cl::Buffer>(1, b);
    kernel_full.setArg<cl::Buffer>(2, ab_full);
#endif

    // -------------------------------------------------------------------------------

#ifdef EXECNOOP
    // create program 2 (only copying memory between host and device) from source
    cl::Program program_noop(context, OpenCLUtils::loadSingleKernel("matmul_noop.cl"), &error);
    CL_CHECK(error);

    // compile program 2
    error = program_noop.build(devices, (boost::format("-D MATRIX_SIZE=%d") % size).str().c_str(), 0, 0);
    if (error == CL_BUILD_PROGRAM_FAILURE)
        printProgramBuildLog(program_noop, devices);
    CL_CHECK(error);

    // create kernel 2
    cl::Kernel kernel_noop(program_noop, "MatMulNoop", &error);
    CL_CHECK(error);

    // create buffer for output matrix
    cl::Buffer ab_noop(context, CL_MEM_WRITE_ONLY, sizeof(float) * size * size, 0, &error);
    CL_CHECK(error);

    // set kernel args
    kernel_noop.setArg<cl::Buffer>(0, a);
    kernel_noop.setArg<cl::Buffer>(1, b);
    kernel_noop.setArg<cl::Buffer>(2, ab_noop);
#endif

    // --- END set up kernels and output structures ---------------------------


    // --- BEGIN create command queue ---------------------------

    cl::CommandQueue queue(context, devices[devIndex], CL_QUEUE_PROFILING_ENABLE, &error);
    CL_CHECK(error);

    // --- END create command queue ---------------------------


    // --- BEGIN execute kernels ---------------------------
    cl::Event event;

#ifdef EXECFULL
    // execute full multiplication
    CL_CHECK(queue.enqueueNDRangeKernel(kernel_full, cl::NullRange, cl::NDRange(size, size), cl::NullRange, 0, &event));
    CL_CHECK(event.wait());
    // examine contents of ab_full ... TBD
    const float msecs_full = elapsedMilliseconds(event);
#else
    const float msecs_full = -1;
#endif

#ifdef EXECNOOP
    // execute noop (only copying memory between host and device)
    CL_CHECK(queue.enqueueNDRangeKernel(kernel_noop, cl::NullRange, cl::NDRange(size, size), cl::NullRange, 0, &event));
    CL_CHECK(event.wait());
    // examine contents of ab_noop ... TBD
    const float msecs_noop = elapsedMilliseconds(event);
#else
    const float msecs_noop = -1;
#endif
    // --- END execute kernels ---------------------------

#ifndef NDEBUG
    cout << (boost::format("matrix size: %4d x %4d;   msecs: %12.6f;   noop msecs: %12.6f   (platform: %s; device: %s)")
            % size % size % msecs_full % msecs_noop % platformName % deviceName) << endl;
#endif
}
