#include "oclutils.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "boost/format.hpp"

#define NDEBUG

using namespace std;

static void errorCallback(const char *errInfo, const void *privateInfo, size_t cb, void *userData)
{
    cout << "errorCallback:" << errInfo << endl;
}

class MatMulExecutor
{
public:
    MatMulExecutor(int, const vector<float> &ha, const vector<float> &hb, cl_device_type, const string &);
    virtual ~MatMulExecutor();
    bool initialize();
    void execute();
    float elapsedMilliseconds() const;
    string platformName() const;
    string deviceName() const;

private:
    void releaseObjects() const;
    cl_int createContext(vector<cl_device_id> *);
    cl_int createInputBuffers();

    struct MatMulExecutorImpl;
    MatMulExecutorImpl *pimpl;
};

struct MatMulExecutor::MatMulExecutorImpl
{
    int size;
    cl_device_type deviceType;
    string kernelFileName;
    bool initialized;
    cl_ulong cmdStart;
    cl_ulong cmdEnd;
    string platformName;
    string deviceName;

    vector<float> ha;
    vector<float> hb;

    cl_context context;
    cl_program program;
    cl_kernel kernel;
    cl_mem a;
    cl_mem b;
    cl_mem ab;
    cl_command_queue queue;
    cl_event event;

    MatMulExecutorImpl(int, const vector<float> &, const vector<float> &, cl_device_type, const string &);
};

MatMulExecutor::MatMulExecutorImpl::MatMulExecutorImpl(
        int size, const vector<float> &ha, const vector<float> &hb, cl_device_type deviceType, const string &kernelFileName)
    : size(size)
    , ha(ha)
    , hb(hb)
    , deviceType(deviceType)
    , kernelFileName(kernelFileName)
    , initialized(false)
    , cmdStart(0)
    , cmdEnd(0)
{
}

MatMulExecutor::MatMulExecutor(
        int size, const vector<float> &ha, const vector<float> &hb, cl_device_type deviceType, const string &kernelFileName)
    : pimpl(new MatMulExecutorImpl(size, ha, hb, deviceType, kernelFileName))
{
}

MatMulExecutor::~MatMulExecutor()
{
    if (pimpl->initialized)
        releaseObjects();
}

bool MatMulExecutor::initialize()
{
    if (pimpl->size <= 0)
        throw std::runtime_error((boost::format("size (%1%) <= 0") % pimpl->size).str());

    cl_int error;
    vector<cl_device_id> deviceIds;

    // create context
    CL_CHECK(createContext(&deviceIds));

    // create random input matrices
    CL_CHECK(createInputBuffers());

    // create program from source
    pimpl->program = OpenCLUtils::createProgram(pimpl->context, OpenCLUtils::loadKernel(pimpl->kernelFileName.c_str()));

    // compile program
    error = clBuildProgram(
                pimpl->program, deviceIds.size(), deviceIds.data(), (boost::format("-D MATRIX_SIZE=%d") % pimpl->size).str().c_str(), 0, 0);
    if (error == CL_BUILD_PROGRAM_FAILURE) {
        string log;
        log.resize(2048);
        cerr << "build program failure detected:\n";
        for (unsigned int i = 0; i < deviceIds.size(); ++i) {
            cerr << "============ build log for device " << i << ": ============\n";
            cl_int error2 = clGetProgramBuildInfo(
                        pimpl->program, deviceIds[i], CL_PROGRAM_BUILD_LOG, log.size(),
                        const_cast<char *>(log.data()), 0);
            CL_CHECK(error2);
            cerr << log << endl;
        }
    }
    CL_CHECK(error);

    // create kernel
    pimpl->kernel = clCreateKernel(pimpl->program, "MatMul", &error);
    CL_CHECK(error);

    // create buffer for output matrix
    pimpl->ab = clCreateBuffer(pimpl->context, CL_MEM_WRITE_ONLY, sizeof(float) * pimpl->size * pimpl->size, 0, &error);
    CL_CHECK(error);

    clSetKernelArg(pimpl->kernel, 0, sizeof(cl_mem), &pimpl->a);
    clSetKernelArg(pimpl->kernel, 1, sizeof(cl_mem), &pimpl->b);
    clSetKernelArg(pimpl->kernel, 2, sizeof(cl_mem), &pimpl->ab);

    // create command queue
    pimpl->queue = clCreateCommandQueue(pimpl->context, deviceIds[0], CL_QUEUE_PROFILING_ENABLE, &error); // for now, use first device
    CL_CHECK(error);

    pimpl->initialized = true;
    return true;
}

void MatMulExecutor::execute()
{
    size_t offset[3] = { 0 };
    size_t size[3] = { pimpl->size, pimpl->size, 1 };

    CL_CHECK(clEnqueueNDRangeKernel(pimpl->queue, pimpl->kernel, 2, offset, size, 0, 0, 0, &pimpl->event));

    CL_CHECK(clWaitForEvents(1, &pimpl->event));

    CL_CHECK(clGetEventProfilingInfo(pimpl->event, CL_PROFILING_COMMAND_START, sizeof(pimpl->cmdStart), &pimpl->cmdStart, 0));
    CL_CHECK(clGetEventProfilingInfo(pimpl->event, CL_PROFILING_COMMAND_END, sizeof(pimpl->cmdEnd), &pimpl->cmdEnd, 0));

    // copy result back to host ... TBD
}

void MatMulExecutor::releaseObjects() const
{
    if (!pimpl->initialized)
        return;

    CL_CHECK(clReleaseEvent(pimpl->event));
    CL_CHECK(clReleaseCommandQueue(pimpl->queue));
    CL_CHECK(clReleaseMemObject(pimpl->ab));
    CL_CHECK(clReleaseKernel(pimpl->kernel));
    CL_CHECK(clReleaseProgram(pimpl->program));
    CL_CHECK(clReleaseMemObject(pimpl->b));
    CL_CHECK(clReleaseMemObject(pimpl->a));
    CL_CHECK(clReleaseContext(pimpl->context));
}

float MatMulExecutor::elapsedMilliseconds() const
{
    return (pimpl->cmdEnd - pimpl->cmdStart) * .000001; // note: cmdEnd and cmdStart are both in nanoseconds
}

string MatMulExecutor::platformName() const
{
    return pimpl->platformName;
}

string MatMulExecutor::deviceName() const
{
    return pimpl->deviceName;
}

cl_int MatMulExecutor::createContext(vector<cl_device_id> *deviceIds)
{
    vector<cl_platform_id> platforms;
    OpenCLUtils::getPlatforms(platforms);

    if (platforms.empty()) {
        throw runtime_error("No OpenCL platform found");
    } else {
#ifndef NDEBUG
        cout << "Found " << platforms.size() << " platform(s)" << endl;
#endif
    }

#ifndef NDEBUG
    for (cl_uint i = 0; i < platforms.size(); ++i) {
        cout << "\t (" << (i + 1) << ") : " << OpenCLUtils::getPlatformName(platforms[i]) << endl;
    }
#endif

    cl_uint deviceIdCount = 0;
    cl_uint pfIndex = 0;

    for (pfIndex = 0; pfIndex < platforms.size(); ++pfIndex) {
        clGetDeviceIDs(platforms[pfIndex], pimpl->deviceType, 0, 0, &deviceIdCount);
        if (deviceIdCount > 0)
            break; // found device for this platform
    }
    if (pfIndex == platforms.size()) {
        throw runtime_error("No OpenCL devices found for any platform");
    } else {
#ifndef NDEBUG
        cout << "Found " << deviceIdCount << " device(s)" << endl;
#endif
    }

    pimpl->platformName = OpenCLUtils::getPlatformName(platforms[pfIndex]);

    deviceIds->resize(deviceIdCount);
    clGetDeviceIDs(platforms[pfIndex], pimpl->deviceType, deviceIds->size(), deviceIds->data(), 0);

#ifndef NDEBUG
    for (cl_uint i = 0; i < deviceIds->size(); ++i) {
        cout << "\t (" << (i + 1) << ") : " << OpenCLUtils::getDeviceName(deviceIds->at(i)) << endl;
    }
#endif

    pimpl->deviceName = OpenCLUtils::getDeviceName(deviceIds->front()); // for now, use first device

    const cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[pfIndex]),
        0, 0
    };
    cl_int error = CL_SUCCESS;
    pimpl->context = clCreateContext(contextProperties, deviceIds->size(), deviceIds->data(), errorCallback, 0, &error);
    return error;
}

cl_int MatMulExecutor::createInputBuffers()
{
    cl_int error = CL_SUCCESS;

    pimpl->a = clCreateBuffer(
                pimpl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * pimpl->size * pimpl->size, pimpl->ha.data(),
                &error);
    if (error != CL_SUCCESS)
        return error;

    pimpl->b = clCreateBuffer(
                pimpl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * pimpl->size * pimpl->size, pimpl->hb.data(),
                &error);
    if (error != CL_SUCCESS)
        return error;

    return CL_SUCCESS;
}

static void createInputMatrices(int size, vector<float> &a, vector<float> &b)
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

static bool execKernel(
        int size, const vector<float> &a, const vector<float> &b, cl_device_type deviceType, const string &kernelFileName,
        float &msecs, string *platformName = 0, string *deviceName = 0)
{
    MatMulExecutor matMulExec(size, a, b, deviceType, kernelFileName);
    if (!matMulExec.initialize())
        return false;

    matMulExec.execute();

    msecs = matMulExec.elapsedMilliseconds();
    if (platformName)
        *platformName = matMulExec.platformName();
    if (deviceName)
        *deviceName = matMulExec.deviceName();

    return true;
}

/**
 * Multiplies two NxN matrices.
 * @param size: The size of N
 * @param execOnCpu Input: Whether to execute the kernel on the CPU
 * @return Success status
 */
bool matmul(int size, bool execOnCpu)
{
    // generate random input
    vector<float> a;
    vector<float> b;
    createInputMatrices(size, a, b);

    const cl_device_type deviceType = execOnCpu ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU;

    string platformName;
    string deviceName;

    // execute full multiplication
    float msecs_full = -1.0;
    if (!execKernel(size, a, b, deviceType, "matmul.cl", msecs_full, &platformName, &deviceName))
        return false;

    // execute noop (only copying memory between host and device)
    float msecs_noop = -1.0;
    if (!execKernel(size, a, b, deviceType, "matmul_noop.cl", msecs_noop))
        return false;

#ifndef NDEBUG
    cout << (boost::format("matrix size: %4d x %4d;   msecs: %12.6f;   noop msecs: %12.6f   (platform: %s; device: %s)")
            % size % size % msecs_full % msecs_noop % platformName % deviceName) << endl;
#endif

    return true;
}
