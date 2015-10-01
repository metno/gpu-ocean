#include "oclutils.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "boost/format.hpp"

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
    int size_;
    cl_device_type deviceType_;
    string kernelFileName_;
    bool initialized_;
    cl_ulong cmdStart_;
    cl_ulong cmdEnd_;
    string platformName_;
    string deviceName_;

    vector<float> ha_;
    vector<float> hb_;

    cl_context context_;
    cl_program program_;
    cl_kernel kernel_;
    cl_mem a_;
    cl_mem b_;
    cl_mem ab_;
    cl_command_queue queue_;
    cl_event event_;

    void releaseObjects() const;
    cl_int createContext(vector<cl_device_id> *);
    cl_int createInputBuffers();
};

MatMulExecutor::MatMulExecutor(
        int size, const vector<float> &ha, const vector<float> &hb, cl_device_type deviceType, const string &kernelFileName)
    : size_(size)
    , ha_(ha)
    , hb_(hb)
    , deviceType_(deviceType)
    , kernelFileName_(kernelFileName)
    , initialized_(false)
    , cmdStart_(0)
    , cmdEnd_(0)
{
}

MatMulExecutor::~MatMulExecutor()
{
    if (initialized_)
        releaseObjects();
}

bool MatMulExecutor::initialize()
{
    if (size_ <= 0)
        throw std::runtime_error((boost::format("size_ (%1%) <= 0") % size_).str());

    cl_int error;
    vector<cl_device_id> deviceIds;

    // create context
    CL_CHECK(createContext(&deviceIds));

    // create random input matrices
    CL_CHECK(createInputBuffers());

    // create program from source
    program_ = OpenCLUtils::createProgram(context_, OpenCLUtils::loadKernel(kernelFileName_.c_str()));

    // compile program
    error = clBuildProgram(program_, deviceIds.size(), deviceIds.data(), (boost::format("-D MATRIX_SIZE=%d") % size_).str().c_str(), 0, 0);
    if (error == CL_BUILD_PROGRAM_FAILURE) {
        string log;
        log.resize(2048);
        cerr << "build program failure detected:\n";
        for (unsigned int i = 0; i < deviceIds.size(); ++i) {
            cerr << "============ build log for device " << i << ": ============\n";
            cl_int error2 = clGetProgramBuildInfo(
                        program_, deviceIds[i], CL_PROGRAM_BUILD_LOG, log.size(),
                        const_cast<char *>(log.data()), 0);
            CL_CHECK(error2);
            cerr << log << endl;
        }
    }
    CL_CHECK(error);

    // create kernel
    kernel_ = clCreateKernel(program_, "MatMul", &error);
    CL_CHECK(error);

    // create buffer for output matrix
    ab_ = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, sizeof(float) * size_ * size_, 0, &error);
    CL_CHECK(error);

    clSetKernelArg(kernel_, 0, sizeof(cl_mem), &a_);
    clSetKernelArg(kernel_, 1, sizeof(cl_mem), &b_);
    clSetKernelArg(kernel_, 2, sizeof(cl_mem), &ab_);

    // create command queue
    queue_ = clCreateCommandQueue(context_, deviceIds[0], CL_QUEUE_PROFILING_ENABLE, &error); // for now, use first device
    CL_CHECK(error);

    initialized_ = true;
    return true;
}

void MatMulExecutor::execute()
{
    size_t offset[3] = { 0 };
    size_t size[3] = { size_, size_, 1 };

    CL_CHECK(clEnqueueNDRangeKernel(queue_, kernel_, 2, offset, size, 0, 0, 0, &event_));

    CL_CHECK(clWaitForEvents(1, &event_));

    CL_CHECK(clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_START, sizeof(cmdStart_), &cmdStart_, 0));
    CL_CHECK(clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_END, sizeof(cmdEnd_), &cmdEnd_, 0));

    // copy result back to host ... TBD
}

void MatMulExecutor::releaseObjects() const
{
    if (!initialized_)
        return;

    CL_CHECK(clReleaseEvent(event_));
    CL_CHECK(clReleaseCommandQueue(queue_));
    CL_CHECK(clReleaseMemObject(ab_));
    CL_CHECK(clReleaseKernel(kernel_));
    CL_CHECK(clReleaseProgram(program_));
    CL_CHECK(clReleaseMemObject(b_));
    CL_CHECK(clReleaseMemObject(a_));
    CL_CHECK(clReleaseContext(context_));
}

float MatMulExecutor::elapsedMilliseconds() const
{
    return (cmdEnd_ - cmdStart_) * .000001; // note: cmdEnd and cmdStart are both in nanoseconds
}

string MatMulExecutor::platformName() const
{
    return platformName_;
}

string MatMulExecutor::deviceName() const
{
    return deviceName_;
}

cl_int MatMulExecutor::createContext(vector<cl_device_id> *deviceIds)
{
    vector<cl_platform_id> platforms;
    OpenCLUtils::getPlatforms(platforms);

    if (platforms.empty()) {
        throw runtime_error("No OpenCL platform found");
    } else {
        cout << "Found " << platforms.size() << " platform(s)" << endl;
    }

    for (cl_uint i = 0; i < platforms.size(); ++i) {
        cout << "\t (" << (i + 1) << ") : " << OpenCLUtils::getPlatformName(platforms[i]) << endl;
    }

    cl_uint deviceIdCount = 0;
    cl_uint pfIndex = 0;

    for (pfIndex = 0; pfIndex < platforms.size(); ++pfIndex) {
        clGetDeviceIDs(platforms[pfIndex], deviceType_, 0, 0, &deviceIdCount);
        if (deviceIdCount > 0)
            break; // found device for this platform
    }
    if (pfIndex == platforms.size()) {
        throw runtime_error("No OpenCL devices found for any platform");
    } else {
        cout << "Found " << deviceIdCount << " device(s)" << endl;
    }

    platformName_ = OpenCLUtils::getPlatformName(platforms[pfIndex]);

    deviceIds->resize(deviceIdCount);
    clGetDeviceIDs(platforms[pfIndex], deviceType_, deviceIds->size(), deviceIds->data(), 0);

    for (cl_uint i = 0; i < deviceIds->size(); ++i) {
        cout << "\t (" << (i + 1) << ") : " << OpenCLUtils::getDeviceName(deviceIds->at(i)) << endl;
    }

    deviceName_ = OpenCLUtils::getDeviceName(deviceIds->front()); // for now, use first device

    const cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[pfIndex]),
        0, 0
    };
    cl_int error = CL_SUCCESS;
    context_ = clCreateContext(contextProperties, deviceIds->size(), deviceIds->data(), errorCallback, 0, &error);
    return error;
}

cl_int MatMulExecutor::createInputBuffers()
{
    cl_int error = CL_SUCCESS;

    a_ = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size_ * size_, ha_.data(), &error);
    if (error != CL_SUCCESS)
        return error;

    b_ = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * size_ * size_, hb_.data(), &error);
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

    cout << (boost::format("matrix size: %4d x %4d;   msecs: %12.6f;   noop msecs: %12.6f   (platform: %s; device: %s)")
            % size % size % msecs_full % msecs_noop % platformName % deviceName) << endl;

    return true;
}
