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

using namespace std;

static void errorCallback(const char *errInfo, const void *privateInfo, size_t cb, void *userData)
{
    cout << "errorCallback:" << errInfo << endl;
}

class MatMulExecutor
{
public:
    MatMulExecutor(size_t, const vector<float> &ha, const vector<float> &hb, cl_device_type, const string &);
    virtual ~MatMulExecutor() {}
    void initialize();
    void execute();
    float elapsedMilliseconds() const;
    string platformName() const;
    string deviceName() const;

private:
    cl_int createContext(vector<cl::Device> *);
    cl_int createInputBuffers();

    struct MatMulExecutorImpl;
    MatMulExecutorImpl *pimpl;
};

struct MatMulExecutor::MatMulExecutorImpl
{
    size_t size;
    cl_device_type deviceType;
    string kernelFileName;
    bool initialized;
    cl_ulong cmdStart;
    cl_ulong cmdEnd;
    string platformName;
    string deviceName;

    vector<float> ha;
    vector<float> hb;

    cl::Context context;
    cl::Program program;
    cl::Kernel kernel;
    cl::Buffer a;
    cl::Buffer b;
    cl::Buffer ab;
    cl::CommandQueue queue;
    cl::Event event;

    MatMulExecutorImpl(size_t, const vector<float> &, const vector<float> &, cl_device_type, const string &);
};

MatMulExecutor::MatMulExecutorImpl::MatMulExecutorImpl(
        size_t size, const vector<float> &ha, const vector<float> &hb, cl_device_type deviceType, const string &kernelFileName)
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
        size_t size, const vector<float> &ha, const vector<float> &hb, cl_device_type deviceType, const string &kernelFileName)
    : pimpl(new MatMulExecutorImpl(size, ha, hb, deviceType, kernelFileName))
{
}

/**
 * Initializes the executor. Throws std::runtime_error if the matrix size is null.
 */
void MatMulExecutor::initialize()
{
    if (pimpl->size <= 0)
        throw std::runtime_error((boost::format("size (%1%) <= 0") % pimpl->size).str());

    cl_int error;
    vector<cl::Device> devices;

    // create context
    CL_CHECK(createContext(&devices));

    // create random input matrices
    CL_CHECK(createInputBuffers());

    // create program from source
    pimpl->program = cl::Program(pimpl->context, OpenCLUtils::loadSingleKernel(pimpl->kernelFileName.c_str()), &error);
    CL_CHECK(error);

    // compile program
    error = pimpl->program.build(devices, (boost::format("-D MATRIX_SIZE=%d") % pimpl->size).str().c_str(), 0, 0);
    if (error == CL_BUILD_PROGRAM_FAILURE) {
        string log;
        log.resize(2048);
        cerr << "build program failure detected:\n";
        for (unsigned int i = 0; i < devices.size(); ++i) {
            cerr << "============ build log for device " << i << ": ============\n";
            cl_int error2;
            std::string log = pimpl->program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[i], &error2);
            CL_CHECK(error2);
            cerr << log << endl;
        }
    }
    CL_CHECK(error);

    // create kernel
    pimpl->kernel = cl::Kernel(pimpl->program, "MatMul", &error);
    CL_CHECK(error);

    // create buffer for output matrix
    pimpl->ab = cl::Buffer(pimpl->context, CL_MEM_WRITE_ONLY, sizeof(float) * pimpl->size * pimpl->size, 0, &error);
    CL_CHECK(error);

    pimpl->kernel.setArg<cl::Buffer>(0, pimpl->a);
    pimpl->kernel.setArg<cl::Buffer>(1, pimpl->b);
    pimpl->kernel.setArg<cl::Buffer>(2, pimpl->ab);

    // create command queue
    pimpl->queue = cl::CommandQueue(pimpl->context, devices[0], CL_QUEUE_PROFILING_ENABLE, &error); // for now, use first device
    CL_CHECK(error);

    pimpl->initialized = true;
}

/**
 * Executes the OpenCL kernel.
 */
void MatMulExecutor::execute()
{
    CL_CHECK(pimpl->queue.enqueueNDRangeKernel(
                 pimpl->kernel, cl::NullRange, cl::NDRange(pimpl->size, pimpl->size), cl::NullRange, 0, &pimpl->event));

    CL_CHECK(pimpl->event.wait());

    pimpl->cmdStart = pimpl->event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    pimpl->cmdEnd = pimpl->event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    // copy result back to host ... TBD
}

/**
 * Returns the execution time.
 * @return Elapsed execution time in milliseconds
 */
float MatMulExecutor::elapsedMilliseconds() const
{
    return (pimpl->cmdEnd - pimpl->cmdStart) * .000001; // note: cmdEnd and cmdStart are both in nanoseconds
}

/**
 * Returns the platform on which the kernel will be executed.
 * @return Platform name
 */
string MatMulExecutor::platformName() const
{
    return pimpl->platformName;
}

/**
 * Returns the device on which the kernel will be executed.
 * @return Device name
 */
string MatMulExecutor::deviceName() const
{
    return pimpl->deviceName;
}

/**
 * Creates the OpenCL context. Throws std::runtime_error if no relevant device is found.
 * @param devices Output: Device objects matching this->pimpl->deviceType
 * @return OpenCL error code
 */
cl_int MatMulExecutor::createContext(vector<cl::Device> *devices)
{
    vector<cl::Platform> platforms;
    OpenCLUtils::getPlatforms(&platforms);

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

    cl_uint pfIndex = 0;

    // loop over platforms
    for (pfIndex = 0; pfIndex < platforms.size(); ++pfIndex) {
        platforms[pfIndex].getDevices(pimpl->deviceType, devices);
        if (!devices->empty())
            break; // found at last one relevant device on this platform
    }
    if (pfIndex == platforms.size()) {
        throw runtime_error("No relevant OpenCL devices found on any platform");
    } else {
#ifndef NDEBUG
        cout << "Found " << devices->size() << " device(s)" << endl;
#endif
    }

    pimpl->platformName = OpenCLUtils::getPlatformName(platforms[pfIndex]);

#ifndef NDEBUG
    for (cl_uint i = 0; i < devices.size(); ++i) {
        cout << "\t (" << (i + 1) << ") : " << OpenCLUtils::getDeviceName(devices[i]) << endl;
    }
#endif

    pimpl->deviceName = OpenCLUtils::getDeviceName(devices->front()); // for now, use first device

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[pfIndex])(),
        0
    };
    cl_int error = CL_SUCCESS;

    pimpl->context = cl::Context(*devices, contextProperties, errorCallback, 0, &error);

    return error;
}

/**
 * Creates OpenCL buffer objects, one for each of the two input matrices A and B.
 * @return OpenCL error code
 */
cl_int MatMulExecutor::createInputBuffers()
{
    cl_int error = CL_SUCCESS;

    pimpl->a = cl::Buffer(
                pimpl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * pimpl->size * pimpl->size,
                pimpl->ha.data(), &error);
    if (error != CL_SUCCESS)
        return error;

    pimpl->b = cl::Buffer(
                pimpl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * pimpl->size * pimpl->size,
                pimpl->hb.data(), &error);
    if (error != CL_SUCCESS)
        return error;

    return CL_SUCCESS;
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
 * Multiplies two NxN matrices A and B.
 * @param size Input: The size of N
 * @param a Input: The A matrix
 * @param b Input: The B matrix
 * @param deviceType Input: The device type: CL_DEVICE_TYPE_CPU or CL_DEVICE_TYPE_GPU
 * @param kernelFileName Input: The name of the kernel file (note: the value of the KERNELDIR environment variable will be prepended)
 * @param msecs Output: The elapsed execution time in milliseconds
 * @param platformName Output: If non-null, the name of the platform used for the execution
 * @param deviceName Output: If non-null, the name of the device used for the execution
 */
static void matmulExec(
        size_t size, const vector<float> &a, const vector<float> &b, cl_device_type deviceType, const string &kernelFileName,
        float &msecs, string *platformName = 0, string *deviceName = 0)
{
    MatMulExecutor executor(size, a, b, deviceType, kernelFileName);
    executor.initialize();
    executor.execute();
    msecs = executor.elapsedMilliseconds();
    if (platformName)
        *platformName = executor.platformName();
    if (deviceName)
        *deviceName = executor.deviceName();
}

/**
 * Public function for multiplying two NxN matrices. Throws std::runtime_error if something goes wrong.
 * @param size: The size of N
 * @param execOnCpu Input: Whether to execute the kernel on the CPU
 */
void matmul(size_t size, bool execOnCpu)
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
    matmulExec(size, a, b, deviceType, "matmul.cl", msecs_full, &platformName, &deviceName);

    // execute noop (only copying memory between host and device)
    float msecs_noop = -1.0;
    matmulExec(size, a, b, deviceType, "matmul_noop.cl", msecs_noop);

#ifndef NDEBUG
    cout << (boost::format("matrix size: %4d x %4d;   msecs: %12.6f;   noop msecs: %12.6f   (platform: %s; device: %s)")
            % size % size % msecs_full % msecs_noop % platformName % deviceName) << endl;
#endif
}
