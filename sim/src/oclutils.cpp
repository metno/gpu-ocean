#include "oclutils.h"
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <fstream>
#include <stdexcept>

#define NDEBUG

using namespace std;

/**
 * Finds all available OpenCL platforms on the current node.
 * @param clPlatformIDs Output: Vector of detected platforms
 * @return Number of detected platforms
 */
cl_uint OpenCLUtils::getPlatforms(vector<cl_platform_id> &clPlatformIDs)
{
    cl_uint platformCount;
    CL_CHECK(clGetPlatformIDs(0, 0, &platformCount));

    clPlatformIDs.resize(platformCount);

    CL_CHECK(clGetPlatformIDs(platformCount, clPlatformIDs.data(), 0));

#ifndef NDEBUG
    cout << "Number of available OpenCL platforms: " << platformCount << endl;
#endif

    return platformCount;
}

/**
 * Finds number of available devices for a given OpenCL platform.
 * @param clPlatformID Input: The platform to search
 * @return Number of detected devices for given platform
 */
cl_uint OpenCLUtils::countDevices(cl_platform_id clPlatformID)
{
    cl_uint deviceCount;

    CL_CHECK(clGetDeviceIDs(clPlatformID, CL_DEVICE_TYPE_ALL, 0, 0, &deviceCount));

#ifndef NDEBUG
    cout << "Number of available OpenCL devices: " << deviceCount << endl;
#endif

    return deviceCount;
}

/**
 * @brief The NameGetter class extracts the name of different types of OpenCL entities
 * by means of the template method pattern.
 */
class NameGetter
{
public:
    string get() const;
private:
    virtual void infoGetter(size_t, void *, size_t *) const = 0;
};

string NameGetter::get() const
{
    size_t size = 0;
    infoGetter(0, 0, &size);

    string result;
    result.resize(size);
    infoGetter(size, const_cast<char *>(result.data()), 0);

    boost::trim(result);
    return result;
}

class PlatformNameGetter : public NameGetter
{
public:
    PlatformNameGetter(cl_platform_id id) : id(id) {}
private:
    cl_platform_id id;
    virtual void infoGetter(size_t size, void *value, size_t *size_ret) const
    {
        clGetPlatformInfo(id, CL_PLATFORM_NAME, size, value, size_ret);
    }
};

class DeviceNameGetter : public NameGetter
{
public:
    DeviceNameGetter(cl_device_id id) : id(id) {}
private:
    cl_device_id id;
    virtual void infoGetter(size_t size, void *value, size_t *size_ret) const
    {
        clGetDeviceInfo(id, CL_DEVICE_NAME, size, value, size_ret);
    }
};

/**
 * Gets platform name from ID.
 * @param id Input: Platform ID
 * @return Platform name
 */
string OpenCLUtils::getPlatformName(cl_platform_id id)
{
    return PlatformNameGetter(id).get();
}

/**
 * Gets device name from ID.
 * @param id Input: Device ID
 * @return Device name
 */
string OpenCLUtils::getDeviceName(cl_device_id id)
{
    return DeviceNameGetter(id).get();
}

/**
 * Lists available devices for each platform.
 */
void OpenCLUtils::listDevices()
{
    vector<cl_platform_id> platforms;
    OpenCLUtils::getPlatforms(platforms);
    for (int i = 0; i < platforms.size(); ++i) {
        cout << "  platform " << i << endl;
        cout << "    name: " << getPlatformName(platforms[i]) << endl;
        cout << "    devices:\n";

        cl_uint deviceCount = 0;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, 0, &deviceCount);

        std::vector<cl_device_id> devices(deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices.data (), 0);

        for (cl_uint j = 0; j < deviceCount; ++j)
            cout << "      " << (j + 1) << ": " << getDeviceName(devices[j]) << endl;
    }
}

/**
 * Loads a kernel file.
 * @param name Input: Kernel file name
 * @return Contents of kernel file name as a string
 */
string OpenCLUtils::loadKernel(const char *name)
{
    const char *kernelDir = getenv("KERNELDIR");
    if (!kernelDir)
        throw runtime_error("KERNELDIR environment variable not set");
    ifstream in((boost::format("%s/%s") % kernelDir % name).str().c_str());
    if (!in.good())
        throw runtime_error((boost::format("failed to open kernel file >%s<") % name).str());
    string result((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
    return result;
}

/**
 * Creates a program from a kernel.
 * @param source Input: Kernel source
 * @return Program
 */
cl_program OpenCLUtils::createProgram(const cl_context &context, const string &source)
{
    size_t lengths[1] = { source.size() };
    const char *sources[1] = { source.data() };

    cl_int error = 0;
    cl_program program = clCreateProgramWithSource(context, 1, sources, lengths, &error);
    CL_CHECK(error);

    return program;
}
