#include "oclutils.h"
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <fstream>
#include <stdexcept>

using namespace std;

#define NDEBUG

/**
 * Finds all available OpenCL platforms on the current node.
 * @param patforms Output: Vector of platforms found
 * @return Number of platforms found
 */
cl_uint OpenCLUtils::getPlatforms(std::vector<cl::Platform> *platforms)
{
    CL_CHECK(cl::Platform::get(platforms));

#ifndef NDEBUG
    cout << "Number of available OpenCL platforms: " << platforms->size() << endl;
#endif

    return platforms->size();
}

/**
 * Finds number of available devices for a given OpenCL platform.
 * @param platform Input: The platform object to search
 * @return Number of detected devices for given platform
 */
cl_uint OpenCLUtils::countDevices(const cl::Platform &platform)
{
    vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

#ifndef NDEBUG
    cout << "Number of available OpenCL devices: " << devices.size() << endl;
#endif

    return devices.size();
}

/**
 * Gets platform name from object.
 * @param platform Input: Platform object
 * @return Platform name
 */
string OpenCLUtils::getPlatformName(const cl::Platform &platform)
{
    string name;
    platform.getInfo(CL_PLATFORM_NAME, &name);
    return name;
}

/**
 * Gets device name from object.
 * @param device Input: Device object
 * @return Device name
 */
string OpenCLUtils::getDeviceName(const cl::Device &device)
{
    string name;
    device.getInfo(CL_DEVICE_NAME, &name);
    return name;
}

/**
 * Lists available devices for each platform.
 */
void OpenCLUtils::listDevices()
{
    vector<cl::Platform> platforms;
    OpenCLUtils::getPlatforms(&platforms);
    for (int i = 0; i < platforms.size(); ++i) {
        cout << "  platform " << i << endl;
        cout << "    name: " << getPlatformName(platforms[i]) << endl;
        cout << "    devices:\n";
        vector<cl::Device> devices;
        platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);
        for (int j = 0; j < devices.size(); ++j)
            cout << "      " << (j + 1) << ": " << getDeviceName(devices[j]) << endl;

    }
}

/**
 * Loads kernel files.
 * @param names Input: Kernel file names
 * @return The string contents and size of the kernel
 */
cl::Program::Sources OpenCLUtils::loadKernels(const vector<string> &names)
{
    const char *kernelDir = getenv("KERNELDIR");
    if (!kernelDir)
        throw runtime_error("KERNELDIR environment variable not set");

    cl::Program::Sources sources;
    for (vector<string>::const_iterator it = names.begin(); it != names.end(); ++it) {
        ifstream in((boost::format("%s/%s") % kernelDir % *it).str().c_str());
        if (!in.good())
            throw runtime_error((boost::format("failed to open kernel file >%s<") % *it).str());
        string result((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
        sources.push_back(std::make_pair(result.c_str(), result.size()));
    }

    return sources;
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
