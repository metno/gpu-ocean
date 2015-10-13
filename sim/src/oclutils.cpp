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
 * Returns the execution time from an event.
 * @param event Input: The event object
 * @return Elapsed execution time in milliseconds
 */
float OpenCLUtils::elapsedMilliseconds(const cl::Event &event)
{
    return (event.getProfilingInfo<CL_PROFILING_COMMAND_END>()
            - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * .000001; // note: _START and _END values are both in nanoseconds
}

/**
 * Loads kernel files.
 * @param names Input: Kernel file names
 * @return The string contents and size of the kernel
 */
static cl::Program::Sources loadKernels(const vector<string> &names)
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
 * Prints the build log of a program on stderr.
 * @param program Input: The program object
 * @param devices Input: The device objects
 */
static void printProgramBuildLog(const cl::Program &program, const vector<cl::Device> &devices)
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

// Initializes kernels.
void OpenCLUtils::initKernels(
        const cl::Context &context, const vector<cl::Device> &devices,
        const vector<pair<string, string> > &sources, const string &programOptions)
{
    if (kernelsInit)
        throw runtime_error("kernels already initialized");

    cl_int error;

    // create program
    vector<string> srcFiles;
    for (vector<pair<string, string> >::const_iterator it = sources.begin(); it != sources.end(); ++it) {
        const string srcFile = it->second;
        srcFiles.push_back(srcFile);
    }
    program = cl::Program(context, loadKernels(srcFiles), &error);
    CL_CHECK(error);

    // compile program
    error = program.build(devices, programOptions.c_str(), 0, 0);
    if (error == CL_BUILD_PROGRAM_FAILURE)
        printProgramBuildLog(program, devices);
    CL_CHECK(error);

    // create kernels
    for (vector<pair<string, string> >::const_iterator it = sources.begin(); it != sources.end(); ++it) {
        const string tag = it->first;
        kernels[tag] = cl::Kernel(program, tag.c_str(), &error);
        CL_CHECK(error);
    }

    kernelsInit = true;
}

bool OpenCLUtils::kernelsInit = false;
map<string, cl::Kernel> OpenCLUtils::kernels;
cl::Program OpenCLUtils::program;

// Returns the kernel object for a given tag.
cl::Kernel &OpenCLUtils::getKernel(const string &tag)
{
    if (!kernelsInit)
        throw runtime_error("kernels not initialized");

    return kernels[tag];
}
