#ifndef OCLUTILS_H
#define OCLUTILS_H

#define CL_HPP_MINIMUM_OPENCL_VERSION 110 
#define CL_HPP_TARGET_OPENCL_VERSION 110

/*
 * Source code should be fixed to avoid this, see cl2.hpp:
 * Finally, the program construction interface used a clumsy vector-of-pairs
 * design in the earlier versions. We have replaced that with a cleaner 
 * vector-of-vectors and vector-of-strings design. However, for backward 
 * compatibility old behaviour can be regained with the
 */
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY

#include "config.h"
#include <CL/cl2.hpp>
#include <cstdio>
#include <vector>
#include <string>
#include <map>
#include <memory>

// define macros to assert that an expression evaluates to CL_SUCCESS
// version 1:
#define CL_CHECK(_expr)                                                          \
    do {                                                                         \
        cl_int _err = _expr;                                                     \
        if (_err == CL_SUCCESS)                                                  \
            break;                                                               \
        fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
        abort();                                                                 \
    } while (0)
// version 2:
#define CL_CHECK_ERR(_expr)                                                          \
    ({                                                                               \
        cl_int _err = CL_INVALID_VALUE;                                              \
        typeof(_expr) _ret = _expr;                                                  \
        if (_err != CL_SUCCESS) {                                                    \
            fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
            abort();                                                                 \
        }                                                                            \
        _ret;                                                                        \
    })

class OpenCLUtils
{
public:

    /**
     * Finds all available OpenCL platforms on the current node.
     * @param platforms Output: Vector of platforms found.
     * @returns Number of platforms found.
     */
    static cl_uint getPlatforms(std::vector<cl::Platform> *platforms);

    /**
     * Gets platform name from object.
     * @param platform: Platform object
     * @returns Platform name
     */
    static std::string getPlatformName(const cl::Platform &platform);

    /**
     * Gets device name from object.
     * @param device: Device object.
     * @returns Device name.
     */
    static std::string getDeviceName(const cl::Device &device);

    /**
     * Finds the number of available devices for a given OpenCL platform.
     * @param platform: The platform object to search.
     * @returns Number of detected devices for given platform.
     */
    static cl_uint countDevices(const cl::Platform &platform);

    /**
     * Lists available devices for each platform.
     */
    static void listDevices();

    /**
     * Returns the maximum local memory size of the current device in bytes.
     */
    static cl_ulong getDeviceLocalMemSize();

    /**
     * Returns the execution time from an event.
     * @param event: The event object.
     * @returns Elapsed execution time in milliseconds.
     */
    static float elapsedMilliseconds(const cl::Event &event);

    /**
     * Initializes OpenCL structures.
     * @param sources: Pairs of (tag, source file name) for kernels.
     * @param deviceType: Device type, typically CL_DEVICE_TYPE_CPU or CL_DEVICE_TYPE_GPU.
     * @param buildOptions: Options passed to cl::Program::build().
     */
    static void init(
            const std::vector<std::pair<std::string, std::string> > &sources, cl_device_type deviceType, const std::string &buildOptions = std::string());

    /**
     * Returns the current context.
     */
    static cl::Context *getContext();

    /**
     * Returns the kernel object for a given tag.
     * @param tag: Kernel tag (corresponding to one of those passed in the sources parameter in init()).
     */
    static cl::Kernel *getKernel(const std::string &tag);

    /**
     * Returns the command queue for the current device.
     */
    static cl::CommandQueue *getQueue();

    /**
     * Returns the name of the current platform.
     */
    static std::string getPlatformName();

    /**
     * Returns the name of the current device.
     */
    static std::string getDeviceName();

    /**
     * Returns the value of the mandatory KERNELDIR environment variable.
     * @note The function asserts that the variable is set.
     */
    static std::string getKernelDir();

private:
    static bool isInit;
    static std::vector<cl::Platform> platforms;
    static cl_uint pfmIndex; // index of current platform
    static std::vector<cl::Device> devices;
    static cl_uint devIndex; // index of current device
    static std::shared_ptr<cl::Context> context;
    static std::shared_ptr<cl::Program> program;
    static std::map<std::string, std::shared_ptr<cl::Kernel> > kernels; // tag-to-kernel mapping
    static std::shared_ptr<cl::CommandQueue> queue;
};

#endif // OCLUTILS_H
