#ifndef OCLUTILS_H
#define OCLUTILS_H

// All OpenCL headers
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

#include <cstdio>
#include <vector>
#include <string>

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
    static cl_uint getPlatforms(std::vector<cl_platform_id> &);
    static std::string getPlatformName(cl_platform_id);
    static std::string getDeviceName(cl_device_id);
    static cl_uint countDevices(cl_platform_id);
    static void listDevices();
    static std::string loadKernel(const char *);
    static cl_program createProgram(const cl_context &, const std::string &);
};

#endif // OCLUTILS_H
