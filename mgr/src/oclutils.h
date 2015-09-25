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
    static cl_uint getOCLPlatforms(std::vector<cl_platform_id> &);
    static cl_uint countOCLDevices(cl_platform_id);
};

#endif // OCLUTILS_H
