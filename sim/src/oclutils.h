#ifndef OCLUTILS_H
#define OCLUTILS_H

#include <CL/cl.hpp>
#include <cstdio>
#include <vector>
#include <string>
#include <map>

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
    static cl_uint getPlatforms(std::vector<cl::Platform> *);
    static std::string getPlatformName(const cl::Platform &);
    static std::string getDeviceName(const cl::Device &);
    static cl_uint countDevices(const cl::Platform &);
    static void listDevices();
    static float elapsedMilliseconds(const cl::Event &);
    static void init(const std::vector<std::pair<std::string, std::string> > &, const std::string &, cl_device_type);
    static cl::Context *getContext();
    static cl::Kernel *getKernel(const std::string &);
    static cl::CommandQueue *getQueue();
    static std::string getPlatformName();
    static std::string getDeviceName();
private:
    static bool isInit;
    static std::vector<cl::Platform> platforms;
    static cl_uint pfmIndex;
    static std::vector<cl::Device> devices;
    static cl_uint devIndex;
    static cl::Context *context;
    static cl::Program *program;
    static std::map<std::string, cl::Kernel *> kernels; // tag-to-kernel mapping
    static cl::CommandQueue *queue;
};

#endif // OCLUTILS_H
