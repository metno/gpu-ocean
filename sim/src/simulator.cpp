#include "simulator.h"

#include "programoptions.h"
#include <boost/format.hpp>
#include <iostream>
#include <stdexcept>
#include <cstdio>

///XXX
//#define NDEBUG

/*
 * XXX: Extract this to OCL utils class
 */
#define CL_CHECK(_expr)                                                          \
	do {                                                                         \
		cl_int _err = _expr;                                                     \
		if (_err == CL_SUCCESS)                                                  \
			break;                                                               \
		fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
		abort();                                                                 \
	} while (0)

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

using namespace std;

struct Simulator::SimulatorImpl
{
    OptionsPtr options;
    InitCondPtr initCond;
    int nextStep;
    int finalStep;
    SimulatorImpl(const OptionsPtr &, const InitCondPtr &);
};

Simulator::SimulatorImpl::SimulatorImpl(const OptionsPtr &options, const InitCondPtr &initCond)
    : options(options)
    , initCond(initCond)
    , nextStep(-1)
    , finalStep(-1)
{
}

Simulator::Simulator(const OptionsPtr &options, const InitCondPtr &initCond)
    : pimpl(new SimulatorImpl(options, initCond))
{
}

Simulator::~Simulator()
{
}

void Simulator::init()
{
    pimpl->nextStep = 0;

    //pimpl->finalStep = calculate from options_->duration();
    pimpl->finalStep = 4; // ### for now
}

int Simulator::nextStep() const
{
    return pimpl->nextStep;
}

int Simulator::finalStep() const
{
    return pimpl->finalStep;
}

void Simulator::execNextStep()
{
    if (pimpl->nextStep > pimpl->finalStep)
        throw runtime_error((boost::format("error: next_step_ (%1%) > final_step_ (%2%)") % pimpl->nextStep % pimpl->finalStep).str());

    // executing next step ... TBD

    pimpl->nextStep++;
}

void Simulator::printStatus() const
{
    cout << "Simulator::printStatus(); options: " << *pimpl->options << endl;
}

cl_uint Simulator::getOCLPlatforms(vector<cl_platform_id> &clPlatformIDs)
{
	cl_uint countPlatforms;
	CL_CHECK(clGetPlatformIDs(0, NULL, &countPlatforms));

	clPlatformIDs.resize(countPlatforms);

	CL_CHECK(clGetPlatformIDs(countPlatforms, clPlatformIDs.data(), NULL));

#ifndef NDEBUG
    cout << "Number of available OpenCL platforms: " << countPlatforms << endl;
#endif

    return countPlatforms;
}

cl_uint Simulator::countOCLDevices(cl_platform_id clPlatformID) const
{
	cl_uint countDevices;

	CL_CHECK(clGetDeviceIDs(clPlatformID, CL_DEVICE_TYPE_ALL, 0, NULL, &countDevices));

#ifndef NDEBUG
	cout << "Number of available OpenCL devices: " << countDevices << endl;
#endif

    return countDevices;
}
