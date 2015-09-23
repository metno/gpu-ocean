#include "simulator.h"

#include "programoptions.h"
#include <boost/format.hpp>
#include <iostream>
#include <stdexcept>

// All OpenCL headers
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

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

void Simulator::countOCLDevices() const
{
	cl_uint countEntries = 999;
	cl_platform_id clSelectedPlatformID = NULL;
	cl_uint countPlatforms;
	cl_int ciErrNum = clGetPlatformIDs(countEntries, &clSelectedPlatformID, &countPlatforms);

	cl_uint ciDeviceCount;
	ciErrNum = clGetDeviceIDs(clSelectedPlatformID, CL_DEVICE_TYPE_ALL, 0, NULL, &ciDeviceCount);

    cout << "Number of available OpenCL devices: " << ciDeviceCount << endl;
}
