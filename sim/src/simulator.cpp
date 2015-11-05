#include "simulator.h"
#include "oclutils.h"
#include "config.h"
#include "reconstructH_types.h"
#include "computeV_types.h"
#include <boost/format.hpp>
#include <iostream>
#include <stdexcept>

using namespace std;

struct Simulator::SimulatorImpl
{
    // H reconstructed
    cl::Buffer Hr_u; // x-dimension
    cl::Buffer Hr_v; // y-dimension

    SimulatorImpl();
    void reconstructH(const OptionsPtr &, const InitCondPtr &);
    void computeV(const OptionsPtr &, const InitCondPtr &);
};

Simulator::SimulatorImpl::SimulatorImpl()
{
}

/**
 * Reconstructs H, i.e. computes Hr_u and Hr_v from initCond->H().
 */
void Simulator::SimulatorImpl::reconstructH(const OptionsPtr &options, const InitCondPtr &initCond)
{
    cerr << "reconstructing H ...\n";

    const int nx = options->nx();
    const int ny = options->ny();
    const FieldInfo Hfi = initCond->H();

    // check preconditions
    assert(Hfi.data->size() == Hfi.nx * Hfi.ny);
    assert(Hfi.nx == nx + 1);
    assert(Hfi.ny == ny + 1);
    assert(Hfi.nx > 2);
    assert(Hfi.ny > 2);

    cl_int error = CL_SUCCESS;

    // create buffer for H (released from device after reconstruction is complete)
    cl::Buffer H = cl::Buffer(
                *OpenCLUtils::getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(float) * Hfi.nx * Hfi.ny, Hfi.data->data(), &error);
    CL_CHECK(error);

    // create buffers for Hr_u and Hr_v (kept on device throughout the simulation)
    Hr_u = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE, sizeof(float) * nx * (ny + 1), 0, &error);
    CL_CHECK(error);
    Hr_v = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE, sizeof(float) * (nx + 1) * ny, 0, &error);
    CL_CHECK(error);

    // ensure that we have enough local memory for the work-group
    const int H_local_size = sizeof(float) * (WGNX + 1) * (WGNY + 1);
    const cl_ulong localDevMemAvail = OpenCLUtils::getDeviceLocalMemSize();
    if (H_local_size > localDevMemAvail)
        throw runtime_error(
                (boost::format("not enough local device memory for reconstructing H: requested: %d bytes, available: %d bytes")
                 % H_local_size % localDevMemAvail).str());

    cl::Kernel *kernel = OpenCLUtils::getKernel("ReconstructH");

    // set up kernel arguments
    kernel->setArg<cl::Buffer>(0, H);
    kernel->setArg<cl::Buffer>(1, Hr_u);
    kernel->setArg<cl::Buffer>(2, Hr_v);
    ReconstructH_args args;
    args.nx = options->nx();
    args.ny = options->ny();
    kernel->setArg(3, args);

    // execute kernel (computes Hr_u and Hr_v in device memory and returns pointers)
    cl::Event event;
    CL_CHECK(OpenCLUtils::getQueue()->enqueueNDRangeKernel(
                 *kernel, cl::NullRange, cl::NDRange(nx + 1, ny + 1), cl::NDRange(WGNX, WGNY), 0, &event));
    CL_CHECK(event.wait());

    // ...
}

/**
 * Computes V.
 */
void Simulator::SimulatorImpl::computeV(const OptionsPtr &options, const InitCondPtr &initCond)
{
    cerr << "computing V ...\n";

    // ... more code here ...

    cl::Kernel *kernel = OpenCLUtils::getKernel("computeV");

    // set up kernel arguments
    // ...
    computeV_args args;
    args.nx = options->nx();
    args.ny = options->ny();
    args.dt = -1; // ### replace -1 with actual value!
    args.dy = -1; // ### replace -1 with actual value!
    args.R = -1; // ### replace -1 with actual value!
    args.F = -1; // ### replace -1 with actual value!
    args.g = -1; // ### replace -1 with actual value!
    kernel->setArg(-1, args); // ### replace -1 with actual index!

    // ... more code here ...
}

Simulator::Simulator(const OptionsPtr &options, const InitCondPtr &initCond)
    : SimBase(options, initCond)
    , pimpl(new SimulatorImpl)
{
}

Simulator::~Simulator()
{
}

bool Simulator::_init()
{
    // initialize OpenCL structures
    vector<pair<string, string> > sources;
    sources.push_back(make_pair("ReconstructH", "reconstructH.cl"));
    OpenCLUtils::init(
                sources, options()->cpu() ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU,
                (boost::format("-I %s") % OpenCLUtils::getKernelDir()).str());

    // reconstruct H
    pimpl->reconstructH(options(), initCond());

    return true;
}

double Simulator::_currTime() const
{
    return 0; // ### for now
}

double Simulator::_maxTime() const
{
    return -1; // ### for now
}

void Simulator::_execNextStep()
{
    // compute U ... TBD
    // compute V ... TBD
    // compute eta ... TBD
}

FieldInfo Simulator::_U() const
{
    return FieldInfo(); // ### for now
}

FieldInfo Simulator::_V() const
{
    return FieldInfo(); // ### for now
}

FieldInfo Simulator::_eta() const
{
    return FieldInfo(); // ### for now
}

void Simulator::_printStatus() const
{
    cout << "Simulator::_printStatus(); options: " << *options() << endl;
}
