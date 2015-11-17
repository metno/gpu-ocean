#include "simulator.h"
#include "oclutils.h"
#include "config.h"
#include "reconstructH_types.h"
#include "computeU_types.h"
#include "computeV_types.h"
#include "computeEta_types.h"
#include <boost/format.hpp>
#include <iostream>
#include <stdexcept>

using namespace std;

struct Simulator::SimulatorImpl
{
    // H reconstructed
    cl::Buffer Hr_u; // x-dimension
    cl::Buffer Hr_v; // y-dimension

    // U, V, eta
    cl::Buffer U;
    cl::Buffer V;
    cl::Buffer eta;

    SimulatorImpl();
    void init(const OptionsPtr &);
    void reconstructH(const OptionsPtr &, const InitCondPtr &);
    void computeU(const OptionsPtr &, const InitCondPtr &);
    void computeV(const OptionsPtr &, const InitCondPtr &);
    void computeEta(const OptionsPtr &, const InitCondPtr &);
};

Simulator::SimulatorImpl::SimulatorImpl()
{
}

void Simulator::SimulatorImpl::init(const OptionsPtr &options)
{
    const int nx = options->nx();
    const int ny = options->ny();

    cl_int error = CL_SUCCESS;

    // create buffers that will be kept in global device memory throughout the simulation
    Hr_u = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE, sizeof(float) * nx * (ny + 1), 0, &error);
    CL_CHECK(error);
    Hr_v = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE, sizeof(float) * (nx + 1) * ny, 0, &error);
    CL_CHECK(error);
    U = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE, sizeof(float) * (nx + 2) * (ny - 1), 0, &error);
    CL_CHECK(error);
    V = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE, sizeof(float) * (nx - 1) * (ny + 2), 0, &error);
    CL_CHECK(error);
    eta = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE, sizeof(float) * (nx + 1) * (ny + 1), 0, &error);
    CL_CHECK(error);
}

// Returns the ceiling of the result of dividing two integers.
static int idivceil(int a, int b)
{
    assert(b > 0);
    return a / b + ((a % b) ? 1 : 0);
}

/**
 * Returns the global 2D work size for a kernel.
 * @param nx, ny: Requested global work size.
 * @param lnx, lny: Local work size (i.e. work-group size).
 * @returns The smallest global work size that will accommodate the requested global work size and still divide the local work size.
 */
static cl::NDRange global2DWorkSize(int nx, int ny, int lnx, int lny)
{
    assert(nx > 0);
    assert(ny > 0);
    assert(lnx > 0);
    assert(lny > 0);
    return cl::NDRange(lnx * idivceil(nx, lnx), lny * idivceil(ny, lny));
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

    // check preconditions on H
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
    args.nx = nx;
    args.ny = ny;
    kernel->setArg(3, args);

    // execute kernel (computes Hr_u and Hr_v in device memory)
    cl::Event event;
    CL_CHECK(OpenCLUtils::getQueue()->enqueueNDRangeKernel(
                 *kernel, cl::NullRange, global2DWorkSize(nx + 1, ny + 1, WGNX, WGNY), cl::NDRange(WGNX, WGNY), 0, &event));
    CL_CHECK(event.wait());

    // ...
}

/**
 * Computes U.
 */
void Simulator::SimulatorImpl::computeU(const OptionsPtr &options, const InitCondPtr &initCond)
{
    cerr << "computing U ...\n";

    const int nx = options->nx();
    const int ny = options->ny();

    // ... more code here ...

    cl::Kernel *kernel = OpenCLUtils::getKernel("computeU");

    // set up kernel arguments
    // ...
    computeU_args args;
    args.nx = nx;
    args.ny = ny;
    args.dt = -1; // ### replace -1 with actual value!
    args.dx = -1; // ### replace -1 with actual value!
    args.R = -1; // ### replace -1 with actual value!
    args.F = -1; // ### replace -1 with actual value!
    args.g = -1; // ### replace -1 with actual value!
    kernel->setArg(-1, args); // ### replace -1 with actual index!

    // ... more code here ...

    // execute kernel (computes U in device memory, excluding western sides of western ghost cells and eastern
    // side of eastern ghost cells)
    cl::Event event;
    CL_CHECK(OpenCLUtils::getQueue()->enqueueNDRangeKernel(
                 *kernel, cl::NullRange, global2DWorkSize(nx - 1, ny - 1, WGNX, WGNY), cl::NDRange(WGNX, WGNY), 0, &event));
    CL_CHECK(event.wait());

    // ...
}

/**
 * Computes V.
 */
void Simulator::SimulatorImpl::computeV(const OptionsPtr &options, const InitCondPtr &initCond)
{
    cerr << "computing V ...\n";

    const int nx = options->nx();
    const int ny = options->ny();

    // ... more code here ...

    cl::Kernel *kernel = OpenCLUtils::getKernel("computeV");

    // set up kernel arguments
    // ...
    computeV_args args;
    args.nx = nx;
    args.ny = ny;
    args.dt = -1; // ### replace -1 with actual value!
    args.dy = -1; // ### replace -1 with actual value!
    args.R = -1; // ### replace -1 with actual value!
    args.F = -1; // ### replace -1 with actual value!
    args.g = -1; // ### replace -1 with actual value!
    kernel->setArg(-1, args); // ### replace -1 with actual index!

    // ... more code here ...

    // execute kernel (to do exactly what?)
    // ### is a computational domain of (nx + 1, ny + 1) correct? (i.e. one work-item per cell, including ghost cells) ... TBD
//    cl::Event event;
//    CL_CHECK(OpenCLUtils::getQueue()->enqueueNDRangeKernel(
//                 *kernel, cl::NullRange, cl::NDRange(nx + 1, ny + 1), cl::NDRange(WGNX, WGNY), 0, &event));
//    CL_CHECK(event.wait());

    // ...
}

/**
 * Computes eta.
 */
void Simulator::SimulatorImpl::computeEta(const OptionsPtr &options, const InitCondPtr &initCond)
{
    cerr << "computing eta ...\n";

    const int nx = options->nx();
    const int ny = options->ny();

    cl::Kernel *kernel = OpenCLUtils::getKernel("computeEta");

    // set up kernel arguments
    kernel->setArg<cl::Buffer>(0, U);
    kernel->setArg<cl::Buffer>(1, V);
    kernel->setArg<cl::Buffer>(2, eta);
    computeEta_args args;
    args.nx = nx;
    args.ny = ny;
    args.dt = -1; // ### replace -1 with actual value!
    args.dx = options->width() / float(nx - 1);
    args.dy = options->height() / float(ny - 1);
    kernel->setArg(3, args);

    // execute kernel (computes eta in device memory, excluding ghost cells (hence (nx - 1, ny - 1) instead of (nx + 1, ny + 1));
    // note: eta in ghost cells are part of the boundary conditions and updated separately)
    cl::Event event;
    CL_CHECK(OpenCLUtils::getQueue()->enqueueNDRangeKernel(
                 *kernel, cl::NullRange, global2DWorkSize(nx - 1, ny - 1, WGNX, WGNY), cl::NDRange(WGNX, WGNY), 0, &event));
    CL_CHECK(event.wait());

    // ...
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
    sources.push_back(make_pair("computeU", "computeU.cl"));
    sources.push_back(make_pair("computeV", "computeV.cl"));
    sources.push_back(make_pair("computeEta", "computeEta.cl"));
    OpenCLUtils::init(
                sources, options()->cpu() ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU,
                (boost::format("-I %s") % OpenCLUtils::getKernelDir()).str());

    pimpl->init(options());

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
    // compute U
    pimpl->computeU(options(), initCond());

    // compute V ... TBD

    // compute eta
    pimpl->computeEta(options(), initCond());
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
