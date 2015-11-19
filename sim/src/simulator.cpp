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
    int nx; // number of grid points excluding points on the outside of ghost cells (number of cells is nx - 1)
    int ny; // ...
    float dx; // grid cell size in meters
    float dy; // ...
    float dt; // seconds by which to advance simulation in each step
    float R; // friction
    float F; // Coriolis effect
    float g; // standard gravity

    // H reconstructed
    cl::Buffer Hr_u; // x-dimension
    cl::Buffer Hr_v; // y-dimension

    // U, V, eta
    // device buffers:
    cl::Buffer U;
    cl::Buffer V;
    cl::Buffer eta;
    // host buffers:
    FieldInfo _U;
    FieldInfo _V;
    FieldInfo _eta;

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
    nx = options->nx();
    ny = options->ny();
    dx = options->width() / nx;
    dy = options->height() / ny;
    dt = std::min(dx, dy) * 0.001; // ### for now
    R = 1; // ### no influence for now
    F = 1; // ### no influence for now
    g = 9.8;

    cl_int error = CL_SUCCESS;

    // create buffers that will be kept in global device memory throughout the simulation
    Hr_u = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE, sizeof(float) * nx * (ny + 1), 0, &error);
    CL_CHECK(error);
    Hr_v = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE, sizeof(float) * (nx + 1) * ny, 0, &error);
    CL_CHECK(error);

    const int nx_U = nx + 2;
    const int ny_U = ny - 1;
    const int size_U = nx_U * ny_U;
    U = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE, sizeof(float) * size_U, 0, &error);
    CL_CHECK(error);

    const int nx_V = nx - 1;
    const int ny_V = ny + 2;
    const int size_V = nx_V * ny_V;
    V = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE, sizeof(float) * size_V, 0, &error);
    CL_CHECK(error);

    const int nx_eta = nx + 1;
    const int ny_eta = ny + 1;
    const int size_eta = nx_eta * ny_eta;
    eta = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE, sizeof(float) * size_eta, 0, &error);
    CL_CHECK(error);

    // create host buffers for U, V, and eta
    _U.data->resize(sizeof(float) * size_U);
    _U.nx = nx_U;
    _U.ny = ny_U;

    _V.data->resize(sizeof(float) * size_V);
    _V.nx = nx_V;
    _V.ny = ny_V;

    _eta.data->resize(sizeof(float) * size_eta);
    _eta.nx = nx_eta;
    _eta.ny = ny_eta;
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

    // ... more code here ...

    cl::Kernel *kernel = OpenCLUtils::getKernel("computeU");

    // set up kernel arguments
    kernel->setArg<cl::Buffer>(0, eta);
    kernel->setArg<cl::Buffer>(1, V);
    kernel->setArg<cl::Buffer>(2, Hr_u);
    kernel->setArg<cl::Buffer>(3, U);
    computeU_args args;
    args.nx = nx;
    args.ny = ny;
    args.dt = dt;
    args.dx = dx;
    args.R = R;
    args.F = F;
    args.g = g;
    kernel->setArg(4, args);

    // ... more code here ...

    // execute kernel (computes U in device memory, excluding western sides of western ghost cells and eastern
    // side of eastern ghost cells)
    cl::Event event;
    cl::NDRange r = global2DWorkSize(nx - 1, ny - 1, WGNX, WGNY);
    CL_CHECK(OpenCLUtils::getQueue()->enqueueNDRangeKernel(
                 *kernel, cl::NullRange, global2DWorkSize(nx - 1, ny - 1, WGNX, WGNY), cl::NDRange(WGNX, WGNY), 0, &event));
    CL_CHECK(event.wait());

    // copy result from device to host
    CL_CHECK(OpenCLUtils::getQueue()->enqueueReadBuffer(U, CL_TRUE, 0, sizeof(float) * _U.nx * _U.ny, _U.data->data(), 0, 0));
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
    args.nx = nx;
    args.ny = ny;
    args.dt = dt;
    args.dy = dy;
    args.R = R;
    args.F = F;
    args.g = g;
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

    // copy result from device to host
    CL_CHECK(OpenCLUtils::getQueue()->enqueueReadBuffer(eta, CL_TRUE, 0, sizeof(float) * _eta.nx * _eta.ny, _eta.data->data(), 0, 0));
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
    return pimpl->_U;
}

FieldInfo Simulator::_V() const
{
    return pimpl->_V;
}

FieldInfo Simulator::_eta() const
{
    return pimpl->_eta;
}

void Simulator::_printStatus() const
{
    cout << "Simulator::_printStatus(); options: " << *options() << endl;
}
