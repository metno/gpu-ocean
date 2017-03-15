#include "simulator.h"
#include "oclutils.h"
#include "reconstructH_types.h"
#include "computeU_types.h"
#include "computeV_types.h"
#include "computeEta_types.h"
#include "windStress_types.h"
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
    float r; // friction
    float f; // Coriolis effect
    float g; // standard gravity

    float currTime; // current simulation time
    float maxTime; // maximum simulation time

    // H reconstructed
    cl::Buffer Hr_u; // x-dimension
    cl::Buffer Hr_v; // y-dimension

    // H, U, V, eta
    // device buffers:
    cl::Buffer H;
    cl::Buffer U;
    cl::Buffer V;
    cl::Buffer eta;
    // host buffers:
    Field2D _H;
    Field2D _U;
    Field2D _V;
    Field2D _eta;

    SimulatorImpl();
    void init(const OptionsPtr &, const InitCondPtr &);
    void reconstructH(const OptionsPtr &, const InitCondPtr &);
    void computeU(const OptionsPtr &, const InitCondPtr &, ProfileInfo *);
    void computeV(const OptionsPtr &, const InitCondPtr &, ProfileInfo *);
    void computeEta(const OptionsPtr &, const InitCondPtr &, ProfileInfo *);
};

Simulator::SimulatorImpl::SimulatorImpl()
{
}

void Simulator::SimulatorImpl::init(const OptionsPtr &options, const InitCondPtr &initCond)
{
	nx = initCond->nx();
	ny = initCond->ny();
	dx = initCond->dx();
	dy = initCond->dy();
	dt = std::min(dx, dy) * 0.01; // ### for now
	r = 0.0024;
	f = 0.f; // ### no influence for now
	g = 9.8;
	currTime = 0;
	maxTime = options->duration();

    cl_int error = CL_SUCCESS;

    // create buffers ...
    // ... H reconstructed
    Hr_u = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE, sizeof(float) * nx * (ny + 1), 0, &error);
    CL_CHECK(error);
    Hr_v = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE, sizeof(float) * (nx + 1) * ny, 0, &error);
    CL_CHECK(error);

    // ... H
    const int nx_H = nx + 1;
    const int ny_H = ny + 1;
    _H = Field2D(new vector<float>(nx_H * ny_H), nx_H, ny_H, dx, dy);
    _H.fill(initCond->H());
    H = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * _H.data()->size(), _H.data()->data(), &error);
    CL_CHECK(error);

    // ... U
    const int nx_U = nx + 2;
    const int ny_U = ny - 1;
    _U = Field2D(new vector<float>(nx_U * ny_U), nx_U, ny_U, dx, dy);
    _U.fill(0.0f);
    U = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * _U.data()->size(), _U.data()->data(), &error);
    CL_CHECK(error);

    // ... V
    const int nx_V = nx - 1;
    const int ny_V = ny + 2;
    _V = Field2D(new vector<float>(nx_V * ny_V), nx_V, ny_V, dx, dy);
    _V.fill(0.0f);
    V = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * _V.data()->size(), _V.data()->data(), &error);
    CL_CHECK(error);

    // ... eta
    const int nx_eta = nx + 1;
    const int ny_eta = ny + 1;
    _eta = Field2D(new vector<float>(nx_eta * ny_eta), nx_eta, ny_eta, dx, dy);
    _eta.fill(initCond->eta());
    eta = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * _eta.data()->size(), _eta.data()->data(), &error);
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
    const Field2D Hfi = initCond->H();

    // check preconditions on H
    assert(Hfi.data()->size() == Hfi.nx() * Hfi.ny());
    assert(Hfi.nx() == nx + 1);
    assert(Hfi.ny() == ny + 1);
    assert(Hfi.nx() > 2);
    assert(Hfi.ny() > 2);

    cl_int error = CL_SUCCESS;

    // create buffer for H (released from device after reconstruction is complete)
    H = cl::Buffer(
                *OpenCLUtils::getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(float) * Hfi.nx() * Hfi.ny(), Hfi.data()->data(), &error);
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
void Simulator::SimulatorImpl::computeU(const OptionsPtr &options, const InitCondPtr &initCond, ProfileInfo *profInfo)
{
    cl::Kernel *kernel = OpenCLUtils::getKernel("computeU");

    // set up kernel arguments
    computeU_args args;
    args.nx = nx;
    args.ny = ny;
    args.dt = dt;
    args.dx = dx;
    args.dy = dy;
    args.r = r;
    args.f = f;
    args.g = g;
    kernel->setArg(0, args);
    windStress_args ws;
    ws.wind_stress_type = 99;
	ws.tau0 = 0.0f;
	ws.rho = 0.0f;
	ws.alpha = 0.0f;
	ws.xm = 0.0f;
	ws.Rc = 0.0f;
	ws.x0 = 0.0f;
	ws.y0 = 0.0f;
	ws.u0 = 0.0f;
	ws.v0 = 0.0f;
    kernel->setArg(1, ws);
    kernel->setArg<cl::Buffer>(2, H);
    kernel->setArg<int>(3, _H.nx()*sizeof(float));
    kernel->setArg<cl::Buffer>(4, U);
    kernel->setArg<int>(5, _U.nx()*sizeof(float));
    kernel->setArg<cl::Buffer>(6, V);
    kernel->setArg<int>(7, _V.nx()*sizeof(float));
    kernel->setArg<cl::Buffer>(8, eta);
    kernel->setArg<int>(9, _eta.nx()*sizeof(float));
    kernel->setArg<float>(10, currTime);

    // execute kernel
    cl::Event event;
    CL_CHECK(OpenCLUtils::getQueue()->enqueueNDRangeKernel(
                 *kernel, cl::NullRange, global2DWorkSize(nx+1, ny+1, WGNX, WGNY), cl::NDRange(WGNX, WGNY), 0, &event));
    CL_CHECK(event.wait());
    if (profInfo)
        profInfo->time_computeU = OpenCLUtils::elapsedMilliseconds(event);

    // copy result from device to host
    CL_CHECK(OpenCLUtils::getQueue()->enqueueReadBuffer(U, CL_TRUE, 0, sizeof(float) * _U.data()->size(), _U.data()->data(), 0, 0));
}

/**
 * Computes V.
 */
void Simulator::SimulatorImpl::computeV(const OptionsPtr &options, const InitCondPtr &initCond, ProfileInfo *profInfo)
{
    cl::Kernel *kernel = OpenCLUtils::getKernel("computeV");

    // set up kernel arguments
	computeV_args args;
	args.nx = nx;
	args.ny = ny;
	args.dt = dt;
	args.dx = dx;
	args.dy = dy;
	args.r = r;
	args.f = f;
	args.g = g;
	kernel->setArg(0, args);
	windStress_args ws;
	ws.wind_stress_type = 99;
	ws.tau0 = 0.0f;
	ws.rho = 0.0f;
	ws.alpha = 0.0f;
	ws.xm = 0.0f;
	ws.Rc = 0.0f;
	ws.x0 = 0.0f;
	ws.y0 = 0.0f;
	ws.u0 = 0.0f;
	ws.v0 = 0.0f;
	kernel->setArg(1, ws);
	kernel->setArg<cl::Buffer>(2, H);
	kernel->setArg<int>(3, _H.nx()*sizeof(float));
	kernel->setArg<cl::Buffer>(4, U);
	kernel->setArg<int>(5, _U.nx()*sizeof(float));
	kernel->setArg<cl::Buffer>(6, V);
	kernel->setArg<int>(7, _V.nx()*sizeof(float));
	kernel->setArg<cl::Buffer>(8, eta);
	kernel->setArg<int>(9, _eta.nx()*sizeof(float));
	kernel->setArg<float>(10, currTime);

    // execute kernel
    cl::Event event;
    CL_CHECK(OpenCLUtils::getQueue()->enqueueNDRangeKernel(
                 *kernel, cl::NullRange, global2DWorkSize(nx+1, ny+1, WGNX, WGNY), cl::NDRange(WGNX, WGNY), 0, &event));
    CL_CHECK(event.wait());
    if (profInfo)
        profInfo->time_computeV = OpenCLUtils::elapsedMilliseconds(event);

    // copy result from device to host
    CL_CHECK(OpenCLUtils::getQueue()->enqueueReadBuffer(V, CL_TRUE, 0, sizeof(float) * _V.data()->size(), _V.data()->data(), 0, 0));
}

/**
 * Computes eta.
 */
void Simulator::SimulatorImpl::computeEta(const OptionsPtr &options, const InitCondPtr &initCond, ProfileInfo *profInfo)
{
    cl::Kernel *kernel = OpenCLUtils::getKernel("computeEta");

    // set up kernel arguments
    computeEta_args args;
    args.nx = nx;
    args.ny = ny;
    args.dt = dt;
    args.dx = dx;
    args.dy = dy;
    args.g = g;
	args.f = f;
	args.r = r;
    kernel->setArg(0, args);
    kernel->setArg<cl::Buffer>(1, U);
    kernel->setArg<int>(2, _U.nx()*sizeof(float));
    kernel->setArg<cl::Buffer>(3, V);
    kernel->setArg<int>(4, _V.nx()*sizeof(float));
    kernel->setArg<cl::Buffer>(5, eta);
    kernel->setArg<int>(6, _eta.nx()*sizeof(float));

    // execute kernel
    cl::Event event;
    CL_CHECK(OpenCLUtils::getQueue()->enqueueNDRangeKernel(
                 *kernel, cl::NullRange, global2DWorkSize(nx, ny, WGNX, WGNY), cl::NDRange(WGNX, WGNY), 0, &event));
    CL_CHECK(event.wait());
    if (profInfo)
        profInfo->time_computeEta = OpenCLUtils::elapsedMilliseconds(event);

    // copy result from device to host
    CL_CHECK(OpenCLUtils::getQueue()->enqueueReadBuffer(eta, CL_TRUE, 0, sizeof(float) * _eta.data()->size(), _eta.data()->data(), 0, 0));
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

    pimpl->init(options(), initCond());

    // reconstruct H
    // NOTE!
    // This is now done in the different compute kernels for the numerical schemes
    // Since we may want to precompute the reconstructed H (Hr_u and Hr_v) at a
    // later time, the code was not removed.
    //pimpl->reconstructH(options(), initCond());

    return true;
}

double Simulator::_currTime() const
{
    return pimpl->currTime;
}

double Simulator::_maxTime() const
{
    return pimpl->maxTime;
}

float Simulator::_deltaTime() const
{
    return pimpl->dt;
}

void Simulator::_execNextStep(ProfileInfo *profInfo)
{
    // compute U
    pimpl->computeU(options(), initCond(), profInfo);

    // compute V
    pimpl->computeV(options(), initCond(), profInfo);

    // compute eta
    pimpl->computeEta(options(), initCond(), profInfo);

    pimpl->currTime += pimpl->dt; // advance simulation time
}

Field2D Simulator::_H() const
{
    return pimpl->_H;
}

Field2D Simulator::_U() const
{
    return pimpl->_U;
}

Field2D Simulator::_V() const
{
    return pimpl->_V;
}

Field2D Simulator::_eta() const
{
    return pimpl->_eta;
}

float Simulator::_f() const
{
    return pimpl->f;
}

float Simulator::_r() const
{
    return pimpl->r;
}

void Simulator::_printStatus() const
{
    cout << "Simulator::_printStatus(); options: " << *options() << endl;
}
