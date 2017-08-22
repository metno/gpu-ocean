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
#include <vector>

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
    Field2D H_host;
    Field2D U_host;
    Field2D V_host;
    Field2D eta_host;

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
	nx = initCond->getNx();
	ny = initCond->getNy();
	dx = initCond->getDx();
	dy = initCond->getDy();
	dt = std::min(dx, dy) * 0.05; // ### for now
	r = 0.0024;
	f = 0.f; // ### no influence for now
	g = 9.8;
	currTime = 0;
	maxTime = options->duration();

    cl_int error = CL_SUCCESS;

    // create buffers ...
    // ... H reconstructed
    Hr_u = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE, sizeof(float) * (nx-1) * ny, 0, &error);
    CL_CHECK(error);
    Hr_v = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE, sizeof(float) * nx * (ny-1), 0, &error);
    CL_CHECK(error);

    // ... H
    const int nx_H = nx;
    const int ny_H = ny;
    H_host = Field2D(new vector<float>(nx_H * ny_H), nx_H, ny_H, dx, dy);
    H_host.fill(initCond->H());
    H = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * H_host.getData()->size(), H_host.getData()->data(), &error);
    CL_CHECK(error);

    // ... U
    const int nx_U = nx + 1; //including ghost cells
    const int ny_U = ny;
    U_host = Field2D(new vector<float>(nx_U * ny_U), nx_U, ny_U, dx, dy);
    U_host.fill(0.0f);
    U = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * U_host.getData()->size(), U_host.getData()->data(), &error);
    CL_CHECK(error);

    // ... V
    const int nx_V = nx;
    const int ny_V = ny + 1; //including ghost cells
    V_host = Field2D(new vector<float>(nx_V * ny_V), nx_V, ny_V, dx, dy);
    V_host.fill(0.0f);
    V = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * V_host.getData()->size(), V_host.getData()->data(), &error);
    CL_CHECK(error);

    // ... eta
    const int nx_eta = nx;
    const int ny_eta = ny;
    eta_host = Field2D(new vector<float>(nx_eta * ny_eta), nx_eta, ny_eta, dx, dy);
    eta_host.fill(initCond->eta());
    eta = cl::Buffer(*OpenCLUtils::getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * eta_host.getData()->size(), eta_host.getData()->data(), &error);
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
    assert(Hfi.getData()->size() == Hfi.getNx() * Hfi.getNy());
    assert(Hfi.getNx() == nx + 1);
    assert(Hfi.getNy() == ny + 1);
    assert(Hfi.getNx() > 2);
    assert(Hfi.getNy() > 2);

    cl_int error = CL_SUCCESS;

    // create buffer for H (released from device after reconstruction is complete)
    H = cl::Buffer(
                *OpenCLUtils::getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(float) * Hfi.getNx() * Hfi.getNy(), Hfi.getData()->data(), &error);
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
    kernel->setArg<int>(3, H_host.getNx()*sizeof(float));
    kernel->setArg<cl::Buffer>(4, U);
    kernel->setArg<int>(5, U_host.getNx()*sizeof(float));
    kernel->setArg<cl::Buffer>(6, V);
    kernel->setArg<int>(7, V_host.getNx()*sizeof(float));
    kernel->setArg<cl::Buffer>(8, eta);
    kernel->setArg<int>(9, eta_host.getNx()*sizeof(float));
    kernel->setArg<float>(10, currTime);

    // execute kernel
    cl::Event event;
    CL_CHECK(OpenCLUtils::getQueue()->enqueueNDRangeKernel(
                 *kernel, cl::NullRange, global2DWorkSize(nx+1, ny+1, WGNX, WGNY), cl::NDRange(WGNX, WGNY), 0, &event));
    CL_CHECK(event.wait());
    if (profInfo)
        profInfo->time_computeU = OpenCLUtils::elapsedMilliseconds(event);

    // copy result from device to host
    CL_CHECK(OpenCLUtils::getQueue()->enqueueReadBuffer(U, CL_TRUE, 0, sizeof(float) * U_host.getData()->size(), U_host.getData()->data(), 0, 0));
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
	kernel->setArg<int>(3, H_host.getNx()*sizeof(float));
	kernel->setArg<cl::Buffer>(4, U);
	kernel->setArg<int>(5, U_host.getNx()*sizeof(float));
	kernel->setArg<cl::Buffer>(6, V);
	kernel->setArg<int>(7, V_host.getNx()*sizeof(float));
	kernel->setArg<cl::Buffer>(8, eta);
	kernel->setArg<int>(9, eta_host.getNx()*sizeof(float));
	kernel->setArg<float>(10, currTime);

    // execute kernel
    cl::Event event;
    CL_CHECK(OpenCLUtils::getQueue()->enqueueNDRangeKernel(
                 *kernel, cl::NullRange, global2DWorkSize(nx+1, ny+1, WGNX, WGNY), cl::NDRange(WGNX, WGNY), 0, &event));
    CL_CHECK(event.wait());
    if (profInfo)
        profInfo->time_computeV = OpenCLUtils::elapsedMilliseconds(event);

    // copy result from device to host
    CL_CHECK(OpenCLUtils::getQueue()->enqueueReadBuffer(V, CL_TRUE, 0, sizeof(float) * V_host.getData()->size(), V_host.getData()->data(), 0, 0));
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
    kernel->setArg<int>(2, U_host.getNx()*sizeof(float));
    kernel->setArg<cl::Buffer>(3, V);
    kernel->setArg<int>(4, V_host.getNx()*sizeof(float));
    kernel->setArg<cl::Buffer>(5, eta);
    kernel->setArg<int>(6, eta_host.getNx()*sizeof(float));

    // execute kernel
    cl::Event event;
    CL_CHECK(OpenCLUtils::getQueue()->enqueueNDRangeKernel(
                 *kernel, cl::NullRange, global2DWorkSize(nx, ny, WGNX, WGNY), cl::NDRange(WGNX, WGNY), 0, &event));
    CL_CHECK(event.wait());
    if (profInfo)
        profInfo->time_computeEta = OpenCLUtils::elapsedMilliseconds(event);

    // copy result from device to host
    CL_CHECK(OpenCLUtils::getQueue()->enqueueReadBuffer(eta, CL_TRUE, 0, sizeof(float) * eta_host.getData()->size(), eta_host.getData()->data(), 0, 0));
}

Simulator::Simulator(const OptionsPtr &options, const InitCondPtr &initCond)
    : SimBase(options, initCond)
    , pimpl(new SimulatorImpl)
{
}

Simulator::~Simulator()
{
}

bool Simulator::init()
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

double Simulator::currTime() const
{
    return pimpl->currTime;
}

double Simulator::maxTime() const
{
    return pimpl->maxTime;
}

float Simulator::deltaTime() const
{
    return pimpl->dt;
}

bool Simulator::execNextStep(ProfileInfo *profInfo)
{
	// check if a time-bounded simulation is exhausted
	if ((options()->duration() >= 0) && (currTime() >= maxTime()))
		return false;

    // compute U
    pimpl->computeU(options(), initCond(), profInfo);

    // compute V
    pimpl->computeV(options(), initCond(), profInfo);

    // compute eta
    pimpl->computeEta(options(), initCond(), profInfo);

    pimpl->currTime += pimpl->dt; // advance simulation time

    return true;
}

Field2D Simulator::H() const
{
    return pimpl->H_host;
}

Field2D Simulator::U() const
{
    return pimpl->U_host;
}

Field2D Simulator::V() const
{
    return pimpl->V_host;
}

Field2D Simulator::eta() const
{
    return pimpl->eta_host;
}

float Simulator::f() const
{
    return pimpl->f;
}

float Simulator::r() const
{
    return pimpl->r;
}

void Simulator::printStatus() const
{
    cout << "Simulator::_printStatus(); options: " << *options() << endl;
}
