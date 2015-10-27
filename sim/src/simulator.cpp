#include "simulator.h"
#include "oclutils.h"
#include <boost/format.hpp>
#include <iostream>

using namespace std;

struct Simulator::SimulatorImpl
{
    vector<float> Hr_u; // H reconstructed in x-dimension
    vector<float> Hr_v; // H reconstructed in y-dimension

    SimulatorImpl();
    void reconstructH(const InitCondPtr &);
};

Simulator::SimulatorImpl::SimulatorImpl()
{
}

/**
 * Reconstructs H, i.e. computes Hr_u and Hr_v from initCond->H().
 */
void Simulator::SimulatorImpl::reconstructH(const InitCondPtr &initCond)
{
    cerr << "reconstructing H ...\n";

    cl_int error = CL_SUCCESS;

    // create buffers for input field
    cl::Buffer H = cl::Buffer(
                *OpenCLUtils::getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(float) * initCond->H().data->size(), initCond->H().data->data(), &error);
    CL_CHECK(error);

    // set up kernel arguments ...
    // ... in:
    OpenCLUtils::getKernel("MatMulNoop")->setArg<cl::Buffer>(0, H);
    // ... out:
    // pointers to Hr_u and Hr_v in device memory ... TBD

    // execute kernel (computes Hr_u and Hr_v in device memory and returns pointers)
    cl::Event event;
    CL_CHECK(OpenCLUtils::getQueue()->enqueueNDRangeKernel(
                 *OpenCLUtils::getKernel("ReconstructH"), cl::NullRange, cl::NDRange(initCond->H().nx, initCond->H().ny), cl::NullRange, 0, &event));
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
    OpenCLUtils::init(
                sources,
                (boost::format("-D NX=%d -D NY=%d") % options()->nx() % options()->ny()).str(),
                options()->cpu() ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU);

    // reconstruct H
    pimpl->reconstructH(initCond());

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
