#include "testsim.h"
#include "oclutils.h"
#include "boost/format.hpp"
#include <iostream>

#undef NDEBUG
//#define NDEBUG

#define EXECFULL // execute the kernel that does an actual matrix multiplication
#define EXECNOOP // execute the kernel that does nothing

using namespace std;

struct TestSim::TestSimImpl
{
    int nextStep;
    int finalStep;
    int size;
    TestSimImpl(int);
};

TestSim::TestSimImpl::TestSimImpl(int size)
    : nextStep(-1)
    , finalStep(-1)
    , size(size)
{
    if (size <= 0)
        throw runtime_error((boost::format("size (%1%) <= 0") % size).str());
}

TestSim::TestSim(const OptionsPtr &options, const InitCondPtr &initCond)
    : SimBase(options, initCond)
    , pimpl(new TestSimImpl(options->nx())) // use nx for matrix size
{
}

TestSim::~TestSim()
{
}

bool TestSim::_init()
{
    pimpl->nextStep = 0;
    pimpl->finalStep = 10; // run 10 "simulation" steps

    // initialize OpenCL structures
    vector<pair<string, string> > sources;
#ifdef EXECFULL
    sources.push_back(make_pair("MatMul", "matmul.cl"));
#endif
#ifdef EXECNOOP
    sources.push_back(make_pair("MatMulNoop", "matmul_noop.cl"));
#endif
    OpenCLUtils::init(
                sources,
                (boost::format("-D MATRIX_SIZE=%d") % pimpl->size).str(),
                options()->cpu() ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU);

    return true;
}

int TestSim::_nextStep() const
{
    return pimpl->nextStep;
}

int TestSim::_finalStep() const
{
    return pimpl->finalStep;
}

/**
 * Creates two NxN input matrices A and B with random values.
 * @param size Input: The size of N
 * @param a Output: The A matrix with random values
 * @param b Output: The B matrix with random values
 */
static void createInputMatrices(size_t size, vector<float> &a, vector<float> &b)
{
    srand48(time(0));
    a.resize(size * size);
    b.resize(size * size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            a[j * size + i] = (lrand48() / float(RAND_MAX)) * 100;
            b[j * size + i] = (lrand48() / float(RAND_MAX)) * 100;
        }
    }
}

void TestSim::_execNextStep()
{
    if (pimpl->nextStep > pimpl->finalStep)
        throw runtime_error((boost::format("error: next_step_ (%1%) > final_step_ (%2%)") % pimpl->nextStep % pimpl->finalStep).str());


    cl_int error = CL_SUCCESS;


    // --- BEGIN set up kernel arguments ---------------------------

    // generate input matrices with random values
    vector<float> ha;
    vector<float> hb;
    createInputMatrices(pimpl->size, ha, hb);

    // create buffers for input matrices
    cl::Buffer a = cl::Buffer(
                *OpenCLUtils::getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * pimpl->size * pimpl->size, ha.data(), &error);
    CL_CHECK(error);
    cl::Buffer b = cl::Buffer(
                *OpenCLUtils::getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * pimpl->size * pimpl->size, hb.data(), &error);
    CL_CHECK(error);

#ifdef EXECFULL
    // create buffer for output matrix
    cl::Buffer ab_full(*OpenCLUtils::getContext(), CL_MEM_WRITE_ONLY, sizeof(float) * pimpl->size * pimpl->size, 0, &error);
    CL_CHECK(error);

    // set kernel args
    OpenCLUtils::getKernel("MatMul")->setArg<cl::Buffer>(0, a);
    OpenCLUtils::getKernel("MatMul")->setArg<cl::Buffer>(1, b);
    OpenCLUtils::getKernel("MatMul")->setArg<cl::Buffer>(2, ab_full);
#endif
#ifdef EXECNOOP
    // create buffer for output matrix
    cl::Buffer ab_noop(*OpenCLUtils::getContext(), CL_MEM_WRITE_ONLY, sizeof(float) * pimpl->size * pimpl->size, 0, &error);
    CL_CHECK(error);

    // set kernel args
    OpenCLUtils::getKernel("MatMulNoop")->setArg<cl::Buffer>(0, a);
    OpenCLUtils::getKernel("MatMulNoop")->setArg<cl::Buffer>(1, b);
    OpenCLUtils::getKernel("MatMulNoop")->setArg<cl::Buffer>(2, ab_noop);
#endif
    // --- END set up kernel arguments ---------------------------


    // --- BEGIN execute kernels ---------------------------
    cl::Event event;

#ifdef EXECFULL
    // execute full multiplication
    CL_CHECK(OpenCLUtils::getQueue()->enqueueNDRangeKernel(
                 *OpenCLUtils::getKernel("MatMul"), cl::NullRange, cl::NDRange(pimpl->size, pimpl->size), cl::NullRange, 0, &event));
    CL_CHECK(event.wait());
    // examine contents of ab_full ... TBD
    const float msecs_full = OpenCLUtils::elapsedMilliseconds(event);
#else
    const float msecs_full = -1;
#endif

#ifdef EXECNOOP
    // execute noop (only copying memory between host and device)
    CL_CHECK(OpenCLUtils::getQueue()->enqueueNDRangeKernel(
                 *OpenCLUtils::getKernel("MatMulNoop"), cl::NullRange, cl::NDRange(pimpl->size, pimpl->size), cl::NullRange, 0, &event));
    CL_CHECK(event.wait());
    // examine contents of ab_noop ... TBD
    const float msecs_noop = OpenCLUtils::elapsedMilliseconds(event);
#else
    const float msecs_noop = -1;
#endif
    // --- END execute kernels ---------------------------

#ifndef NDEBUG
    cout << (boost::format("matrix size: %4d x %4d;   msecs: %12.6f;   noop msecs: %12.6f   (platform: %s; device: %s)")
             % pimpl->size % pimpl->size % msecs_full % msecs_noop % OpenCLUtils::getPlatformName() % OpenCLUtils::getDeviceName()) << endl;
#endif

    pimpl->nextStep++;
}

vector<float> TestSim::_results() const
{
    return vector<float>(); // ### for now
}

void TestSim::_printStatus() const
{
    cout << "TestSim::printStatus(); options: " << *options() << endl;
}
