#include "dummysim.h"
#include "matmulcl.h"
#include "boost/format.hpp"

using namespace std;

struct DummySim::DummySimImpl
{
    int nextStep;
    int finalStep;
    DummySimImpl();
};

DummySim::DummySimImpl::DummySimImpl()
    : nextStep(-1)
    , finalStep(-1)
{
}

DummySim::DummySim(const OptionsPtr &options, const InitCondPtr &initCond)
    : SimBase(options, initCond)
    , pimpl(new DummySimImpl)
{
}

DummySim::~DummySim()
{
}

void DummySim::init()
{
    pimpl->nextStep = 0;
    pimpl->finalStep = 0; // this computation consists of a single step only
}

int DummySim::nextStep() const
{
    return pimpl->nextStep;
}

int DummySim::finalStep() const
{
    return pimpl->finalStep;
}

void DummySim::execNextStep()
{
    if (pimpl->nextStep > pimpl->finalStep)
        throw runtime_error((boost::format("error: next_step_ (%1%) > final_step_ (%2%)") % pimpl->nextStep % pimpl->finalStep).str());

    const bool execOnCpu = true; // for now
    const int size = options()->nx(); // for now
    matmul(size, execOnCpu); // multiply random square matrices

    pimpl->nextStep++;
}

vector<float> DummySim::results() const
{
    return vector<float>(); // ### for now
}

void DummySim::printStatus() const
{
    cout << "DummySim::printStatus(); options: " << *options() << endl;
}
