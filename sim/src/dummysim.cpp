#include "dummysim.h"
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

    //pimpl->finalStep = calculate from options()->duration();
    pimpl->finalStep = 4; // ### for now
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

    // executing next step ... TBD

    pimpl->nextStep++;
}

vector<float> DummySim::results() const
{
    return vector<float>(); // ### for now
}

void DummySim::printStatus() const
{
    cout << "DummySim::printStatus(); options: " << options() << endl;
}
