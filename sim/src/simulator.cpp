#include "simulator.h"
#include <boost/format.hpp>
#include <iostream>

using namespace std;

struct Simulator::SimulatorImpl
{
    int nextStep;
    int finalStep;
    SimulatorImpl();
};

Simulator::SimulatorImpl::SimulatorImpl()
    : nextStep(-1)
    , finalStep(-1)
{
}

Simulator::Simulator(const OptionsPtr &options, const InitCondPtr &initCond)
    : SimBase(options, initCond)
    , pimpl(new SimulatorImpl)
{
}

Simulator::~Simulator()
{
}

void Simulator::_init()
{
    pimpl->nextStep = 0;

    //pimpl->finalStep = calculate from options()->duration();
    pimpl->finalStep = 4; // ### for now
}

int Simulator::_nextStep() const
{
    return pimpl->nextStep;
}

int Simulator::_finalStep() const
{
    return pimpl->finalStep;
}

void Simulator::_execNextStep()
{
    if (pimpl->nextStep > pimpl->finalStep)
        throw runtime_error((boost::format("error: next_step_ (%1%) > final_step_ (%2%)") % pimpl->nextStep % pimpl->finalStep).str());

    // executing next step ... TBD

    pimpl->nextStep++;
}

vector<float> Simulator::_results() const
{
    return vector<float>(); // ### for now
}

void Simulator::_printStatus() const
{
    cout << "Simulator::_printStatus(); options: " << *options() << endl;
}
