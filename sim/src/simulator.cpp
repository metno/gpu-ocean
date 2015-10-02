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

void Simulator::init()
{
    pimpl->nextStep = 0;

    //pimpl->finalStep = calculate from options()->duration();
    pimpl->finalStep = 4; // ### for now
}

int Simulator::nextStep() const
{
    return pimpl->nextStep;
}

int Simulator::finalStep() const
{
    return pimpl->finalStep;
}

void Simulator::execNextStep()
{
    if (pimpl->nextStep > pimpl->finalStep)
        throw runtime_error((boost::format("error: next_step_ (%1%) > final_step_ (%2%)") % pimpl->nextStep % pimpl->finalStep).str());

    // executing next step ... TBD

    pimpl->nextStep++;
}

vector<float> Simulator::results() const
{
    return vector<float>(); // ### for now
}

void Simulator::printStatus() const
{
    cout << "Simulator::printStatus(); options: " << *options() << endl;
}
