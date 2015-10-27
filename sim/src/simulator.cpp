#include "simulator.h"
#include <boost/format.hpp>
#include <iostream>

using namespace std;

struct Simulator::SimulatorImpl
{
    // ### nothing here yet
    SimulatorImpl();
};

Simulator::SimulatorImpl::SimulatorImpl()
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

bool Simulator::_init()
{
    // ### nothing here yet
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
    // ### nothing here yet
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
