#include "simulator.h"
#include "programoptions.h"
#include <boost/format.hpp>
#include <iostream>
#include <stdexcept>

using namespace std;

Simulator::Simulator(ProgramOptions *options)
    : options_(options)
    , next_step_(-1)
    , final_step_(-1)
{
}

Simulator::~Simulator()
{
}

void Simulator::init()
{
    next_step_ = 0;

    //final_step_ = stepsFromDuraton(options_->duration());
    final_step_ = 4; // ### for now
}

int Simulator::nextStep() const
{
    return next_step_;
}

int Simulator::finalStep() const
{
    return final_step_;
}

void Simulator::execNextStep()
{
    if (next_step_ > final_step_)
        throw runtime_error((boost::format("error: next_step_ (%1%) > final_step_ (%2%)") % next_step_ % final_step_).str());

    cout << "Simulator::execNextStep() ...\n";
    next_step_++;
}

void Simulator::printStatus() const
{
    cout << "Simulator::printStatus(); options: " << *options_ << endl;
}
