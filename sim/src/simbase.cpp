#include "simbase.h"

#include "programoptions.h"
#include "initconditions.h"
#include <iostream>
#include <stdexcept>

using namespace std;

struct SimBase::SimBaseImpl
{
    OptionsPtr options;
    InitCondPtr initCond;
    bool initCalled;
    SimBaseImpl(const OptionsPtr &, const InitCondPtr &);
};

SimBase::SimBaseImpl::SimBaseImpl(const OptionsPtr &options, const InitCondPtr &initCond)
    : options(options)
    , initCond(initCond)
    , initCalled(false)
{
}

/**
 * Constructor.
 * @param options Input: Options.
 * @param initCond Input: Initial conditions.
 */
SimBase::SimBase(const OptionsPtr &options, const InitCondPtr &initCond)
    : pimpl(new SimBaseImpl(options, initCond))
{
}

SimBase::~SimBase()
{
}

void SimBase::assertInitCalled() const
{
    if (!pimpl->initCalled)
        throw runtime_error("SimBase: init() not called");
}

void SimBase::assertInitNotCalled() const
{
    if (pimpl->initCalled)
        throw runtime_error("SimBase: init() already called");
}

/**
 * Initializes the simulator. This includes resetting the current and final step values.
 */
void SimBase::init()
{
    assertInitNotCalled();
    _init();
    pimpl->initCalled = true;
}

/**
 * Returns the current step value.
 */
int SimBase::nextStep() const
{
    assertInitCalled();
    return _nextStep();
}

/**
 * Returns the final step value.
 */
int SimBase::finalStep() const
{
    assertInitCalled();
    return _finalStep();
}

/**
 * Executes the next simulation step and increases the step value by one.
 * Throws std::runtime_error if the current step value exceeds the final one.
 */
void SimBase::execNextStep()
{
    assertInitCalled();
    _execNextStep();
}

/**
 * Returns the simulation results at the current step.
 */
std::vector<float> SimBase::results() const
{
    assertInitCalled();
    return _results();
}

/**
 * Prints status.
 */
void SimBase::printStatus() const
{
    assertInitCalled();
    _printStatus();
}

/**
 * Returns the options.
 */
OptionsPtr SimBase::options() const
{
    return pimpl->options;
}

/**
 * Returns the initial conditions.
 */
InitCondPtr SimBase::initCond() const
{
    return pimpl->initCond;
}
