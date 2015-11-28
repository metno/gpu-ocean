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
    bool isInit;
    SimBaseImpl(const OptionsPtr &, const InitCondPtr &);
};

SimBase::SimBaseImpl::SimBaseImpl(const OptionsPtr &options, const InitCondPtr &initCond)
    : options(options)
    , initCond(initCond)
    , isInit(false)
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

void SimBase::assertInitialized() const
{
    if (!pimpl->isInit)
        throw runtime_error("SimBase: not initialized");
}

/**
 * Initializes the simulator. This includes resetting the current and final step values.
 * @return Initialization status (true iff initialization was successful)
 */
bool SimBase::init()
{
    return pimpl->isInit = _init();
}

/**
 * Returns the current simulation time in seconds.
 */
double SimBase::currTime() const
{
    assertInitialized();
    return _currTime();
}

/**
 * Returns the maximum simulation time in seconds.
 */
double SimBase::maxTime() const
{
    assertInitialized();
    return _maxTime();
}

/**
 * Returns the time (in seconds) by which to advance the simulation in each step.
 */
float SimBase::deltaTime() const
{
    assertInitialized();
    return _deltaTime();
}

/**
 * Executes the next simulation step and returns true if the simulation is not time-bounded
 * or the current simulation time is still less than the maximum simulation time.
 * Otherwise, the function returns false without executing the next simulation step.
 * Note: The simulation is time-bounded iff the program option 'duration' is non-negative.
 * @return true iff a time-bounded simulation was exhausted
 */
bool SimBase::execNextStep()
{
    assertInitialized();

    // check if a time-bounded simulation is exhausted
    if ((pimpl->options->duration() >= 0) && (_currTime() >= _maxTime()))
        return false;

    _execNextStep();
    return true;
}

/**
 * Returns U at the current simulation time.
 */
FieldInfo SimBase::U() const
{
    assertInitialized();
    return _U();
}

/**
 * Returns V at the current simulation time.
 */
FieldInfo SimBase::V() const
{
    assertInitialized();
    return _V();
}

/**
 * Returns eta at the current simulation time.
 */
FieldInfo SimBase::eta() const
{
    assertInitialized();
    return _eta();
}

/**
 * Returns the friction.
 */
float SimBase::F() const
{
    assertInitialized();
    return _F();
}

/**
 * Returns the Coriolis effect.
 */
float SimBase::R() const
{
    assertInitialized();
    return _R();
}

/**
 * Prints status.
 */
void SimBase::printStatus() const
{
    assertInitialized();
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
