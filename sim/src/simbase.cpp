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

void SimBase::assertNotInitialized() const
{
    if (pimpl->isInit)
        throw runtime_error("SimBase: already initialized");
}

/**
 * Initializes the simulator. This includes resetting the current and final step values.
 * @return Initialization status (true iff initialization was successful)
 */
bool SimBase::init()
{
    assertNotInitialized();
    return pimpl->isInit = _init();
}

/**
 * Returns the current step value.
 */
int SimBase::nextStep() const
{
    assertInitialized();
    return _nextStep();
}

/**
 * Returns the final step value.
 */
int SimBase::finalStep() const
{
    assertInitialized();
    return _finalStep();
}

/**
 * Executes the next simulation step and increases the step value by one.
 * Throws std::runtime_error if the current step value exceeds the final one.
 */
void SimBase::execNextStep()
{
    assertInitialized();
    _execNextStep();
}

/**
 * Returns the simulation results at the current step.
 */
std::vector<float> SimBase::results() const
{
    assertInitialized();
    return _results();
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
