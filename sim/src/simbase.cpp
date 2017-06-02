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

bool SimBase::init()
{
    return pimpl->isInit = _init();
}

double SimBase::currTime() const
{
    assertInitialized();
    return _currTime();
}

double SimBase::maxTime() const
{
    assertInitialized();
    return _maxTime();
}

float SimBase::deltaTime() const
{
    assertInitialized();
    return _deltaTime();
}

bool SimBase::execNextStep(ProfileInfo *profInfo)
{
    assertInitialized();

    // check if a time-bounded simulation is exhausted
    if ((pimpl->options->duration() >= 0) && (_currTime() >= _maxTime()))
        return false;

    _execNextStep(profInfo);
    return true;
}

Field2D SimBase::H() const
{
    assertInitialized();
    return _H();
}

Field2D SimBase::eta() const
{
    assertInitialized();
    return _eta();
}

Field2D SimBase::U() const
{
    assertInitialized();
    return _U();
}

Field2D SimBase::V() const
{
    assertInitialized();
    return _V();
}

float SimBase::f() const
{
    assertInitialized();
    return _f();
}

float SimBase::r() const
{
    assertInitialized();
    return _r();
}

void SimBase::printStatus() const
{
    assertInitialized();
    _printStatus();
}

OptionsPtr SimBase::options() const
{
    return pimpl->options;
}

InitCondPtr SimBase::initCond() const
{
    return pimpl->initCond;
}
