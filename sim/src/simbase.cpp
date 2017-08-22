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

OptionsPtr SimBase::options() const
{
    return pimpl->options;
}

InitCondPtr SimBase::initCond() const
{
    return pimpl->initCond;
}
