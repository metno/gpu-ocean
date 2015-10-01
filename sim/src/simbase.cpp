#include "simbase.h"

#include "programoptions.h"
#include "initconditions.h"
#include <iostream>

using namespace std;

struct SimBase::SimBaseImpl
{
    OptionsPtr options;
    InitCondPtr initCond;
    SimBaseImpl(const OptionsPtr &, const InitCondPtr &);
};

SimBase::SimBaseImpl::SimBaseImpl(const OptionsPtr &options, const InitCondPtr &initCond)
    : options(options)
    , initCond(initCond)
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

OptionsPtr SimBase::options() const
{
    return pimpl->options;
}

InitCondPtr SimBase::initCond() const
{
    return pimpl->initCond;
}
