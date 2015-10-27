#include "manager.h"
#include "programoptions.h"
#include "initconditions.h"
#include "simbase.h"
#include "simulator.h"
#include "testsim.h"
#include <stdexcept>
#include <vector>
#include <iostream>

// comment out the following line to run the real simulator:
//#define TESTSIM

using namespace std;

struct Manager::ManagerImpl
{
    SimBasePtr sim;
    ManagerImpl(const OptionsPtr &, const InitCondPtr &);
};

Manager::ManagerImpl::ManagerImpl(const OptionsPtr &options, const InitCondPtr &initCond)
#ifdef TESTSIM
    : sim(new TestSim(options, initCond))
#else
    : sim(new Simulator(options, initCond))
#endif
{
}

// Initializes the manager.
void Manager::init(int argc, char *argv[])
{
    if (isInit)
        throw runtime_error("manager already initialized");

    opts.reset(new ProgramOptions);
    if (!opts->parse(argc, argv)) {
        cerr << opts->message() << endl;
        return;
    }

    initCond.reset(new InitConditions);
    initCond->init(opts);

    isInit = true;
}

// Returns true iff the manager is initialized.
bool Manager::initialized()
{
    return isInit;
}

// Returns the singleton instance (thread-safe in C++11).
Manager &Manager::instance()
{
    if (!isInit)
        throw runtime_error("manager not initialized");

    static Manager mgr;
    return mgr;
}

// Returns the program options object.
OptionsPtr Manager::options() const
{
    return opts;
}

// Returns the simulator object.
SimBasePtr Manager::sim() const
{
    return pimpl->sim;
}

// Initializes a new simulation run (aborting one that is already in progress, if necessary).
void Manager::initSim()
{
    pimpl->sim->init();
}

// Executes the next simulation step and advances the simulation time.
bool Manager::execNextStep()
{
    return pimpl->sim->execNextStep();
}

// Returns U at the current simulation step.
FieldInfo Manager::U() const
{
    return pimpl->sim->U();
}

// Returns V at the current simulation step.
FieldInfo Manager::V() const
{
    return pimpl->sim->V();
}

// Returns eta at the current simulation step.
FieldInfo Manager::eta() const
{
    return pimpl->sim->eta();
}

Manager::Manager()
    : pimpl(new ManagerImpl(opts, initCond))
{
}

bool Manager::isInit = false;
OptionsPtr Manager::opts;
InitCondPtr Manager::initCond;
