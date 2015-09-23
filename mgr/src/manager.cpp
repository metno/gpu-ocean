#include "manager.h"
#include "programoptions.h"
#include "../../sim/src/mockup_sim.h"
#include <stdexcept>
#include <vector>

using namespace std;

struct Manager::ManagerImpl
{
    SimPtr sim;
    ManagerImpl(const OptionsPtr &);
};

Manager::ManagerImpl::ManagerImpl(const OptionsPtr &options)
    : sim(new Simulator(options))
{
}

// Initializes the manager.
void Manager::init(int argc, char *argv[])
{
    if (INITIALIZED)
        throw runtime_error("manager already initialized");

    OPTIONS.reset(new ProgramOptions);
    if (!OPTIONS->parse(argc, argv)) {
        cerr << OPTIONS->message() << endl;
        return;
    }

    INITIALIZED = true;
}

// Returns true iff the manager is initialized.
bool Manager::initialized()
{
    return INITIALIZED;
}

// Returns the singleton instance (thread-safe in C++11).
Manager &Manager::instance()
{
    if (!INITIALIZED)
        throw runtime_error("manager not initialized");

    static Manager MGR;
    return MGR;
}

// Returns the program options object.
OptionsPtr Manager::options() const
{
    return OPTIONS;
}

// Returns the simulator object.
SimPtr Manager::sim() const
{
    return pimpl->sim;
}

// Initializes a new simulation run (aborting one that is already in progress, if necessary).
void Manager::initSim()
{
    pimpl->sim->init();
}

// Returns the next simulation step.
int Manager::nextStep() const
{
    return pimpl->sim->nextStep();
}

// Returns the final simulation step.
int Manager::finalStep() const
{
    return pimpl->sim->finalStep();
}

// Advances the simulation one time step.
void Manager::execNextStep()
{
    // process next simulation step ... TBD
    pimpl->sim->execNextStep();
}

// Returns simulation results at the current step.
vector<float> Manager::results() const
{
    return vector<float>();
}

Manager::Manager()
    : pimpl(new ManagerImpl(OPTIONS))
{
}

bool Manager::INITIALIZED = false;
OptionsPtr Manager::OPTIONS;
