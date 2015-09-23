#include "manager.h"
#include "programoptions.h"
#include "simulator.h"
#include <stdexcept>
#include <vector>

using namespace std;

// Initializes the manager.
void Manager::init(int argc, char *argv[])
{
    if (initialized_)
        throw runtime_error("manager already initialized");

    options_.reset(new ProgramOptions);
    if (!options_->parse(argc, argv)) {
        cerr << options_->message() << endl;
        return;
    }

    initialized_ = true;
}

bool Manager::initialized()
{
    return initialized_;
}

// Returns the singleton instance (thread-safe in C++11).
Manager &Manager::instance()
{
    if (!initialized_)
        throw runtime_error("manager not initialized");

    static Manager mgr;
    return mgr;
}

OptionsPtr Manager::options() const
{
    return options_;
}

SimPtr Manager::sim() const
{
    return sim_;
}

// Initializes a new simulation run (aborting one that is already in progress, if necessary).
void Manager::initSim()
{
    sim_->init();
}

// Returns the next simulation step.
int Manager::nextStep() const
{
    return sim_->nextStep();
}

// Returns the final simulation step.
int Manager::finalStep() const
{
    return sim_->finalStep();
}

// Advances the simulation one time step.
void Manager::execNextStep()
{
    // process next simulation step ... TBD
    sim_->execNextStep();
}

// Returns simulation results at the current step.
vector<float> Manager::results() const
{
    return vector<float>();
}

Manager::Manager()
    : sim_(new Simulator(options_))
{
}

bool Manager::initialized_ = false;
OptionsPtr Manager::options_;
