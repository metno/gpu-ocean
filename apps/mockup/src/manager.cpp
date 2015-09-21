#include "manager.h"
#include <stdexcept>
#include <vector>

using namespace std;

// Initializes the manager.
void Manager::init(int argc, char *argv[])
{
    if (initialized_)
        throw runtime_error("manager already initialized");

    // configure using boost::program_options ... TBD
    initialized_ = true;
}

// Returns the singleton instance (thread-safe in C++11).
Manager &Manager::instance()
{
    if (!initialized_)
        throw runtime_error("manager not initialized");

    static Manager mgr;
    return mgr;
}

// Initializes a new simulation run (aborting one that is already in progress, if necessary).
void Manager::simInit()
{
}

// Returns true iff a simulation is in progress.
bool Manager::simInProgress() const
{
    return false;
}

// Advances the simulation one time step.
void Manager::simStep()
{
}

vector<float> Manager::simResults() const
{
    return vector<float>();
}

Manager::Manager()
{
    // create simulator object(s) from config ... TBD
}

bool Manager::initialized_ = false;
