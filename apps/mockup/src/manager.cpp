#include "manager.h"
#include "programoptions.h"
#include <stdexcept>
#include <vector>

using namespace std;

// Initializes the manager.
void Manager::init(int argc, char *argv[])
{
    if (initialized_)
        throw runtime_error("manager already initialized");

    options_ = new ProgramOptions;
    if (!options_->parse(argc, argv)) {
        cerr << options_->message() << endl;
        return;
    }

    cout << "successfully parsed program options:\n";
    cout << "nx: " << options_->nx() << endl;
    cout << "ny: " << options_->ny() << endl;
    cout << "width: " << options_->width() << endl;
    cout << "height: " << options_->height() << endl;
    cout << "duration: " << options_->duration() << endl;

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
    // create simulator object(s) based on options_ ... TBD
}

bool Manager::initialized_ = false;
ProgramOptions *Manager::options_ = 0;
