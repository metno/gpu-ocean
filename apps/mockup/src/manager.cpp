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

ProgramOptions *Manager::options() const
{
    return options_;
}

// Initializes a new simulation run (aborting one that is already in progress, if necessary).
void Manager::initSim()
{
    next_step_ = 0;
    //final_step_ = stepsFromDuraton(options_->duration());
    final_step_ = 9; // ### for now
}

// Returns the next simulation step.
int Manager::nextStep() const
{
    return next_step_;
}

// Returns the final simulation step.
int Manager::finalStep() const
{
    return final_step_;
}

// Advances the simulation one time step.
void Manager::execNextStep()
{
    // process next simulation step ... TBD
    next_step_++;
}

// Returns simulation results at the current step.
vector<float> Manager::results() const
{
    return vector<float>();
}

Manager::Manager()
    : next_step_(0)
    , final_step_(-1)
{
    // create simulator object(s) based on options_ ... TBD
}

bool Manager::initialized_ = false;
ProgramOptions *Manager::options_ = 0;
