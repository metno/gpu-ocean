#include <boost/shared_ptr.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace std;

class Simulator
{
};

class Options
{
};

// This class manages a simulation run.
class Manager
{
public:
    // Initializes the manager.
    static void init(int argc, char *argv[])
    {
        if (initialized_)
            throw runtime_error("manager already initialized");

        // configure using boost::program_options ... TBD
        initialized_ = true;
    }

    // Returns the singleton instance (thread-safe in C++11).
    static Manager &instance()
    {
        if (!initialized_)
            throw runtime_error("manager not initialized");

         static Manager mgr;
         return mgr;
    }

    // A dummy function to test initialization behavior.
    void dummy()
    {
        cerr << "Manager::dummy()...\n";
    }

    // Initializes a new simulation run (aborting one that is already in progress, if necessary).
    void simInit()
    {
    }

    // Returns true iff a simulation is in progress.
    bool simInProgress() const
    {
        return false;
    }

    // Advances the simulation one time step.
    void simStep()
    {
    }

    vector<float> simResults() const
    {
        return vector<float>();
    }

private:
    static bool initialized_;

    Manager()
    {
        // create simulator object(s) from config ... TBD
    }

    Manager(const Manager &);
    Manager &operator=(const Manager &);
};

bool Manager::initialized_ = false;


int main(int argc, char *argv[])
{
    // *** Phase 1: Initialize Manager
    Manager::init(argc, argv);

    Manager &mgr = Manager::instance();
    mgr.dummy();


    // *** Phase 2: Initialize a new simulation run
    mgr.simInit();


    // *** Phase 3: Run simulation
    while (mgr.simInProgress())
        mgr.simStep();


    // *** Phase 4: Process results of simulation
    vector<float> results = mgr.simResults();

    return 0;
}
