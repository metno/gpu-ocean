#include "manager.h"
#include "programoptions.h"
#include "simulator.h"
#include <vector>
#include <iostream>

using namespace std;

static void processResults(const vector<float> &results, int step, int final_step, ProgramOptions *options)
{
    cout << "processResults(): results.size(): " << results.size() << ", step: " << step << ", final_step: " << final_step
         << "; options: " << *options << ((step > final_step) ? "; (final results!)" : "") << endl;
}

int main(int argc, char *argv[])
{
    // *** Phase 1: Initialize manager
    Manager::init(argc, argv);
    if (!Manager::initialized())
        return 0;

    Manager &mgr = Manager::instance();


    // *** Phase 2: Initialize a new simulation run
    mgr.initSim();


    // *** Phase 3: Run simulation and process results at each step
    while (mgr.nextStep() <= mgr.finalStep()) {
        processResults(mgr.results(), mgr.nextStep(), mgr.finalStep(), mgr.options());
        mgr.execNextStep();
    }


    // *** Phase 4: Process final simulation results
    processResults(mgr.results(), mgr.nextStep(), mgr.finalStep(), mgr.options());

    cout << "done\n";

    // get pointer to simulator object
    SimPtr sim = mgr.sim();
    sim->printStatus();

    return 0;
}
