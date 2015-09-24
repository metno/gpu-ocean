#include "manager.h"
#include "programoptions.h"
#include "simulator.h"
#include <vector>
#include <iostream>

using namespace std;

static void processResults(const vector<float> &results, int step, int finalStep, const OptionsPtr &options)
{
    cout << "processResults(): results.size(): " << results.size() << ", step: " << step << ", finalStep: " << finalStep
         << "; options: " << *options << ((step > finalStep) ? "; (final results!)" : "") << endl;
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

    // do some OpenCL stuff
    vector<cl_platform_id> oclPlatforms;
    sim->getOCLPlatforms(oclPlatforms);
    if(oclPlatforms.size())
    	sim->countOCLDevices(oclPlatforms[0]);

    return 0;
}
