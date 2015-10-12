#include "manager.h"
#include "programoptions.h"
#include "simulator.h"
#include "oclutils.h"
#include <vector>
#include <iostream>

#define NDEBUG

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

#ifndef NDEBUG
    // get pointer to simulator object and print status
    SimBasePtr sim = mgr.sim();
    sim->printStatus();
#endif

    // *** Phase 2: Initialize a new simulation run
    mgr.initSim();


    // *** Phase 3: Run simulation and process results at each step
    while (mgr.nextStep() <= mgr.finalStep()) {
        processResults(mgr.results(), mgr.nextStep(), mgr.finalStep(), mgr.options());
        mgr.execNextStep();
    }


    // *** Phase 4: Process final simulation results
    processResults(mgr.results(), mgr.nextStep(), mgr.finalStep(), mgr.options());

#ifndef NDEBUG
    cout << "done\n";
#endif

    // do some OpenCL stuff
    vector<cl::Platform> oclPlatforms;
    OpenCLUtils::getPlatforms(&oclPlatforms);
    if (!oclPlatforms.empty())
        OpenCLUtils::countDevices(oclPlatforms[0]);

#ifndef NDEBUG
    cout << "available platforms and devices:\n";
    OpenCLUtils::listDevices();
#endif

    return 0;
}
