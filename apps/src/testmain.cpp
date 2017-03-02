#include "manager.h"
#include "programoptions.h"
#include "simulator.h"
#include "oclutils.h"
#include <vector>
#include <iostream>
#include <ctime>

#undef NDEBUG
#define NDEBUG

using namespace std;

static void processResults(const Field2D &eta, int step, double currTime, double maxTime, const OptionsPtr &options)
{
    cout << "processResults(): eta.data()->size(): " << eta.data()->size() << ", step: " << step << ", currTime: " << currTime
         << ", maxTime: " << maxTime << "; options: " << *options << ((currTime >= maxTime) ? "; (final results!)" : "") << endl;
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
    int step = 0;
#if 0 // Option 1: Run until max simulated time
    processResults(mgr.eta(), -1, mgr.sim()->currTime(), mgr.sim()->maxTime(), mgr.options());
    while (mgr.execNextStep()) {
#else // Option 2: Run until max wall time
    const time_t startTime = time(0);
    processResults(mgr.eta(), -1, mgr.sim()->currTime(), mgr.sim()->maxTime(), mgr.options());
    while (((mgr.options()->wallDuration() < 0) || (difftime(time(0), startTime) < mgr.options()->wallDuration()))
           && mgr.execNextStep()) {
#endif
        processResults(mgr.eta(), step, mgr.sim()->currTime(), mgr.sim()->maxTime(), mgr.options());
        step++;
    }

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
