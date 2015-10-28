#include "manager.h"
#include "programoptions.h"
#include "simulator.h"
#include "oclutils.h"
#include <vector>
#include <iostream>
#include <ctime>
#include <cassert>

#define NDEBUG

using namespace std;

static float computeMass(const FieldInfo &H, const FieldInfo &eta)
{
    assert(H.nx == eta.nx);
    assert(H.ny == eta.ny);
    assert(H.nx > 2);
    float mass = 0;
    for (int j = 1; j < H.ny - 1; ++j)
        for (int i = 1; i < H.nx - 1; ++i)
            mass += (H(i, j) + eta(i, j));
    return mass;
}

static void processResults(const FieldInfo &H, const FieldInfo &eta, int step, double currTime, double maxTime, const OptionsPtr &options)
{
    const float mass = computeMass(H, eta);
    cout << "processResults(): H.nx: " << H.nx << ", H.ny: " << H.ny << ", mass: " << mass << ", step: " << step << ", currTime: " << currTime
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
    processResults(mgr.initCond()->H(), mgr.results(), -1, mgr.sim()->currTime(), mgr.sim()->maxTime(), mgr.options());
    while (mgr.execNextStep()) {
#else // Option 2: Run until max wall time
    const time_t startTime = time(0);
    processResults(mgr.initConditions()->H(), mgr.eta(), -1, mgr.sim()->currTime(), mgr.sim()->maxTime(), mgr.options());
    while (((mgr.options()->wallDuration() < 0) || (difftime(time(0), startTime) < mgr.options()->wallDuration()))
           && mgr.execNextStep()) {
#endif
        processResults(mgr.initConditions()->H(), mgr.eta(), step, mgr.sim()->currTime(), mgr.sim()->maxTime(), mgr.options());
        step++;
    }

#ifndef NDEBUG
    cout << "done\n";
#endif

#ifndef NDEBUG
    cout << "available platforms and devices:\n";
    OpenCLUtils::listDevices();
#endif

    return 0;
}
