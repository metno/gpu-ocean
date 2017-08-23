#include "manager.h"
#include "programoptions.h"
#include "simulator.h"
#include "oclutils.h"
#include <vector>
#include <iostream>
#include <ctime>
#include <cassert>

#undef NDEBUG
#define NDEBUG

using namespace std;

static float computeMass(const Field2D &H, const Field2D &eta)
{
    assert(H.getNx() == eta.getNx());
    assert(H.getNy() == eta.getNy());
    assert(H.getNx() > 2);
    float mass = 0;
    for (int j = 1; j < H.getNy() - 1; ++j)
        for (int i = 1; i < H.getNx() - 1; ++i)
            mass += (H(i, j) + eta(i, j));
    return mass;
}

static void processResults(int step, const ProfileInfo *profInfo = 0)
{
#ifndef PROFILE
    return;
#endif

    Manager &mgr = Manager::instance();
    const float mass = step < 0 ? -1 : computeMass(mgr.initConditions()->H(), mgr.eta());
    cout << "step: " << step << ", mass: " << mass << ", currTime: " << mgr.sim()->currTime()
         << ", maxTime: " << mgr.sim()->maxTime();

    if (profInfo)
        cout << ", U: " << profInfo->time_computeU
             << ", V: " << profInfo->time_computeV
             << ", eta: " << profInfo->time_computeEta;

    cout << ((mgr.sim()->currTime() >= mgr.sim()->maxTime()) ? " (final results!)" : "") << endl;
}

static void runUntilMaxSimulatedTime(ProfileInfo *profInfo)
{
    processResults(-1);
    Manager &mgr = Manager::instance();
    int step = 0;
    while (mgr.execNextStep(profInfo)) {
        processResults(step, profInfo);
        step++;
    }
}

static void runUntilMaxWallTime(ProfileInfo *profInfo)
{
    processResults(-1);
    Manager &mgr = Manager::instance();
    int step = 0;
    const time_t startTime = time(0);
    while (((mgr.options()->wallDuration() < 0) || (difftime(time(0), startTime) < mgr.options()->wallDuration()))
           && mgr.execNextStep(profInfo)) {
        processResults(step, profInfo);
        step++;
    }
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

    shared_ptr<ProfileInfo> profInfo;
#ifdef PROFILE
    profInfo.reset(new ProfileInfo);
#endif

#if 0 // Option 1
    runUntilMaxSimulatedTime(profInfo.get());
#else // Option 2
    runUntilMaxWallTime(profInfo.get());
#endif

#ifndef NDEBUG
    cout << "done\n";
#endif

#ifndef NDEBUG
    cout << "available platforms and devices:\n";
    OpenCLUtils::listDevices();
#endif

    return 0;
}
