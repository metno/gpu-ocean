#ifndef MANAGER_H
#define MANAGER_H

#include "programoptions.h"
#include "initconditions.h"
#include "simulator.h"
#include "profile.h"
#include <iostream>
#include <stdexcept>
#include <vector>

/**
 * This class manages the high-level aspects of a simulation.
 */
class Manager
{
public:

    /**
     * Initializes the manager.
     * @param argc: Number of arguments passed to ProgramOptions::parse().
     * @param argv: Argument vector passed to ProgramOptions::parse().
     */
    static void init(int argc, char *argv[]);

    /**
     * Returns true iff the manager is initialized.
     */
    static bool initialized();

    /**
     * Returns the singleton instance.
     */
    static Manager &instance();

    /**
     * Returns the program options object.
     */
    OptionsPtr options() const;

    /**
     * Returns the initial conditions object.
     */
    InitCondPtr initConditions() const;

    /**
     * Returns the simulator object.
     */
    SimBasePtr sim() const;

    /**
     * Initializes a new simulation run (aborting one that is already in progress, if necessary).
     */
    void initSim();

    /**
     * Executes the next simulation step, advances the simulation time, and writes the current state to file if requested.
     * @param profInfo: If non-null, structure in which profiling is written (if applicable).
     * @returns See SimBase::execNextStep().
     */
    bool execNextStep(ProfileInfo *profInfo = 0);

    /**
     * Returns eta (sea surface deviation away from the equilibrium depth) at the current simulation step.
     */
    FieldInfo eta() const;

    /**
     * Returns U (depth averaged velocity in the x direction) at the current simulation step.
     */
    FieldInfo U() const;

    /**
     * Returns V (depth averaged velocity in the y direction) at the current simulation step.
     */
    FieldInfo V() const;

private:
    static bool isInit;
    static OptionsPtr opts;
    static InitCondPtr initCond;
    struct ManagerImpl;
    ManagerImpl *pimpl;
    Manager();
    Manager(const Manager &);
    Manager &operator=(const Manager &);
};

#endif // MANAGER_H
