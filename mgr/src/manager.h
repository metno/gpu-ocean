#ifndef MANAGER_H
#define MANAGER_H

#include "programoptions.h"
#include "initconditions.h"
#include <iostream>
#include <stdexcept>
#include <vector>
#include "simulator.h"

/**
 * This class manages the high-level aspects of a simulation.
 */
class Manager
{
public:

    /**
     * Initializes the manager.
     * @param argc: Number of arguments.
     * @param argv: Argument vector.
     * @note Passing "--help" as one of the arguments prints a documentation of the available arguments on stderr.
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
     * Executes the next simulation step and advances the simulation time.
     */
    bool execNextStep();

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
