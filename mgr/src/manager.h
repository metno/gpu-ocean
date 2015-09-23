#ifndef MANAGER_H
#define MANAGER_H

#include "programoptions.h"
#include "initconditions.h"
#include <iostream>
#include <stdexcept>
#include <vector>
#include "simulator.h"

// This class manages the high-level aspects of a simulation.
class Manager
{
public:
    static void init(int, char *[]);
    static bool initialized();
    static Manager &instance();
    OptionsPtr options() const;
    SimPtr sim() const;
    void initSim();
    int nextStep() const;
    int finalStep() const;
    void execNextStep();
    std::vector<float> results() const;
private:
    static bool INITIALIZED;
    static OptionsPtr OPTIONS;
    static InitCondPtr INIT_COND;
    struct ManagerImpl;
    ManagerImpl *pimpl;
    Manager();
    Manager(const Manager &);
    Manager &operator=(const Manager &);
};

#endif // MANAGER_H
