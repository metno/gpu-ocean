#ifndef MANAGER_H
#define MANAGER_H

#include "simulator.h"
#include "programoptions.h"
#include <iostream>
#include <stdexcept>
#include <vector>

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
    static bool initialized_;
    static OptionsPtr options_;
    SimPtr sim_;
    Manager();
    Manager(const Manager &);
    Manager &operator=(const Manager &);
};

#endif // MANAGER_H
