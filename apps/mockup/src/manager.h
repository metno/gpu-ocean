#ifndef MANAGER_H
#define MANAGER_H

#include <boost/shared_ptr.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>

class ProgramOptions;
class Simulator;

typedef boost::shared_ptr<Simulator> SimPtr;

// This class manages the high-level aspects of a simulation.
class Manager
{
public:
    static void init(int, char *[]);
    static bool initialized();
    static Manager &instance();
    ProgramOptions *options() const;
    void initSim();
    int nextStep() const;
    int finalStep() const;
    void execNextStep();
    std::vector<float> results() const;
    SimPtr sim() const;
private:
    static bool initialized_;
    static ProgramOptions *options_;
    SimPtr sim_;
    Manager();
    Manager(const Manager &);
    Manager &operator=(const Manager &);
};

#endif // MANAGER_H
