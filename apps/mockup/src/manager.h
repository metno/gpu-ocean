#ifndef MANAGER_H
#define MANAGER_H

#include <iostream>
#include <stdexcept>
#include <vector>

class ProgramOptions;

// This class manages a simulation run.
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
private:
    static bool initialized_;
    static ProgramOptions *options_;
    int next_step_;
    int final_step_;
    Manager();
    Manager(const Manager &);
    Manager &operator=(const Manager &);
};

#endif // MANAGER_H
