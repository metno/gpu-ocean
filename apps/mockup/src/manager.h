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
    void simInit();
    bool simInProgress() const;
    void simStep();
    std::vector<float> simResults() const;
private:
    static bool initialized_;
    static ProgramOptions *options_;
    Manager();
    Manager(const Manager &);
    Manager &operator=(const Manager &);
};

#endif // MANAGER_H
