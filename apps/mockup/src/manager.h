#ifndef MANAGER_H
#define MANAGER_H

#include <boost/shared_ptr.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>

// This class manages a simulation run.
class Manager
{
public:
    static void init(int, char *[]);
    static Manager &instance();
    void simInit();
    bool simInProgress() const;
    void simStep();
    std::vector<float> simResults() const;
private:
    static bool initialized_;
    Manager();
    Manager(const Manager &);
    Manager &operator=(const Manager &);
};

#endif // MANAGER_H
