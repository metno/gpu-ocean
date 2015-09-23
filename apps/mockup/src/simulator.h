#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "programoptions.h"
#include <boost/shared_ptr.hpp>

// This class manages the low-level aspects of a simulation.
class Simulator
{
public:
    Simulator(const OptionsPtr &);
    virtual ~Simulator();
    void init();
    int nextStep() const;
    int finalStep() const;
    void execNextStep();
    void printStatus() const;
private:
    OptionsPtr options_;
    int next_step_;
    int final_step_;
};

typedef boost::shared_ptr<Simulator> SimPtr;

#endif // SIMULATOR_H
