#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "../../mgr/src/programoptions.h"
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
    struct SimulatorImpl;
    SimulatorImpl *pimpl;
};

typedef boost::shared_ptr<Simulator> SimPtr;

#endif // SIMULATOR_H
