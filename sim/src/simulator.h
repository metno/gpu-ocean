#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "simbase.h"

/**
 * @brief This class implements the real simulator.
 */
class Simulator : public SimBase
{
public:
    Simulator(const OptionsPtr &, const InitCondPtr &);
    virtual ~Simulator();

private:
    virtual void init();
    virtual int nextStep() const;
    virtual int finalStep() const;
    virtual void execNextStep();
    virtual std::vector<float> results() const;
    virtual void printStatus() const;

    struct SimulatorImpl;
    SimulatorImpl *pimpl;
};

#endif // SIMULATOR_H
