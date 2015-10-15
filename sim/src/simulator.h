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
    virtual bool _init();
    virtual int _nextStep() const;
    virtual int _finalStep() const;
    virtual void _execNextStep();
    virtual std::vector<float> _results() const;
    virtual void _printStatus() const;

    struct SimulatorImpl;
    SimulatorImpl *pimpl;
};

#endif // SIMULATOR_H
