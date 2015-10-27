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
    virtual double _currTime() const;
    virtual double _maxTime() const;
    virtual void _execNextStep();
    virtual FieldInfo _U() const;
    virtual FieldInfo _V() const;
    virtual FieldInfo _eta() const;
    virtual void _printStatus() const;

    struct SimulatorImpl;
    SimulatorImpl *pimpl;
};

#endif // SIMULATOR_H
