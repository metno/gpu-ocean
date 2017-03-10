#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "simbase.h"

/**
 * @brief This class implements the real simulator.
 */
class Simulator : public SimBase
{
public:

    /**
     * Constructor.
     * @param options: Program options.
     * @param initCond: Initial conditions.
     */
    Simulator(const OptionsPtr &, const InitCondPtr &);

    virtual ~Simulator();

private:
    virtual bool _init();
    virtual double _currTime() const;
    virtual double _maxTime() const;
    virtual float _deltaTime() const;
    virtual void _execNextStep(ProfileInfo *);
    virtual Field2D _U() const;
    virtual Field2D _V() const;
    virtual Field2D _eta() const;
    virtual float _f() const;
    virtual float _r() const;
    virtual void _printStatus() const;

    struct SimulatorImpl;
    SimulatorImpl *pimpl;
};

#endif // SIMULATOR_H
