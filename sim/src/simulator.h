#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "simbase.h"
#include <memory>

using namespace std;

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
    virtual bool init();
    virtual void assertInitialized() const;
    virtual double currTime() const;
    virtual double maxTime() const;
    virtual float deltaTime() const;
    virtual bool execNextStep(ProfileInfo *);
    virtual Field2D H() const;
    virtual Field2D U() const;
    virtual Field2D V() const;
    virtual Field2D eta() const;
    virtual float f() const;
    virtual float r() const;
    virtual void printStatus() const;

    struct SimulatorImpl;
    SimulatorImpl *pimpl;
};

#endif // SIMULATOR_H
