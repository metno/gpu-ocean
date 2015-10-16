#ifndef SIMBASE_H
#define SIMBASE_H

#include "programoptions.h"
#include "initconditions.h"
#include <memory>
#include <vector>

/**
 * @brief This is the base class for managing low-level aspects of a simulation.
 */
class SimBase
{
public:
    bool init();
    double currTime() const;
    double maxTime() const;
    bool execNextStep();
    std::vector<float> results() const;
    void printStatus() const;
    OptionsPtr options() const;
    InitCondPtr initCond() const;

protected:
    SimBase(const OptionsPtr &, const InitCondPtr &);
    virtual ~SimBase();
    virtual bool _init() = 0;
    virtual double _currTime() const = 0;
    virtual double _maxTime() const = 0;
    virtual void _execNextStep() = 0;
    virtual std::vector<float> _results() const = 0;
    virtual void _printStatus() const = 0;

private:
    struct SimBaseImpl;
    SimBaseImpl *pimpl;
    void assertInitialized() const;
    void assertNotInitialized() const;
};

typedef std::shared_ptr<SimBase> SimBasePtr;

#endif // SIMBASE_H
