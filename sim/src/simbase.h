#ifndef SIMBASE_H
#define SIMBASE_H

#include "config.h"
#include "profile.h"
#include "programoptions.h"
#include "initconditions.h"
#include "field.h"
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
    float deltaTime() const;
    bool execNextStep(ProfileInfo * = 0);
    FieldInfo U() const;
    FieldInfo V() const;
    FieldInfo eta() const;
    float F() const;
    float R() const;
    void printStatus() const;
    OptionsPtr options() const;
    InitCondPtr initCond() const;

protected:
    SimBase(const OptionsPtr &, const InitCondPtr &);
    virtual ~SimBase();
    virtual bool _init() = 0;
    virtual double _currTime() const = 0;
    virtual double _maxTime() const = 0;
    virtual float _deltaTime() const = 0;
    virtual void _execNextStep(ProfileInfo *) = 0;
    virtual FieldInfo _U() const = 0;
    virtual FieldInfo _V() const = 0;
    virtual FieldInfo _eta() const = 0;
    virtual float _F() const = 0;
    virtual float _R() const = 0;
    virtual void _printStatus() const = 0;

private:
    struct SimBaseImpl;
    SimBaseImpl *pimpl;

    /**
     * Asserts that the simulator is initialized with a successful call to init().
     * @throws std::runtime_error if init() has not been successfully called.
     */
    void assertInitialized() const;
};

typedef std::shared_ptr<SimBase> SimBasePtr;

#endif // SIMBASE_H
