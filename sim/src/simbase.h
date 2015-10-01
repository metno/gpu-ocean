#ifndef SIMBASE_H
#define SIMBASE_H

#include "programoptions.h"
#include "initconditions.h"
#include <boost/shared_ptr.hpp>
#include <vector>

/**
 * @brief This is the base class for managing low-level aspects of a simulation.
 */
class SimBase
{
public:

    /**
     * Initializes the simulator. This includes resetting the current and final step values.
     */
    virtual void init() = 0;

    /**
     * Returns the current step value.
     */
    virtual int nextStep() const = 0;

    /**
     * Returns the final step value.
     */
    virtual int finalStep() const = 0;

    /**
     * Executes the next simulation step and increases the step value by one.
     * Throws std::runtime_error if the current step value exceeds the final one.
     */
    virtual void execNextStep() = 0;

    /**
     * Returns the simulation results at the current step.
     */
    virtual std::vector<float> results() const = 0;

    virtual void printStatus() const = 0;

    /**
     * Returns the options.
     */
    OptionsPtr options() const;

    /**
     * Returns the initial conditions.
     */
    InitCondPtr initCond() const;

protected:

    SimBase(const OptionsPtr &, const InitCondPtr &);
    virtual ~SimBase();

private:

    struct SimBaseImpl;
    SimBaseImpl *pimpl;
};

typedef boost::shared_ptr<SimBase> SimBasePtr;

#endif // SIMBASE_H
