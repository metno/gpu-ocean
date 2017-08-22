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

    /**
     * Initializes the simulator. This includes resetting the current and final step values.
     * @returns Initialization status (true if initialization was successful).
     */
	virtual bool init() = 0;

    /**
     * Returns the current simulation time in seconds.
     */
    virtual double currTime() const = 0;

    /**
     * Returns the maximum simulation time in seconds.
     */
    virtual double maxTime() const = 0;

    /**
     * Returns the time (in seconds) by which to advance the simulation in each step.
     */
    virtual float deltaTime() const = 0;

    /**
     * Executes the next simulation step and returns true if the simulation is not time-bounded
     * or the current simulation time is still less than the maximum simulation time.
     * Otherwise, the function returns false without executing the next simulation step.
     * Note: The simulation is time-bounded iff the program option 'duration' is non-negative.
     * @param profInfo: If non-null, structure in which profiling is written (if applicable).
     * @returns True iff a time-bounded simulation was not exhausted.
     */
    virtual bool execNextStep(ProfileInfo *profInfo = 0) = 0;

	/**
     * Returns the equilibrium depth at the current simulation time.
     */
    virtual Field2D H() const = 0;
    
    /**
     * Returns the sea surface deviation away from the equilibrium depth at the current simulation time.
     */
    virtual Field2D eta() const = 0;

    /**
     * Returns the depth averaged velocity in the x direction at the current simulation time.
     */
    virtual Field2D U() const = 0;

    /**
     * Returns the depth averaged velocity in the y direction at the current simulation time.
     */
    virtual Field2D V() const = 0;

    /**
     * Returns the friction.
     */
    virtual float f() const = 0;

    /**
     * Returns the Coriolis effect.
     */
    virtual float r() const = 0;

    /**
     * Prints status.
     */
    virtual void printStatus() const = 0;

    /**
     * Returns the program options.
     */
    OptionsPtr options() const;

    /**
     * Returns the initial conditions.
     */
    InitCondPtr initCond() const;

protected:

    /**
     * Constructor.
     * @param options: Program options.
     * @param initCond: Initial conditions.
     */
    SimBase(const OptionsPtr &options, const InitCondPtr &initCond);

    virtual ~SimBase();

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
