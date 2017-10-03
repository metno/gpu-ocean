#ifndef TESTSIM_H
#define TESTSIM_H

#include "simbase.h"

/**
 * This class demonstrates two things:
 * 1: A specific implementation of SimBase.
 * 2: How OpenCL can be used for a simple computation (in this case multiplication of two random matrices).
 */
class TestSim : public SimBase
{
public:

    /**
     * Constructor.
     * @param options: Program options.
     * @param initCond: Initial conditions.
     */
    TestSim(const OptionsPtr &options, const InitCondPtr &initCond);

    virtual ~TestSim();

private:
    virtual bool init();
    virtual double currTime() const;
    virtual double maxTime() const;
    virtual bool execNextStep(ProfileInfo *);
    virtual Field2D U() const;
    virtual Field2D V() const;
    virtual Field2D eta() const;
    virtual void printStatus() const;

    struct TestSimImpl;
    TestSimImpl *pimpl;
};

#endif // TESTSIM_H
