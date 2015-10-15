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
    TestSim(const OptionsPtr &, const InitCondPtr &);
    virtual ~TestSim();

private:
    virtual void init();
    virtual int nextStep() const;
    virtual int finalStep() const;
    virtual void execNextStep();
    virtual std::vector<float> results() const;
    virtual void printStatus() const;

    struct TestSimImpl;
    TestSimImpl *pimpl;
};

#endif // TESTSIM_H
