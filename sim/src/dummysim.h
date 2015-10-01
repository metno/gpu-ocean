#ifndef DUMMYSM_H
#define DUMMYSIM_H

#include "simbase.h"

/**
 * @brief This class tests execution of an OpenCL kernel that does nothing useful.
 */
class DummySim : public SimBase
{
public:
    DummySim(const OptionsPtr &, const InitCondPtr &);
    virtual ~DummySim();

private:
    virtual void init();
    virtual int nextStep() const;
    virtual int finalStep() const;
    virtual void execNextStep();
    virtual std::vector<float> results() const;
    virtual void printStatus() const;

    struct DummySimImpl;
    DummySimImpl *pimpl;
};

#endif // DUMMYSIM_H
