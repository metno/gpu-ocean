#ifndef INITCONDITIONS_H
#define INITCONDITIONS_H

#include "programoptions.h"
#include <boost/shared_ptr.hpp>
#include <vector>

typedef boost::shared_ptr<std::vector<float> > FieldPtr;

// This class holds initial conditions for a simulation.
class InitConditions
{
public:
    InitConditions();
    void init(const OptionsPtr &);
    FieldPtr field() const;
private:
    struct InitConditionsImpl;
    InitConditionsImpl *pimpl;
};

typedef boost::shared_ptr<InitConditions> InitCondPtr;

#endif // INITCONDITIONS_H
