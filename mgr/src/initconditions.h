#ifndef INITCONDITIONS_H
#define INITCONDITIONS_H

#include "programoptions.h"
#include "field.h"
#include <memory>
#include <vector>

typedef std::shared_ptr<std::vector<float> > FieldPtr;

// This class holds initial conditions for a simulation.
class InitConditions
{
public:
    InitConditions();
    void init(const OptionsPtr &options);
    FieldInfo waterElevationField() const;
    FieldInfo bathymetryField() const;
    FieldInfo H() const;
    FieldInfo eta() const;

private:
    struct InitConditionsImpl;
    InitConditionsImpl *pimpl;
};

typedef std::shared_ptr<InitConditions> InitCondPtr;

#endif // INITCONDITIONS_H
