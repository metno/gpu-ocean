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
    int nx() const;
    int ny() const;
    float width() const;
    float height() const;
    float dx() const;
    float dy() const;
    FieldInfo waterElevationField() const;
    FieldInfo bathymetryField() const;
    FieldInfo H() const;
    FieldInfo eta() const;
    FieldInfo U() const;
    FieldInfo V() const;

private:
    struct InitConditionsImpl;
    InitConditionsImpl *pimpl;
};

typedef std::shared_ptr<InitConditions> InitCondPtr;

#endif // INITCONDITIONS_H
