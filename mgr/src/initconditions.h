#ifndef INITCONDITIONS_H
#define INITCONDITIONS_H

#include "programoptions.h"
#include <memory>
#include <vector>

typedef std::shared_ptr<std::vector<float> > FieldPtr;

// This class holds initial conditions for a simulation.
class InitConditions
{
public:
    InitConditions();
    void init(const OptionsPtr &options);

    struct FieldInfo {
        FieldPtr data;
        int nx;
        int ny;
        float dx;
        float dy;
        FieldInfo();
        FieldInfo(const FieldPtr &, int, int, float, float);
    };
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
