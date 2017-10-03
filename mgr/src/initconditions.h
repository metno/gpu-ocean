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

    /**
     * Constructs the object.
     */
    InitConditions();

    /**
     * Initializes the object by reading initial fields from an input file (if specified), doing some validation etc.
     * @param options: The fundamental options specified on the command line and/or in a config file.
     */
    void init(const OptionsPtr &options);

    /**
     * Returns the number of grid points in the x dimension (not including ghost cells).
     */
    int getNx() const;

    /**
     * Returns the number of grid points in the y dimension (not including ghost cells).
     */
    int getNy() const;

    /**
     * Returns the width of the grid in meters (not including ghost cells).
     */
    float width() const;

    /**
     * Returns the height of the grid in meters (not including ghost cells).
     */
    float height() const;

    /**
     * Returns the width of a grid cell in meters.
     */
    float getDx() const;

    /**
     * Returns the height of a grid cell in meters.
     */
    float getDy() const;

    /**
     * Returns the water elevation field.
     */
    Field2D waterElevationField() const;

    /**
     * Returns the bathymetry field.
     */
    Field2D bathymetryField() const;

    /**
     * Returns the equilibrium depth.
     */
    Field2D H() const;

    /**
     * Returns the sea surface deviation away from the equilibrium depth.
     */
    Field2D eta() const;

    /**
     * Returns the depth averaged velocity in the x direction.
     */
    Field2D U() const;

    /**
     * Returns the depth averaged velocity in the y direction.
     */
    Field2D V() const;

private:
    struct InitConditionsImpl;
    InitConditionsImpl *pimpl;
};

typedef std::shared_ptr<InitConditions> InitCondPtr;

#endif // INITCONDITIONS_H
