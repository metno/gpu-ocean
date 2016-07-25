#ifndef NETCDFREADER_H
#define NETCDFREADER_H

#include "field.h"
#include <netcdfcpp.h>
#include <memory>
#include <map>
#include <string>

/**
 * This class is used for reading simulation results and/or initial conditions from a file in NetCDF format.
 */
class NetCDFReader {
public:

    /**
     * Constructs the object and creates an internal NcFile object for a read-only file.
     * @param fname: File name.
     * @throws std::runtime_error if the file cannot be opened as a valid, read-only NetCDF file.
     */
    NetCDFReader(const std::string &fname);

    /**
     * Destructs the object and closes the internal NcFile object.
     */
    ~NetCDFReader();

    /**
     * Returns the number of grid points in the x dimension (not including ghost cells).
     */
    int nx() const;

    /**
     * Returns the number of grid points in the y dimension (not including ghost cells).
     */
    int ny() const;

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
    float dx() const;

    /**
     * Returns the height of a grid cell in meters.
     */
    float dy() const;

    /**
     * Returns H (equilibrium depth).
     */
    Field2D H() const;

    /**
     * Returns the number of timesteps in the eta time series.
     */
    long etaTimesteps() const;

    /**
     * Returns eta (sea surface deviation away from the equilibrium depth) at a given timestep.
     * @param timestep: Valid range: [0, etaTimesteps() - 1]. The last timestep may be implicitly specified by passing -1.
     */
    Field2D eta(long timestep = -1) const;

    /**
     * Returns the number of timesteps in the U time series.
     */
    long UTimesteps() const;

    /**
     * Returns U (depth averaged velocity in the x direction) at a given timestep.
     * @param timestep: Valid range: [0, UTimesteps() - 1]. The last timestep may be implicitly specified by passing -1.
     */
    Field2D U(long timestep = -1) const;

    /**
     * Returns the number of timesteps in the V time series.
     */
    long VTimesteps() const;

    /**
     * Returns V (depth averaged velocity in the y direction) at a given timestep.
     * @param timestep: Valid range: [0, VTimesteps() - 1]. The last timestep may be implicitly specified by passing -1.
     */
    Field2D V(long timestep = -1) const;

private:
    struct NetCDFReaderImpl;
    NetCDFReaderImpl *pimpl;

    /**
     * This function copies a 2D float field from file to memory. If the field variable (NcVar) has three dimensions, it is assumed that the third
     * dimension is time, and the 2D field of the last timestep is copied. Otherwise, the field variable must have two dimensions, and the
     * field is copied directly.
     * @param name: Field name.
     * @param nx_exp: Expected size of X-dimension.
     * @param ny_exp: Expected size of Y-dimension.
     * @param timestep: Timestep (if applicable, i.e. if the field variable is 3D). The first and last timestep is indicated by 0 and -1 respectively.
     * @returns The Field2D object.
     * @note An empty object is returned if the field doesn't exist (which is not considered an error).
     * @throws std::runtime_error if an error occurs.
     */
    Field2D read2DFloatField(const std::string &name, int nx_exp, int ny_exp, long timestep = -1) const;

    /**
     * Returns the number of timesteps of a 2D field.
     * @param name: Field name.
     * @returns The number of timesteps (>= 0) of the 2D field if the field exists and the field variable (NcVar) has three dimensions. Otherwise -1.
     * @throws std::runtime_error if an error occurs.
     */
    long timesteps(const std::string &) const;
};

typedef std::shared_ptr<NetCDFReader> NetCDFReaderPtr;

#endif // NETCDFREADER_H
