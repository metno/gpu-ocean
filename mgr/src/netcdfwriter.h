#ifndef NETCDFWRITER_H
#define NETCDFWRITER_H

#include <netcdfcpp.h>
#include <memory>

class NetCDFWriter {
public:
    /**
     * Constructs an object and creates an internal NcFile object for a writable file with an implicitly defined file name of the form
     * results_<year>-<month>-<day>T<hour>:<minute>:<second>.nc, for example: results_2015-11-20T13:50:26.nc.
     * @throws std::runtime_error if the file already exists.
     */
	NetCDFWriter();

    /**
     * Constructs an object and creates an internal NcFile object for a writable file.
     * @param fname: File name.
     * @throws std::runtime_error if the file already exists, the path is invalid, the disk is full, etc.
     */
    NetCDFWriter(const std::string &fname);

    /**
     * Destructs the object and closes the internal NcFile object (flushing any unwritten data to disk).
     */
	~NetCDFWriter();

    /**
     * Initializes the writer and writes the initial state (at timestep 0) to file.
     * @note: This function must be called once before the first call to writeTimestep().
     * @param nx: Number of grid points in the x dimension.
     * @param ny: Number of grid points in the y dimension.
     * @param dt: Duration of a simulation step in seconds.
     * @param dx: Width of a grid cell in meters.
     * @param dy: Height of a grid cell in meters.
     * @param f: Coriolis effect.
     * @param r: Bottom friction coefficient.
     * @param H: Equilibrium depth.
     * @param eta: Initial sea surface deviation away from the equilibrium depth.
     * @param U: Initial depth averaged velocity in the x direction.
     * @param V: Initial depth averaged velocity in the y direction.
     */
    void init(int nx, int ny, float dt, float dx, float dy, float f, float r, float *H, float *eta, float *U, float *V);

    /**
     * Writes the next timestep (value > 0) to file.
     * @param eta: Sea surface deviation away from the equilibrium depth.
     * @param U: Depth averaged velocity in the x direction.
     * @param V: Depth averaged velocity in the y direction.
     * @param t: Time in seconds.
     */
    void writeTimestep(float *eta, float *U, float *V, float t);

private:

    /**
     * Helper function called by constructors.
     */
    void initFile(const std::string &fname);

    struct NetCDFWriterImpl;
    NetCDFWriterImpl *pimpl;
};

typedef std::shared_ptr<NetCDFWriter> NetCDFWriterPtr;

#endif // NETCDFWRITER_H
