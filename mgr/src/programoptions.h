#ifndef PROGRAMOPTIONS_H
#define PROGRAMOPTIONS_H

#include <memory>
#include <string>
#include <ostream>

// This class holds the program options.
class ProgramOptions
{
public:
    ProgramOptions();

    /**
     * Initializes the object by parsing program options specified on command line and/or config file.
     * @param argc: Number of arguments.
     * @param argv: Argument vector.
     * @returns True iff parsing was successful. Before false is returned, the latest parsing message is updated.
     * @note Passing "--help" as one of the arguments prints a documentation of the available arguments on stderr.
     */
    bool init(int argc, char *argv[]);

    /**
     * Returns the message resulting from the most recent call to parse() that returned false.
     */
    std::string message() const;

    /**
     * Returns the global initial water elevation level in meters, or -1 if not set.
     */
    float wGlobal() const;

    /**
     * Returns the type of initial, synthesized sea surface deviation (0..4), or < 0 if not set.
     */
    int etaNo() const;

    /**
     * Returns the type of initial, synthesized water elevation (0..6), or < 0 if not set.
     */
    int waterElevationNo() const;

    /**
     * Returns the type of initial, synthesized athymetry (0..4), or < 0 if not set.
     */
    int bathymetryNo() const;

    /**
     * Returns the maximum simulated time duration in seconds, or < 0 for infinite duration.
     */
    double duration() const;

    /**
     * Returns the maximum wall time duration in seconds, or < 0 for infinite duration.
     */
    double wallDuration() const;

    /**
     * Returns true or false to indicate if the computation runs on the CPU or GPU respectively.
     */
    float cpu() const;

    /**
     * Returns the name of the file used for reading input in NetCDF format, or an empty string for no input.
     */
    std::string inputFile() const;

    /**
     * Returns the name of the file used for writing output in NetCDF format, or an empty string for no output.
     */
    std::string outputFile() const;

private:
    struct ProgramOptionsImpl;
    ProgramOptionsImpl *pimpl;

    // NOTE: The the grid dimensions set in ProgramOptions should be accessible only to InitConditions (hence the friend declaration below)
    // since the latter sets the final values (possibly read from an input file, if such one exists). Other parts of the program
    // (like functions in the the Simulator class) then access the final grid dimensions through public functions in InitConditions.
    int nx() const;
    int ny() const;
    float width() const;
    float height() const;
    float dx() const;
    float dy() const;

    friend class InitConditions;
    friend std::ostream &operator<<(std::ostream &, const ProgramOptions &);

    /**
     * Asserts that the object is initialized with a successful call to init().
     * @throws std::runtime_error if init() has not been successfully called.
     */
    void assertInitialized() const;
};

/**
 * Formats output of a ProgramOptions object.
 */
std::ostream &operator<<(std::ostream &, const ProgramOptions &);

typedef std::shared_ptr<ProgramOptions> OptionsPtr;

#endif // PROGRAMOPTIONS_H
