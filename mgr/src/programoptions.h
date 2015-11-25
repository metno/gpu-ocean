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
    bool parse(int, char *[]);
    std::string message() const;
    float wGlobal() const;
    int etaNo() const;
    int waterElevationNo() const;
    int bathymetryNo() const;
    double duration() const;
    double wallDuration() const;
    float cpu() const;
    std::string inputFile() const;
    std::string outputFile() const;
private:
    struct ProgramOptionsImpl;
    ProgramOptionsImpl *pimpl;

    // NOTE: The the grid dimensions set in ProgramOptions should be accessible only to InitConditions (hence the friend declaration below)
    // since the latter sets the final values (possibly read from an input file, if such one exists). Other parts of the program
    // (like functions in the the Simulator class) then access the final grid dimensions through the the InitConditions API.
    int nx() const;
    int ny() const;
    float width() const;
    float height() const;
    float dx() const;
    float dy() const;

    friend class InitConditions;
    friend std::ostream &operator<<(std::ostream &, const ProgramOptions &);
};

// Formats output of a ProgramOptions object.
std::ostream &operator<<(std::ostream &, const ProgramOptions &);

typedef std::shared_ptr<ProgramOptions> OptionsPtr;

#endif // PROGRAMOPTIONS_H
