#ifndef PROGRAMOPTIONS_H
#define PROGRAMOPTIONS_H

#include <string>
#include <ostream>

// This class holds the program options.
class ProgramOptions
{
public:
    ProgramOptions();
    bool parse(int, char *[]);
    std::string message() const;
    int nx() const;
    int ny() const;
    float width() const;
    float height() const;
    float duration() const;
private:
    std::string msg_;
    int nx_; // number of grid horizontal grid cells
    int ny_; // number of vertical grid cells
    float width_; // horizontal extension of grid (in meters)
    float height_; // vertical extension of grid (in meters)
    float duration_; // duration of simulation (in seconds)
};

// Formats output of a ProgramOptions object.
std::ostream &operator<<(std::ostream &, const ProgramOptions &);

#endif // PROGRAMOPTIONS_H
