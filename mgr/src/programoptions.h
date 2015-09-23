#ifndef PROGRAMOPTIONS_H
#define PROGRAMOPTIONS_H

#include <boost/shared_ptr.hpp>
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
    struct ProgramOptionsImpl;
    ProgramOptionsImpl *pimpl;
};

// Formats output of a ProgramOptions object.
std::ostream &operator<<(std::ostream &, const ProgramOptions &);

typedef boost::shared_ptr<ProgramOptions> OptionsPtr;

#endif // PROGRAMOPTIONS_H
