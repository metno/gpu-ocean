#include "programoptions.h"
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <fstream>
#include <sstream>

namespace po = boost::program_options;

using namespace std;

ProgramOptions::ProgramOptions()
    : nx_(-1)
    , ny_(-1)
    , width_(-1)
    , height_(-1)
    , duration_(-1)
{
}

// Parses program options specified on command line and/or config file.
// Returns true if parsing was successful, otherwise updates the latest parsing message and returns false.
bool ProgramOptions::parse(int argc, char *argv[])
{
    try {
        string config_file;

        // ----- PHASE 1: Load options from command-line and/or config file -----

        // declare options that will be allowed only on command line
        po::options_description cmdline_only_opts("Options allowed only on command line");
        const char *cfd_cstr = getenv("CFGFILE");
        const string cfgfile_default(cfd_cstr ? cfd_cstr : "");
        cmdline_only_opts.add_options()
                ("version,v", "print version string")
                ("help", "print this message")
                ("config,c", po::value<string>(&config_file)->default_value(cfgfile_default), "configuration file")
                ;

        // declare options that will be allowed both on command line and in config file
        po::options_description cfgfile_opts("Options allowed both on command line and in config file (the former overrides the latter)");
        cfgfile_opts.add_options()
                ("nx", po::value<int>(&nx_)->default_value(10), "number of horizontal grid cells")
                ("ny", po::value<int>(&ny_)->default_value(10), "number of vertical grid cells")
                ("width", po::value<float>(&width_)->default_value(1000), "horizontal extension of grid (in meters)")
                ("height", po::value<float>(&height_)->default_value(1000), "vertical extension of grid (in meters)")
                ("duration", po::value<float>(&duration_)->default_value(100), "duration of simulation (in seconds)")
                ;

        po::options_description all_options;
        all_options.add(cmdline_only_opts).add(cfgfile_opts);

        po::options_description cfgfile_options;
        cfgfile_options.add(cfgfile_opts);

        po::options_description visible_options("Allowed options");
        visible_options.add(cmdline_only_opts).add(cfgfile_opts);

        // declare positional ("nameless") options
        po::positional_options_description p;
        // none for the time being

        // parse all options from command line (including specific name of config file)
        po::variables_map vm;
        store(po::command_line_parser(argc, argv).
              options(all_options).positional(p).run(), vm);
        notify(vm);

        // parse config file
        ifstream ifs(config_file.c_str());
        if ((!ifs) && (!config_file.empty())) {
            msg_ = (boost::format("error: can not open config file: %1%") % config_file).str();
            return false;
        } else {
            store(parse_config_file(ifs, cfgfile_options), vm);
            notify(vm);
        }


        // ----- PHASE 2: Extract options from vm structure -----

        if (vm.count("help")) {
            ostringstream oss;
            visible_options.print(oss);
            msg_ = oss.str();
            return false;
        }

        if (vm.count("version")) {
            msg_ = "GPU EPS HAV, version 1.0\n";
            return false;
        }
    }
    catch(exception &e)
    {
        msg_ = e.what();
        return false;
    }

    return true;
}

// Returns the message resulting when the latest call to parse() returned false.
string ProgramOptions::message() const
{
    return msg_;
}

int ProgramOptions::nx() const
{
    return nx_;
}

int ProgramOptions::ny() const
{
    return ny_;
}

float ProgramOptions::width() const
{
    return width_;
}

float ProgramOptions::height() const
{
    return height_;
}

float ProgramOptions::duration() const
{
    return duration_;
}

// Formats output of a ProgramOptions object.
ostream &operator<<(ostream &os, const ProgramOptions &po)
{
    os << "nx: " << po.nx() << ", ny: " << po.ny() << ", width: " << po.width() << ", height: "
       << po.height() << ", duration: " << po.duration();
    return os;
}
