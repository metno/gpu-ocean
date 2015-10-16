#include "programoptions.h"
#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace po = boost::program_options;

using namespace std;

struct ProgramOptions::ProgramOptionsImpl
{
    std::string msg;
    int waterElevationNo; // which water elevation to generate for IC
    int bathymetryNo; // which bathymetry to generate for IC
    int nx; // number of grid horizontal grid cells
    int ny; // number of vertical grid cells
    float width; // horizontal extension of grid (in meters)
    float height; // vertical extension of grid (in meters)
    double duration; // duration of simulation (in simulated seconds)
    double wallDuration; // duration of simulation (in wall time seconds)
    bool cpu; // whether to run kernels on the CPU instead of the GPU
    ProgramOptionsImpl();
};

ProgramOptions::ProgramOptionsImpl::ProgramOptionsImpl()
    : waterElevationNo(-1)
	, bathymetryNo(-1)
    , nx(-1)
    , ny(-1)
    , width(-1)
    , height(-1)
    , duration(-1)
    , wallDuration(-1)
    , cpu(false)
{
};

ProgramOptions::ProgramOptions()
    : pimpl(new ProgramOptionsImpl)
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
                ("cpu", "run on CPU (default is to run on GPU)")
                ;

        // declare options that will be allowed both on command line and in config file
        po::options_description cfgfile_opts("Options allowed both on command line and in config file (the former overrides the latter)");
        cfgfile_opts.add_options()
				("waterElevationNo", po::value<int>(&pimpl->waterElevationNo)->default_value(-1), "initial water elevation")
		        ("bathymetryNo", po::value<int>(&pimpl->bathymetryNo)->default_value(-1), "initial bathymetry")
                ("nx", po::value<int>(&pimpl->nx)->default_value(10), "number of horizontal grid cells")
                ("ny", po::value<int>(&pimpl->ny)->default_value(10), "number of vertical grid cells")
                ("width", po::value<float>(&pimpl->width)->default_value(1000), "horizontal extension of grid (in meters)")
                ("height", po::value<float>(&pimpl->height)->default_value(1000), "vertical extension of grid (in meters)")
                ("duration", po::value<double>(&pimpl->duration)->default_value(5), "max duration of simulation (in seconds) (< 0 = infinite duration)")
                ("wallduration", po::value<double>(&pimpl->wallDuration)->default_value(0.5), "max wall time duration (in seconds) (< 0 = infinite duration)")
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
            pimpl->msg = (boost::format("error: can not open config file: %1%") % config_file).str();
            return false;
        } else {
            store(parse_config_file(ifs, cfgfile_options), vm);
            notify(vm);
        }


        // ----- PHASE 2: Extract options from vm structure -----

        if (vm.count("help")) {
            ostringstream oss;
            visible_options.print(oss);
            pimpl->msg = oss.str();
            return false;
        }

        if (vm.count("version")) {
            pimpl->msg = "GPU EPS HAV, version 1.0\n";
            return false;
        }

        if (vm.count("cpu"))
            pimpl->cpu = true;

        // final validation
        if (pimpl->waterElevationNo < -1)
            throw runtime_error((boost::format("error: waterElevationNo (%1%) < -1") % pimpl->waterElevationNo).str());
        if (pimpl->bathymetryNo < -1)
            throw runtime_error((boost::format("error: bathymetryNo (%1%) < -1") % pimpl->bathymetryNo).str());
        if (pimpl->nx <= 0)
            throw runtime_error((boost::format("error: nx (%1%) <= 0") % pimpl->nx).str());
        if (pimpl->ny <= 0)
            throw runtime_error((boost::format("error: ny (%1%) <= 0") % pimpl->ny).str());
        if (pimpl->width <= 0)
            throw runtime_error((boost::format("error: width (%1%) <= 0") % pimpl->width).str());
        if (pimpl->height <= 0)
            throw runtime_error((boost::format("error: height (%1%) <= 0") % pimpl->height).str());
        if (pimpl->duration <= 0)
            throw runtime_error((boost::format("error: duration (%1%) <= 0") % pimpl->duration).str());
    }
    catch(exception &e)
    {
        pimpl->msg = e.what();
        return false;
    }

    return true;
}

// Returns the message resulting when the latest call to parse() returned false.
string ProgramOptions::message() const
{
    return pimpl->msg;
}

int ProgramOptions::waterElevationNo() const
{
	return pimpl->waterElevationNo;
}

int ProgramOptions::bathymetryNo() const
{
	return pimpl->bathymetryNo;
}

int ProgramOptions::nx() const
{
    return pimpl->nx;
}

int ProgramOptions::ny() const
{
    return pimpl->ny;
}

float ProgramOptions::width() const
{
    return pimpl->width;
}

float ProgramOptions::height() const
{
    return pimpl->height;
}

double ProgramOptions::duration() const
{
    return pimpl->duration;
}

double ProgramOptions::wallDuration() const
{
    return pimpl->wallDuration;
}

float ProgramOptions::cpu() const
{
    return pimpl->cpu;
}

// Formats output of a ProgramOptions object.
ostream &operator<<(ostream &os, const ProgramOptions &po)
{
    os << "nx: " << po.nx() << ", ny: " << po.ny() << ", width: " << po.width() << ", height: "
       << po.height() << ", duration: " << po.duration();
    return os;
}
