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
    bool isInit;
    std::string msg;
    float wGlobal; // global water elevation level used to generate H in IC
    int etaNo; // type of sea surface deviation field (eta) to generate for IC
    int waterElevationNo; // type of water elevation field (w) to generate for IC
    int bathymetryNo; // type of bathymetry field (B) to generate for IC
    int nx; // number of grid horizontal grid cells
    int ny; // number of vertical grid cells
    float width; // horizontal extension of grid (in meters)
    float height; // vertical extension of grid (in meters)
    double duration; // duration of simulation (in simulated seconds)
    double wallDuration; // duration of simulation (in wall time seconds)
    bool cpu; // whether to run kernels on the CPU instead of the GPU
    string inputFile; // name of file for reading output in NetCDF format
    string outputFile; // name of file for writing output in NetCDF format
    ProgramOptionsImpl();
};

ProgramOptions::ProgramOptionsImpl::ProgramOptionsImpl()
    : isInit(false)
    , wGlobal(-1.0f)
	, etaNo(-1)
	, waterElevationNo(-1)
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

// Initializes the object by parsing program options specified on command line and/or config file.
// Returns true if parsing was successful, otherwise updates the latest parsing message and returns false.
bool ProgramOptions::init(int argc, char *argv[])
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
                ("wGlobal", po::value<float>(&pimpl->wGlobal)->default_value(-1.0f), "global initial water elevation level (in meters)")
                ("etaNo", po::value<int>(&pimpl->etaNo)->default_value(-1), "type of initial, generated sea surface deviation field (0..4)")
                ("waterElevationNo", po::value<int>(&pimpl->waterElevationNo)->default_value(-1), "type of initial, generated water elevation field (0..6)")
                ("bathymetryNo", po::value<int>(&pimpl->bathymetryNo)->default_value(-1), "type of initial, generated bathymetry field (0..4)")
                ("nx", po::value<int>(&pimpl->nx)->default_value(-1), "number of horizontal grid cells")
                ("ny", po::value<int>(&pimpl->ny)->default_value(-1), "number of vertical grid cells")
                ("width", po::value<float>(&pimpl->width)->default_value(-1), "horizontal extension of grid (in meters)")
                ("height", po::value<float>(&pimpl->height)->default_value(-1), "vertical extension of grid (in meters)")
                ("duration", po::value<double>(&pimpl->duration)->default_value(-1), "max simulated time duration (in seconds) (< 0 = infinite duration)")
                ("wallDuration", po::value<double>(&pimpl->wallDuration)->default_value(-1), "max wall time duration (in seconds) (< 0 = infinite duration)")
                ("inputFile", po::value<string>(&pimpl->inputFile)->default_value(""), "name of file for reading input in NetCDF format (empty = no input)")
                ("outputFile", po::value<string>(&pimpl->outputFile)->default_value(""), "name of file for writing output in NetCDF format (empty = no output)")
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
            pimpl->msg = "GPU Ocean, version 0.8\n";
            return false;
        }

        if (vm.count("cpu"))
            pimpl->cpu = true;

        // final validation
        if (pimpl->wGlobal < -1.0f)
            throw runtime_error((boost::format("error: wGlobal (%1%) < -1.0f") % pimpl->wGlobal).str());
        if (pimpl->etaNo < -1)
            throw runtime_error((boost::format("error: etaNo (%1%) < -1") % pimpl->etaNo).str());
        if (pimpl->waterElevationNo < -1)
            throw runtime_error((boost::format("error: waterElevationNo (%1%) < -1") % pimpl->waterElevationNo).str());
        if (pimpl->bathymetryNo < -1)
            throw runtime_error((boost::format("error: bathymetryNo (%1%) < -1") % pimpl->bathymetryNo).str());
        if (pimpl->duration < 0 && pimpl->wallDuration < 0)
            throw runtime_error(
                    (boost::format("error: duration (%1%) and wallduration (%2%) cannot both be negative")
                     % pimpl->duration % pimpl->wallDuration).str());
    }
    catch(exception &e)
    {
        pimpl->msg = e.what();
        return false;
    }

    pimpl->isInit = true;

    return true;
}

void ProgramOptions::assertInitialized() const
{
    if (!pimpl->isInit)
        throw runtime_error("ProgramOptions: not initialized");
}

string ProgramOptions::message() const
{
    return pimpl->msg;
}

float ProgramOptions::wGlobal() const
{
    assertInitialized();
    return pimpl->wGlobal;
}

int ProgramOptions::etaNo() const
{
    assertInitialized();
    return pimpl->etaNo;
}

int ProgramOptions::waterElevationNo() const
{
    assertInitialized();
    return pimpl->waterElevationNo;
}

int ProgramOptions::bathymetryNo() const
{
    assertInitialized();
    return pimpl->bathymetryNo;
}

int ProgramOptions::nx() const
{
    assertInitialized();
    return pimpl->nx;
}

int ProgramOptions::ny() const
{
    assertInitialized();
    return pimpl->ny;
}

float ProgramOptions::width() const
{
    assertInitialized();
    return pimpl->width;
}

float ProgramOptions::height() const
{
    assertInitialized();
    return pimpl->height;
}

float ProgramOptions::dx() const
{
    assertInitialized();
    assert(pimpl->nx > 1);
    return pimpl->width / pimpl->nx;
}

float ProgramOptions::dy() const
{
    assertInitialized();
    assert(pimpl->ny > 1);
    return pimpl->height / pimpl->ny;
}

double ProgramOptions::duration() const
{
    assertInitialized();
    return pimpl->duration;
}

double ProgramOptions::wallDuration() const
{
    assertInitialized();
    return pimpl->wallDuration;
}

bool ProgramOptions::cpu() const
{
    assertInitialized();
    return pimpl->cpu;
}

string ProgramOptions::inputFile() const
{
    assertInitialized();
    return pimpl->inputFile;
}

string ProgramOptions::outputFile() const
{
    assertInitialized();
    return pimpl->outputFile;
}

ostream &operator<<(ostream &os, const ProgramOptions &po)
{
    os << "nx: " << po.nx() << ", ny: " << po.ny() << ", width: " << po.width() << ", height: "
       << po.height() << ", duration: " << po.duration();
    return os;
}
