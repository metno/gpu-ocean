#include "config.h"

#include "manager.h"
#include "programoptions.h"
#include "initconditions.h"
#include "simbase.h"
#include "simulator.h"
#include "testsim.h"
#ifdef mgr_USE_NETCDF
#include "netcdfwriter.h"
#endif
#include <boost/format.hpp>
#include <stdexcept>
#include <vector>
#include <iostream>

// comment out the following line to run the real simulator:
//#define TESTSIM

using namespace std;

struct Manager::ManagerImpl
{
    SimBasePtr sim;

	#ifdef mgr_USE_NETCDF
    NetCDFWriterPtr fileWriter;
	#endif

    ManagerImpl(const OptionsPtr &, const InitCondPtr &);
};

Manager::ManagerImpl::ManagerImpl(const OptionsPtr &options, const InitCondPtr &initCond)
#ifdef TESTSIM
    : sim(new TestSim(options, initCond))
#else
    : sim(new Simulator(options, initCond))
#endif
{
}

void Manager::init(int argc, char *argv[])
{
    opts.reset(new ProgramOptions);
    if (!opts->init(argc, argv))
        throw runtime_error((boost::format("failed to initialize manager: %s") % opts->message()).str());

    initCond.reset(new InitConditions);
    initCond->init(opts);

    isInit = true;
}

bool Manager::initialized()
{
    return isInit;
}

Manager &Manager::instance()
{
    if (!isInit)
        throw runtime_error("manager not initialized");

    static Manager mgr;
    return mgr;
}

OptionsPtr Manager::options() const
{
    return opts;
}

InitCondPtr Manager::initConditions() const
{
    return initCond;
}

SimBasePtr Manager::sim() const
{
    return pimpl->sim;
}

void Manager::initSim()
{
    // initialize simulation
    pimpl->sim->init();

    // initialize output file (if requested)
	#ifdef mgr_USE_NETCDF
    const string ofname = options()->outputFile();
    if (!ofname.empty()) {
        pimpl->fileWriter.reset(
                    (ofname == "*")
                    ? new NetCDFWriter() // generate file names automatically
                    : new NetCDFWriter(ofname)); // use explicit file name
        pimpl->fileWriter->init(
                    initConditions()->getNx(), initConditions()->getNy(), pimpl->sim->deltaTime(), initConditions()->getDx(), initConditions()->getDy(),
                    pimpl->sim->f(), pimpl->sim->r(), initConditions()->H().getData()->data(), initConditions()->eta().getData()->data(),
                    pimpl->sim->U().getData()->data(), pimpl->sim->V().getData()->data());
    }
	#endif
}

bool Manager::execNextStep(ProfileInfo *profInfo)
{
	cerr << "!!! 1!" << endl;
    // execute next step
    const bool status = pimpl->sim->execNextStep(profInfo);

    // append to output file (if requested)
	#ifdef mgr_USE_NETCDF
    if (pimpl->fileWriter.get())
        pimpl->fileWriter->writeTimestep(
                    pimpl->sim->eta().getData()->data(),
                    pimpl->sim->U().getData()->data(),
                    pimpl->sim->V().getData()->data(),
                    pimpl->sim->currTime());
	#endif

    return status;
}

Field2D Manager::eta() const
{
    return pimpl->sim->eta();
}

Field2D Manager::U() const
{
    return pimpl->sim->U();
}

Field2D Manager::V() const
{
    return pimpl->sim->V();
}

Manager::Manager()
    : pimpl(new ManagerImpl(opts, initCond))
{
}

bool Manager::isInit = false;
OptionsPtr Manager::opts;
InitCondPtr Manager::initCond;
