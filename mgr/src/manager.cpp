#include "manager.h"
#include "programoptions.h"
#include "initconditions.h"
#include "simbase.h"
#include "simulator.h"
#include "testsim.h"
#include "netcdfwriter.h"
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
    NetCDFWriterPtr fileWriter;
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
    if (!opts->parse(argc, argv))
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
    const string ofname = options()->outputFile();
    if (!ofname.empty()) {
        pimpl->fileWriter.reset(
                    (ofname == "*")
                    ? new NetCDFWriter() // generate file names automatically
                    : new NetCDFWriter(ofname)); // use explicit file name
        pimpl->fileWriter->init(
                    initConditions()->nx(), initConditions()->ny(), pimpl->sim->deltaTime(), initConditions()->dx(), initConditions()->dy(),
                    pimpl->sim->F(), pimpl->sim->R(), initConditions()->H().data->data(), initConditions()->eta().data->data(),
                    pimpl->sim->U().data->data(), pimpl->sim->V().data->data());
    }
}

bool Manager::execNextStep()
{
    // execute next step
    const bool status = pimpl->sim->execNextStep();

    // append to output file (if requested)
    if (pimpl->fileWriter.get())
        pimpl->fileWriter->writeTimestep(
                    pimpl->sim->eta().data->data(),
                    pimpl->sim->U().data->data(),
                    pimpl->sim->V().data->data(),
                    pimpl->sim->currTime());

    return status;
}

FieldInfo Manager::eta() const
{
    return pimpl->sim->eta();
}

FieldInfo Manager::U() const
{
    return pimpl->sim->U();
}

FieldInfo Manager::V() const
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
