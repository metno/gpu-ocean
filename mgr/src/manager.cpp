#include "manager.h"
#include "programoptions.h"
#include "initconditions.h"
#include "simbase.h"
#include "simulator.h"
#include "testsim.h"
#include "netcdfwriter.h"
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

// Initializes the manager.
void Manager::init(int argc, char *argv[])
{
    if (isInit)
        throw runtime_error("manager already initialized");

    opts.reset(new ProgramOptions);
    if (!opts->parse(argc, argv)) {
        cerr << opts->message() << endl;
        return;
    }

    initCond.reset(new InitConditions);
    initCond->init(opts);

    isInit = true;
}

// Returns true iff the manager is initialized.
bool Manager::initialized()
{
    return isInit;
}

// Returns the singleton instance (thread-safe in C++11).
Manager &Manager::instance()
{
    if (!isInit)
        throw runtime_error("manager not initialized");

    static Manager mgr;
    return mgr;
}

// Returns the program options object.
OptionsPtr Manager::options() const
{
    return opts;
}

// Returns the initial conditions object.
InitCondPtr Manager::initConditions() const
{
    return initCond;
}

// Returns the simulator object.
SimBasePtr Manager::sim() const
{
    return pimpl->sim;
}

// Initializes a new simulation run (aborting one that is already in progress, if necessary).
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
                    options()->nx(), options()->ny(), pimpl->sim->deltaTime(), options()->dx(), options()->dy(),
                    pimpl->sim->F(), pimpl->sim->R(), initConditions()->H().data->data(), initConditions()->eta().data->data(),
                    pimpl->sim->U().data->data(), pimpl->sim->V().data->data());
    }
}

// Executes the next simulation step and advances the simulation time.
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

// Returns U at the current simulation step.
FieldInfo Manager::U() const
{
    return pimpl->sim->U();
}

// Returns V at the current simulation step.
FieldInfo Manager::V() const
{
    return pimpl->sim->V();
}

// Returns eta at the current simulation step.
FieldInfo Manager::eta() const
{
    return pimpl->sim->eta();
}

Manager::Manager()
    : pimpl(new ManagerImpl(opts, initCond))
{
}

bool Manager::isInit = false;
OptionsPtr Manager::opts;
InitCondPtr Manager::initCond;
