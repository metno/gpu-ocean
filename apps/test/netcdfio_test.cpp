#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE netcdfio test
#include <boost/test/unit_test.hpp>
#define BOOST_TEST_LOG_LEVEL "all"

#include "utils/testutils.h"
#include "manager.h"
#include "programoptions.h"
#include "initconditions.h"
#include "simulator.h"
#include "netcdfwriter.h"
#include "netcdfreader.h"

using namespace std;

static string outputFile("/tmp/out.nc");

// *** fixtures ***

// Define a fixture to be set up and torn down once for the entire set of tests.
struct GlobalFixture {
    GlobalFixture() { BOOST_TEST_MESSAGE("GlobalFixture() ..."); }
    ~GlobalFixture() { BOOST_TEST_MESSAGE("~GlobalFixture() ..."); }
};
BOOST_GLOBAL_FIXTURE(GlobalFixture);

// Define a fixture to be set up and torn down for every test case invocation (assuming all test cases belong to the master test suite).
struct MasterTestSuiteFixture {
    MasterTestSuiteFixture()
    {
        BOOST_TEST_MESSAGE("MasterTestSuiteFixture() ...");
        remove(outputFile.c_str());

        // initialize manager
        vector<string> words;
        words.push_back("--duration");
        words.push_back("1000");
        words.push_back("--nx");
        words.push_back("10");
        words.push_back("--ny");
        words.push_back("10");
        words.push_back("--width");
        words.push_back("1000");
        words.push_back("--height");
        words.push_back("1000");
        words.push_back("--bathymetryNo");
        words.push_back("0");
        words.push_back("--etaNo");
        words.push_back("0");
        words.push_back("--outputFile");
        words.push_back(outputFile);
        pair<int, char **> args = createArgs(words);
        Manager::init(args.first, args.second);
    }

    ~MasterTestSuiteFixture()
    {
        BOOST_TEST_MESSAGE("~MasterTestSuiteFixture() ...");
        remove(outputFile.c_str());
    }
};
BOOST_FIXTURE_TEST_SUITE(MasterTestSuite, MasterTestSuiteFixture) // note: the first arg can be anything in this case


// *** test cases ***

BOOST_AUTO_TEST_CASE(ManagerInitialized)
{
    BOOST_CHECK(Manager::instance().initialized());
}

BOOST_AUTO_TEST_CASE(InitStateWrittenAndReadBack)
{
    Manager &mgr = Manager::instance();
    mgr.initSim(); // note: this call also writes the initial state to the output file (timestep 0)

    NetCDFReaderPtr fileReader(new NetCDFReader(outputFile));

    // check time series sizes
    BOOST_CHECK_EQUAL(fileReader->etaTimesteps(), 1);
    BOOST_CHECK_EQUAL(fileReader->UTimesteps(), 1);
    BOOST_CHECK_EQUAL(fileReader->VTimesteps(), 1);

    // check other contents
    const InitCondPtr initCond = mgr.initConditions();
    BOOST_CHECK_EQUAL(initCond->getNx(), fileReader->nx());
    BOOST_CHECK_EQUAL(initCond->getNy(), fileReader->ny());
    BOOST_CHECK_EQUAL(initCond->width(), fileReader->width());
    BOOST_CHECK_EQUAL(initCond->height(), fileReader->height());
    BOOST_CHECK_EQUAL(initCond->getDx(), fileReader->dx());
    BOOST_CHECK_EQUAL(initCond->getDy(), fileReader->dy());
    CHECK_FIELDS_EQUAL(initCond->H(), fileReader->H());
    CHECK_FIELDS_EQUAL(initCond->eta(), mgr.eta()); // note this!
    CHECK_FIELDS_EQUAL(mgr.eta(), fileReader->eta());
    CHECK_FIELDS_EQUAL(mgr.U(), fileReader->U());
    CHECK_FIELDS_EQUAL(mgr.V(), fileReader->V());
}

BOOST_AUTO_TEST_CASE(TimeSeriesWrittenAndReadBack)
{
    Manager &mgr = Manager::instance();
    mgr.initSim(); // note: this call also writes the initial state to the output file (timestep 0)

    vector<Field2D> eta;
    vector<Field2D> U;
    vector<Field2D> V;

    // copy initial state of each time series (timestep 0)
    //CHECK_FIELDS_EQUAL(mgr.initConditions()->eta(), mgr.eta()); // note this!
    eta.push_back(mgr.eta());
    U.push_back(mgr.U());
    V.push_back(mgr.V());

    // simulate a few steps
    const int nsteps = 10;
    for (int i = 0; i < nsteps; ++i) {
        mgr.execNextStep();
        eta.push_back(mgr.eta());
        U.push_back(mgr.U());
        V.push_back(mgr.V());
    }

    NetCDFReaderPtr fileReader(new NetCDFReader(outputFile));

    // check that H hasn't changed from its original value
    CHECK_FIELDS_EQUAL(mgr.initConditions()->H(), fileReader->H());

    // check time series sizes
    const int tssize = nsteps + 1; // note that we must include the initial state (timestep 0, before the first simulation step)
    BOOST_CHECK_EQUAL(tssize, fileReader->etaTimesteps());
    BOOST_CHECK_EQUAL(tssize, fileReader->UTimesteps());
    BOOST_CHECK_EQUAL(tssize, fileReader->VTimesteps());

    // check that fields at explicit timesteps are equal
    for (int i = 0; i < nsteps; ++i) {
        CHECK_FIELDS_EQUAL(eta.at(i), fileReader->eta(i));
        CHECK_FIELDS_EQUAL(U.at(i), fileReader->U(i));
        CHECK_FIELDS_EQUAL(V.at(i), fileReader->V(i));
    }

    // check that the last timestep may be implicitly specified
    CHECK_FIELDS_EQUAL(mgr.eta(), fileReader->eta());
    CHECK_FIELDS_EQUAL(mgr.U(), fileReader->U());
    CHECK_FIELDS_EQUAL(mgr.V(), fileReader->V());
    CHECK_FIELDS_EQUAL(mgr.eta(), fileReader->eta(-1));
    CHECK_FIELDS_EQUAL(mgr.U(), fileReader->U(-1));
    CHECK_FIELDS_EQUAL(mgr.V(), fileReader->V(-1));
}

BOOST_AUTO_TEST_SUITE_END()
