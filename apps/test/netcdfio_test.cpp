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
#include <cstdio>

using namespace std;

static string outputFile("/tmp/out.nc");

// *** fixtures ***

// Define a fixture to be set up and torn down once for the entire set of tests.
struct GlobalFixture {
    GlobalFixture()
    {
        BOOST_TEST_MESSAGE("GlobalFixture() ...");

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

    ~GlobalFixture()
    {
        BOOST_TEST_MESSAGE("~GlobalFixture() ...");
    }
};
BOOST_GLOBAL_FIXTURE(GlobalFixture);

// Define a fixture to be set up and torn down for every test case invocation (assuming all test cases belong to the master test suite).
struct MasterTestSuiteFixture {
    MasterTestSuiteFixture()
    {
        BOOST_TEST_MESSAGE("MasterTestSuiteFixture() ...");
        remove(outputFile.c_str());
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
    mgr.initSim(); // note: this call writes initial state to file

    NetCDFReaderPtr fileReader(new NetCDFReader(outputFile));

    const InitCondPtr initCond = mgr.initConditions();
    BOOST_CHECK_EQUAL(initCond->nx(), fileReader->nx());
    BOOST_CHECK_EQUAL(initCond->ny(), fileReader->ny());
    BOOST_CHECK_EQUAL(initCond->width(), fileReader->width());
    BOOST_CHECK_EQUAL(initCond->height(), fileReader->height());
    BOOST_CHECK_EQUAL(initCond->dx(), fileReader->dx());
    BOOST_CHECK_EQUAL(initCond->dy(), fileReader->dy());
    CHECK_FIELDS_EQUAL(initCond->H(), fileReader->H());
    CHECK_FIELDS_EQUAL(initCond->eta(), fileReader->eta());
    CHECK_FIELDS_EQUAL(mgr.U(), fileReader->U());
    CHECK_FIELDS_EQUAL(mgr.V(), fileReader->V());
}

BOOST_AUTO_TEST_CASE(StateAtNTimestepsWrittenAndReadBack)
{
    Manager &mgr = Manager::instance();
    mgr.initSim();

    NetCDFReaderPtr fileReader(new NetCDFReader(outputFile));

    // simulate a few steps
    const int n = 5;
    for (int i = 0; i < n; ++i)
        mgr.execNextStep();

    CHECK_FIELDS_EQUAL(mgr.initConditions()->H(), fileReader->H());

    BOOST_CHECK_EQUAL(n, fileReader->etaTimesteps());
    CHECK_FIELDS_EQUAL(mgr.eta(), fileReader->eta());

    BOOST_CHECK_EQUAL(n, fileReader->UTimesteps());
    CHECK_FIELDS_EQUAL(mgr.U(), fileReader->U());

    BOOST_CHECK_EQUAL(n, fileReader->VTimesteps());
    CHECK_FIELDS_EQUAL(mgr.V(), fileReader->V());
}

BOOST_AUTO_TEST_SUITE_END()
