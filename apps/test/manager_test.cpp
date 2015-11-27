#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mgr test
#include <boost/test/unit_test.hpp>
#define BOOST_TEST_LOG_LEVEL "all"

#include "manager.h"
#include "programoptions.h"
#include "simulator.h"
#include <vector>
#include <string>
#include <utility>
#include <boost/algorithm/string.hpp>

using namespace std;

// Returns an (argc, argv) pair from the words in a string. The word "argv0" is automatically prepended.
static pair<int, char **> s2args(const string &s)
{
    vector<string> words;
    boost::split(words, s, boost::is_any_of("\t "));
    words.insert(words.begin(), "argv0");
    const int argc = words.size();
    char **argv = (char **)malloc(argc * sizeof(char *));
    for (int i = 0; i < words.size(); ++i) {
        *(argv + i) = (char *)malloc(words.at(i).size() + 1);
        strcpy(*(argv + i), words.at(i).c_str());
    }

    return make_pair(argc, argv);
}


struct ManagerFixture {
	ManagerFixture()
	{
		BOOST_TEST_MESSAGE( "setup fixture" );

        pair<int, char **> args = s2args("--duration 0 --nx 2 --ny 2 --width 1 --height 1 --bathymetryNo 0 --etaNo 0");
        Manager::init(args.first, args.second);

		manager = &Manager::instance();
	}

    ~ManagerFixture()
    {
    	BOOST_TEST_MESSAGE( "teardown fixture" );
    }

    Manager *manager;
};

BOOST_FIXTURE_TEST_SUITE(TestManager, ManagerFixture)

BOOST_AUTO_TEST_CASE(TestManagerInitialized)
{
	BOOST_CHECK_EQUAL(manager->initialized(), true);
}

BOOST_AUTO_TEST_SUITE_END()
