#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mgr test
#include <boost/test/unit_test.hpp>
#define BOOST_TEST_LOG_LEVEL "all"

#include "utils/testutils.h"
#include "manager.h"
#include "programoptions.h"
#include "simulator.h"

struct ManagerFixture {
	ManagerFixture()
	{
		BOOST_TEST_MESSAGE( "setup fixture" );

        std::pair<int, char **> args = s2args("--duration 0 --nx 2 --ny 2 --width 1 --height 1 --bathymetryNo 0 --etaNo 0");
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
