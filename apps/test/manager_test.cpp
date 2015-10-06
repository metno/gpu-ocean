#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE mgr test
#include <boost/test/unit_test.hpp>

#define BOOST_TEST_LOG_LEVEL "all"

#include "manager.h"
#include "programoptions.h"
#include "simulator.h"

struct ManagerFixture {
	ManagerFixture()
	{
		BOOST_TEST_MESSAGE( "setup fixture" );

		Manager::init(0, NULL);

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
