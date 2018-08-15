import unittest
import sys
import time

import xmlrunner
# install xmlrunner by
# $ sudo easy_install unittest-xml-reporting

#import testUtils
from pyopenclTests.OpenCLArray2D_test import OpenCLArray2DTest

if (len(sys.argv) < 1):
    print("Usage:")
    print("\t %s  [jenkins]" % sys.argv[0])
    exit()

# In order to format the test report so that Jenkins can read it:
jenkins = False
if (len(sys.argv) > 1):
    if (sys.argv[1].lower() == "jenkins"):
        jenkins = True

if (jenkins):
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports'))


# Define the tests that will be part of our test suite:
test_classes_to_run = None
test_classes_to_run = [OpenCLArray2DTest]


loader = unittest.TestLoader()
suite_list = []
for test_class in test_classes_to_run:
    suite = loader.loadTestsFromTestCase(test_class)
    suite_list.append(suite)

big_suite = unittest.TestSuite(suite_list)
results = unittest.TextTestRunner(verbosity=2).run(big_suite)

sys.exit(not results.wasSuccessful())
