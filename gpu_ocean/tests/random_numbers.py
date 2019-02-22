import unittest
import sys
import time

import xmlrunner
# install xmlrunner by
# $ sudo easy_install unittest-xml-reporting

#import testUtils
from stochastic.OceanStateNoise_test import OceanStateNoiseTest
from stochastic.RandomNumbers_test import RandomNumbersTest
from stochastic.RandomNumbers_LCG_test import RandomNumbersLCGTest

def printSupportedTests():
    print ("Supported tests:")
    print ("0: All, 1: RandomNumbers, 2: OceanStateNoise, \
            3: RandomNumbersLCGTest")


if (len(sys.argv) < 2):
    print ("Usage:")
    print ("\t %s tests  [jenkins]" % sys.argv[0])
    printSupportedTests()
    exit()
tests = int(sys.argv[1])

# In order to format the test report so that Jenkins can read it:
jenkins = False
if (len(sys.argv) > 2):
    if (sys.argv[1].lower() == "jenkins"):
        jenkins = True

if (jenkins):
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports'))


# Define the tests that will be part of our test suite:
test_classes_to_run = None
if tests == 0:
    test_classes_to_run = [RandomNumbersTest, 
                           OceanStateNoiseTest,
                           RandomNumbersLCGTest]
elif tests == 1:
    test_classes_to_run = [RandomNumbersTest]
elif tests == 2:
    test_classes_to_run = [OceanStateNoiseTest]
elif tests == 3:
    test_classes_to_run = [RandomNumbersLCGTest ]
else:
    print ("Error: " + str(tests) + " is not a supported test number...")
    printSupportedTests()
    exit()

    
loader = unittest.TestLoader()
suite_list = []
for test_class in test_classes_to_run:
    suite = loader.loadTestsFromTestCase(test_class)
    suite_list.append(suite)

big_suite = unittest.TestSuite(suite_list)
results = unittest.TextTestRunner(verbosity=2).run(big_suite)

sys.exit(not results.wasSuccessful())
