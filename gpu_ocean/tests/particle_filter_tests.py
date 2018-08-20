import unittest
import sys
import time

import xmlrunner
# install xmlrunner by
# $ sudo easy_install unittest-xml-reporting

#import testUtils
from dataAssimilation.CPUDrifter_test import CPUDrifterTest
from dataAssimilation.GPUDrifter_test import GPUDrifterTest
from dataAssimilation.OceanStateNoise_test import OceanStateNoiseTest
from dataAssimilation.DrifterEnsemble_test import DrifterEnsembleTest
from dataAssimilation.CPUDrifterEnsemble_test import CPUDrifterEnsembleTest

def printSupportedTests():
    print("Supported tests:")
    print("0: All, 1: CPUDrifter, 2: GPUDrifter, 3: DrifterEnsembleTest, "
           + "4: CPUDrifterEnsembleTest, 5: OceanStateNoise")


if (len(sys.argv) < 2):
    print("Usage:")
    print("\t %s tests  [jenkins]" % sys.argv[0])
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
    test_classes_to_run = [CPUDrifterTest, GPUDrifterTest,
                           DrifterEnsembleTest, CPUDrifterEnsembleTest,
                           OceanStateNoiseTest]
elif tests == 1:
    test_classes_to_run = [CPUDrifterTest]
elif tests == 2:
    test_classes_to_run = [GPUDrifterTest]
elif tests == 3:
    test_classes_to_run = [DrifterEnsembleTest]
elif tests == 4:
    test_classes_to_run = [CPUDrifterEnsembleTest]
elif tests == 5:
    test_classes_to_run = [OceanStateNoiseTest]
else:
    print("Error: " + str(tests) + " is not a supported test number...")
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
