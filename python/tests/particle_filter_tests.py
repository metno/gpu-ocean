import unittest
import sys
import time

import xmlrunner
# install xmlrunner by
# $ sudo easy_install unittest-xml-reporting

#import testUtils
from dataAssimilation.CPUDrifterTest import CPUDrifterTest
from dataAssimilation.GPUDrifterTest import GPUDrifterTest
from dataAssimilation.DrifterEnsembleTest import DrifterEnsembleTest

if (len(sys.argv) < 1):
    print ("Usage:")
    print ("\t %s [cpuOnly]  [jenkins]" % sys.argv[0])
    exit()

# In order to format the test report so that Jenkins can read it:
jenkins = False
cpuOnly = False
if (len(sys.argv) > 1):
    if sys.argv[1] == "1":
        cpuOnly = True
if (len(sys.argv) > 2):
    if (sys.argv[1].lower() == "jenkins"):
        jenkins = True

if (jenkins):
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports'))


# Define the tests that will be part of our test suite:
test_classes_to_run = None
if cpuOnly:
    test_classes_to_run = [CPUDrifterTest]
else:
    test_classes_to_run = [CPUDrifterTest, GPUDrifterTest, DrifterEnsembleTest]

    
loader = unittest.TestLoader()
suite_list = []
for test_class in test_classes_to_run:
    suite = loader.loadTestsFromTestCase(test_class)
    suite_list.append(suite)

big_suite = unittest.TestSuite(suite_list)
unittest.TextTestRunner(verbosity=2).run(big_suite)
