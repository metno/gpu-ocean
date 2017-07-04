import unittest
import sys
import time

import xmlrunner
# install xmlrunner by
# $ sudo easy_install unittest-xml-reporting

#import testUtils
from schemes.FBLtest import FBLtest
from schemes.CTCStest import CTCStest
from schemes.CDKLM16test import CDKLM16test
from schemes.KP07test import KP07test

def printSupportedSchemes():
    print ("Supported schemes:")
    print ("1: FBL, 2: CTCS, 3: CDKLM16, 4: KP07")
    

if (len(sys.argv) < 2):
    print ("Usage:")
    print ("\t %s scheme [jenkins]" % sys.argv[0])
    printSupportedSchemes()
    exit()
scheme = int(sys.argv[1])

# In order to format the test report so that Jenkins can read it:
jenkins = False
if (len(sys.argv) > 2):
    if (sys.argv[2].lower() == "jenkins"):
        jenkins = True

if (jenkins):
    unittest.main(testRunner=xmlrunner.XMLTestRunner(output='test-reports'))


# Define the tests that will be part of our test suite:
test_classes_to_run = None
if scheme == 1:
    test_classes_to_run = [FBLtest]
elif scheme == 2:
    test_classes_to_run = [CTCStest]
elif scheme == 3:
    test_classes_to_run = [CDKLM16test]
elif scheme == 4:
    test_classes_to_run = [KP07test]
else:
    print("Error: " + str(scheme) + " is not a supported scheme...")
    printSupportedSchemes()
    exit()


loader = unittest.TestLoader()
suite_list = []
for test_class in test_classes_to_run:
    suite = loader.loadTestsFromTestCase(test_class)
    suite_list.append(suite)

big_suite = unittest.TestSuite(suite_list)
unittest.TextTestRunner(verbosity=2).run(big_suite)
