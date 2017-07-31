"""
Module to run the tests within the scikit-criteria tests folder using the unittest architecture. This script looks for
all modules matching the tests_*.py syntax. Tests currently utilized are as follows:

"""

### Imports ###
import unittest, os

### Begin main testing run ###
if __name__ == "__main__":
    # Create a test suite. This will hold the identified tests.
    suite = unittest.TestSuite()

    # Discover all tests in the tests folder and add them to the suite
    suite.addTests(unittest.TestLoader().discover(start_dir=os.getcwd(), pattern='test_*.py'))

    # Create the result object
    res = unittest.TestResult()

    # Run the test suite
    unittest.TextTestRunner(verbosity=2).run(suite)
