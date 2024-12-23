import unittest
import sys
import os

def load_all_tests():
    """
    Aggregates all tests across the project by discovering
    any files named 'test_*.py' within this directory.
    """
    test_loader = unittest.TestLoader()
    # Discover tests in the current directory (test folder)
    tests = test_loader.discover(os.path.dirname(__file__), pattern='test_*.py')
    test_suite = unittest.TestSuite()
    test_suite.addTests(tests)
    return test_suite

if __name__ == '__main__':
    suite = load_all_tests()
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)