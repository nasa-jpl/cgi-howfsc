# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit tests for generic YAML loading
"""

import unittest
import os

from .loadyaml import loadyaml

class TestOnlyException(Exception):
    """Exception to be used below for the custom_exception input"""
    pass

class TestLoadYAML(unittest.TestCase):
    """
    Test successful and failed loads
    Test default exception behavior
    """

    def test_good_input(self):
        """
        Verify a valid input loads successfully
        """
        localpath = os.path.dirname(os.path.abspath(__file__))
        fn = os.path.join(localpath, 'testdata', 'ut_valid.yaml')
        loadyaml(fn)
        pass


    def test_missing_file(self):
        """
        Fail when input file is missing
        """
        fn = 'does_not_exist'

        with self.assertRaises(TestOnlyException):
            loadyaml(fn, custom_exception=TestOnlyException)
            pass
        pass


    def test_not_yaml(self):
        """
        Fail when input is not valid YAML
        """
        localpath = os.path.dirname(os.path.abspath(__file__))
        fn = os.path.join(localpath, 'testdata', 'ut_not_yaml.fits')

        with self.assertRaises(TestOnlyException):
            loadyaml(fn, custom_exception=TestOnlyException)
            pass
        pass

    def test_default_exception(self):
        """
        Default behavior raises as expected
        """
        localpath = os.path.dirname(os.path.abspath(__file__))
        fn = os.path.join(localpath, 'testdata', 'ut_not_yaml.fits')

        with self.assertRaises(Exception):
            loadyaml(fn)
            pass
        pass
