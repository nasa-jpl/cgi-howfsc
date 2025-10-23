# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit tests for generic YAML dictionary writing
"""

import unittest
from unittest.mock import patch
import tempfile
import os

import yaml

from .writeyaml import writeyaml
from .loadyaml import loadyaml

class TestOnlyException(Exception):
    """Exception to be used below for the custom_exception input"""
    pass

class TestWriteYAML(unittest.TestCase):
    """
    Test successful and failed loads
    Test default exception behavior
    """

    def test_good_input(self):
        """
        Verify a valid input writes successfully
        """
        outdict = {'a': 1, 'b':{'c':2, 'd':3}}

        try:
            (fd, name) = tempfile.mkstemp()
            writeyaml(outdict, name)
        finally:
            # mkstemp() does not clean up after itself, but it does have a name
            os.close(fd)
            os.unlink(name)
            pass
        pass


    def test_readback(self):
        """
        Verify a valid input writes successfully and can be reread to give the
        same contents
        """
        outdict = {'a': 1, 'b':{'c':2, 'd':3}}

        try:
            (fd, name) = tempfile.mkstemp()
            writeyaml(outdict, name)
            readback = loadyaml(name)
            self.assertTrue(outdict == readback)
        finally:
            # mkstemp() does not clean up after itself, but it does have a name
            os.close(fd)
            os.unlink(name)
            pass
        pass



    @patch('builtins.open')
    def test_unable_to_open(self, mock_open):
        """
        Fail when output file is not writable
        """
        outdict = {'a': 1, 'b':{'c':2, 'd':3}}
        path = 'does_not_exist_or_matter' # not written to anyway
        mock_open.side_effect = [OSError]

        with self.assertRaises(TestOnlyException):
            writeyaml(outdict, path, custom_exception=TestOnlyException)
            pass
        pass


    @patch('yaml.dump')
    def test_unable_to_dump(self, mock_dump):
        """
        Fail when output file is not writable
        """
        outdict = {'a': 1, 'b':{'c':2, 'd':3}}
        mock_dump.side_effect = [yaml.YAMLError]

        try:
            (fd, name) = tempfile.mkstemp()
            with self.assertRaises(TestOnlyException):
                writeyaml(outdict, name, custom_exception=TestOnlyException)
                pass
            pass
        finally:
            # mkstemp() does not clean up after itself, but it does have a name
            os.close(fd)
            os.unlink(name)
            pass
        pass


    def test_invalid_outdict(self):
        """Invalid inputs caught"""
        # not dicts
        perrlist = [(1, 1, 2), [2, 3, 3], 'txt', 0, None, {3, 3}]
        path = 'does_not_matter'

        for perr in perrlist:
            with self.assertRaises(TypeError):
                writeyaml(perr, path)
            pass
        pass


    def test_invalid_path(self):
        """Invalid inputs caught"""
        # not str
        perrlist = [(1, 1, 2), [2, 3, 3], {'a': 1}, 0, None, {3, 3}]
        outdict = {'a': 1, 'b':{'c':2, 'd':3}}

        for perr in perrlist:
            with self.assertRaises(TypeError):
                writeyaml(outdict, perr)
            pass
        pass
