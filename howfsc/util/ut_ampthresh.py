# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
# pylint: disable=maybe-no-member
"""Unit test suite for for ampthresh."""
import unittest
import os

import numpy as np
from astropy.io import fits

from .ampthresh import ampthresh


class TestAmpthreshInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""

    def test_ampthresh_incorrect_type_input_0(self):
        """Verify array inputs with invalid format are caught."""
        with self.assertRaises(ValueError):
            for badInput in ([], np.ones((3, 3, 3)), 'string', np.ones((10,))):
                ampthresh(badInput)

    def test_ampthresh_incorrect_type_input_1(self):
        """Verify nBin inputs with invalid formats are caught."""
        with self.assertRaises(ValueError):
            for nBin in ([], np.ones((3, 3, 3)), 'string', np.ones((10,)),
                         0, -10, 5.5):
                ampthresh(np.eye(10), nBin=nBin)


class TestAmpthresh(unittest.TestCase):
    """Perform unit tests for AMPTHRESH."""

    def test_ampthresh_with_noise(self):
        """Check that ampthresh recovers the pupil from a noisy image."""
        localpath = os.path.dirname(os.path.abspath(__file__))
        pupilfile = os.path.join(localpath, 'testdata',
                                 'amp_xc20.30_yc-10.50.fits')
        pupil0 = fits.getdata(pupilfile)
        pupil = pupil0 + 0.1*np.random.rand(pupil0.shape[0], pupil0.shape[1])
        boolMask = ampthresh(pupil)

        self.assertTrue(np.sum(boolMask == pupil0)/pupil0.size > 0.99)

    def test_ampthresh_uniform_false(self):
        """Verify the function fails as expected if all values are the same."""
        array_shape = (100, 100)
        array_in = np.zeros(array_shape, dtype=complex)
        array_out = ampthresh(array_in)
        array_expected = np.zeros(array_shape, dtype=bool)

        self.assertTrue(np.array_equal(array_out, array_expected))
        
    def test_ampthresh_uniform_true(self):
        """Verify the function fails as expected if all values are the same."""
        array_shape = (100, 100)
        coef_list = [0.1, 1, 2.5, 2j, 1+1j]
        for coef in coef_list:
            array_in = coef * np.ones(array_shape, dtype=complex)
            array_out = ampthresh(array_in)
            array_expected = np.ones(array_shape, dtype=bool)

            self.assertTrue(np.array_equal(array_out, array_expected))


if __name__ == '__main__':
    unittest.main()
