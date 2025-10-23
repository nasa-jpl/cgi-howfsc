# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Test suite for remove_ptt.py."""
from math import isclose
import os
import unittest

import numpy as np

from . import mft, remove_ptt


HERE = os.path.dirname(os.path.abspath(__file__))

not_a_2d_array = (True, -1, 1, 1.1, 1j, np.ones((5, 5, 4)), 'string', {'a': 2})
not_a_1d_array = (True, -1, 1, 1.1, 1j, np.ones((5, 5)), 'string', {'a': 2})
not_a_dict = (True, -1, 1, 1.1, 1j, np.ones(5), np.ones((5, 5)), 'string')
not_a_real_scalar = (1j, np.ones(5), 'string', {'a': 2})
not_a_real_positive_scalar = (-1, 0, 1j, np.ones(5), 'string', {'a': 2})
not_a_positive_scalar_integer = (-1, 0, 1.1, 1j, np.ones(5), np.ones((5, 5)),
                                 'string')
not_a_string = (True, -1, 0, 1.1, 1j, np.ones(5), np.ones((5, 5)),  {'a': 2})


class TestRemovePTTDirectly(unittest.TestCase):
    """Unit tests for remove_ptt.fit_and_remove_ptt_directly()."""

    def test_remove_ptt_directly(self):
        """Test that piston, tip, and tilt are fully removed."""
        N = 100
        x = np.arange(-N/2, N/2)/N
        [X, Y] = np.meshgrid(x, x)

        arrayIn = 2*X - 3*Y + 0.8
        mask = np.ones_like(X)
        mask[X*X+Y*Y > 0.5**2] = 0

        arrayOut, arrayFitted = remove_ptt.fit_and_remove_ptt_directly(
            arrayIn, mask)

        self.assertTrue(isclose(np.mean(arrayOut), 0, abs_tol=1e-12))
        self.assertTrue(np.max(np.abs(arrayIn - arrayFitted)) < 1e-8)

    def test_remove_ptt_directly_inputs(self):
        """Test the inputs of fit_and_remove_ptt_directly."""
        arrayToFit = np.ones((10, 11))
        mask = np.ones((10, 11))

        for bad_val in not_a_2d_array:
            with self.assertRaises(ValueError):
                remove_ptt.fit_and_remove_ptt_directly(bad_val, mask)
        for bad_val in not_a_2d_array:
            with self.assertRaises(ValueError):
                remove_ptt.fit_and_remove_ptt_directly(arrayToFit, bad_val)

        # Check shape equality
        with self.assertRaises(ValueError):
            remove_ptt.fit_and_remove_ptt_directly(
                np.ones((10, 10)), np.ones((11, 12)))


if __name__ == '__main__':
    unittest.main()
