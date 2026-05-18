# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Unit tests for dict_tools functions"""

import unittest

import numpy as np

from .gitl_tools import validate_dict_keys, param_order_to_list, \
    remove_subnormals, as_f32_normal

class TestValidateDictKeys(unittest.TestCase):
    """
    Tests for the function that checks if a dict's keys match the keys in a
    given iterable
    """

    def test_success(self):
        """Good inputs success without issues"""
        d = {'a':0, 'b':1, 'c':3}
        keys = ['a', 'b', 'c']
        validate_dict_keys(d, keys)
        pass


    def test_invalid_d(self):
        """invalid inputs caught"""
        d = ['a', 'b', 'c']
        keys = ['a', 'b', 'c']
        with self.assertRaises(TypeError):
            validate_dict_keys(d, keys, custom_exception=Exception)
        pass


    def test_invalid_keys(self):
        """invalid inputs caught"""
        d = {'a':0, 'b':1, 'c':3}
        keys = [{'a':0}, 'b', 'c']
        with self.assertRaises(TypeError):
            validate_dict_keys(d, keys, custom_exception=Exception)
        pass


    def test_invalid_custom_exception(self):
        """invalid inputs caught"""
        d = {'a':0, 'b':1, 'c':3}
        keys = ['a', 'b', 'c']
        with self.assertRaises(TypeError):
            validate_dict_keys(d, keys, custom_exception=None)
        pass


    def test_ok_iterables(self):
        """check that hashable inputs for keys are OK"""
        d = {'a':0, 'b':1, 'c':3}
        keyslist = [{'a', 'b', 'c'},
                    ['a', 'b', 'c'],
                    ('a', 'b', 'c'),
                    d.keys(),
                    ]

        for keys in keyslist:
            validate_dict_keys(d, keys)
            pass
        pass


    def test_missing_keys(self):
        """missing keys caught as expected"""
        d = {'a':0, 'b':1, 'c':3}
        keys = ['a', 'b']
        with self.assertRaises(TypeError):
            validate_dict_keys(d, keys)
        with self.assertRaises(KeyError):
            validate_dict_keys(d, keys, custom_exception=KeyError)
        pass


    def test_extra_keys(self):
        """extra keys caught as expected"""
        d = {'a':0, 'b':1, 'c':3}
        keys = ['a', 'b', 'c', 'd']
        with self.assertRaises(TypeError):
            validate_dict_keys(d, keys)
        with self.assertRaises(KeyError):
            validate_dict_keys(d, keys, custom_exception=KeyError)
        pass


    def test_missing_and_extra_keys(self):
        """missing/extra keys caught as expected"""
        d = {'a':0, 'b':1, 'c':3}
        keys = ['a', 'b', 'd']
        with self.assertRaises(TypeError):
            validate_dict_keys(d, keys)
        with self.assertRaises(KeyError):
            validate_dict_keys(d, keys, custom_exception=KeyError)
        pass



class TestParamOrderToList(unittest.TestCase):
    """
    Test reordering between HOWFSC output form (to parameters) and input form
    (from telemetry)

    success
    exact answer
    invalid inputs (outer not list, inner not list, inner lengths not right)
    output length right
    """

    def test_success(self):
        """good inputs return successfully"""
        nestlist = []
        nestlist.append([1, 2, 3, 4])
        nestlist.append([5, 6, 7, 8])
        nestlist.append([9, 10, 11, 12])
        param_order_to_list(nestlist)
        pass


    def test_exact(self):
        """Get expected answer"""
        nestlist = []
        nestlist.append([1, 2, 3, 4])
        nestlist.append([5, 6, 7, 8])
        nestlist.append([9, 10, 11, 12])
        target = [1, 2, 2, 3, 3, 4, 4,
                  5, 6, 6, 7, 7, 8, 8,
                  9, 10, 10, 11, 11, 12, 12]

        output = param_order_to_list(nestlist)
        self.assertTrue(output == target)
        pass


    def test_outer_not_list(self):
        """fails as expected if outer loop not iterable"""
        perrlist = [1, 1j, None]

        for p in perrlist:
            with self.assertRaises(TypeError):
                param_order_to_list(p)
            pass
        pass


    def test_inner_not_list(self):
        """fails as expected if inner loop not iterable"""
        perrlist = [[1, 2, 3, 4],
                    [[1], [2], [3], 4], # must all be iterable
                    ]

        for p in perrlist:
            with self.assertRaises(TypeError):
                param_order_to_list(p)
            pass
        pass


    def test_inner_empty(self):
        """fails as expected if inner loop iterable but empty"""
        perrlist = [[[], [], [], []],
                    ]

        for p in perrlist:
            with self.assertRaises(TypeError):
                param_order_to_list(p)
            pass
        pass


    def test_inner_not_same_length(self):
        """fails as expected if inner loop not iterable"""
        perrlist = [[[1], [2], [3], [4, 5]],
                    ]

        for p in perrlist:
            with self.assertRaises(TypeError):
                param_order_to_list(p)
            pass
        pass


    def test_edge_cases_OK(self):
        """some edge cases that should be supported succeed"""
        perrlist = [[[1], [2], [3], [4]], # length 1 inner
                    [[1, 2, 3]], # length 1 outer
                    [[1]], # length 1 inner and outer
                    # indifferent to internal type
                    [[1, 0], ['t', None], [-1.5, 1j], [(4, 5), 16.6]],
                    ]

        for p in perrlist:
            param_order_to_list(p)
            pass
        pass


    def test_output_length(self):
        """Verify output length is as expected"""
        Alist = [1, 3, 7, 19]
        Blist = [1, 5, 10, 17]

        for A in Alist:
            for B in Blist:
                inner = [0]*B
                nestlist = [inner]*A
                output = param_order_to_list(nestlist)
                self.assertTrue(len(output) == A*(2*B - 1))
                pass
            pass
        pass


class TestRemoveSubnormals(unittest.TestCase):
    """
    Tests for stripping subnormals out of numpy arrays
    """

    def setUp(self):
        self.rng = np.random.default_rng(17273747)
        self.data = self.rng.random(25).reshape((5, 5))
        # random in [0, 1], random + 2 in [2, 3] and can't be subnormal or -0
        self.data += 2
        pass


    def test_valid(self):
        """Correct answer with valid inputs"""
        output = remove_subnormals(self.data)
        self.assertTrue((output == self.data).all())
        pass


    def test_valid_infs(self):
        """Correct answer with valid inputs and infs"""
        d2 = self.data.copy()
        d2[0, 0] = np.inf
        output = remove_subnormals(d2)
        self.assertTrue((output[~np.isinf(d2)] == d2[~np.isinf(d2)]).all())
        self.assertTrue(np.isinf(output[np.isinf(d2)]).all())
        pass


    def test_valid_nans(self):
        """Correct answer with valid inputs and nans"""
        d2 = self.data.copy()
        d2[0, 0] = np.nan
        output = remove_subnormals(d2)
        self.assertTrue((output[~np.isnan(d2)] == d2[~np.isnan(d2)]).all())
        self.assertTrue(np.isnan(output[np.isnan(d2)]).all())
        pass


    def test_subnormals_float32(self):
        """Correct answer with subnormals, float32 edition"""
        dsub = self.data.copy().astype('float32')
        dsub *= np.finfo(dsub.dtype).tiny
        dsub *= 0.1 # for margin
        target = np.zeros_like(dsub)
        output = remove_subnormals(dsub)
        self.assertTrue((output == target).all())
        pass


    def test_subnormals_float64(self):
        """Correct answer with subnormals, float64 edition"""
        dsub = self.data.copy().astype('float64')
        dsub *= np.finfo(dsub.dtype).tiny
        dsub *= 0.1 # for margin
        target = np.zeros_like(dsub)
        output = remove_subnormals(dsub)
        self.assertTrue((output == target).all())
        pass


    def test_negzero_float32(self):
        """Correct answer with negative zero, float32 edition"""
        nrow = 2
        ncol = 2
        # don't astype from float64 to avoid overflow (default is float64)
        dneg = np.empty((nrow, ncol), dtype=np.float32)
        for j in range(nrow):
            for k in range(ncol):
                dneg[j, k] = -0
                pass
            pass

        target = np.zeros((nrow, ncol)).astype(dneg.dtype)
        output = remove_subnormals(dneg)
        self.assertTrue((output == target).all())
        pass


    def test_negzero_float64(self):
        """Correct answer with negative zero, float64 edition"""
        nrow = 2
        ncol = 2
        dneg = np.empty((nrow, ncol), dtype=np.float64)
        for j in range(nrow):
            for k in range(ncol):
                dneg[j, k] = -0
                pass
            pass

        target = np.zeros((nrow, ncol)).astype(dneg.dtype)
        output = remove_subnormals(dneg)
        self.assertTrue((output == target).all())
        pass


    def test_invalid_input(self):
        """Invalid inputs caught"""
        perrlist = [(1, 2, 3), None, 'txt', {'a':1, 'b':2}]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                remove_subnormals(perr)
            pass
        pass



class TestAsF32Normal(unittest.TestCase):
    """
    Tests for stripping subnormals out of scalars (assumed float32)
    """

    def test_valid(self):
        """Correct answer with valid inputs"""
        for val in [2, 2.1, 0, -1.5, 100000.5]:
            output = as_f32_normal(val)
            self.assertTrue(output == val)
            pass
        pass


    def test_valid_infs(self):
        """Correct answer with infs"""
        for val in [np.inf, -np.inf]:
            output = as_f32_normal(val)
            self.assertTrue(output == val)
            pass
        pass


    def test_valid_nans(self):
        """Correct answer with nan"""
        val = np.nan
        output = as_f32_normal(val)
        self.assertTrue(np.isnan(output))
        pass


    def test_subnormals_float32(self):
        """Correct answer with subnormals, float32 edition"""
        dsub = np.finfo(np.float32).tiny
        dsub *= 0.1

        for val in [dsub, -dsub]:
            output = as_f32_normal(val)
            self.assertFalse(output == val)
            self.assertTrue(output == 0)
            pass
        pass


    def test_negzero_float32(self):
        """Correct answer with negative 0, float32 edition"""
        output = as_f32_normal(-0.0)
        self.assertTrue(output == 0)
        pass


    def test_invalid_input(self):
        """Invalid inputs caught"""
        perrlist = [(1, 2, 3), None, 'txt', {'a':1, 'b':2}, np.eye(5)]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                as_f32_normal(perr)
            pass
        pass



if __name__ == '__main__':
    unittest.main()
