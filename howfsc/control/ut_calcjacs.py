# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Unit tests for calcjacs.py"""
import unittest
from unittest.mock import patch
import os
import numpy as np

from howfsc.model.mode import CoronagraphMode

from .calcjacs import get_ndhpix, calcjacs_mp, calcjacs_sp, calcjacs, \
                      generate_ijlist, CalcJacsException

cfgpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'model', 'testdata', 'widefov', 'widefov.yaml')

class Testget_ndhpix(unittest.TestCase):
    """Unit test suite for utility get_ndhpix(cfg) """

    def setUp(self):
        """Preload configuration so we don't have to waste time reloading."""
        self.cfg = CoronagraphMode(cfgpath)

    def test_successful_run(self):
        """valid cfg"""
        get_ndhpix(self.cfg)
        pass

    def test_correct_answer(self):
        """test get correct answer """

        ndhpix_test = get_ndhpix(self.cfg)
        ndhpix_ref = np.cumsum([0]+[len(sl.dh_inds) for sl in self.cfg.sl_list])

        # ndhpix is a list, check same length, then same values
        self.assertTrue(len(ndhpix_test) == len(ndhpix_ref))
        self.assertTrue(np.all(np.array(ndhpix_test) == np.array(ndhpix_ref)))

    def test_invalid_config(self):
        """Fails as expected on bad cfg input."""
        for perr in [0, np.ones((5, 5)), 'txt']:
            with self.assertRaises(TypeError):
                get_ndhpix(cfg=perr)

class TestCalcJacs_mp(unittest.TestCase):
    """
       Unit test suite for Jacobian calculator with parallel processes
       Most parameters are checked in calcjacs() which is called
       by each process created by calcjacs_mp(). Here we only test
       parameters required for creating the processes, and success
       of the calculation using parallel processes.
       calcjacs_mp(cfg, ijlist, dm0list=None, jacmethod='normal',
                   num_process=1):
    """

    def setUp(self):
        """Preload configuration so we don't have to waste time reloading."""
        self.cfg = CoronagraphMode(cfgpath)

    def test_successful_run(self):
        """Verify default single process"""
        calcjacs_mp(self.cfg, [0])
        pass

    def test_successful_run_mp(self):
        """Verify multiprocessing runs"""
        calcjacs_mp(self.cfg, [0, 1, 2, 3], num_process=4)
        pass

    def test_sp_mp_same_answer(self):
        """calcjacs() and calcjacs_mp() should get same answer"""
        jac_s = calcjacs_sp(self.cfg, [0, 1, 2, 3])
        jac_m = calcjacs_mp(self.cfg, [0, 1, 2, 3], num_process=4)

        # jac_s and jac_m should be identical
        self.assertTrue(jac_s.shape == jac_m.shape)

        # raises assertion error if fails (default rtol = 1e-7):
        self.assertTrue(np.allclose(jac_s, jac_m, rtol=1e-7, equal_nan=True))

    def test_invalid_ijlist(self):
        """Fails as expected on bad ijlist input."""
        for perr in [0, np.ones((5, 5)), 'txt']:
            with self.assertRaises(TypeError):
                calcjacs_mp(cfg=self.cfg, ijlist=perr)

class TestCalcJacs_sp(unittest.TestCase):
    """
       Unit test suite for Jacobian calculator with single process.
       Most parameters are checked in calcjacs() which then calls
       calcjacs_sp() if specified. Here we only test simple functionality
    """

    def setUp(self):
        """Preload configuration so we don't have to waste time reloading."""
        self.cfg = CoronagraphMode(cfgpath)

    def test_successful_run(self):
        """Verify default single process"""
        calcjacs_sp(self.cfg, [0])
        pass

    def test_invalid_ijlist(self):
        """Fails as expected on bad ijlist input."""
        for perr in [0, np.ones((5, 5)), 'txt']:
            with self.assertRaises(TypeError):
                calcjacs_sp(cfg=self.cfg, ijlist=perr)

class TestCalcJacs(unittest.TestCase):
    """Unit test suite for Jacobian calculator."""

    def setUp(self):
        """Preload configuration so we don't have to waste time reloading."""
        self.cfg = CoronagraphMode(cfgpath)

    def test_successful_run(self):
        """Verify no main-body errors in FT logic."""
        calcjacs(self.cfg, [0])
        pass

    def test_output_size(self):
        """Verify output sizes are as documented."""
        ijlist = [0, 1, 2, 3, 4]
        out = calcjacs(self.cfg, ijlist)
        npix = 0
        for sl in self.cfg.sl_list:
            npix += len(sl.dh_inds)

        self.assertTrue(out.shape == (2, len(ijlist), npix))

    def test_fast_jac_against_normal_jac(self):
        """Verify that fast and regular Jacobians are the same."""
        # choose actuators within the open pupil on both DMs
        ijlist = [504, 506, 48*48+504, 48*48+506]
        relTol = 1e-3

        outNormal = calcjacs(self.cfg, ijlist, jacmethod='normal')
        outFast = calcjacs(self.cfg, ijlist, jacmethod='fast')

        maxVal = np.max(np.abs(outNormal))
        maxDiff = np.max(np.abs(outFast - outNormal))

        self.assertTrue(maxDiff/maxVal < relTol)

    # Failure tests
    @patch('howfsc.model.singlelambda.SingleLambda.get_inorm')
    def test_sl_inorm_zero(self, mock_inorm):
        """Test the pathological case where sl.inorm = 0."""
        mock_inorm.return_value = 0
        with self.assertRaises(CalcJacsException):
            calcjacs_sp(self.cfg, [0])

    @patch('howfsc.model.singlelambda.SingleLambda.proptolyot_nofpm')
    def test_epk0_zero(self, mock_epk0):
        """Test the pathological case where epk0 = 0."""
        mock_epk0.return_value = np.zeros(self.cfg.sl_list[0].lyot.e.shape, dtype='complex128')
        # ValueError should be triggered by ampthresh()
        with self.assertRaises(ValueError):
            calcjacs_sp(self.cfg, [0])

    def test_invalid_config(self):
        """Fails as expected on bad cfg input."""
        for perr in [0, np.ones((5, 5)), 'txt']:
            with self.assertRaises(TypeError):
                calcjacs(cfg=perr, ijlist=[0])

    def test_mangled_sl_list_in_config(self):
        """Fails if cfg.sl_list is a list, but not made of SingleLambdas."""
        self.cfg.sl_list = [None]*len(self.cfg.sl_list)
        with self.assertRaises(TypeError):
            calcjacs(cfg=self.cfg, ijlist=[0])

    def test_sl_list_in_config_not_list(self):
        """Fails if cfg.sl_list is not a list (or iterable equivalent)."""
        self.cfg.sl_list = 3
        with self.assertRaises(TypeError):
            calcjacs(cfg=self.cfg, ijlist=[0])

    def test_sl_list_not_in_config(self):
        """Fails if cfg.sl_list attribute is missing."""
        del self.cfg.sl_list
        with self.assertRaises(TypeError):
            calcjacs(cfg=self.cfg, ijlist=[0])

    def test_mangled_dmlist_in_config(self):
        """Fails if cfg.dmlist is a list, but not made of DMFaces."""
        self.cfg.dmlist = [None]*len(self.cfg.dmlist)
        with self.assertRaises(TypeError):
            calcjacs(cfg=self.cfg, ijlist=[0])

    def test_dmlist_in_config_not_list(self):
        """Fails if cfg.dmlist is not a list (or iterable equivalent)."""
        self.cfg.dmlist = 3
        with self.assertRaises(TypeError):
            calcjacs(cfg=self.cfg, ijlist=[0])

    def test_dmlist_not_in_config(self):
        """Fails if cfg.dmlist attribute is missing."""
        del self.cfg.dmlist
        with self.assertRaises(TypeError):
            calcjacs(cfg=self.cfg, ijlist=[0])

    def test_invalid_ijlist(self):
        """Fails as expected on bad ijlist input."""
        for perr in [0, np.ones((5, 5)), 'txt']:
            with self.assertRaises(TypeError):
                calcjacs(cfg=self.cfg, ijlist=perr)

    def test_ijlist_is_list_of_DM_indices(self):
        """Succeeds if actuators are within total number."""
        nactall = 0
        for dm in self.cfg.dmlist:
            nactall += dm.registration['nact']**2
            pass
        calcjacs(cfg=self.cfg, ijlist=[nactall-1])

    def test_ijlist_not_in_list_of_DM_indices_invalid(self):
        """Fails if actuators in ijlist aren't positive indices."""
        for perr in [[-1], [1.5]]:
            with self.assertRaises(TypeError):
                calcjacs(cfg=self.cfg, ijlist=perr)

    def test_ijlist_not_in_list_of_DM_indices(self):
        """Fails if actuators out of bounds."""
        nactall = 0
        for dm in self.cfg.dmlist:
            nactall += dm.registration['nact']**2
            pass

        for perr in [[nactall], [0, 1, nactall+1]]:
            with self.assertRaises(ValueError):
                calcjacs(cfg=self.cfg, ijlist=perr)

    def test_dm0list_not_list(self):
        """Fails if dm0list is not an iterable (other than None)."""
        for perr in [0, 'txt']:
            with self.assertRaises(TypeError):
                calcjacs(cfg=self.cfg, ijlist=[0], dm0list=perr)

    def test_dm0list_None_does_correct_default(self):
        """
        Verify that the default case and None case are the same, and both are
        equal to initmaps (which is the right default now that we're no longer
        assuming our DMs start at all zeroes).
        """
        j0 = calcjacs(cfg=self.cfg, ijlist=[0])
        j1 = calcjacs(cfg=self.cfg, ijlist=[0], dm0list=None)
        j2 = calcjacs(cfg=self.cfg, ijlist=[0], dm0list=self.cfg.initmaps)

        self.assertTrue((j1 == j0).all())
        self.assertTrue((j1 == j2).all())

    def test_dm0list_fails_when_contains_nonarray_elements(self):
        """Fails if dm0list doesn't contain arrays."""
        for perr in [[np.zeros((48, 48)), None],
                     ['a', 'a'],
                     [0, 0]]:
            with self.assertRaises(TypeError):
                calcjacs(cfg=self.cfg, ijlist=[0], dm0list=perr)

    def test_dm0list_fails_when_arrays_wrong_size(self):
        """Fails if dm0list arrays don't match cfg sizes."""
        for perr in [[np.eye(47), np.eye(49)],
                     [np.eye(48), np.eye(47)],
                     [np.eye(49), np.eye(48)],
                     [np.ones((48, 47)), np.ones((48, 48))],
                     [np.ones((48, 48)), np.ones((48, 47))],
                     [np.eye(48), np.eye(48), np.eye(48)],
                     [np.ones((48, 48))]]:
            with self.assertRaises(TypeError):
                calcjacs(cfg=self.cfg, ijlist=[0], dm0list=perr)


class TestGenerateIJList(unittest.TestCase):
    """Unit test suite for function to generate lists of valid actuators."""

    def setUp(self):
        """Preload configuration so we don't have to waste time reloading."""
        self.cfg = CoronagraphMode(cfgpath)
        pass

    def test_run_as_expected(self):
        """Check output for known case."""
        dm0 = np.eye(48)
        dm1 = np.eye(48)
        output = list(np.concatenate((np.arange(0, 48**2, 49),
                                      np.arange(48**2, 2*48**2, 49))))  # diags

        ijlist = generate_ijlist(self.cfg, [dm0, dm1])
        self.assertTrue(output == ijlist)

    def test_invalid_cfg(self):
        """Fails if not given a CoronagraphMode object."""
        dm0 = np.eye(48)
        dm1 = np.eye(48)

        for perr in [0, np.ones((5, 5)), 'txt']:
            with self.assertRaises(TypeError):
                generate_ijlist(cfg=perr, dmarrlist=[dm0, dm1])

    def test_invalid_dmlist(self):
        """Fails if not given a list of arrays."""
        for perr in [0, np.ones((5, 5)), 'txt']:
            with self.assertRaises(TypeError):
                generate_ijlist(cfg=self.cfg, dmarrlist=perr)
                pass
            pass
        pass

    def test_mismatched_DM_sizes(self):
        """Fails if DM sizes don't match config."""
        for perr in [[np.eye(47), np.eye(49)],
                     [np.eye(48), np.eye(47)],
                     [np.eye(49), np.eye(48)],
                     [np.ones((48, 47)), np.ones((48, 48))],
                     [np.ones((48, 48)), np.ones((48, 47))],
                     [np.eye(48), np.eye(48), np.eye(48)],
                     [np.ones((48, 48))]]:
            with self.assertRaises(TypeError):
                generate_ijlist(cfg=self.cfg, dmarrlist=perr)


if __name__ == '__main__':
    unittest.main()
