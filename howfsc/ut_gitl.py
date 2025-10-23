# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
# pylint: disable=unsubscriptable-object
"""
Unit tests for the main HOWFSC computation block
"""

import unittest
from unittest.mock import patch
import os
import copy
import logging
import sys

import numpy as np

from eetc.thpt_tools import ThptToolsException
from eetc.excam_tools import EXCAMOptimizeException

from howfsc.model.mode import CoronagraphMode
from howfsc.model.singlelambda import SingleLambdaException
from howfsc.model.parse_mdf import MDFException

from howfsc.control.calcjacs import calcjacs, generate_ijlist
from howfsc.control.cs import ControlStrategy
from howfsc.control.calcjtwj import JTWJMap
from howfsc.control.inversion import InversionException
from howfsc.control.calcjacs import CalcJacsException
from howfsc.control.parse_cs import CSException

from howfsc.util.loadyaml import loadyaml
from howfsc.util.constrain_dm import ConstrainDMException
from howfsc.util.actlimits import ActLimitException
from howfsc.util.check import CheckException

from .gitl import (
    howfsc_computation,
    _main_howfsc_computation,
    efc_computation,
)
from .status_codes import status_codes

# Keep logger spam out of unit test results
if sys.platform.startswith('win'):
    logging.basicConfig(filename='NUL')
else:
    logging.basicConfig(filename='/dev/null')

# use small array and DM to keep Jacobian manageable
cfgpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'model', 'testdata', 'ut', 'ut_smalljac.yaml')
cfg = CoronagraphMode(cfgpath)
jac = calcjacs(cfg, generate_ijlist(cfg, cfg.initmaps))

cstratfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'control', 'testdata', 'ut_good_cs.yaml')
cstrat = ControlStrategy(cstratfile)

subcroplist = []

nlam = 3
npix = 7
for _ in range(nlam):
    subcroplist.append((0, 0, npix, npix))
    pass
jtwj_map = JTWJMap(cfg, jac, cstrat, subcroplist)

hconffile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'control', 'testdata', 'ut_howfsc_config_good.yaml')
hconf = loadyaml(hconffile, custom_exception=TypeError)


class TestMainHOWFSCComputation(unittest.TestCase):
    """
    Tests for the inner meat

    Have to rely on functional test to make sure it's doing the right thing
    """

    def setUp(self):
        # for small jac: 7x7 images, 4x4 DMs, nlam = 3
        self.cfg = cfg
        self.jac = jac
        self.jtwj_map = jtwj_map

        self.nlam = nlam
        self.npix = npix
        self.nprobepair = 3
        self.ndm = 2*self.nprobepair + 1
        self.nact = 4

        rng = np.random.default_rng(5551212)
        self.framelist = []
        self.dm1_list = []
        self.dm2_list = []
        self.croplist = []
        self.prev_exptime_list = []
        self.n2clist = []

        dm1 = 50 + rng.random((self.nact, self.nact))
        dm2 = 50 + rng.random((self.nact, self.nact))
        p0 = rng.random((self.nact, self.nact))
        p1 = rng.random((self.nact, self.nact))
        p2 = rng.random((self.nact, self.nact))
        for _ in range(self.nlam):
            for dummy in range(self.ndm):
                self.framelist.append(100*rng.random((self.npix, self.npix)))
                self.dm2_list.append(dm2)
                self.croplist.append((0, 0, self.npix, self.npix))
                self.prev_exptime_list.append(10)
                pass
            self.dm1_list.append(dm1)
            self.dm1_list.append(dm1 + p0)
            self.dm1_list.append(dm1 - p0)
            self.dm1_list.append(dm1 + p1)
            self.dm1_list.append(dm1 - p1)
            self.dm1_list.append(dm1 + p2)
            self.dm1_list.append(dm1 - p2)
            self.n2clist.append(np.ones((self.npix, self.npix)))
            pass

        self.cstrat = cstrat
        self.hconf = hconf
        self.iteration = 1

        pass


    def test_success(self):
        """Good inputs return without incident"""
        _main_howfsc_computation(self.framelist, self.dm1_list, self.dm2_list,
                           self.cfg, self.jac, self.jtwj_map,
                           self.croplist, self.prev_exptime_list,
                           self.cstrat, self.n2clist, self.hconf,
                           self.iteration)
        pass


    def test_outputs(self):
        """Outputs are as expected in shape/type"""
        abs_dm1, abs_dm2, scale_factor_list, gain_list, exptime_list, \
        nframes_list, prev_c, next_c, next_time, status, other = \
        _main_howfsc_computation(self.framelist, self.dm1_list, self.dm2_list,
                           self.cfg, self.jac, self.jtwj_map,
                           self.croplist, self.prev_exptime_list,
                           self.cstrat, self.n2clist, self.hconf,
                           self.iteration)

        self.assertTrue(abs_dm1.shape == (self.nact, self.nact))
        self.assertTrue(abs_dm2.shape == (self.nact, self.nact))
        self.assertTrue(len(scale_factor_list) == 2*self.nprobepair)
        for s in scale_factor_list:
            self.assertTrue(np.isscalar(s))
            pass

        self.assertTrue(len(gain_list) == self.nlam)
        self.assertTrue(len(exptime_list) == self.nlam)
        self.assertTrue(len(nframes_list) == self.nlam)
        for g in gain_list:
            self.assertTrue(len(g) == (1 + self.nprobepair))
            for gg in g:
                self.assertTrue(np.isscalar(gg))
                pass
            pass
        for e in exptime_list:
            self.assertTrue(len(e) == (1 + self.nprobepair))
            for ee in e:
                self.assertTrue(np.isscalar(ee))
                pass
            pass
        for n in nframes_list:
            self.assertTrue(len(n) == (1 + self.nprobepair))
            for nn in n:
                self.assertTrue(np.isscalar(nn))
                pass
            pass
        self.assertTrue(np.isscalar(prev_c))
        self.assertTrue(np.isscalar(next_c))
        self.assertTrue(np.isscalar(next_time))
        self.assertTrue(np.isscalar(status))
        self.assertTrue(isinstance(other, dict))
        pass


    def test_invalid_framelist(self):
        """Invalid inputs caught as expected"""
        N = self.ndm*self.nlam
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cfg, # wrong type
                    [1]*N, [np.ones((self.npix,))]*N, # wrong inner type
                    [np.ones((self.npix+1, self.npix+1))]*N, # wrong in size
                    [np.ones((self.npix, self.npix))]*(N+1), # wrong out size
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                _main_howfsc_computation(perr, self.dm1_list,
                                   self.dm2_list, self.cfg, self.jac,
                                   self.jtwj_map, self.croplist,
                                   self.prev_exptime_list, self.cstrat,
                                   self.n2clist, self.hconf, self.iteration)
            pass
        pass


    def test_invalid_dm1_list(self):
        """Invalid inputs caught as expected"""
        N = self.ndm*self.nlam
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cfg, # wrong type
                    [1]*N, [np.ones((self.npix,))]*N, # wrong inner type
                    [np.ones((self.nact+1, self.nact+1))]*N, # wrong in size
                    [np.ones((self.nact, self.nact))]*(N+1), # wrong out size
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                _main_howfsc_computation(self.framelist, perr,
                                   self.dm2_list, self.cfg, self.jac,
                                   self.jtwj_map, self.croplist,
                                   self.prev_exptime_list, self.cstrat,
                                   self.n2clist, self.hconf, self.iteration)
            pass
        pass


    def test_invalid_dm2_list(self):
        """Invalid inputs caught as expected"""
        N = self.ndm*self.nlam
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cfg, # wrong type
                    [1]*N, [np.ones((self.npix,))]*N, # wrong inner type
                    [np.ones((self.nact+1, self.nact+1))]*N, # wrong in size
                    [np.ones((self.nact, self.nact))]*(N+1), # wrong out size
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                _main_howfsc_computation(self.framelist, self.dm1_list,
                                   perr, self.cfg, self.jac,
                                   self.jtwj_map, self.croplist,
                                   self.prev_exptime_list, self.cstrat,
                                   self.n2clist, self.hconf, self.iteration)
            pass
        pass


    def test_invalid_cfg(self):
        """Invalid inputs caught as expected"""
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cstrat, # wrong type
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                _main_howfsc_computation(self.framelist, self.dm1_list,
                                   self.dm2_list, perr, self.jac,
                                   self.jtwj_map, self.croplist,
                                   self.prev_exptime_list, self.cstrat,
                                   self.n2clist, self.hconf, self.iteration)
            pass
        pass


    def test_invalid_jac(self):
        """Invalid inputs caught as expected"""
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cfg, # wrong type
                    np.ones((2,)), np.ones((self.nact, self.npix)), #wrong Ndim
                    np.ones((1, self.nact, self.npix)), # wrong dim size
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                _main_howfsc_computation(self.framelist, self.dm1_list,
                                   self.dm2_list, self.cfg, perr,
                                   self.jtwj_map, self.croplist,
                                   self.prev_exptime_list, self.cstrat,
                                   self.n2clist, self.hconf, self.iteration)
            pass
        pass


    def test_invalid_jtwj_map(self):
        """Invalid inputs caught as expected"""
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cstrat, # wrong type
                    np.eye(2*self.nact**2), # not a 2D ndm X ndm array
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                _main_howfsc_computation(self.framelist, self.dm1_list,
                                   self.dm2_list, self.cfg, self.jac,
                                   perr, self.croplist,
                                   self.prev_exptime_list, self.cstrat,
                                   self.n2clist, self.hconf, self.iteration)
            pass
        pass


    def test_invalid_croplist(self):
        """Invalid inputs caught as expected"""
        N = self.ndm*self.nlam
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cfg, # wrong type
                    [1]*N, [np.ones((self.npix,))]*N, # wrong inner type
                    [(0, 0, self.npix, self.npix, 0)]*N, # wrong in size
                    [(0, 0, self.npix, self.npix)]*(N+1), # wrong out size
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                _main_howfsc_computation(self.framelist, self.dm1_list,
                                   self.dm2_list, self.cfg, self.jac,
                                   self.jtwj_map, perr,
                                   self.prev_exptime_list, self.cstrat,
                                   self.n2clist, self.hconf, self.iteration)
            pass
        pass


    def test_croplist_not_internally_consistent(self):
        """
        Croplist data is derived from parameters that should change in a
        consistent way based on the data collection procedure.  If that is not
        the case, catch these as potentially indicative of corruption or some
        other mismatch
        """
        N = self.ndm*self.nlam

        # case 1: nrow/col not constant
        c1a = [(0, 0, self.npix+1, self.npix)] + \
            [(0, 0, self.npix, self.npix)]*(N-1)
        c1b = [(0, 0, self.npix, self.npix+1)] + \
            [(0, 0, self.npix, self.npix)]*(N-1)

        # case 2: crop params not same across lambda
        c2 = [(1, 1, npix, npix)] + [(0, 0, npix, npix)]*(N-1)

        perrlist = [c1a, c1b, c2]

        for perr in perrlist:
            with self.assertRaises(ValueError):
                _main_howfsc_computation(self.framelist, self.dm1_list,
                                   self.dm2_list, self.cfg, self.jac,
                                   self.jtwj_map, perr,
                                   self.prev_exptime_list, self.cstrat,
                                   self.n2clist, self.hconf, self.iteration)
            pass
        pass



    def test_invalid_prev_exptime_list(self):
        """Invalid inputs caught as expected"""
        N = self.ndm*self.nlam
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cfg, # wrong type
                    [np.ones((self.npix,))]*N, # wrong inner type
                    [1]*(N+1), # wrong out size
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                _main_howfsc_computation(self.framelist, self.dm1_list,
                                   self.dm2_list, self.cfg, self.jac,
                                   self.jtwj_map, self.croplist,
                                   perr, self.cstrat,
                                   self.n2clist, self.hconf, self.iteration)
            pass
        pass


    def test_invalid_cstrat(self):
        """Invalid inputs caught as expected"""
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cfg, # wrong type
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                _main_howfsc_computation(self.framelist, self.dm1_list,
                                   self.dm2_list, self.cfg, self.jac,
                                   self.jtwj_map, self.croplist,
                                   self.prev_exptime_list, perr,
                                   self.n2clist, self.hconf, self.iteration)
            pass
        pass


    def test_invalid_n2clist(self):
        """Invalid inputs caught as expected"""
        N = self.nlam # short list
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cfg, # wrong type
                    [1]*N, [np.ones((self.npix,))]*N, # wrong inner type
                    [np.ones((self.npix+1, self.npix+1))]*N, # wrong in size
                    [np.ones((self.npix, self.npix))]*(N+1), # wrong out size
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                _main_howfsc_computation(self.framelist, self.dm1_list,
                                   self.dm2_list, self.cfg, self.jac,
                                   self.jtwj_map, self.croplist,
                                   self.prev_exptime_list, self.cstrat,
                                   perr, self.hconf, self.iteration)
            pass
        pass


    def test_invalid_hconf(self):
        """Invalid inputs caught as expected"""

        hc2 = copy.deepcopy(hconf)
        hc2.popitem() # pop first level

        hc3a = copy.deepcopy(hconf)
        hc3a['overhead'].popitem() # pop second level
        hc3b = copy.deepcopy(hconf)
        hc3b['star'].popitem() # pop second level
        hc3c = copy.deepcopy(hconf)
        hc3c['excam'].popitem() # pop second level
        hc3d = copy.deepcopy(hconf)
        hc3d['hardware'].popitem() # pop second level
        hc3e = copy.deepcopy(hconf)
        hc3e['howfsc'].popitem() # pop second level
        hc3f = copy.deepcopy(hconf)
        hc3f['probe'].popitem() # pop second level

        hc4 = copy.deepcopy(hconf)
        hc4.update({'not_a_key':0})

        hc5a = copy.deepcopy(hconf)
        hc5a['overhead'].update({'not_a_key':0})
        hc5b = copy.deepcopy(hconf)
        hc5b['star'].update({'not_a_key':0})
        hc5c = copy.deepcopy(hconf)
        hc5c['excam'].update({'not_a_key':0})
        hc5d = copy.deepcopy(hconf)
        hc5d['hardware'].update({'not_a_key':0})
        hc5e = copy.deepcopy(hconf)
        hc5e['howfsc'].update({'not_a_key':0})
        hc5f = copy.deepcopy(hconf)
        hc5f['probe'].update({'not_a_key':0})

        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cfg, # wrong type
                    hc2, hc3a, hc3b, hc3c, hc3d, hc3e, hc3f, # missing keys
                    hc4, hc5a, hc5b, hc5c, hc5d, hc5e, hc5f, # extra keys
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                _main_howfsc_computation(self.framelist, self.dm1_list,
                                   self.dm2_list, self.cfg, self.jac,
                                   self.jtwj_map, self.croplist,
                                   self.prev_exptime_list, self.cstrat,
                                   self.n2clist, perr, self.iteration)
            pass
        pass


    def test_invalid_iteration(self):
        """Invalid inputs caught as expected"""
        perrlist = [-1, 1.5, 0, 1j, 'txt', None, self.cfg, # wrong type/value
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                _main_howfsc_computation(self.framelist, self.dm1_list,
                                   self.dm2_list, self.cfg, self.jac,
                                   self.jtwj_map, self.croplist,
                                   self.prev_exptime_list, self.cstrat,
                                   self.n2clist, self.hconf, perr)
            pass
        pass


class TestHOWFSCComputation(unittest.TestCase):
    """
    Tests for the outer wrapper on the end-to-end block
    """

    def setUp(self):
        # for small jac: 7x7 images, 4x4 DMs, nlam = 3
        self.cfg = cfg
        self.jac = jac
        self.jtwj_map = jtwj_map

        self.nlam = nlam
        self.npix = npix
        self.nprobepair = 3
        self.ndm = 2*self.nprobepair + 1
        self.nact = 4

        rng = np.random.default_rng(5551212)
        self.framelist = []
        self.dm1_list = []
        self.dm2_list = []
        self.croplist = []
        self.prev_exptime_list = []
        self.n2clist = []

        dm1 = 50 + rng.random((self.nact, self.nact))
        dm2 = 50 + rng.random((self.nact, self.nact))
        p0 = rng.random((self.nact, self.nact))
        p1 = rng.random((self.nact, self.nact))
        p2 = rng.random((self.nact, self.nact))
        for _ in range(self.nlam):
            for dummy in range(self.ndm):
                self.framelist.append(100*rng.random((self.npix, self.npix)))
                self.dm2_list.append(dm2)
                self.croplist.append((0, 0, self.npix, self.npix))
                self.prev_exptime_list.append(10)
                pass
            self.dm1_list.append(dm1)
            self.dm1_list.append(dm1 + p0)
            self.dm1_list.append(dm1 - p0)
            self.dm1_list.append(dm1 + p1)
            self.dm1_list.append(dm1 - p1)
            self.dm1_list.append(dm1 + p2)
            self.dm1_list.append(dm1 - p2)
            self.n2clist.append(np.ones((self.npix, self.npix)))
            pass

        self.cstrat = cstrat
        self.hconf = hconf
        self.iteration = 1

        pass


    @patch('howfsc.gitl._main_howfsc_computation')
    def test_all_known_exceptions(self, mock_howfsc):
        """
        Run through all the known exceptions that the HOWFSC repository can
        throw, and verify that they are all caught by their own status code
        """
        list_of_exceptions = [TypeError,
                              ValueError,
                              ConstrainDMException,
                              InversionException,
                              ZeroDivisionError,
                              KeyError,
                              IOError,
                              CalcJacsException,
                              CSException,
                              MDFException,
                              ActLimitException,
                              CheckException,
                              SingleLambdaException,
                              ThptToolsException,
                              EXCAMOptimizeException,
                              ]
        nexc = len(list_of_exceptions)
        mock_howfsc.side_effect = list_of_exceptions

        setofvalids = {
            status_codes['nominal'],
            status_codes['LowerThanExpectedSNR'],
        }
        codenum = set(status_codes.values()) \
            - setofvalids \
            - {status_codes['Exception']}

        for _ in range(nexc):
            out = howfsc_computation(self.framelist, self.dm1_list,
                                     self.dm2_list,
                                     self.cfg, self.jac, self.jtwj_map,
                                     self.croplist, self.prev_exptime_list,
                                     self.cstrat, self.n2clist, self.hconf,
                                     self.iteration)
            # It failed...
            self.assertFalse(out[-2] in setofvalids)
            # ..but not blindly...
            self.assertFalse(out[-2] == status_codes['Exception'])
            # ...instead matching a specific known code
            self.assertTrue(out[-2] in codenum)

            pass
        pass


    @patch('howfsc.gitl._main_howfsc_computation')
    def test_catch_unknown_exceptions(self, mock_howfsc):
        """
        Test catching an exception that the HOWFSC code shouldn't throw
        """
        mock_howfsc.side_effect = [RuntimeError] # use the python catchall
        out = howfsc_computation(self.framelist, self.dm1_list,
                                 self.dm2_list,
                                 self.cfg, self.jac, self.jtwj_map,
                                 self.croplist, self.prev_exptime_list,
                                 self.cstrat, self.n2clist, self.hconf,
                                 self.iteration)

        self.assertTrue(out[-2] == status_codes['Exception'])

        pass

    @patch('eetc.cgi_eetc.CGIEETC.calc_exp_time')
    def test_met_target_snr(self, mock_calc):
        """
        Check the right output code is sent if all optimizations succeeded
        """
        # only value that matters is the last (optflag)
        mock_calc.return_value = (1, 1, 1, 1, 0) # 0 = 1st opt success
        out = howfsc_computation(self.framelist, self.dm1_list,
                                 self.dm2_list,
                                 self.cfg, self.jac, self.jtwj_map,
                                 self.croplist, self.prev_exptime_list,
                                 self.cstrat, self.n2clist, self.hconf,
                                 self.iteration)

        self.assertTrue(out[-2] == status_codes['nominal'])

        pass


    @patch('eetc.cgi_eetc.CGIEETC.calc_exp_time')
    def test_did_not_meet_target_snr(self, mock_calc):
        """
        Check the right output code is sent if at least one initial
        optimization failed.  Second optimizer succeeded (or an exception would
        have been thrown)
        """
        # only value that matters is the last (optflag)
        mock_calc.return_value = (1, 1, 1, 1, 1) # 1 = 1st opt failed
        out = howfsc_computation(self.framelist, self.dm1_list,
                                 self.dm2_list,
                                 self.cfg, self.jac, self.jtwj_map,
                                 self.croplist, self.prev_exptime_list,
                                 self.cstrat, self.n2clist, self.hconf,
                                 self.iteration)

        self.assertTrue(out[-2] == status_codes['LowerThanExpectedSNR'])

        pass


class TestEFCComputation(unittest.TestCase):
    """
    Tests for efc_computation().

    Have to rely on functional test to make sure it's doing the right thing
    """
    def setUp(self):
        # for small jac: 7x7 images, 4x4 DMs, nlam = 3
        self.cfg = cfg
        self.jac = jac
        self.jtwj_map = jtwj_map

        self.nlam = nlam
        self.npix = npix
        self.ndm = 1
        self.nact = 4

        rng = np.random.default_rng(5551212)
        self.dm1v = 50 + rng.random((self.nact, self.nact))
        self.dm2v = 50 + rng.random((self.nact, self.nact))

        self.croplist = []
        for _ in range(self.nlam):
            for dummy in range(self.ndm):
                self.croplist.append((0, 0, self.npix, self.npix))

        self.cstrat = cstrat
        self.iteration = 1

    def test_success(self):
        """Good inputs return without incident"""
        efc_computation(self.dm1v, self.dm2v, self.cfg, self.jac,
                        self.jtwj_map, self.cstrat, self.croplist,
                        self.iteration)

    def test_outputs(self):
        """Outputs are as expected in shape/type"""
        abs_dm1, abs_dm2, dh_cube, next_c, prev_c, beta, dmmultgain = \
            efc_computation(
            self.dm1v, self.dm2v, self.cfg, self.jac, self.jtwj_map,
            self.cstrat, self.croplist, self.iteration)

        self.assertTrue(abs_dm1.shape == (self.nact, self.nact))
        self.assertTrue(abs_dm2.shape == (self.nact, self.nact))
        self.assertTrue(dh_cube.shape == (nlam, npix, npix))
        self.assertTrue(np.isscalar(next_c))
        self.assertTrue(np.isscalar(prev_c))
        self.assertTrue(np.isscalar(beta))
        self.assertTrue(np.isscalar(dmmultgain))

    def test_invalid_dm1_list(self):
        """Invalid inputs caught as expected"""
        N = self.ndm*self.nlam
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cfg, # wrong type
                    [1]*N, [np.ones((self.npix,))]*N, # wrong inner type
                    [np.ones((self.nact+1, self.nact+1))]*N, # wrong in size
                    [np.ones((self.nact, self.nact))]*(N+1), # wrong out size
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                efc_computation(
                    perr, self.dm2v, self.cfg, self.jac, self.jtwj_map,
                    self.cstrat, self.croplist, self.iteration)

    def test_invalid_dm2_list(self):
        """Invalid inputs caught as expected"""
        N = self.ndm*self.nlam
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cfg, # wrong type
                    [1]*N, [np.ones((self.npix,))]*N, # wrong inner type
                    [np.ones((self.nact+1, self.nact+1))]*N, # wrong in size
                    [np.ones((self.nact, self.nact))]*(N+1), # wrong out size
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                efc_computation(
                    self.dm1v, perr, self.cfg, self.jac, self.jtwj_map,
                    self.cstrat, self.croplist, self.iteration)

    def test_invalid_cfg(self):
        """Invalid inputs caught as expected"""
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cstrat, # wrong type
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                efc_computation(
                    self.dm1v, self.dm2v, perr, self.jac, self.jtwj_map,
                    self.cstrat, self.croplist, self.iteration)

    def test_invalid_jac(self):
        """Invalid inputs caught as expected"""
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cfg, # wrong type
                    np.ones((2,)), np.ones((self.nact, self.npix)), #wrong Ndim
                    np.ones((1, self.nact, self.npix)), # wrong dim size
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                efc_computation(
                    self.dm1v, self.dm2v, self.cfg, perr, self.jtwj_map,
                    self.cstrat, self.croplist, self.iteration)

    def test_invalid_jtwj_map(self):
        """Invalid inputs caught as expected"""
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cstrat, # wrong type
                    np.eye(2*self.nact**2), # not a 2D ndm X ndm array
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                efc_computation(
                    self.dm1v, self.dm2v, self.cfg, self.jac, perr,
                    self.cstrat, self.croplist, self.iteration)

    def test_invalid_croplist(self):
        """Invalid inputs caught as expected"""
        N = self.ndm*self.nlam
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cfg, # wrong type
                    [1]*N, [np.ones((self.npix,))]*N, # wrong inner type
                    [(0, 0, self.npix, self.npix, 0)]*N, # wrong in size
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                efc_computation(
                    self.dm1v, self.dm2v, self.cfg, self.jac, self.jtwj_map,
                    self.cstrat, perr, self.iteration)

    def test_invalid_cstrat(self):
        """Invalid inputs caught as expected"""
        perrlist = [1, -1, 1.5, 0, 1j, 'txt', None, self.cfg, # wrong type
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                efc_computation(
                    self.dm1v, self.dm2v, self.cfg, self.jac, self.jtwj_map,
                    perr, self.croplist, self.iteration)

    def test_invalid_iteration(self):
        """Invalid inputs caught as expected"""
        perrlist = [-1, 1.5, 0, 1j, 'txt', None, self.cfg, # wrong type/value
                    ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                efc_computation(
                    self.dm1v, self.dm2v, self.cfg, self.jac, self.jtwj_map,
                    self.cstrat, self.croplist, perr)


if __name__ == '__main__':
    unittest.main()
