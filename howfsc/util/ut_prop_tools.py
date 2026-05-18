# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Unit tests for prop_tools functions"""

import unittest
from unittest.mock import patch
import os
import copy

import numpy as np

from howfsc.model.mode import CoronagraphMode

from .prop_tools import efield, open_efield, model_pm0, make_dmrel_probe, make_dmrel_probe_gaussian

cfgpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
               '..', 'model', 'testdata', 'ut', 'ut_smalljac.yaml')
cfg = CoronagraphMode(cfgpath)

class TestEfield(unittest.TestCase):
    """
    Tests for a basic single-lambda e-field calculation
    """

    def setUp(self):
        self.cfg = cfg
        self.dmlist = self.cfg.initmaps
        pass


    def test_success(self):
        """Good inputs success without issues"""
        for ind in range(len(self.cfg.sl_list)):
            efield(cfg=self.cfg, dmlist=self.dmlist, ind=ind)
            pass
        pass


    def test_invalid_cfg(self):
        """invalid inputs caught"""
        perrlist = ['cfgfn', None, np.ones((5, 5)), 0]
        ind = 0

        for perr in perrlist:
            with self.assertRaises(TypeError):
                efield(cfg=perr, dmlist=self.dmlist, ind=ind)
            pass
        pass


    def test_invalid_dmlist(self):
        """invalid inputs caught"""
        perrlist = [
            'cfgfn', None, np.ones((5, 5)), 0, # wrong outer type
            [np.eye(4), np.eye(4), np.eye(4)], # wrong list length
            [None, np.eye(4)], [np.eye(4), None], # wrong inner type
            [np.eye(3), np.eye(4)], # wrong first size
            [np.eye(4), np.eye(3)], # wrong second size
        ]
        ind = 0

        for perr in perrlist:
            with self.assertRaises(TypeError):
                efield(cfg=self.cfg, dmlist=perr, ind=ind)
            pass
        pass


    def test_invalid_ind(self):
        """invalid inputs caught"""
        perrlist = [
            'cfgfn', None, np.ones((5, 5)), 1j, 1.5, # not integer
            -2, -1, # not nonnegative
            len(self.cfg.sl_list), len(self.cfg.sl_list)+1, # too long
        ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                efield(cfg=self.cfg, dmlist=self.dmlist, ind=perr)
            pass
        pass


class TestOpenEfield(unittest.TestCase):
    """
    Tests for a basic single-lambda e-field calculation with no FPM
    """

    def setUp(self):
        self.cfg = cfg
        self.dmlist = self.cfg.initmaps
        pass


    def test_success(self):
        """Good inputs success without issues"""
        for ind in range(len(self.cfg.sl_list)):
            open_efield(cfg=self.cfg, dmlist=self.dmlist, ind=ind)
            pass
        pass


    def test_invalid_cfg(self):
        """invalid inputs caught"""
        perrlist = ['cfgfn', None, np.ones((5, 5)), 0]
        ind = 0

        for perr in perrlist:
            with self.assertRaises(TypeError):
                open_efield(cfg=perr, dmlist=self.dmlist, ind=ind)
            pass
        pass


    def test_invalid_dmlist(self):
        """invalid inputs caught"""
        perrlist = [
            'cfgfn', None, np.ones((5, 5)), 0, # wrong outer type
            [np.eye(4), np.eye(4), np.eye(4)], # wrong list length
            [None, np.eye(4)], [np.eye(4), None], # wrong inner type
            [np.eye(3), np.eye(4)], # wrong first size
            [np.eye(4), np.eye(3)], # wrong second size
        ]
        ind = 0

        for perr in perrlist:
            with self.assertRaises(TypeError):
                open_efield(cfg=self.cfg, dmlist=perr, ind=ind)
            pass
        pass


    def test_invalid_ind(self):
        """invalid inputs caught"""
        perrlist = [
            'cfgfn', None, np.ones((5, 5)), 1j, 1.5, # not integer
            -2, -1, # not nonnegative
            len(self.cfg.sl_list), len(self.cfg.sl_list)+1, # too long
        ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                open_efield(cfg=self.cfg, dmlist=self.dmlist, ind=perr)
            pass
        pass


class TestModelPM0(unittest.TestCase):
    """
    Tests for tool to calculate a model-based +/-/0 intensity residual
    """

    def setUp(self):
        self.cfg = cfg
        self.dm0 = cfg.initmaps[0]
        self.dmplus = self.dm0.copy()
        self.dmplus[2, 2] += 1
        self.dmminus = self.dm0.copy()
        self.dmminus[2, 2] -= 1
        self.otherdm = cfg.initmaps[1]
        self.ind = 0
        pass

    def test_success(self):
        """Valid inputs succeed with no errors"""
        model_pm0(
            cfg=self.cfg,
            dm0=self.dm0,
            dmplus=self.dmplus,
            dmminus=self.dmminus,
            otherdm=self.otherdm,
            ind=self.ind,
        )


    @patch('howfsc.model.singlelambda.SingleLambda.proptodh')
    def test_exact(self, mock_edh):
        """
        Test that the summing math (I+ + I-)/2 - I0  works as expected

        (I+ + I-)/2 - I0 =>
        - = |E + D|**2/2 + |E - D|**2/2 - |E|**2
        - = (|E|**2 + E*D + D*E + |D|**2 + |E|**2 - E*D - D*E + |D|**2)/2
             - |E|**2
        - = (|E|**2  + |D|**2 + |E|**2 + |D|**2)/2 - |E|**2 as cross-terms
            cancel
        - = |E|**2 + |D|**2 - |E|**2
        - = |D|**2

        This only works in the linear limit (small motion), so mock out the
        propagator, which will always be nonlinear, and check the rest of the
        math.
        """
        tol = 1e-13

        rng = np.random.default_rng(30303030)
        enom = rng.random((5, 5)) + 1j*rng.random((5, 5))
        edel = 0.01*rng.random((5, 5)) + 0.01*1j*rng.random((5, 5))
        edh0 = enom
        edhp = enom + edel
        edhm = enom - edel
        mock_edh.side_effect = [edh0, edhp, edhm]

        target = np.abs(edel)**2
        out = model_pm0(
            cfg=self.cfg,
            dm0=self.dm0,
            dmplus=self.dmplus,
            dmminus=self.dmminus,
            otherdm=self.otherdm,
            ind=self.ind,
        )

        self.assertTrue(np.max(np.abs(out - target)) < tol)
        pass


    def test_swap_dms(self):
        """Verify swap_dms works as expected"""
        tol = 1e-13

        regular = model_pm0(
            cfg=self.cfg,
            dm0=self.dm0,
            dmplus=self.dmplus,
            dmminus=self.dmminus,
            otherdm=self.otherdm,
            ind=self.ind,
        )

        # swap the DM registrations, too, so we should get the same answer
        cfg2 = copy.deepcopy(self.cfg)
        tmp = copy.deepcopy(cfg2.dmlist[0])
        cfg2.dmlist[0] = cfg2.dmlist[1]
        cfg2.dmlist[1] = tmp

        swapped = model_pm0(
            cfg=cfg2,
            dm0=self.dm0,
            dmplus=self.dmplus,
            dmminus=self.dmminus,
            otherdm=self.otherdm,
            ind=self.ind,
            swap_dms=True,
        )

        self.assertTrue(np.max(np.abs(regular - swapped)) < tol)
        pass


    def test_invalid_cfg(self):
        """Invalid inputs fail as expected"""
        perrlist = ['cfgfn', None, np.ones((5, 5)), 0]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                model_pm0(
                    cfg=perr,
                    dm0=self.dm0,
                    dmplus=self.dmplus,
                    dmminus=self.dmminus,
                    otherdm=self.otherdm,
                    ind=self.ind,
                )
            pass
        pass


    def test_invalid_dm0(self):
        """Invalid inputs fail as expected"""
        perrlist = [
            'cfgfn', None, 0, # not array
            np.ones((5,)), np.ones((2, 2, 2)), # not 2D
            self.cfg.initmaps[0][:, :-1], # wrong size
        ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                model_pm0(
                    cfg=self.cfg,
                    dm0=perr,
                    dmplus=self.dmplus,
                    dmminus=self.dmminus,
                    otherdm=self.otherdm,
                    ind=self.ind,
                )
            pass
        pass


    def test_invalid_dmplus(self):
        """Invalid inputs fail as expected"""
        perrlist = [
            'cfgfn', None, 0, # not array
            np.ones((5,)), np.ones((2, 2, 2)), # not 2D
            self.cfg.initmaps[0][:, :-1], # wrong size
        ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                model_pm0(
                    cfg=self.cfg,
                    dm0=self.dm0,
                    dmplus=perr,
                    dmminus=self.dmminus,
                    otherdm=self.otherdm,
                    ind=self.ind,
                )
            pass
        pass


    def test_invalid_dmminus(self):
        """Invalid inputs fail as expected"""
        perrlist = [
            'cfgfn', None, 0, # not array
            np.ones((5,)), np.ones((2, 2, 2)), # not 2D
            self.cfg.initmaps[0][:, :-1], # wrong size
        ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                model_pm0(
                    cfg=self.cfg,
                    dm0=self.dm0,
                    dmplus=self.dmplus,
                    dmminus=perr,
                    otherdm=self.otherdm,
                    ind=self.ind,
                )
            pass
        pass


    def test_invalid_otherdm(self):
        """Invalid inputs fail as expected"""
        perrlist = [
            'cfgfn', None, 0, # not array
            np.ones((5,)), np.ones((2, 2, 2)), # not 2D
            self.cfg.initmaps[1][:, :-1], # wrong size
        ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                model_pm0(
                    cfg=self.cfg,
                    dm0=self.dm0,
                    dmplus=self.dmplus,
                    dmminus=self.dmminus,
                    otherdm=perr,
                    ind=self.ind,
                )
            pass
        pass


    def test_invalid_ind(self):
        """Invalid inputs fail as expected"""
        perrlist = [
            'cfgfn', None, np.ones((5, 5)), 1j, 1.5, # not integer
            -2, -1, # not nonnegative
            len(self.cfg.sl_list), len(self.cfg.sl_list)+1, # too long
        ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                model_pm0(
                    cfg=self.cfg,
                    dm0=self.dm0,
                    dmplus=self.dmplus,
                    dmminus=self.dmminus,
                    otherdm=self.otherdm,
                    ind=perr,
                )
            pass
        pass



class TestMakeDMRelProbe(unittest.TestCase):
    """
    Tests for a function to make a HOWFSC DM relative probe of a desired
    amplitude
    """

    def setUp(self):
        self.cfg = cfg
        self.dmlist = self.cfg.initmaps
        self.dact = 4
        self.xcenter = 0
        self.ycenter = 0
        self.clock = 0
        self.ximin = 0
        self.ximax = 1.5
        self.etamin = -1.5
        self.etamax = 1.5
        self.phase = 0
        self.target = 1e-5
        self.lod_min = 0
        self.lod_max = 1.2
        self.ind = 0
        self.maxiter = 5
        self.verbose = False
        pass


    def test_success(self):
        """Valid inputs complete as expected"""
        dpv, probe_int, lod_mask, dm_surface, pupil_mask = \
            make_dmrel_probe(
            cfg=self.cfg,
            dmlist=self.dmlist,
            dact=self.dact,
            xcenter=self.xcenter,
            ycenter=self.ycenter,
            clock=self.clock,
            ximin=self.ximin,
            ximax=self.ximax,
            etamin=self.etamin,
            etamax=self.etamax,
            phase=self.phase,
            target=self.target,
            lod_min=self.lod_min,
            lod_max=self.lod_max,
            ind=self.ind,
            maxiter=self.maxiter,
            verbose=self.verbose,
        )

        # Check outputs
        sl = cfg.sl_list[self.ind]
        self.assertTrue(dpv.shape == (cfg.dmlist[0].registration['nact'],
                                      cfg.dmlist[0].registration['nact']))
        self.assertTrue(probe_int.shape == sl.dh.e.shape)
        self.assertTrue(lod_mask.shape == sl.dh.e.shape)
        self.assertTrue(lod_mask.dtype == 'bool')
        self.assertTrue(pupil_mask.shape == dm_surface.shape)
        self.assertTrue(pupil_mask.shape[0] >= sl.epup.e.shape[0])
        self.assertTrue(pupil_mask.shape[1] >= sl.epup.e.shape[1])
        self.assertTrue(pupil_mask.shape[0] >= sl.pupil.e.shape[0])
        self.assertTrue(pupil_mask.shape[1] >= sl.pupil.e.shape[1])
        self.assertTrue(pupil_mask.shape[0] >= sl.lyot.e.shape[0])
        self.assertTrue(pupil_mask.shape[1] >= sl.lyot.e.shape[1])
        pass


    def test_invalid_cfg(self):
        """Invalid inputs fail as expected"""
        perrlist = ['cfgfn', None, np.ones((5, 5)), 0]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe(
                    cfg=perr,
                    dmlist=self.dmlist,
                    dact=self.dact,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    clock=self.clock,
                    ximin=self.ximin,
                    ximax=self.ximax,
                    etamin=self.etamin,
                    etamax=self.etamax,
                    phase=self.phase,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass
        pass


    def test_invalid_dmlist(self):
        """Invalid inputs fail as expected"""
        perrlist = [
            'cfgfn', None, np.ones((5, 5)), 0, # wrong outer type
            [np.eye(4), np.eye(4), np.eye(4)], # wrong list length
            [None, np.eye(4)], [np.eye(4), None], # wrong inner type
            [np.eye(3), np.eye(4)], # wrong first size
            [np.eye(4), np.eye(3)], # wrong second size
        ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe(
                    cfg=self.cfg,
                    dmlist=perr,
                    dact=self.dact,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    clock=self.clock,
                    ximin=self.ximin,
                    ximax=self.ximax,
                    etamin=self.etamin,
                    etamax=self.etamax,
                    phase=self.phase,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass
        pass


    def test_invalid_rps(self):
        """Invalid inputs fail as expected"""
        # case: real positive scalar
        perrlist = ['cfgfn', None, np.ones((5, 5)), 0, 1j, -1.5]

        # dact
        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    dact=perr,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    clock=self.clock,
                    ximin=self.ximin,
                    ximax=self.ximax,
                    etamin=self.etamin,
                    etamax=self.etamax,
                    phase=self.phase,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass

        # target
        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    dact=self.dact,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    clock=self.clock,
                    ximin=self.ximin,
                    ximax=self.ximax,
                    etamin=self.etamin,
                    etamax=self.etamax,
                    phase=self.phase,
                    target=perr,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass

        # lod_max
        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    dact=self.dact,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    clock=self.clock,
                    ximin=self.ximin,
                    ximax=self.ximax,
                    etamin=self.etamin,
                    etamax=self.etamax,
                    phase=self.phase,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=perr,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass

        pass


    def test_invalid_rns(self):
        """Invalid inputs fail as expected"""
        # case: real nonnegative scalar
        perrlist = ['cfgfn', None, np.ones((5, 5)), 1j, -1.5]

        # lod_min
        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    dact=self.dact,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    clock=self.clock,
                    ximin=self.ximin,
                    ximax=self.ximax,
                    etamin=self.etamin,
                    etamax=self.etamax,
                    phase=self.phase,
                    target=self.target,
                    lod_min=perr,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass
        pass


    def test_invalid_psi(self):
        """Invalid inputs fail as expected"""
        # case: positive scalar integer
        perrlist = ['cfgfn', None, np.ones((5, 5)), 0, 1j, -1.5, 1.5]

        # maxiter
        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    dact=self.dact,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    clock=self.clock,
                    ximin=self.ximin,
                    ximax=self.ximax,
                    etamin=self.etamin,
                    etamax=self.etamax,
                    phase=self.phase,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=perr,
                    verbose=self.verbose,
                )
            pass
        pass


    def test_invalid_nsi(self):
        """Invalid inputs fail as expected"""
        # case: non-negative scalar integer
        perrlist = ['cfgfn', None, np.ones((5, 5)), 1j, -1.5, 1.5]

        # ind
        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    dact=self.dact,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    clock=self.clock,
                    ximin=self.ximin,
                    ximax=self.ximax,
                    etamin=self.etamin,
                    etamax=self.etamax,
                    phase=self.phase,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=perr,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass
        pass


    def test_invalid_rs(self):
        """Invalid inputs fail as expected"""
        # case: real scalar
        perrlist = ['cfgfn', None, np.ones((5, 5)), 1j]

        # xcenter
        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    dact=self.dact,
                    xcenter=perr,
                    ycenter=self.ycenter,
                    clock=self.clock,
                    ximin=self.ximin,
                    ximax=self.ximax,
                    etamin=self.etamin,
                    etamax=self.etamax,
                    phase=self.phase,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass

        # ycenter
        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    dact=self.dact,
                    xcenter=self.xcenter,
                    ycenter=perr,
                    clock=self.clock,
                    ximin=self.ximin,
                    ximax=self.ximax,
                    etamin=self.etamin,
                    etamax=self.etamax,
                    phase=self.phase,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass

        # clock
        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    dact=self.dact,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    clock=perr,
                    ximin=self.ximin,
                    ximax=self.ximax,
                    etamin=self.etamin,
                    etamax=self.etamax,
                    phase=self.phase,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass

        # ximin
        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    dact=self.dact,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    clock=self.clock,
                    ximin=perr,
                    ximax=self.ximax,
                    etamin=self.etamin,
                    etamax=self.etamax,
                    phase=self.phase,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass

        # ximax
        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    dact=self.dact,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    clock=self.clock,
                    ximin=self.ximin,
                    ximax=perr,
                    etamin=self.etamin,
                    etamax=self.etamax,
                    phase=self.phase,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass

        # etamin
        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    dact=self.dact,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    clock=self.clock,
                    ximin=self.ximin,
                    ximax=self.ximax,
                    etamin=perr,
                    etamax=self.etamax,
                    phase=self.phase,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass

        # etamax
        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    dact=self.dact,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    clock=self.clock,
                    ximin=self.ximin,
                    ximax=self.ximax,
                    etamin=self.etamin,
                    etamax=perr,
                    phase=self.phase,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass

        # phase
        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    dact=self.dact,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    clock=self.clock,
                    ximin=self.ximin,
                    ximax=self.ximax,
                    etamin=self.etamin,
                    etamax=self.etamax,
                    phase=perr,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass
        pass


    def test_invalid_bool(self):
        """Invalid inputs fail as expected"""
        # case: bool
        perrlist = [None]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    dact=self.dact,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    clock=self.clock,
                    ximin=self.ximin,
                    ximax=self.ximax,
                    etamin=self.etamin,
                    etamax=self.etamax,
                    phase=self.phase,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=perr,
                )
            pass
        pass


    def test_invalid_ranges(self):
        """Invalid inputs fail as expected"""

        # ximin/ximax
        with self.assertRaises(ValueError):
            make_dmrel_probe(
                cfg=self.cfg,
                dmlist=self.dmlist,
                dact=self.dact,
                xcenter=self.xcenter,
                ycenter=self.ycenter,
                clock=self.clock,
                ximin=self.ximax, # same
                ximax=self.ximax, # same
                etamin=self.etamin,
                etamax=self.etamax,
                phase=self.phase,
                target=self.target,
                lod_min=self.lod_min,
                lod_max=self.lod_max,
                ind=self.ind,
                maxiter=self.maxiter,
                verbose=self.verbose,
            )

        # etamin/etamax
        with self.assertRaises(ValueError):
            make_dmrel_probe(
                cfg=self.cfg,
                dmlist=self.dmlist,
                dact=self.dact,
                xcenter=self.xcenter,
                ycenter=self.ycenter,
                clock=self.clock,
                ximin=self.ximin,
                ximax=self.ximax,
                etamin=self.etamax, # same
                etamax=self.etamax, # same
                phase=self.phase,
                target=self.target,
                lod_min=self.lod_min,
                lod_max=self.lod_max,
                ind=self.ind,
                maxiter=self.maxiter,
                verbose=self.verbose,
            )

        # lod_min/lod_max
        with self.assertRaises(ValueError):
            make_dmrel_probe(
                cfg=self.cfg,
                dmlist=self.dmlist,
                dact=self.dact,
                xcenter=self.xcenter,
                ycenter=self.ycenter,
                clock=self.clock,
                ximin=self.ximin,
                ximax=self.ximax,
                etamin=self.etamin,
                etamax=self.etamax,
                phase=self.phase,
                target=self.target,
                lod_min=self.lod_max, # same
                lod_max=self.lod_max, # same
                ind=self.ind,
                maxiter=self.maxiter,
                verbose=self.verbose,
            )


        pass


    def test_ind_out_of_range(self):
        """Invalid inputs fail as expected"""
        with self.assertRaises(TypeError):
            make_dmrel_probe(
                cfg=self.cfg,
                dmlist=self.dmlist,
                dact=self.dact,
                xcenter=self.xcenter,
                ycenter=self.ycenter,
                clock=self.clock,
                ximin=self.ximin,
                ximax=self.ximax,
                etamin=self.etamin,
                etamax=self.etamax,
                phase=self.phase,
                target=self.target,
                lod_min=self.lod_min,
                lod_max=self.lod_max,
                ind=len(self.cfg.sl_list),
                maxiter=self.maxiter,
                verbose=self.verbose,
            )
        pass


class TestMakeDMRelProbeGaussian(unittest.TestCase):
    """
    Tests for a function to make a HOWFSC Gaussian DM relative probe of a
    desired amplitude
    """

    def setUp(self):
        self.cfg = cfg
        self.dmlist = self.cfg.initmaps
        self.dact = 4
        self.xcenter = 0
        self.ycenter = 0
        self.sigma = 1.0
        self.target = 1e-5
        self.lod_min = 0
        self.lod_max = 1.2
        self.ind = 0
        self.maxiter = 5
        self.verbose = False
        pass


    def test_success(self):
        """Valid inputs complete as expected"""
        dpv, probe_int, lod_mask, dm_surface, pupil_mask = \
            make_dmrel_probe_gaussian(
            cfg=self.cfg,
            dmlist=self.dmlist,
            xcenter=self.xcenter,
            ycenter=self.ycenter,
            sigma=self.sigma,
            target=self.target,
            lod_min=self.lod_min,
            lod_max=self.lod_max,
            ind=self.ind,
            maxiter=self.maxiter,
            verbose=self.verbose,
        )

        # Check outputs
        sl = cfg.sl_list[self.ind]
        self.assertTrue(dpv.shape == (cfg.dmlist[0].registration['nact'],
                                      cfg.dmlist[0].registration['nact']))
        self.assertTrue(probe_int.shape == sl.dh.e.shape)
        self.assertTrue(lod_mask.shape == sl.dh.e.shape)
        self.assertTrue(lod_mask.dtype == 'bool')
        self.assertTrue(pupil_mask.shape == dm_surface.shape)
        self.assertTrue(pupil_mask.shape[0] >= sl.epup.e.shape[0])
        self.assertTrue(pupil_mask.shape[1] >= sl.epup.e.shape[1])
        self.assertTrue(pupil_mask.shape[0] >= sl.pupil.e.shape[0])
        self.assertTrue(pupil_mask.shape[1] >= sl.pupil.e.shape[1])
        self.assertTrue(pupil_mask.shape[0] >= sl.lyot.e.shape[0])
        self.assertTrue(pupil_mask.shape[1] >= sl.lyot.e.shape[1])
        pass


    def test_invalid_cfg(self):
        """Invalid inputs fail as expected"""
        perrlist = ['cfgfn', None, np.ones((5, 5)), 0]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe_gaussian(
                    cfg=perr,
                    dmlist=self.dmlist,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    sigma=self.sigma,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass
        pass


    def test_invalid_dmlist(self):
        """Invalid inputs fail as expected"""
        perrlist = [
            'cfgfn', None, np.ones((5, 5)), 0, # wrong outer type
            [np.eye(4), np.eye(4), np.eye(4)], # wrong list length
            [None, np.eye(4)], [np.eye(4), None], # wrong inner type
            [np.eye(3), np.eye(4)], # wrong first size
            [np.eye(4), np.eye(3)], # wrong second size
        ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe_gaussian(
                    cfg=self.cfg,
                    dmlist=perr,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    sigma=self.sigma,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass
        pass


    def test_invalid_rps(self):
        """Invalid inputs fail as expected"""
        # case: real positive scalar
        perrlist = ['cfgfn', None, np.ones((5, 5)), 0, 1j, -1.5]

        # target
        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe_gaussian(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    sigma=self.sigma,
                    target=perr,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass

        # lod_max
        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe_gaussian(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    sigma=self.sigma,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=perr,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass

        # sigma
        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe_gaussian(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    sigma=perr,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass
        pass

    def test_invalid_rns(self):
        """Invalid inputs fail as expected"""
        # case: real nonnegative scalar
        perrlist = ['cfgfn', None, np.ones((5, 5)), 1j, -1.5]

        # lod_min
        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe_gaussian(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    sigma=self.sigma,
                    target=self.target,
                    lod_min=perr,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass
        pass


    def test_invalid_psi(self):
        """Invalid inputs fail as expected"""
        # case: positive scalar integer
        perrlist = ['cfgfn', None, np.ones((5, 5)), 0, 1j, -1.5, 1.5]

        # maxiter
        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe_gaussian(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    sigma=self.sigma,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=perr,
                    verbose=self.verbose,
                )
            pass
        pass


    def test_invalid_nsi(self):
        """Invalid inputs fail as expected"""
        # case: non-negative scalar integer
        perrlist = ['cfgfn', None, np.ones((5, 5)), 1j, -1.5, 1.5]

        # ind
        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe_gaussian(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    sigma=self.sigma,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=perr,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass
        pass


    def test_invalid_rs(self):
        """Invalid inputs fail as expected"""
        # case: real scalar
        perrlist = ['cfgfn', None, np.ones((5, 5)), 1j]

        # xcenter
        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe_gaussian(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    xcenter=perr,
                    ycenter=self.ycenter,
                    sigma=self.sigma,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass

        # ycenter
        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe_gaussian(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    xcenter=self.xcenter,
                    ycenter=perr,
                    sigma=self.sigma,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=self.verbose,
                )
            pass
        pass


    def test_invalid_bool(self):
        """Invalid inputs fail as expected"""
        # case: bool
        perrlist = [None]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                make_dmrel_probe_gaussian(
                    cfg=self.cfg,
                    dmlist=self.dmlist,
                    xcenter=self.xcenter,
                    ycenter=self.ycenter,
                    sigma=self.sigma,
                    target=self.target,
                    lod_min=self.lod_min,
                    lod_max=self.lod_max,
                    ind=self.ind,
                    maxiter=self.maxiter,
                    verbose=perr,
                )
            pass
        pass


    def test_invalid_ranges(self):
        """Invalid inputs fail as expected"""

        # lod_min/lod_max
        with self.assertRaises(ValueError):
            make_dmrel_probe_gaussian(
                cfg=self.cfg,
                dmlist=self.dmlist,
                xcenter=self.xcenter,
                ycenter=self.ycenter,
                sigma=self.sigma,
                target=self.target,
                lod_min=self.lod_max, # same
                lod_max=self.lod_max, # same
                ind=self.ind,
                maxiter=self.maxiter,
                verbose=self.verbose,
            )
        pass


    def test_ind_out_of_range(self):
        """Invalid inputs fail as expected"""
        with self.assertRaises(TypeError):
            make_dmrel_probe_gaussian(
                cfg=self.cfg,
                dmlist=self.dmlist,
                xcenter=self.xcenter,
                ycenter=self.ycenter,
                sigma=self.sigma,
                target=self.target,
                lod_min=self.lod_min,
                lod_max=self.lod_max,
                ind=len(self.cfg.sl_list),
                maxiter=self.maxiter,
                verbose=self.verbose,
            )
        pass



if __name__ == '__main__':
    unittest.main()
