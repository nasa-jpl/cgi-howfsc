# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
# pylint: disable=unsubscriptable-object
"""
Unit tests for post-wavefront-control steps
"""

import unittest
from unittest.mock import patch
import os

import numpy as np

from howfsc.model.mode import CoronagraphMode
from howfsc.util.insertinto import insertinto
from .nextiter import get_next_c, expected_time, get_scale_factor_list

cfgpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'model', 'testdata', 'widefov', 'widefov.yaml')
cfg = CoronagraphMode(cfgpath)

class TestGetNextC(unittest.TestCase):
    """
    Tests for computation of the expected contrast for next iteration
    """

    def setUp(self):
        self.cfg = cfg
        self.dmlist = self.cfg.initmaps
        #436 = (1024 - 1024//2) - 153//2
        self.croplist = [(436, 436, 153, 153)]*len(self.cfg.sl_list)
        self.fixedbp = np.zeros((1024, 1024)).astype('bool')
        self.n2clist = [np.ones((153, 153))]*len(self.cfg.sl_list)
        self.destlist = []
        for sl in self.cfg.sl_list:
            self.destlist.append(np.zeros((sl.dh.e.shape), dtype='complex128'))
            pass
        pass


    def test_success(self):
        """Completely successfully with valid inputs"""
        get_next_c(cfg=self.cfg,
                   dmlist=self.dmlist,
                   croplist=self.croplist,
                   fixedbp=self.fixedbp,
                   n2clist=self.n2clist,
                   destlist=self.destlist)
        pass


    def test_clean_defaults(self):
        """Verify defaults work as documented"""
        out0 = get_next_c(cfg=self.cfg,
                          dmlist=self.dmlist,
                          croplist=self.croplist,
                          fixedbp=self.fixedbp,
                          n2clist=self.n2clist,
                          destlist=self.destlist)

        out1 = get_next_c(cfg=self.cfg,
                          dmlist=self.dmlist,
                          croplist=self.croplist,
                          fixedbp=self.fixedbp,
                          n2clist=self.n2clist,
                          destlist=self.destlist,
                          cleanrow=1024,
                          cleancol=1024)

        self.assertTrue(out0 == out1)
        pass


    def test_mean_defaults(self):
        """Verify defaults work as documented"""
        out0 = get_next_c(cfg=self.cfg,
                          dmlist=self.dmlist,
                          croplist=self.croplist,
                          fixedbp=self.fixedbp,
                          n2clist=self.n2clist,
                          destlist=self.destlist,
        )

        out1 = get_next_c(cfg=self.cfg,
                          dmlist=self.dmlist,
                          croplist=self.croplist,
                          fixedbp=self.fixedbp,
                          n2clist=self.n2clist,
                          destlist=self.destlist,
                          method='mean',
        )

        self.assertTrue(out0 == out1)
        pass


    def test_percentile_defaults(self):
        """Verify defaults work as documented"""
        out0 = get_next_c(cfg=self.cfg,
                          dmlist=self.dmlist,
                          croplist=self.croplist,
                          fixedbp=self.fixedbp,
                          n2clist=self.n2clist,
                          destlist=self.destlist,
                          method='percentile',
        )

        out1 = get_next_c(cfg=self.cfg,
                          dmlist=self.dmlist,
                          croplist=self.croplist,
                          fixedbp=self.fixedbp,
                          n2clist=self.n2clist,
                          destlist=self.destlist,
                          method='percentile',
                          percentile=50,
        )

        self.assertTrue(out0 == out1)
        pass


    def test_index_list_defaults(self):
        """Verify defaults work as documented"""
        out0 = get_next_c(cfg=self.cfg,
                          dmlist=self.dmlist,
                          croplist=self.croplist,
                          fixedbp=self.fixedbp,
                          n2clist=self.n2clist,
                          destlist=self.destlist,
        )

        out1 = get_next_c(cfg=self.cfg,
                          dmlist=self.dmlist,
                          croplist=self.croplist,
                          fixedbp=self.fixedbp,
                          n2clist=self.n2clist,
                          destlist=self.destlist,
                          index_list=[0, 1, 2], # 3 wavelengths in self.cfg
        )

        self.assertTrue(out0 == out1)
        pass


    def test_fullframe_defaults(self):
        """Verify defaults work as documented"""
        out0 = get_next_c(cfg=self.cfg,
                          dmlist=self.dmlist,
                          croplist=self.croplist,
                          fixedbp=self.fixedbp,
                          n2clist=self.n2clist,
                          destlist=self.destlist,
        )

        out1 = get_next_c(cfg=self.cfg,
                          dmlist=self.dmlist,
                          croplist=self.croplist,
                          fixedbp=self.fixedbp,
                          n2clist=self.n2clist,
                          destlist=self.destlist,
                          fullframe=False,
        )

        self.assertTrue(out0 == out1)
        pass


    def test_percentile_ignored_with_mean(self):
        """Verify percentile argument does not affect output when using mean"""
        out0 = get_next_c(cfg=self.cfg,
                          dmlist=self.dmlist,
                          croplist=self.croplist,
                          fixedbp=self.fixedbp,
                          n2clist=self.n2clist,
                          destlist=self.destlist,
                          method='mean',
                          percentile=0,
        )

        out1 = get_next_c(cfg=self.cfg,
                          dmlist=self.dmlist,
                          croplist=self.croplist,
                          fixedbp=self.fixedbp,
                          n2clist=self.n2clist,
                          destlist=self.destlist,
                          method='mean',
                          percentile=100,
        )

        self.assertTrue(out0 == out1)
        pass


    @patch('howfsc.model.singlelambda.SingleLambda.proptodh')
    def test_exact_NI(self, mock_edh):
        """
        Verify that we get an exact result when expected

        This is specifically checking that we got NI right by excluding pixels
        outside the dark hole

        """
        target = 1e-8
        tol = 1e-13

        edhlist = []
        for index, sl in enumerate(cfg.sl_list):
            edh = np.sqrt(target)*np.ones_like(sl.dh.e)

            if index == 0:
                r_in, c_in = np.where(sl.dh.e != 0)
                r_out, c_out = np.where(sl.dh.e != 0)

                # set bad value outside, to be clipped by dh
                edh[r_out[0], c_out[0]] = 3.8

                # set bad value inside, to be clipped by fixedbp
                edh[r_in[0], c_in[0]] = 4.97

                # set bad value inside, to be caught by nan in destlist
                edh[r_in[1], c_in[1]] = 5.559
                pass

            edhlist.append(edh)
            pass
        mock_edh.side_effect = edhlist

        tmp = np.zeros_like(self.cfg.sl_list[0].dh.e)
        tmp[r_in[0], c_in[0]] = 1
        fixedbp = insertinto(tmp, (1024, 1024)).astype('bool')

        self.destlist[0][r_in[1], c_in[1]] = np.nan

        out = get_next_c(cfg=self.cfg,
                         dmlist=self.dmlist,
                         croplist=self.croplist,
                         fixedbp=fixedbp,
                         n2clist=self.n2clist,
                         destlist=self.destlist)

        self.assertTrue(np.max(np.abs(out - target)) < tol)

        pass


    @patch('howfsc.model.singlelambda.SingleLambda.proptodh')
    def test_different_NI(self, mock_edh):
        """
        Verify that we get an exact result when expected

        This is specifically checking that NI changes by NOT excluding
        pixels outside the dark hole, by request

        """
        target = 1e-8
        bright_target = 3.8
        tol = 1e-13

        edhlist = []
        for index, sl in enumerate(cfg.sl_list):
            edh = np.sqrt(target)*np.ones_like(sl.dh.e)

            if index == 0:
                r_in, c_in = np.where(sl.dh.e != 0)
                r_out, c_out = np.where(sl.dh.e != 0)

                # set bad values outside, NOT to be clipped by dh
                edh[r_out, c_out] = np.sqrt(bright_target)

                # set bad value inside, to be clipped by fixedbp
                edh[r_in[0], c_in[0]] = 4.97

                # set bad value inside, to be caught by nan in destlist
                edh[r_in[1], c_in[1]] = 5.559
                pass

            edhlist.append(edh)
            pass
        mock_edh.side_effect = edhlist

        tmp = np.zeros_like(self.cfg.sl_list[0].dh.e)
        tmp[r_in[0], c_in[0]] = 1
        fixedbp = insertinto(tmp, (1024, 1024)).astype('bool')

        self.destlist[0][r_in[1], c_in[1]] = np.nan

        out = get_next_c(cfg=self.cfg,
                         dmlist=self.dmlist,
                         croplist=self.croplist,
                         fixedbp=fixedbp,
                         n2clist=self.n2clist,
                         destlist=self.destlist,
                         method='percentile',
                         percentile=100,
                         fullframe=True)

        self.assertTrue(np.max(np.abs(out - bright_target)) < tol)

        pass


    @patch('howfsc.model.singlelambda.SingleLambda.proptodh')
    def test_exact_mean_percentile_0(self, mock_edh):
        """
        Verify that we get an exact result following the 'method' argument
        """
        tol = 1e-13

        # we'll make all the pixels open and uniform, but differing per
        # wavelength.  Metrics will be different too:  mean = 14/3,
        # median = 4, 100% percentile/max = 9
        edhlist = []
        pixlist = []
        n2clist = []
        destlist = []
        for index, sl in enumerate(cfg.sl_list):
            edh = np.zeros_like(sl.dh.e)
            self.cfg.sl_list[index].dh.e = np.ones_like(sl.dh.e)
            edhlist.append(edh)

            n2c = np.ones_like(sl.dh.e)
            n2clist.append(n2c)

            dest = (index+1)*np.ones((sl.dh.e.shape), dtype='complex128')
            destlist.append(dest)

            pixlist.append((index+1)**2)
            pass
        mock_edh.side_effect = edhlist

        fixedbp = np.zeros((1024, 1024), dtype=bool)

        mean_target = np.mean(pixlist)
        self.assertTrue(np.max(np.abs(mean_target - 14/3)) < tol)
        mean_out = get_next_c(cfg=self.cfg,
                              dmlist=self.dmlist,
                              croplist=self.croplist,
                              fixedbp=fixedbp,
                              n2clist=n2clist,
                              destlist=destlist,
                              method='mean',
        )

        self.assertTrue(np.max(np.abs(mean_out - mean_target)) < tol)

        pass


    @patch('howfsc.model.singlelambda.SingleLambda.proptodh')
    def test_exact_mean_percentile_1(self, mock_edh):
        """
        Verify that we get an exact result following the 'method' argument
        """
        tol = 1e-13

        # we'll make all the pixels open and uniform, but differing per
        # wavelength.  Metrics will be different too:  mean = 14/3,
        # median = 4, 100% percentile/max = 9
        edhlist = []
        pixlist = []
        n2clist = []
        destlist = []
        for index, sl in enumerate(cfg.sl_list):
            edh = np.zeros_like(sl.dh.e)
            self.cfg.sl_list[index].dh.e = np.ones_like(sl.dh.e)
            edhlist.append(edh)

            n2c = np.ones_like(sl.dh.e)
            n2clist.append(n2c)

            dest = (index+1)*np.ones((sl.dh.e.shape), dtype='complex128')
            destlist.append(dest)

            pixlist.append((index+1)**2)
            pass
        mock_edh.side_effect = edhlist

        fixedbp = np.zeros((1024, 1024), dtype=bool)

        median_target = np.median(pixlist)
        self.assertTrue(np.max(np.abs(median_target - 4)) < tol)
        median_out = get_next_c(cfg=self.cfg,
                                dmlist=self.dmlist,
                                croplist=self.croplist,
                                fixedbp=fixedbp,
                                n2clist=n2clist,
                                destlist=destlist,
                                method='percentile',
                                percentile=50,
        )

        self.assertTrue(np.max(np.abs(median_out - median_target)) < tol)

        pass


    @patch('howfsc.model.singlelambda.SingleLambda.proptodh')
    def test_exact_mean_percentile_2(self, mock_edh):
        """
        Verify that we get an exact result following the 'method' argument
        """
        tol = 1e-13

        # we'll make all the pixels open and uniform, but differing per
        # wavelength.  Metrics will be different too:  mean = 14/3,
        # median = 4, 100% percentile/max = 9
        edhlist = []
        pixlist = []
        n2clist = []
        destlist = []
        for index, sl in enumerate(cfg.sl_list):
            edh = np.zeros_like(sl.dh.e)
            self.cfg.sl_list[index].dh.e = np.ones_like(sl.dh.e)
            edhlist.append(edh)

            n2c = np.ones_like(sl.dh.e)
            n2clist.append(n2c)

            dest = (index+1)*np.ones((sl.dh.e.shape), dtype='complex128')
            destlist.append(dest)

            pixlist.append((index+1)**2)
            pass
        mock_edh.side_effect = edhlist

        fixedbp = np.zeros((1024, 1024), dtype=bool)

        max_target = np.max(pixlist)
        self.assertTrue(np.max(np.abs(max_target - 9)) < tol)
        max_out = get_next_c(cfg=self.cfg,
                             dmlist=self.dmlist,
                             croplist=self.croplist,
                             fixedbp=fixedbp,
                             n2clist=n2clist,
                             destlist=destlist,
                             method='percentile',
                             percentile=100,
        )

        self.assertTrue(np.max(np.abs(max_out - max_target)) < tol)

        pass


    @patch('howfsc.model.singlelambda.SingleLambda.proptodh')
    def test_exact_mean_two_lambda(self, mock_edh):
        """
        Verify that we get an exact result when using a smaller index_list
        """
        tol = 1e-13
        index_list = [0, 2]

        # we'll make all the pixels open and uniform, but differing per
        # wavelength.  With only indices 0 and 2, mean = 5
        edhlist = []
        pixlist = []
        n2clist = []
        destlist = []
        for index, sl in enumerate(cfg.sl_list):

            edh = np.zeros_like(sl.dh.e)
            self.cfg.sl_list[index].dh.e = np.ones_like(sl.dh.e)
            edhlist.append(edh)

            n2c = np.ones_like(sl.dh.e)
            n2clist.append(n2c)

            dest = (index+1)*np.ones((sl.dh.e.shape), dtype='complex128')
            destlist.append(dest)
            pass
        mock_edh.side_effect = edhlist

        fixedbp = np.zeros((1024, 1024), dtype=bool)

        for index in index_list:
            pixlist.append((index+1)**2)
            pass

        mean_target = np.mean(pixlist)
        self.assertTrue(np.max(np.abs(mean_target - 5)) < tol)
        mean_out = get_next_c(cfg=self.cfg,
                              dmlist=self.dmlist,
                              croplist=self.croplist,
                              fixedbp=fixedbp,
                              n2clist=n2clist,
                              destlist=destlist,
                              method='mean',
                              index_list=index_list,
        )

        self.assertTrue(np.max(np.abs(mean_out - mean_target)) < tol)

        pass



    @patch('howfsc.model.singlelambda.SingleLambda.proptodh')
    def test_exact_c(self, mock_edh):
        """
        Verify that we get an exact result when expected

        This is specifically checking that we got contrast right with the n2c
        multiplier

        """
        target = 2e-8
        NI = 1e-8
        n2crat = target/NI
        tol = 1e-13

        edhlist = []
        n2clist = []
        for sl in cfg.sl_list:
            edh = np.sqrt(NI)*np.ones_like(sl.dh.e)
            edhlist.append(edh)

            n2c = 1.782e3*np.ones(sl.dh.e.shape)
            n2c[sl.dh.e == 1] = n2crat
            n2clist.append(n2c)
            pass
        mock_edh.side_effect = edhlist

        out = get_next_c(cfg=self.cfg,
                         dmlist=self.dmlist,
                         croplist=self.croplist,
                         fixedbp=self.fixedbp,
                         n2clist=n2clist,
                         destlist=self.destlist)

        self.assertTrue(np.max(np.abs(out - target)) < tol)

        pass


    def test_no_good_pixels(self):
        """fails as expected when there are no acceptable pixels"""
        allbad = np.ones_like(self.fixedbp)
        with self.assertRaises(ZeroDivisionError):
            get_next_c(cfg=self.cfg,
                       dmlist=self.dmlist,
                       croplist=self.croplist,
                       fixedbp=allbad,
                       n2clist=self.n2clist,
                       destlist=self.destlist,
                       fullframe=False,
            )
            pass

        with self.assertRaises(ZeroDivisionError):
            get_next_c(cfg=self.cfg,
                       dmlist=self.dmlist,
                       croplist=self.croplist,
                       fixedbp=allbad,
                       n2clist=self.n2clist,
                       destlist=self.destlist,
                       fullframe=True,
            )
            pass
        pass


    def test_not_lists(self):
        """Verify invalid inputs caught"""
        xlist = [1, None, 'txt', (5, 5)]

        # dmlist
        for x in xlist:
            with self.assertRaises(TypeError):
                get_next_c(self.cfg, x, self.croplist,
                           self.fixedbp, self.n2clist, self.destlist)
            pass

        # croplist
        for x in xlist:
            with self.assertRaises(TypeError):
                get_next_c(self.cfg, self.dmlist, x,
                           self.fixedbp, self.n2clist, self.destlist)
            pass

        # n2clist
        for x in xlist:
            with self.assertRaises(TypeError):
                get_next_c(self.cfg, self.dmlist, self.croplist,
                           self.fixedbp, x, self.destlist)
            pass

        # destlist
        for x in xlist:
            with self.assertRaises(TypeError):
                get_next_c(self.cfg, self.dmlist, self.croplist,
                           self.fixedbp, self.n2clist, x)
            pass

        pass


    def test_invalid_cfg(self):
        """Verify invalid inputs caught"""
        xlist = [1, None, 'txt', (5, 5)]

        for x in xlist:
            with self.assertRaises(TypeError):
                get_next_c(x, self.dmlist, self.croplist,
                           self.fixedbp, self.n2clist, self.destlist)
            pass
        pass


    def test_invalid_fixedbp(self):
        """Verify invalid inputs caught"""
        xlist = [1, None, 'txt', (5, 5), np.ones((2,)).astype('bool'),
                 np.ones((2, 2, 2)).astype('bool')]

        for x in xlist:
            with self.assertRaises(TypeError):
                get_next_c(self.cfg, self.dmlist, self.croplist,
                           x, self.n2clist, self.destlist)
            pass
        pass


    def test_dmlist_elements_wrong_size(self):
        """Verify invalid inputs caught"""
        nact = cfg.dmlist[0].registration['nact']
        baddmlist = [np.ones((nact+1, nact+1))]*len(cfg.dmlist)

        with self.assertRaises(TypeError):
            get_next_c(self.cfg, baddmlist, self.croplist,
                       self.fixedbp, self.n2clist, self.destlist)
        pass


    def test_dmlist_elements_wrong_type(self):
        """Verify invalid inputs caught"""
        nact = cfg.dmlist[0].registration['nact']
        xlist = [[np.ones((nact,))]*len(cfg.dmlist),
                 [np.ones((nact, nact, 2))]*len(cfg.dmlist),
                 [1]*len(cfg.dmlist),
                 [None]*len(cfg.dmlist),
                 ['txt']*len(cfg.dmlist),
                 [(5, 5)]*len(cfg.dmlist),
                 ]

        for x in xlist:
            with self.assertRaises(TypeError):
                get_next_c(self.cfg, x, self.croplist,
                           self.fixedbp, self.n2clist, self.destlist)
            pass
        pass


    def test_dmlist_wrong_length(self):
        """Verify invalid inputs caught"""
        nact = cfg.dmlist[0].registration['nact']
        baddmlist = [np.ones((nact, nact))]*(len(cfg.dmlist) + 1)

        with self.assertRaises(TypeError):
            get_next_c(self.cfg, baddmlist, self.croplist,
                       self.fixedbp, self.n2clist, self.destlist)
        pass


    def test_croplist_wrong_length(self):
        """Verify invalid inputs caught"""
        badcroplist = [(436, 436, 153, 153)]*(len(cfg.sl_list) + 1)

        with self.assertRaises(TypeError):
            get_next_c(self.cfg, self.dmlist, badcroplist,
                       self.fixedbp, self.n2clist, self.destlist)
        pass


    def test_croplist_elements_wrong_type(self):
        """Verify invalid inputs caught"""
        xlist = [[np.ones((4,))]*len(cfg.sl_list),
                 [np.ones((4, 4))]*len(cfg.sl_list),
                 [np.ones((4, 4, 2))]*len(cfg.sl_list),
                 [1]*len(cfg.sl_list),
                 [None]*len(cfg.sl_list),
                 ['txt']*len(cfg.sl_list),
                 ]

        for x in xlist:
            with self.assertRaises(TypeError):
                get_next_c(self.cfg, self.dmlist, x,
                           self.fixedbp, self.n2clist, self.destlist)
            pass
        pass


    def test_croplist_elements_wrong_size(self):
        """Verify invalid inputs caught"""
        xlist = [[(436, 436, 153)]*len(cfg.sl_list),
                 [(436, 436, 153, 153, 0)]*len(cfg.sl_list),
                 ]

        for x in xlist:
            with self.assertRaises(TypeError):
                get_next_c(self.cfg, self.dmlist, x,
                           self.fixedbp, self.n2clist, self.destlist)
            pass
        pass


    def test_croplist_tuple_elements_wrong_type(self):
        """Verify invalid inputs caught"""
        xlist = [[(-1, 436, 153, 153)]*len(cfg.sl_list),
                 [(1.5, 436, 153, 153)]*len(cfg.sl_list),
                 [(436, -1, 153, 153)]*len(cfg.sl_list),
                 [(436, 1.5, 153, 153)]*len(cfg.sl_list),
                 [(436, 436, -1, 153)]*len(cfg.sl_list),
                 [(436, 436, 0, 153)]*len(cfg.sl_list),
                 [(436, 436, 1.5, 153)]*len(cfg.sl_list),
                 [(436, 436, -1, 153)]*len(cfg.sl_list),
                 [(436, 436, 0, 153)]*len(cfg.sl_list),
                 [(436, 436, 1.5, 153)]*len(cfg.sl_list),
                 ]

        for x in xlist:
            with self.assertRaises(TypeError):
                get_next_c(self.cfg, self.dmlist, x,
                           self.fixedbp, self.n2clist, self.destlist)
            pass
        pass


    def test_n2clist_wrong_length(self):
        """Verify invalid inputs caught"""
        badn2clist = [np.ones((153, 153))]*(len(cfg.sl_list) + 1)

        with self.assertRaises(TypeError):
            get_next_c(self.cfg, self.dmlist, self.croplist,
                       self.fixedbp, badn2clist, self.destlist)
        pass


    def test_n2clist_elements_wrong_type(self):
        """Verify invalid inputs caught"""
        xlist = [[np.ones((153,))]*len(cfg.sl_list),
                 [np.ones((153, 153, 2))]*len(cfg.sl_list),
                 [1]*len(cfg.sl_list),
                 [None]*len(cfg.sl_list),
                 ['txt']*len(cfg.sl_list),
                 [(5, 5)]*len(cfg.sl_list),
                 ]

        for x in xlist:
            with self.assertRaises(TypeError):
                get_next_c(self.cfg, self.dmlist, self.croplist,
                           self.fixedbp, x, self.destlist)
            pass
        pass

    def test_n2clist_elements_too_small(self):
        """Verify invalid inputs caught"""
        okcroplist = [(436, 436, 153, 153)]*len(cfg.sl_list)
        badn2clist = [np.ones((152, 152))]*len(cfg.sl_list)

        with self.assertRaises(TypeError):
            get_next_c(self.cfg, self.dmlist, okcroplist,
                       self.fixedbp, badn2clist, self.destlist)
        pass


    def test_destlist_wrong_length(self):
        """Verify invalid inputs caught"""
        baddestlist = [self.destlist[0]]*(len(cfg.sl_list) + 1)

        with self.assertRaises(TypeError):
            get_next_c(self.cfg, self.dmlist, self.croplist,
                       self.fixedbp, self.n2clist, baddestlist)
        pass


    def test_destlist_elements_wrong_type(self):
        """Verify invalid inputs caught"""
        xlist = [[np.ones((153,))]*len(cfg.sl_list),
                 [np.ones((153, 153, 2))]*len(cfg.sl_list),
                 [1]*len(cfg.sl_list),
                 [None]*len(cfg.sl_list),
                 ['txt']*len(cfg.sl_list),
                 [(5, 5)]*len(cfg.sl_list),
                 ]

        for x in xlist:
            with self.assertRaises(TypeError):
                get_next_c(self.cfg, self.dmlist, self.croplist,
                           self.fixedbp, self.n2clist, x)
            pass
        pass


    def test_destlist_elements_wrong_size(self):
        """Verify invalid inputs caught"""
        baddestlist = []
        for dest in self.destlist:
            baddestlist.append(dest[:-1, :-1])
            pass

        with self.assertRaises(TypeError):
            get_next_c(self.cfg, self.dmlist, self.croplist,
                       self.fixedbp, self.n2clist, baddestlist)
        pass



    def test_invalid_cleanrow(self):
        """Verify invalid inputs caught"""
        xlist = [0, -1, 1.5, None, 'txt', (5, 5), np.ones((2,)).astype('bool'),
                 np.ones((2, 2, 2)).astype('bool')]

        for x in xlist:
            with self.assertRaises(TypeError):
                get_next_c(self.cfg, self.dmlist, self.croplist,
                           self.fixedbp, self.n2clist, self.destlist,
                           cleanrow=x)
            pass
        pass


    def test_invalid_cleancol(self):
        """Verify invalid inputs caught"""
        xlist = [0, -1, 1.5, None, 'txt', (5, 5), np.ones((2,)).astype('bool'),
                 np.ones((2, 2, 2)).astype('bool')]

        for x in xlist:
            with self.assertRaises(TypeError):
                get_next_c(self.cfg, self.dmlist, self.croplist,
                           self.fixedbp, self.n2clist, self.destlist,
                           cleancol=x)
            pass
        pass


    def test_fixedbp_wrong_size(self):
        """Verify invalid inputs caught"""

        with self.assertRaises(TypeError):
            get_next_c(self.cfg, self.dmlist, self.croplist,
                       self.fixedbp, self.n2clist, self.destlist,
                       cleanrow=self.fixedbp.shape[0]+1,
                       cleancol=self.fixedbp.shape[1])

        with self.assertRaises(TypeError):
            get_next_c(self.cfg, self.dmlist, self.croplist,
                       self.fixedbp, self.n2clist, self.destlist,
                       cleanrow=self.fixedbp.shape[0],
                       cleancol=self.fixedbp.shape[1]+1)

        with self.assertRaises(TypeError):
            get_next_c(self.cfg, self.dmlist, self.croplist,
                       self.fixedbp, self.n2clist, self.destlist,
                       cleanrow=self.fixedbp.shape[0]+1,
                       cleancol=self.fixedbp.shape[1]+1)
        pass


    def test_invalid_method(self):
        """Verify invalid inputs caught"""
        xlist = [0, -1, 1.5, None, (5, 5), # wrong type
                 'txt', 'mode', # wrong content
                 ]

        for x in xlist:
            with self.assertRaises(TypeError):
                get_next_c(self.cfg, self.dmlist, self.croplist,
                           self.fixedbp, self.n2clist, self.destlist,
                           method=x)
            pass
        pass


    def test_invalid_percentile(self):
        """Verify invalid inputs caught"""
        xlist = [1j, None, (5, 5), np.ones((2,)), 'txt'] # not float

        for x in xlist:
            with self.assertRaises(TypeError):
                get_next_c(self.cfg, self.dmlist, self.croplist,
                           self.fixedbp, self.n2clist, self.destlist,
                           method='percentile', percentile=x)
            pass
        pass

        ylist = [-0.001, 100.001, 1000, -1000] # not in [0, 100]

        for y in ylist:
            with self.assertRaises(ValueError):
                get_next_c(self.cfg, self.dmlist, self.croplist,
                           self.fixedbp, self.n2clist, self.destlist,
                           method='percentile', percentile=y)
            pass
        pass


    def test_invalid_index_list(self):
        """Verify invalid inputs caught"""
        xlist = [0, -1, 1.5, (5, 5), 'txt', # wrong type
                 [], # empty list
                 [0, 1, 'txt'], [0, 1, 3.5], [0, 1, None], # wrong list types
                 [0, 1, -1], [0, 1, 10000000000], # elements out of range
                 ]

        for x in xlist:
            with self.assertRaises(TypeError):
                get_next_c(self.cfg, self.dmlist, self.croplist,
                           self.fixedbp, self.n2clist, self.destlist,
                           index_list=x)
            pass
        pass


    def test_invalid_fullframe(self):
        """Verify invalid inputs caught"""
        xlist = [-1, 1.5, 'False', None, [2, 3]]

        for x in xlist:
            with self.assertRaises(TypeError):
                get_next_c(self.cfg, self.dmlist, self.croplist,
                           self.fixedbp, self.n2clist, self.destlist,
                           fullframe=x)
            pass
        pass



class TestExpectedTime(unittest.TestCase):
    """
    Tests for function to compute expected time of operation for the next
    iteration

    success
    invalid inputs
    exact answer(s)
    """

    def setUp(self):
        self.ndm = 7
        self.nfilt = 3
        self.exptime = [12.0]*21
        self.nframes = [5]*21
        self.overdm = 5
        self.overfilt = 60
        self.overboth = 10
        self.overfixed = 5
        self.overframe = 2
        pass

    def test_success(self):
        """good inputs complete successfully"""
        expected_time(self.ndm, self.nfilt, self.exptime, self.nframes,
                      self.overdm, self.overfilt, self.overboth,
                      self.overfixed, self.overframe)


    def test_exact_no_overhead(self):
        """exposure-time only calculated as expected"""
        ndm = 19
        nfilt = 11
        exptime = list(range(1, ndm*nfilt+1)) # 1 to ndm*nfilt
        nframes = [1]*(ndm*nfilt)

        tol = 1e-13
        target = (ndm*nfilt)*(ndm*nfilt + 1)/2 # Gauss sum of 1st n ints

        out = expected_time(ndm, nfilt, exptime, nframes, 0, 0, 0, 0, 0)
        self.assertTrue(np.max(np.abs(out - target)) < tol)

        pass


    def test_exact_overhead(self):
        """calculation with overhead runs as expected"""
        ndm = 13
        nfilt = 15

        exptime = [1]*(ndm*nfilt)
        nframes = [1]*(ndm*nfilt)

        overdm = 1
        overfilt = 1
        overboth = 1
        overfixed = 1
        overframe = 1

        tol = 1e-13
        target = 1 + ndm + nfilt + ndm*nfilt + 2*ndm*nfilt

        out = expected_time(ndm, nfilt, exptime, nframes,
                            overdm, overfilt, overboth, overfixed, overframe)
        self.assertTrue(np.max(np.abs(out - target)) < tol)

        pass


    def test_invalid_ndm(self):
        """Verify invalid data caught"""
        xlist = [-1, 0, 1.5, 'txt', None, (5,), np.ones((2,)), np.ones((2, 2))]

        for x in xlist:
            with self.assertRaises(TypeError):
                expected_time(x, self.nfilt, self.exptime, self.nframes,
                              self.overdm, self.overfilt, self.overboth,
                              self.overfixed, self.overframe)
            pass
        pass


    def test_invalid_nfilt(self):
        """Verify invalid data caught"""
        xlist = [-1, 0, 1.5, 'txt', None, (5,), np.ones((2,)), np.ones((2, 2))]

        for x in xlist:
            with self.assertRaises(TypeError):
                expected_time(self.ndm, x, self.exptime, self.nframes,
                              self.overdm, self.overfilt, self.overboth,
                              self.overfixed, self.overframe)
            pass
        pass


    def test_invalid_exptime(self):
        """Verify invalid data caught"""
        xlist = [-1, 0, 1.5, 'txt', None, (5,), np.ones((2, 2))]

        for x in xlist:
            with self.assertRaises(TypeError):
                expected_time(self.ndm, self.nfilt, x, self.nframes,
                              self.overdm, self.overfilt, self.overboth,
                              self.overfixed, self.overframe)
            pass
        pass


    def test_invalid_nframes(self):
        """Verify invalid data caught"""
        xlist = [-1, 0, 1.5, 'txt', None, (5,), np.ones((2, 2))]

        for x in xlist:
            with self.assertRaises(TypeError):
                expected_time(self.ndm, self.nfilt, self.exptime, x,
                              self.overdm, self.overfilt, self.overboth,
                              self.overfixed, self.overframe)
            pass
        pass


    def test_invalid_overdm(self):
        """Verify invalid data caught"""
        xlist = [-1, 'txt', None, (5,), np.ones((2,)), np.ones((2, 2))]

        for x in xlist:
            with self.assertRaises(TypeError):
                expected_time(self.ndm, self.nfilt, self.exptime, self.nframes,
                              x, self.overfilt, self.overboth,
                              self.overfixed, self.overframe)
            pass
        pass


    def test_invalid_overfilt(self):
        """Verify invalid data caught"""
        xlist = [-1, 'txt', None, (5,), np.ones((2,)), np.ones((2, 2))]

        for x in xlist:
            with self.assertRaises(TypeError):
                expected_time(self.ndm, self.nfilt, self.exptime, self.nframes,
                              self.overdm, x, self.overboth,
                              self.overfixed, self.overframe)
            pass
        pass


    def test_invalid_overboth(self):
        """Verify invalid data caught"""
        xlist = [-1, 'txt', None, (5,), np.ones((2,)), np.ones((2, 2))]

        for x in xlist:
            with self.assertRaises(TypeError):
                expected_time(self.ndm, self.nfilt, self.exptime, self.nframes,
                              self.overdm, self.overfilt, x,
                              self.overfixed, self.overframe)
            pass
        pass


    def test_invalid_overfixed(self):
        """Verify invalid data caught"""
        xlist = [-1, 'txt', None, (5,), np.ones((2,)), np.ones((2, 2))]

        for x in xlist:
            with self.assertRaises(TypeError):
                expected_time(self.ndm, self.nfilt, self.exptime, self.nframes,
                              self.overdm, self.overfilt, self.overboth,
                              x, self.overframe)
            pass
        pass


    def test_invalid_overframe(self):
        """Verify invalid data caught"""
        xlist = [-1, 'txt', None, (5,), np.ones((2,)), np.ones((2, 2))]

        for x in xlist:
            with self.assertRaises(TypeError):
                expected_time(self.ndm, self.nfilt, self.exptime, self.nframes,
                              self.overdm, self.overfilt, self.overboth,
                              self.overfixed, x)
            pass
        pass


    def test_exptime_length(self):
        """Verify invalid data caught"""
        xlist = [[1]*(self.ndm*self.nfilt + 1)]

        for x in xlist:
            with self.assertRaises(TypeError):
                expected_time(self.ndm, self.nfilt, x, self.nframes,
                              self.overdm, self.overfilt, self.overboth,
                              self.overfixed, self.overframe)
            pass
        pass


    def test_exptime_elements_positive(self):
        """Verify invalid data caught"""
        xlist = [[-1]*(self.ndm*self.nfilt),
                 [0]*(self.ndm*self.nfilt)]

        for x in xlist:
            with self.assertRaises(TypeError):
                expected_time(self.ndm, self.nfilt, x, self.nframes,
                              self.overdm, self.overfilt, self.overboth,
                              self.overfixed, self.overframe)
            pass
        pass


    def test_nframes_length(self):
        """Verify invalid data caught"""
        xlist = [[1]*(self.ndm*self.nfilt + 1)]

        for x in xlist:
            with self.assertRaises(TypeError):
                expected_time(self.ndm, self.nfilt, self.exptime, x,
                              self.overdm, self.overfilt, self.overboth,
                              self.overfixed, self.overframe)
            pass
        pass


    def test_nframes_elements_positive_integers(self):
        """Verify invalid data caught"""
        xlist = [[-1]*(self.ndm*self.nfilt),
                 [0]*(self.ndm*self.nfilt),
                 [1.5]*(self.ndm*self.nfilt),
                 [1.0]*(self.ndm*self.nfilt),
                 ]

        for x in xlist:
            with self.assertRaises(TypeError):
                expected_time(self.ndm, self.nfilt, self.exptime, x,
                              self.overdm, self.overfilt, self.overboth,
                              self.overfixed, self.overframe)
            pass
        pass


class TestGetScaleFactorList(unittest.TestCase):
    """
    Tests for function to compute scale factors for the next iteration
    """

    def setUp(self):
        self.dmrel_ph_list = [1e-5, 1e-7, 1e-9]
        self.current_ph = 1e-7
        self.target = [0.1, 1, 10, -0.1, -1, -10] # based on sqrt relation


    def test_success(self):
        """Good inputs return as expected"""
        get_scale_factor_list(self.dmrel_ph_list, self.current_ph)
        pass


    def test_exact(self):
        """Inputs produce expected outputs"""
        tol = 1e-13

        out = get_scale_factor_list(self.dmrel_ph_list, self.current_ph)
        self.assertTrue(np.max(np.abs(np.asarray(out) -
                                      np.asarray(self.target))) < tol)
        pass


    def test_output_size(self):
        """Check output list size matches 2x input list size"""

        current_ph = 1e-7

        for n in np.arange(1, 10):
            dmrel = [1e-5]*n
            out = get_scale_factor_list(dmrel, current_ph)
            self.assertTrue(2*len(dmrel) == len(out))
            pass
        pass


    def test_second_half_negatives_of_first_half(self):
        """Check output list has second half as negatives of 1st half"""
        current_ph = 1e-7

        for n in np.arange(1, 10):
            dmrel = [1e-5]*n
            out = get_scale_factor_list(dmrel, current_ph)
            self.assertTrue(
                (np.asarray(out[:n]) == -np.asarray(out[n:])).all())
            pass
        pass


    def test_bad_dmrel_ph_list(self):
        """Invalid inputs caught"""
        badphlist = [(1, 2, 3), 1, 'txt', None, # not lists
                     [1j*1e-5, 1e-7], [None, 1e-7], # not real
                     [-1e-5, -1e-7], # not positive
                     [np.eye(2), np.ones((2, 2))], # not scalar
                     ]

        for badph in badphlist:
            with self.assertRaises(TypeError):
                get_scale_factor_list(badph, self.current_ph)
            pass
        pass


    def test_bad_current_ph(self):
        """Invalid inputs caught"""

        badclist = [-1, 0, 1j, None, 'txt', np.eye(3), [1e-5, 1e-7, 1e-9]]

        for badc in badclist:
            with self.assertRaises(TypeError):
                get_scale_factor_list(self.dmrel_ph_list, badc)
            pass
        pass



if __name__ == '__main__':
    unittest.main()
