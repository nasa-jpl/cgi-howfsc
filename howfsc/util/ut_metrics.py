# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Unit tests for metrics"""

import unittest
import warnings

import numpy as np

from .metrics import de_metrics

warnings.simplefilter('ignore')

class TestDEMetrics(unittest.TestCase):
    """
    Tests for offline delta-electric-field metrics
    """

    def setUp(self):
        self.nrow = 4
        self.ncol = 4

        self.dem_factor = -1.1 + 0.1j
        self.ded_factor = -1.2 + 0.2j

        self.meas_old = (4.9+0.1j)*np.ones((self.nrow, self.ncol))
        self.model_old = (5.0+0.0j)*np.ones((self.nrow, self.ncol))
        self.meas_new = self.meas_old + self.ded_factor
        self.model_new = self.model_old + self.dem_factor

        pass


    def test_success(self):
        """Valid inputs complete without error"""
        de_metrics(
            meas_old=self.meas_old,
            model_old=self.model_old,
            meas_new=self.meas_new,
            model_new=self.model_new,
        )
        pass


    def test_outputs(self):
        """Standard outputs match spec"""
        mdict = de_metrics(
            meas_old=self.meas_old,
            model_old=self.model_old,
            meas_new=self.meas_new,
            model_new=self.model_new,
        )

        self.assertTrue('CC' in mdict)
        self.assertTrue('dE_rat' in mdict)
        self.assertTrue('NI_rat' in mdict)
        self.assertTrue('exp_NI_rat' in mdict)

        # only last three are real
        self.assertTrue(mdict['dE_rat'].imag == 0)
        self.assertTrue(mdict['NI_rat'].imag == 0)
        self.assertTrue(mdict['exp_NI_rat'].imag == 0)

        pass


    def test_analytic_values(self):
        """numbers come out as expected for known inputs"""
        tol = 1e-13

        mdict = de_metrics(
            meas_old=self.meas_old,
            model_old=self.model_old,
            meas_new=self.meas_new,
            model_new=self.model_new,
        )

        CC = self.ded_factor*self.dem_factor.conjugate()
        CC /= np.sqrt(np.abs(self.dem_factor)**2)
        CC /= np.sqrt(np.abs(self.ded_factor)**2)

        self.assertTrue(np.abs(CC.real - mdict['CC'].real) < tol)
        self.assertTrue(np.abs(CC.imag - mdict['CC'].imag) < tol)

        dE_rat = np.abs(self.dem_factor)/np.abs(self.ded_factor)

        self.assertTrue(np.abs(dE_rat.real - mdict['dE_rat'].real) < tol)
        self.assertTrue(np.abs(dE_rat.imag - mdict['dE_rat'].imag) < tol)

        Eold = self.meas_old[0, 0]
        Enew = self.meas_new[0, 0]

        NI_rat = (1 - np.mean(np.abs(Enew)**2)/
                  np.mean(np.abs(Eold)**2))
        exp_NI_rat = (1 - np.mean(np.abs(Eold + self.dem_factor)**2)/
                  np.mean(np.abs(Eold)**2))

        self.assertTrue(np.abs(NI_rat.real - mdict['NI_rat'].real) < tol)
        self.assertTrue(np.abs(NI_rat.imag - mdict['NI_rat'].imag) < tol)

        self.assertTrue(np.abs(exp_NI_rat.real -
                               mdict['exp_NI_rat'].real) < tol)
        self.assertTrue(np.abs(exp_NI_rat.imag -
                               mdict['exp_NI_rat'].imag) < tol)

        pass


    def test_analytic_values_badpix(self):
        """numbers come out as expected for known inputs"""
        tol = 1e-13

        oldbp = np.ones((self.nrow, self.ncol))
        oldbp[0, 0] = np.nan
        newbp = np.ones((self.nrow, self.ncol))
        newbp[-1, -1] = np.nan

        mdict = de_metrics(
            meas_old=self.meas_old*oldbp,
            model_old=self.model_old*oldbp,
            meas_new=self.meas_new*newbp,
            model_new=self.model_new*newbp,
        )

        CC = self.ded_factor*self.dem_factor.conjugate()
        CC /= np.sqrt(np.abs(self.dem_factor)**2)
        CC /= np.sqrt(np.abs(self.ded_factor)**2)

        self.assertTrue(np.abs(CC.real - mdict['CC'].real) < tol)
        self.assertTrue(np.abs(CC.imag - mdict['CC'].imag) < tol)

        dE_rat = np.abs(self.dem_factor)/np.abs(self.ded_factor)

        self.assertTrue(np.abs(dE_rat.real - mdict['dE_rat'].real) < tol)
        self.assertTrue(np.abs(dE_rat.imag - mdict['dE_rat'].imag) < tol)

        Eold = self.meas_old[0, 0]
        Enew = self.meas_new[0, 0]

        NI_rat = (1 - np.mean(np.abs(Enew)**2)/
                  np.mean(np.abs(Eold)**2))
        exp_NI_rat = (1 - np.mean(np.abs(Eold + self.dem_factor)**2)/
                  np.mean(np.abs(Eold)**2))

        self.assertTrue(np.abs(NI_rat.real - mdict['NI_rat'].real) < tol)
        self.assertTrue(np.abs(NI_rat.imag - mdict['NI_rat'].imag) < tol)

        self.assertTrue(np.abs(exp_NI_rat.real -
                               mdict['exp_NI_rat'].real) < tol)
        self.assertTrue(np.abs(exp_NI_rat.imag -
                               mdict['exp_NI_rat'].imag) < tol)

        pass


    def test_allbad_pixels(self):
        """Fails expected if either or both iterations are all bad pixels"""

        with self.assertRaises(TypeError):
            de_metrics(
                meas_old=np.nan*self.meas_old,
                model_old=np.nan*self.model_old,
                meas_new=self.meas_new,
                model_new=self.model_new,
            )
            pass

        with self.assertRaises(TypeError):
            de_metrics(
                meas_old=self.meas_old,
                model_old=self.model_old,
                meas_new=np.nan*self.meas_new,
                model_new=np.nan*self.model_new,
            )
            pass

        with self.assertRaises(TypeError):
            de_metrics(
                meas_old=np.nan*self.meas_old,
                model_old=np.nan*self.model_old,
                meas_new=np.nan*self.meas_new,
                model_new=np.nan*self.model_new,
            )
            pass
        pass

    def test_mixedbad_pixels(self):
        """
        Fails when all pixels are bad on one or the other (but none in common)
        """

        half = np.ones_like(self.meas_old)
        half[:, :self.ncol//2] = np.nan
        otherhalf = np.ones_like(self.meas_old)
        otherhalf[:, self.ncol//2:] = np.nan

        # they should all be bad when put together
        self.assertTrue(np.isnan(half*otherhalf).all())

        with self.assertRaises(TypeError):
            de_metrics(
                meas_old=half*self.meas_old,
                model_old=half*self.model_old,
                meas_new=otherhalf*self.meas_new,
                model_new=otherhalf*self.model_new,
            )
            pass
        pass


    def test_datasame_only(self):
        """Check that expected calculations fail when data deltaE is 0"""
        with self.assertWarns(UserWarning):
            mdict = de_metrics(
                meas_old=self.meas_old,
                model_old=self.model_old,
                meas_new=self.meas_old,
                model_new=self.model_new,
            )

        self.assertTrue(mdict['CC'] is None)
        self.assertTrue(mdict['dE_rat'] is None)
        self.assertTrue(mdict['NI_rat'] is not None)
        self.assertTrue(mdict['exp_NI_rat'] is not None)

        pass


    def test_modelsame_only(self):
        """Check that expected calculations fail when model deltaE is 0"""
        with self.assertWarns(UserWarning):
            mdict = de_metrics(
                meas_old=self.meas_old,
                model_old=self.model_old,
                meas_new=self.meas_new,
                model_new=self.model_old,
            )

        self.assertTrue(mdict['CC'] is None)
        self.assertTrue(mdict['dE_rat'] is not None)
        self.assertTrue(mdict['NI_rat'] is not None)
        self.assertTrue(mdict['exp_NI_rat'] is not None)

        pass


    def test_datasame_modelsame(self):
        """
        Check that expected calculations fail when data deltaE and model deltaE
        are both 0
        """
        with self.assertWarns(UserWarning):
            mdict = de_metrics(
                meas_old=self.meas_old,
                model_old=self.model_old,
                meas_new=self.meas_old,
                model_new=self.model_old,
            )

        self.assertTrue(mdict['CC'] is None)
        self.assertTrue(mdict['dE_rat'] is None)
        self.assertTrue(mdict['NI_rat'] is not None)
        self.assertTrue(mdict['exp_NI_rat'] is not None)

        pass




    def test_invalid_meas_old(self):
        """Invalid inputs caught"""

        perrlist = [0, 1, (1,), np.ones((5,)), 'txt', None, # wrong type
                    np.ones((self.nrow+1, self.ncol+1)), # wrong size
        ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                de_metrics(
                    meas_old=perr,
                    model_old=self.model_old,
                    meas_new=self.meas_new,
                    model_new=self.model_new,
                )
                pass
            pass
        pass


    def test_invalid_model_old(self):
        """Invalid inputs caught"""

        perrlist = [0, 1, (1,), np.ones((5,)), 'txt', None, # wrong type
                    np.ones((self.nrow+1, self.ncol+1)), # wrong size
        ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                de_metrics(
                    meas_old=self.meas_old,
                    model_old=perr,
                    meas_new=self.meas_new,
                    model_new=self.model_new,
                )
                pass
            pass
        pass


    def test_invalid_meas_new(self):
        """Invalid inputs caught"""

        perrlist = [0, 1, (1,), np.ones((5,)), 'txt', None, # wrong type
                    np.ones((self.nrow+1, self.ncol+1)), # wrong size
        ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                de_metrics(
                    meas_old=self.meas_old,
                    model_old=self.model_old,
                    meas_new=perr,
                    model_new=self.model_new,
                )
                pass
            pass
        pass


    def test_invalid_model_new(self):
        """Invalid inputs caught"""

        perrlist = [0, 1, (1,), np.ones((5,)), 'txt', None, # wrong type
                    np.ones((self.nrow+1, self.ncol+1)), # wrong size
        ]

        for perr in perrlist:
            with self.assertRaises(TypeError):
                de_metrics(
                    meas_old=self.meas_old,
                    model_old=self.model_old,
                    meas_new=self.meas_new,
                    model_new=perr,
                )
                pass
            pass
        pass






if __name__ == '__main__':
    unittest.main()
