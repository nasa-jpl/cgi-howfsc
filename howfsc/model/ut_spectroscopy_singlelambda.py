# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Unit tests for singlelambda.py
"""

import unittest
import os

import numpy as np

from howfsc.util.insertinto import insertinto
from howfsc.util.dmhtoph import dmhtoph
from howfsc.model.mode import CoronagraphMode
from howfsc.model.mask import Epup, LyotStop, FieldStop
from .singlelambda import SingleLambda, SingleLambdaException


cfgpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'model', 'testdata', 'spectroscopy',
                       'spectroscopy.yaml')
cfg = CoronagraphMode(cfgpath)


class TestSingleLambda(unittest.TestCase):
    """Unit test suite for base class (SingleLambda) constructor."""

    def setUp(self):
        self.sl = cfg.sl_list[0]
        pass

    def test_lam_realpositivescalar(self):
        """Check wavelength valid."""
        epup = self.sl.epup
        pupil = self.sl.pupil
        fpm = self.sl.fpm
        lyot = self.sl.lyot
        fs = self.sl.fs
        dh = self.sl.dh
        dmlist = [dm for dm in self.sl.dmlist]
        initmaps = self.sl.initmaps
        ft_dir = self.sl.ft_dir

        for perr in [-1.5, -1, 0, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                SingleLambda(lam=perr,
                             epup=epup,
                             dmlist=dmlist,
                             pupil=pupil,
                             fpm=fpm,
                             lyot=lyot,
                             fs=fs,
                             dh=dh,
                             initmaps=initmaps,
                             ft_dir=ft_dir,
                )
                pass
            pass

    def test_pupil_and_focal_classes(self):
        """Run through all ModelElement-child inputs for validity."""
        lam = self.sl.lam
        epup = self.sl.epup
        pupil = self.sl.pupil
        fpm = self.sl.fpm
        lyot = self.sl.lyot
        fs = self.sl.fs
        dh = self.sl.dh
        dmlist = [dm for dm in self.sl.dmlist]
        initmaps = self.sl.initmaps
        ft_dir = self.sl.ft_dir

        # epup
        for perr in [{'test':'test'}, [], 'text', 5.]:
            with self.assertRaises(TypeError):
                SingleLambda(lam=lam,
                             epup=perr,
                             dmlist=dmlist,
                             pupil=pupil,
                             fpm=fpm,
                             lyot=lyot,
                             fs=fs,
                             dh=dh,
                             initmaps=initmaps,
                             ft_dir=ft_dir,
                )
                pass
            pass

        # pupil
        for perr in [{'test':'test'}, [], 'text', 5.]:
            with self.assertRaises(TypeError):
                SingleLambda(lam=lam,
                             epup=epup,
                             dmlist=dmlist,
                             pupil=perr,
                             fpm=fpm,
                             lyot=lyot,
                             fs=fs,
                             dh=dh,
                             initmaps=initmaps,
                             ft_dir=ft_dir,
                )
                pass
            pass

        # fpm
        for perr in [{'test':'test'}, [], 'text', 5.]:
            with self.assertRaises(TypeError):
                SingleLambda(lam=lam,
                             epup=epup,
                             dmlist=dmlist,
                             pupil=pupil,
                             fpm=perr,
                             lyot=lyot,
                             fs=fs,
                             dh=dh,
                             initmaps=initmaps,
                             ft_dir=ft_dir,
                )
                pass
            pass

        # lyot
        for perr in [{'test':'test'}, [], 'text', 5.]:
            with self.assertRaises(TypeError):
                SingleLambda(lam=lam,
                             epup=epup,
                             dmlist=dmlist,
                             pupil=pupil,
                             fpm=fpm,
                             lyot=perr,
                             fs=fs,
                             dh=dh,
                             initmaps=initmaps,
                             ft_dir=ft_dir,
                )
                pass
            pass

        # fs
        for perr in [{'test':'test'}, [], 'text', 5.]:
            with self.assertRaises(TypeError):
                SingleLambda(lam=lam,
                             epup=epup,
                             dmlist=dmlist,
                             pupil=pupil,
                             fpm=fpm,
                             lyot=lyot,
                             fs=perr,
                             dh=dh,
                             initmaps=initmaps,
                             ft_dir=ft_dir,
                )
                pass
            pass

        # dh
        for perr in [{'test':'test'}, [], 'text', 5.]:
            with self.assertRaises(TypeError):
                SingleLambda(lam=lam,
                             epup=epup,
                             dmlist=dmlist,
                             pupil=pupil,
                             fpm=fpm,
                             lyot=lyot,
                             fs=fs,
                             dh=perr,
                             initmaps=initmaps,
                             ft_dir=ft_dir,
                )
                pass
            pass

        # initmaps
        for perr in [{'test':'test'}, [], 'text', 5., [np.zeros((48, 48))]]:
            with self.assertRaises(TypeError):
                SingleLambda(lam=lam,
                             epup=epup,
                             dmlist=dmlist,
                             pupil=pupil,
                             fpm=fpm,
                             lyot=lyot,
                             fs=fs,
                             dh=dh,
                             initmaps=perr,
                             ft_dir=ft_dir,
                )
                pass
            pass

        # ft_dir
        for perr in [{'test':'test'}, [], 'text', 5., [np.zeros((48, 48))]]:
            with self.assertRaises(SingleLambdaException):
                SingleLambda(lam=lam,
                             epup=epup,
                             dmlist=dmlist,
                             pupil=pupil,
                             fpm=fpm,
                             lyot=lyot,
                             fs=fs,
                             dh=dh,
                             initmaps=initmaps,
                             ft_dir=perr,
                )
                pass
            pass
        pass

    def test_dmlist_list(self):
        """Check dmlist is a list of DMFace objects."""
        lam = self.sl.lam
        epup = self.sl.epup
        pupil = self.sl.pupil
        fpm = self.sl.fpm
        lyot = self.sl.lyot
        fs = self.sl.fs
        dh = self.sl.dh
        initmaps = self.sl.initmaps
        ft_dir = self.sl.ft_dir

        sampleDM = self.sl.dmlist[0]

        dmlist_list = [[sampleDM, np.ones((48, 48))],
                       sampleDM,
                       [np.ones((48, 48)), np.ones((48, 48))],
                       [sampleDM, epup]]

        for perr in dmlist_list:
            with self.assertRaises(TypeError):
                SingleLambda(lam=lam,
                             epup=epup,
                             dmlist=perr,
                             pupil=pupil,
                             fpm=fpm,
                             lyot=lyot,
                             fs=fs,
                             dh=dh,
                             initmaps=initmaps,
                             ft_dir=ft_dir,
                )
                pass
            pass


class TestGetJacPrecomp(unittest.TestCase):
    """
    Unit test suite for get_jac_precomp().

    Make a new SingleLambda object for each one so we don't fool ourselves;
    this function changes the storage arrays internal to the object
    """

    def setUp(self):
        self.sl = cfg.sl_list[0]
        pass

    def test_success_and_overwrite(self):
        """Verify that the function can write and overwrite storage arrays."""
        lam = self.sl.lam
        epup = self.sl.epup
        pupil = self.sl.pupil
        fpm = self.sl.fpm
        lyot = self.sl.lyot
        fs = self.sl.fs
        dh = self.sl.dh
        initmaps = self.sl.initmaps
        dmlist = [dm for dm in self.sl.dmlist]
        ft_dir = self.sl.ft_dir

        nactlist = [dm.registration['nact'] for dm in self.sl.dmlist]
        eyelist = [np.eye(nact) for nact in nactlist]

        SL_0 = SingleLambda(lam=lam,
                            epup=epup,
                            dmlist=dmlist,
                            pupil=pupil,
                            fpm=fpm,
                            lyot=lyot,
                            fs=fs,
                            dh=dh,
                            initmaps=initmaps,
                            ft_dir=ft_dir,
        )
        SL_0.get_jac_precomp(eyelist)

        zerolist = [np.zeros((nact, nact)) for nact in nactlist]
        SL_1 = SingleLambda(lam=lam,
                            epup=epup,
                            dmlist=dmlist,
                            pupil=pupil,
                            fpm=fpm,
                            lyot=lyot,
                            fs=fs,
                            dh=dh,
                            initmaps=initmaps,
                            ft_dir=ft_dir,
        )
        SL_1.get_jac_precomp(zerolist)

        # functions of dmset_list
        # dmph_list, e_list
        #
        # not functions of dmset_list
        # zlist, fp_list, e_start
        for index in range(len(SL_0.dmph_list)):
            self.assertFalse((SL_0.dmph_list[index]
                              == SL_1.dmph_list[index]).all())
            pass
        for index in range(len(SL_0.e_list)):
            self.assertFalse((SL_0.e_list[index]
                              == SL_1.e_list[index]).all())
            pass

        for index in range(len(SL_0.zlist)):
            self.assertTrue(SL_0.zlist[index] == SL_1.zlist[index])
            pass
        for index in range(len(SL_0.fp_list)):
            self.assertTrue((SL_0.fp_list[index] == SL_1.fp_list[index]).all())
            pass
        self.assertTrue((SL_0.e_start == SL_1.e_start).all())

        # Overwrite SL_1 with same precomp and check they all match now
        SL_1.get_jac_precomp(eyelist)
        for index in range(len(SL_0.dmph_list)):
            self.assertTrue((SL_0.dmph_list[index]
                             == SL_1.dmph_list[index]).all())
            pass
        for index in range(len(SL_0.e_list)):
            self.assertTrue((SL_0.e_list[index]
                             == SL_1.e_list[index]).all())
            pass

        for index in range(len(SL_0.zlist)):
            self.assertTrue(SL_0.zlist[index] == SL_1.zlist[index])
            pass
        for index in range(len(SL_0.fp_list)):
            self.assertTrue((SL_0.fp_list[index] == SL_1.fp_list[index]).all())
            pass
        self.assertTrue((SL_0.e_start == SL_1.e_start).all())
        pass

    def test_dmset_list_has_same_sizes_as_DMs_in_dmlist(self):
        """Check input DM sizes match SingleLambda DMs."""
        nactlist = [dm.registration['nact'] for dm in self.sl.dmlist]

        # list of lists, each sublist an invalid input
        dmset_list_list = []
        dmset_list_list.append([np.zeros((nact+1, nact+1))
                                for nact in nactlist])
        dmset_list_list.append([np.zeros((nact, nact))
                                for nact in nactlist[:-1]])
        dmset_list_list.append([np.zeros((nact, nact)) for nact in nactlist] +
                          [np.zeros((nactlist[-1], nactlist[-1]))])
        dmset_list_list.append([None for nact in nactlist])
        dmset_list_list.append([np.zeros((nact,)) for nact in nactlist])
        dmset_list_list.append([np.zeros((nact, nact, nact))
                                for nact in nactlist])

        lam = self.sl.lam
        epup = self.sl.epup
        pupil = self.sl.pupil
        fpm = self.sl.fpm
        lyot = self.sl.lyot
        fs = self.sl.fs
        dh = self.sl.dh
        initmaps = self.sl.initmaps
        ft_dir = self.sl.ft_dir
        dmlist = [dm for dm in self.sl.dmlist]

        for dmset_list in dmset_list_list:
            with self.assertRaises(TypeError):
                SL_0 = SingleLambda(lam=lam,
                                    epup=epup,
                                    dmlist=dmlist,
                                    pupil=pupil,
                                    fpm=fpm,
                                    lyot=lyot,
                                    fs=fs,
                                    dh=dh,
                                    initmaps=initmaps,
                                    ft_dir=ft_dir,
                )
                SL_0.get_jac_precomp(dmset_list=dmset_list)
                pass
            pass
        pass

    def test_dmset_list_is_listlike(self):
        """Check the list of DMs is actually list-like."""
        lam = self.sl.lam
        epup = self.sl.epup
        pupil = self.sl.pupil
        fpm = self.sl.fpm
        lyot = self.sl.lyot
        fs = self.sl.fs
        dh = self.sl.dh
        initmaps = self.sl.initmaps
        ft_dir = self.sl.ft_dir
        dmlist = [dm for dm in self.sl.dmlist]

        for dmset_list in [5., 'txt', 1j]:
            with self.assertRaises(TypeError):
                SL_0 = SingleLambda(lam=lam,
                                    epup=epup,
                                    dmlist=dmlist,
                                    pupil=pupil,
                                    fpm=fpm,
                                    lyot=lyot,
                                    fs=fs,
                                    dh=dh,
                                    initmaps=initmaps,
                                    ft_dir=ft_dir,
                )
                SL_0.get_jac_precomp(dmset_list=dmset_list)
                pass
            pass
        pass


class TestGetFastJacPrecomp(unittest.TestCase):
    """
    Unit test suite for get_fast_jac_precomp().

    Make a new SingleLambda object for each one so we don't fool ourselves;
    this function changes the storage arrays internal to the object.
    """

    def setUp(self):
        self.sl = cfg.sl_list[0]

        nactlist = [dm.registration['nact'] for dm in self.sl.dmlist]
        dmset_list = [np.zeros((nact, nact)) for nact in nactlist]
        self.sl.get_jac_precomp(dmset_list)

    def test_no_pad_for_fresnel(self):
        """Verify that nxfresnel_fast_jac stays same for short propagations."""
        self.sl.get_fast_jac_precomp()
        nxfA = self.sl.nxfresnel_fast_jac

        zOrig = self.sl.zlist[1]
        self.sl.zlist[1] = 0.1

        self.sl.get_fast_jac_precomp()
        nxfB = self.sl.nxfresnel_fast_jac

        self.assertTrue(nxfA == nxfB)

        self.sl.zlist[1] = zOrig  # reset

    def test_pad_for_fresnel(self):
        """Verify that nxfresnel_fast_jac increases for long propagations."""
        self.sl.get_fast_jac_precomp()
        nxfA = self.sl.nxfresnel_fast_jac

        zOrig = self.sl.zlist[1]
        self.sl.zlist[1] = 3.0

        self.sl.get_fast_jac_precomp()
        nxfB = self.sl.nxfresnel_fast_jac

        delta_z = np.abs(self.sl.zlist[1] - self.sl.zlist[0])
        nxfMin = self.sl.lam * delta_z * self.sl.dmlist[0].pixpermeter**2

        self.assertTrue(nxfB > nxfA)
        self.assertTrue(nxfB >= nxfMin)

        self.sl.zlist[1] = zOrig  # reset

    def test_overwrite(self):
        """Verify that attributes can be overwritten."""
        self.sl.get_fast_jac_precomp()
        widthA = self.sl.dm1.infMaster.shape[0]
        nSurfA = self.sl.dm1.nSurf
        yLowerLeftA = self.sl.dm1.yLowerLeft
        xLowerLeftA = self.sl.dm1.xLowerLeft
        yOffsetsA = self.sl.dm1.yOffsets
        xOffsetsA = self.sl.dm1.xOffsets
        zlistA = self.sl.zlist

        thactOrig = self.sl.dmlist[0].registration["thact"]
        self.sl.dmlist[0].registration["thact"] = 40.123

        self.sl.get_fast_jac_precomp()
        widthB = self.sl.dm1.infMaster.shape[0]
        yLowerLeftB = self.sl.dm1.yLowerLeft
        xLowerLeftB = self.sl.dm1.xLowerLeft
        yOffsetsB = self.sl.dm1.yOffsets
        xOffsetsB = self.sl.dm1.xOffsets
        nSurfB = self.sl.dm1.nSurf
        zlistB = self.sl.zlist

        # Values should have changed:
        self.assertFalse(widthB == widthA)
        self.assertFalse(nSurfB == nSurfA)
        self.assertFalse(np.all(yLowerLeftA == yLowerLeftB))
        self.assertFalse(np.all(xLowerLeftA == xLowerLeftB))
        self.assertFalse(np.all(yOffsetsA == yOffsetsB))
        self.assertFalse(np.all(xOffsetsA == xOffsetsB))

        # Values should stay the same:
        for index, zA in enumerate(zlistA):
            self.assertTrue(zA == zlistB[index])

        # Reset value for other tests
        self.sl.dmlist[0].registration["thact"] = thactOrig


class TestPropToLyot(unittest.TestCase):
    """Unit test suite for proptolyot()."""

    def setUp(self):
        self.sl = cfg.sl_list[0]
        pass

    def test_output_size(self):
        """Verify output size as per docs."""
        e0 = self.sl.epup.e.copy()

        out = self.sl.proptolyot(e0=e0)
        self.assertTrue(out.shape == self.sl.lyot.e.shape)
        pass

    def test_output_size_nofpm(self):
        """Verify output size as per docs."""
        e0 = self.sl.epup.e.copy()
        out = self.sl.proptolyot_nofpm(e0=e0)
        self.assertTrue(out.shape == self.sl.lyot.e.shape)
        pass

    def test_e0_2darray(self):
        """Verify proptolyot handles bad input."""
        for perr in [np.ones((20,)), np.ones((10, 10, 10)), [], 'txt']:
            with self.assertRaises(TypeError):
                self.sl.proptolyot(e0=perr)
                pass
            pass
        pass

    def test_fft_nofpm(self):
        """Check FFT with no occulter behaves as expected."""
        tol = 1e-13
        e0 = self.sl.epup.e.copy()

        e_fft = self.sl.proptolyot_nofpm(e0)

        # Assumes config has same pupil and Lyot sampling.  True for widefov
        e_mask = self.sl.lyot.applymask(self.sl.pupil.applymask(e0))

        # don't normalize, since this is basically inorm calc
        self.assertTrue(np.max(np.abs(e_fft-e_mask)) < tol)
        pass

    def test_ft_analytic(self):
        """
        TODO: write analytic case; APLC may be the only realistic option
        """
        pass


class TestCroppedPropToLyot(unittest.TestCase):
    """Unit test suite for croppedproptolyot()."""

    def setUp(self):
        self.sl = cfg.sl_list[0]
        self.yxLowerLeft = (4, 11)
        self.nSubarray = 100
        self.e0full = self.sl.epup.e.copy()
        self.nSurf = self.e0full.shape[0]
        self.e0crop = self.e0full[
            self.yxLowerLeft[0]:self.yxLowerLeft[0] + self.nSubarray,
            self.yxLowerLeft[1]:self.yxLowerLeft[1] + self.nSubarray]

    def test_output_size(self):
        """Verify output size as per docs."""
        out = self.sl.croppedproptolyot(e0=self.e0crop,
                                        yxLowerLeft=self.yxLowerLeft,
                                        nSurf=self.nSurf)
        self.assertTrue(out.shape == self.sl.lyot.e.shape)

    def test_propagation_in_croppedproptolyot(self):
        """
        Verify croppedproptolyot() gives same answer as proptolyot()
        when given equivalent inputs.
        """
        e0full = np.zeros((self.nSurf, self.nSurf), dtype=complex)
        e0full[self.yxLowerLeft[0]:self.yxLowerLeft[0] + self.nSubarray,
               self.yxLowerLeft[1]:self.yxLowerLeft[1] + self.nSubarray] = \
            self.e0crop

        outA = self.sl.croppedproptolyot(e0=self.e0crop,
                                        yxLowerLeft=self.yxLowerLeft,
                                        nSurf=self.nSurf)
        outB = self.sl.proptolyot(e0=e0full)

        abs_tol = 10*np.finfo(float).eps
        self.assertTrue(np.max(np.abs(outA - outB)) < abs_tol)

    def test_bad_inputs_to_croppedproptolyot(self):
        """Verify croppedproptolyot handles bad inputs."""
        for badVal in [np.ones((20,)), np.ones((10, 10, 10)), [], 'txt']:
            with self.assertRaises(TypeError):
                self.sl.croppedproptolyot(e0=badVal,
                                          yxLowerLeft=self.yxLowerLeft,
                                          nSurf=self.nSurf)

        for badVal in [np.ones((20,)), np.ones((10, 10, 10)), [], 'txt',
                       (-1, 10), (0, self.nSurf+1), 0, 1j, -2.5]:
            with self.assertRaises(TypeError):
                self.sl.croppedproptolyot(e0=self.e0crop,
                                          yxLowerLeft=badVal,
                                          nSurf=self.nSurf)

        for badVal in [-10, 0, 1.5, 1j, (10,), np.ones((10, 9)), [], 'txt']:
            with self.assertRaises(TypeError):
                self.sl.croppedproptolyot(e0=self.e0crop,
                                          yxLowerLeft=self.yxLowerLeft,
                                          nSurf=badVal)


class TestPropToDH(unittest.TestCase):
    """Unit test suite for proptodh()."""

    def setUp(self):
        self.sl = cfg.sl_list[0]
        pass

    def test_e0_2darray(self):
        """Verify MFT handles bad input."""
        for perr in [np.ones((20,)), np.ones((10, 10, 10)), [], 'txt']:
            with self.assertRaises(TypeError):
                self.sl.proptodh(e0=perr)
                pass
            pass
        pass

    def test_output_size(self):
        """Verify output size as per docs."""
        e0 = self.sl.epup.e.copy()
        out = self.sl.proptodh(e0=e0)
        self.assertTrue(out.shape == self.sl.fs.e.shape)
        pass

    def test_ft_analytic(self):
        """
        Compare numerical FT versus an analytic FT result.

        Note to future test writers: don't be clever and try to make an
        analytic match with a square aperture to a pair of sincs.  The Gibbs
        phenomenon will bite you good and hard.  Use a shape without sharp
        discontinuities.
        """
        # Gaussians go to Gaussians, see table 2.1 in Goodman for convention
        tol = 1e-9

        # widefov has no field stop, so we don't have to try to
        # resubstitute into an self.sl copy
        d = self.sl.lyot.pixperpupil
        outshape = self.sl.fs.e.shape
        pixperlod = self.sl.fs.pixperlod

        x = np.arange(d).astype('float64') - d//2
        XX, YY = np.meshgrid(x, x)
        RR = np.hypot(XX, YY)
        nRR = RR/float(d)  # norm pupil diam of 1
        pRR = RR/float(pixperlod)  # can reuse for focal plane

        gauss_exp = 5
        e = np.exp(-np.pi*(nRR*gauss_exp)**2)

        e_mft = self.sl.proptodh(e)
        e_analytic = np.exp(-np.pi*(pRR/gauss_exp)**2)/(gauss_exp)**2
        ea_sl = insertinto(e_analytic, outshape).astype('complex128')
        ea_sl /= np.sqrt(self.sl.inorm)
        ea_sl *= self.sl.fs.e  # easiest to re-add field stop here

        self.assertTrue(np.max(np.abs(e_mft - ea_sl)) < tol)
        pass

    def test_tip_tilt_in_pixels_lyot(self):
        """Verify that tip/tilt is commanded in pixels."""
        tol = 1e-13

        for ds in ['forward', 'reverse']:
            self.sl.ft_dir = ds
            self.sl.ft_dir_sign = 1 if ds == 'forward' else -1

            # Set an empty field stop, because it won't shift with the star and
            # the roll comparison will get all hashed up.
            fs0 = self.sl.fs
            fs1 = FieldStop(fs0.lam, np.ones_like(fs0.e), fs0.pixperlod)
            self.sl.fs = fs1

            self.sl.ttph_down = np.exp(
                1j*2*np.pi/(self.sl.lyot.pixperpupil*self.sl.fs.pixperlod) *
                self.sl.lyot.ttgrid*self.sl.ft_dir_sign)
            edm = self.sl.eprop(self.sl.initmaps)
            ely = self.sl.proptolyot(edm)
            edh0 = self.sl.proptodh(ely)

            L0 = self.sl.lyot
            tip0 = self.sl.lyot.tip
            tilt0 = self.sl.lyot.tilt

            dtip = 13
            dtilt = 5

            L1 = LyotStop(L0.lam, L0.e, L0.pixperpupil,
                          tip=tip0+dtip, tilt=tilt0+dtilt)
            self.sl.lyot = L1
            self.sl.ttph_down = np.exp(
                1j*2*np.pi/(self.sl.lyot.pixperpupil*self.sl.fs.pixperlod) *
                self.sl.lyot.ttgrid*self.sl.ft_dir_sign)

            edm = self.sl.eprop(self.sl.initmaps)
            ely = self.sl.proptolyot(edm)
            edh1 = self.sl.proptodh(ely)

            # reset
            self.sl.lyot = L0
            self.sl.fs = fs0

            edh2 = np.roll(np.roll(edh0, dtip, axis=1), dtilt, axis=0)
            esize = (self.sl.fs.e.shape[0] - 2*dtilt,
                     self.sl.fs.e.shape[0] - 2*dtip)
            ediff = insertinto(np.abs(edh2-edh1), esize)
            self.assertTrue(np.max(ediff) < tol)
            pass

        pass


class TestPropToDHPeak(unittest.TestCase):
    """Unit test suite for proptodhpeak()."""

    def setUp(self):
        self.sl = cfg.sl_list[0]
        pass

    def test_e0_2darray(self):
        """Verify MFT handles bad input."""
        for perr in [np.ones((20,)), np.ones((10, 10, 10)), [], 'txt']:
            with self.assertRaises(TypeError):
                self.sl.proptodhpeak(e0=perr)

    def test_output_size(self):
        """Verify output is a scalar."""
        e0 = self.sl.epup.e.copy()
        e0[0, 0] = 0  # Can't have all ones pupil for ampthresh
        out = self.sl.proptodhpeak(e0=e0)
        self.assertTrue(np.isscalar(out))

    def test_same_peak_as_full_array(self):
        """Verify that the peak matches the value from full propagation."""
        # Copy as we don't want to modify original
        e0 = self.sl.epup.e.copy()
        e0 = self.sl.proptodm(e0, self.sl.initmaps)
        e0 = self.sl.proptolyot_nofpm(e0)

        e_2d = self.sl.proptodh(e0, normalize=False)
        i_2d_peak = np.abs(np.max(e_2d))**2

        i_direct_peak = np.abs(self.sl.proptodhpeak(e0))**2

        tol=1e-12
        self.assertTrue(np.abs(i_2d_peak-i_direct_peak) < tol)


class TestEProp(unittest.TestCase):
    """Unit test suite for eprop() (similar to pokeprop)."""
    def setUp(self):
        self.sl = cfg.sl_list[0]
        pass

    def test_output_size(self):
        """Verify output size as per docs."""
        dmlist = [np.zeros((dm.registration['nact'],
                            dm.registration['nact'])) for dm in self.sl.dmlist]
        out = self.sl.eprop(dmlist)
        self.assertTrue(out.shape == self.sl.epup.e.shape)
        pass

    def test_eprop_function(self):
        """
        TODO: Write a functional test for this case. This is nontrivial, and
        may not be possible.
        """
        pass

    def test_dmset_list_has_same_sizes_as_DMs_in_dmlist(self):
        """Check input DM sizes match SingleLambda DMs."""
        nactlist = [dm.registration['nact'] for dm in self.sl.dmlist]

        # list of lists, each sublist an invalid input
        dmset_list_list = []
        dmset_list_list.append([np.zeros((nact+1, nact+1))
                                for nact in nactlist])
        dmset_list_list.append([np.zeros((nact, nact))
                                for nact in nactlist[:-1]])
        dmset_list_list.append([np.zeros((nact, nact)) for nact in nactlist] +
                          [np.zeros((nactlist[-1], nactlist[-1]))])
        dmset_list_list.append([None for nact in nactlist])
        dmset_list_list.append([np.zeros((nact,)) for nact in nactlist])
        dmset_list_list.append([np.zeros((nact, nact, nact))
                                for nact in nactlist])

        for dmset_list in dmset_list_list:
            with self.assertRaises(TypeError):
                self.sl.eprop(dmset_list=dmset_list)
                pass
            pass
        pass

    def test_dmset_list_is_listlike(self):
        """Check the list of DMs is actually list-like."""
        for dmset_list in [5., 'txt', 1j]:
            with self.assertRaises(TypeError):
                self.sl.eprop(dmset_list=dmset_list)
                pass
            pass
        pass

    def test_tip_tilt_in_pixels_epup(self):
        """Verify that tip/tilt is commanded in pixels."""
        tol = 1e-13

        for ds in ['forward', 'reverse']:
            self.sl.ft_dir = ds
            self.sl.ft_dir_sign = 1 if ds == 'forward' else -1

            # Set an empty field stop, because it won't shift with the star and
            # the roll comparison will get all hashed up.
            fs0 = self.sl.fs
            fs1 = FieldStop(fs0.lam, np.ones_like(fs0.e), fs0.pixperlod)
            self.sl.fs = fs1

            self.sl.ttph_up = np.exp(
                1j*2*np.pi/(self.sl.epup.pixperpupil*self.sl.fs.pixperlod) *
                self.sl.epup.ttgrid*self.sl.ft_dir_sign)

            # Use nofpm also, for the same reason.
            edm = self.sl.eprop(self.sl.initmaps)
            ely = self.sl.proptolyot_nofpm(edm)
            edh0 = self.sl.proptodh(ely)

            L0 = self.sl.epup
            tip0 = self.sl.epup.tip
            tilt0 = self.sl.epup.tilt

            dtip = 13
            dtilt = 5

            L1 = Epup(L0.lam, L0.e, L0.pixperpupil,
                      tip=tip0+dtip, tilt=tilt0+dtilt)
            self.sl.epup = L1
            self.sl.ttph_up = np.exp(
                1j*2*np.pi/(self.sl.epup.pixperpupil*self.sl.fs.pixperlod) *
                self.sl.epup.ttgrid*self.sl.ft_dir_sign)

            edm = self.sl.eprop(self.sl.initmaps)
            ely = self.sl.proptolyot_nofpm(edm)
            edh1 = self.sl.proptodh(ely)

            # reset
            self.sl.epup = L0
            self.sl.fs = fs0

            edh2 = np.roll(np.roll(edh0, dtip, axis=1), dtilt, axis=0)
            esize = (self.sl.fs.e.shape[0] - 2*dtilt,
                     self.sl.fs.e.shape[0] - 2*dtip)
            ediff = insertinto(np.abs(edh2-edh1), esize)
            self.assertTrue(np.max(ediff) < tol)
            pass

        pass



class TestPokeProp(unittest.TestCase):
    """Unit test suite for pokeprop()."""

    def setUp(self):
        self.sl = cfg.sl_list[0]

        nactlist = [dm.registration['nact'] for dm in self.sl.dmlist]
        dmset_list = [np.zeros((nact, nact)) for nact in nactlist]
        self.sl.get_jac_precomp(dmset_list)
        pass

    def test_output_size(self):
        """Verify output size as per docs."""
        dmlist = [np.zeros((dm.registration['nact'],
                            dm.registration['nact'])) for dm in self.sl.dmlist]
        out = self.sl.pokeprop(0, dmlist)
        self.assertTrue(out.shape == self.sl.epup.e.shape)
        pass

    def test_pokeprop_function(self):
        """
        TODO: Write a functional test for this case.  This is nontrivial, and
        may not be possible
        """
        pass

    def test_catch_missing_sl_data_element(self):
        """Check fails as expected if precomputation was not done first."""
        lam = self.sl.lam
        epup = self.sl.epup
        pupil = self.sl.pupil
        fpm = self.sl.fpm
        lyot = self.sl.lyot
        fs = self.sl.fs
        dh = self.sl.dh
        initmaps = self.sl.initmaps
        dmlist = [dm for dm in self.sl.dmlist]
        ft_dir = self.sl.ft_dir

        SL_0 = SingleLambda(lam=lam,
                            epup=epup,
                            dmlist=dmlist,
                            pupil=pupil,
                            fpm=fpm,
                            lyot=lyot,
                            fs=fs,
                            dh=dh,
                            initmaps=initmaps,
                            ft_dir=ft_dir,
        )
        # precomp would go here normally

        dmind = 0
        nactlist = [dm.registration['nact'] for dm in self.sl.dmlist]
        dmset_list = [np.zeros((nact, nact)) for nact in nactlist]
        with self.assertRaises(SingleLambdaException):
            SL_0.pokeprop(dmind=dmind, dmset_list=dmset_list)
            pass
        pass

    def test_dmind_nonnegative_scalar_integer(self):
        """Check index-of-poked-actuator type valid."""
        nactlist = [dm.registration['nact'] for dm in self.sl.dmlist]
        dmset_list = [np.zeros((nact, nact)) for nact in nactlist]

        for perr in [-1.5, -1, 3.5, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                self.sl.pokeprop(dmind=perr, dmset_list=dmset_list)
                pass
            pass
        pass

    def test_dmind_too_large(self):
        """Check poked actuator is within range of existing DMs."""
        nactlist = [dm.registration['nact'] for dm in self.sl.dmlist]
        dmset_list = [np.zeros((nact, nact)) for nact in nactlist]

        allact = self.sl.ndmact[-1]
        for perr in [allact, allact+1]:
            with self.assertRaises(ValueError):
                self.sl.pokeprop(dmind=perr, dmset_list=dmset_list)
                pass
            pass
        pass


    def test_dmset_list_has_same_sizes_as_DMs_in_dmlist(self):
        """Check input DM sizes match SingleLambda DMs."""
        dmind = 0
        nactlist = [dm.registration['nact'] for dm in self.sl.dmlist]

        # list of lists, each sublist an invalid input
        dmset_list_list = []
        dmset_list_list.append([np.zeros((nact+1, nact+1))
                                for nact in nactlist])
        dmset_list_list.append([np.zeros((nact, nact))
                                for nact in nactlist[:-1]])
        dmset_list_list.append([np.zeros((nact, nact)) for nact in nactlist] +
                          [np.zeros((nactlist[-1], nactlist[-1]))])
        dmset_list_list.append([None for nact in nactlist])
        dmset_list_list.append([np.zeros((nact,)) for nact in nactlist])
        dmset_list_list.append([np.zeros((nact, nact, nact))
                                for nact in nactlist])

        for dmset_list in dmset_list_list:
            with self.assertRaises(TypeError):
                self.sl.pokeprop(dmind=dmind, dmset_list=dmset_list)
                pass
            pass
        pass

    def test_dmset_list_is_listlike(self):
        """Check the list of DMs is actually list-like."""
        dmind = 0
        for dmset_list in [5., 'txt', 1j]:
            with self.assertRaises(TypeError):
                self.sl.pokeprop(dmind=dmind, dmset_list=dmset_list)
                pass
            pass
        pass


class TestCroppedPokeProp(unittest.TestCase):
    """Unit test suite for croppedpokeprop()."""

    def setUp(self):
        self.sl = cfg.sl_list[0]

        nactlist = [dm.registration['nact'] for dm in self.sl.dmlist]
        dmset_list = [np.zeros((nact, nact)) for nact in nactlist]
        self.sl.get_jac_precomp(dmset_list)
        self.sl.get_fast_jac_precomp()

    def test_output_size(self):
        """Verify output size."""
        dmlist = [np.zeros((dm.registration['nact'],
                            dm.registration['nact'])) for dm in self.sl.dmlist]
        e0full, _, yxLowerLeft, nSurf = self.sl.croppedpokeprop(0, dmlist)
        self.assertTrue(e0full.shape == (nSurf, nSurf))
        self.assertTrue(len(yxLowerLeft) == 2)

    def test_compare_croppedpokeprop_to_pokeprop(self):
        """
        Verify croppedpokeprop() against pokeprop() in standard use cases.

        This also acts as a test of proptodm_fast_jac() against proptodm() in
        standard use cases.

        Test for each DM1 and DM2, and for each uniform existing DM commands
        and non-uniform existing DM commands.
        """
        relTol = 5e-3

        nactlist = [dm.registration['nact'] for dm in self.sl.dmlist]
        dmset_list_list = []  # initialize
        dmset_list_list.append([np.zeros((nact, nact)) for nact in nactlist])
        dmset_list_temp = []  # initialize
        for index, nact in enumerate(nactlist):
            dmset_list_temp.append(
                np.rot90(np.linspace(-20, 20, nact*nact).reshape((nact, nact)),
                         index)
                + 50 * np.ones((nact, nact))
            )
            pass
        dmset_list_list.append(dmset_list_temp)

        for dmset_list in dmset_list_list:

            self.sl.get_jac_precomp(dmset_list)
            self.sl.get_fast_jac_precomp()

            for dmind in [504, 48*48+504]:

                e0full, _, _, nSurf = self.sl.croppedpokeprop(
                    dmind, dmset_list)
                e0 = self.sl.pokeprop(dmind, dmset_list)
                e0 = insertinto(e0, (nSurf, nSurf))

                maxAbsDiff = np.max(np.abs(e0 - e0full))
                maxVal = np.max(np.abs(e0))
                self.assertTrue(maxAbsDiff/maxVal < relTol)

    # Failure tests
    def test_catch_missing_sl_data_element(self):
        """Check fails as expected if precomputation was not done first."""
        lam = self.sl.lam
        epup = self.sl.epup
        pupil = self.sl.pupil
        fpm = self.sl.fpm
        lyot = self.sl.lyot
        fs = self.sl.fs
        dh = self.sl.dh
        initmaps = self.sl.initmaps
        dmlist = [dm for dm in self.sl.dmlist]
        ft_dir = self.sl.ft_dir

        SL_0 = SingleLambda(lam=lam,
                            epup=epup,
                            dmlist=dmlist,
                            pupil=pupil,
                            fpm=fpm,
                            lyot=lyot,
                            fs=fs,
                            dh=dh,
                            initmaps=initmaps,
                            ft_dir=ft_dir,
        )
        # precomp would go here normally

        dmind = 0
        nactlist = [dm.registration['nact'] for dm in self.sl.dmlist]
        dmset_list = [np.zeros((nact, nact)) for nact in nactlist]
        with self.assertRaises(SingleLambdaException):
            SL_0.croppedpokeprop(dmind=dmind, dmset_list=dmset_list)
            pass
        pass

    def test_dmind_nonnegative_scalar_integer(self):
        """Check index-of-poked-actuator type valid."""
        nactlist = [dm.registration['nact'] for dm in self.sl.dmlist]
        dmset_list = [np.zeros((nact, nact)) for nact in nactlist]

        for perr in [-1.5, -1, 3.5, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                self.sl.croppedpokeprop(dmind=perr, dmset_list=dmset_list)
                pass
            pass
        pass

    def test_dmind_too_large(self):
        """Check poked actuator is within range of existing DMs."""
        nactlist = [dm.registration['nact'] for dm in self.sl.dmlist]
        dmset_list = [np.zeros((nact, nact)) for nact in nactlist]

        allact = self.sl.ndmact[-1]
        for perr in [allact, allact+1]:
            with self.assertRaises(ValueError):
                self.sl.croppedpokeprop(dmind=perr, dmset_list=dmset_list)
                pass
            pass
        pass

    def test_dmset_list_has_same_sizes_as_DMs_in_dmlist(self):
        """Check input DM sizes match SingleLambda DMs."""
        dmind = 0
        nactlist = [dm.registration['nact'] for dm in self.sl.dmlist]

        # list of lists, each sublist an invalid input
        dmset_list_list = []
        dmset_list_list.append([np.zeros((nact+1, nact+1))
                                for nact in nactlist])
        dmset_list_list.append([np.zeros((nact, nact))
                                for nact in nactlist[:-1]])
        dmset_list_list.append([np.zeros((nact, nact)) for nact in nactlist] +
                          [np.zeros((nactlist[-1], nactlist[-1]))])
        dmset_list_list.append([None for nact in nactlist])
        dmset_list_list.append([np.zeros((nact,)) for nact in nactlist])
        dmset_list_list.append([np.zeros((nact, nact, nact))
                                for nact in nactlist])

        for dmset_list in dmset_list_list:
            with self.assertRaises(TypeError):
                self.sl.croppedpokeprop(dmind=dmind, dmset_list=dmset_list)
                pass
            pass
        pass

    def test_dmset_list_is_listlike(self):
        """Check the list of DMs is actually list-like."""
        dmind = 0
        for dmset_list in [5., 'txt', 1j]:
            with self.assertRaises(TypeError):
                self.sl.croppedpokeprop(dmind=dmind, dmset_list=dmset_list)
                pass
            pass
        pass


class TestPropToDM(unittest.TestCase):
    """Unit test suite for proptodm()."""

    def setUp(self):
        self.sl = cfg.sl_list[0]
        pass

    def test_input_size_equals_output_size(self):
        """Check the sizes on the input and output are the same."""
        dmset_list = []
        for dm in self.sl.dmlist:
            dmset_list.append(np.zeros((dm.registration['nact'],
                                        dm.registration['nact'])))
            pass
        nxfresnel = self.sl.nxfresnel_dm

        for e0 in [np.ones((128, 128)),
                   np.ones((127, 128)),
                   np.ones((128, 127)),
                   np.ones((127, 127))]:
            e1 = self.sl.proptodm(e0=e0, dmset_list=dmset_list,
                                  nxfresnel=nxfresnel)
            self.assertTrue(e1.shape == e0.shape)
            pass
        pass

    def test_input_size_equals_output_size_with_emult(self):
        """
        Check the sizes on the input and output are the same, even if we add
        an additional multiplying field.
        """
        dmset_list = []
        for dm in self.sl.dmlist:
            dmset_list.append(np.zeros((dm.registration['nact'],
                                        dm.registration['nact'])))
            pass
        nxfresnel = self.sl.nxfresnel_dm

        for e0 in [np.ones((128, 128)),
                   np.ones((127, 128)),
                   np.ones((128, 127)),
                   np.ones((127, 127))]:
            emult_list = [None, e0]
            e1 = self.sl.proptodm(e0=e0, dmset_list=dmset_list,
                                  nxfresnel=nxfresnel, emult_list=emult_list)
            self.assertTrue(e1.shape == e0.shape)
            pass
        pass

    def test_e0_2darray(self):
        """Check electric-field type valid."""
        dmset_list = []
        for dm in self.sl.dmlist:
            dmset_list.append(np.zeros((dm.registration['nact'],
                                        dm.registration['nact'])))
            pass

        for perr in [np.ones((20,)), np.ones((10, 10, 10)), [], 'txt']:
            with self.assertRaises(TypeError):
                self.sl.proptodm(e0=perr, dmset_list=dmset_list)
                pass
            pass
        pass

    def test_dmset_list_2darrays(self):
        """Check DM setting type valid."""
        dmset_list = []
        for dm in self.sl.dmlist:
            dmset_list.append(np.zeros((dm.registration['nact'],
                                        dm.registration['nact'])))
            pass
        e0 = np.ones((128, 128))

        for perr in [np.ones((20,)), np.ones((10, 10, 10)), [], 'txt']:
            dmset_list[0] = perr
            with self.assertRaises(TypeError):
                self.sl.proptodm(e0=e0, dmset_list=dmset_list)
                pass
            pass
        pass

    def test_nxfresnel_positive_scalar_integer(self):
        """Check array size type valid."""
        dmset_list = []
        for dm in self.sl.dmlist:
            dmset_list.append(np.zeros((dm.registration['nact'],
                                        dm.registration['nact'])))
            pass
        e0 = np.ones((128, 128))

        for perr in [np.ones((20,)), np.ones((10, 10, 10)), (10,), [], 'txt']:
            with self.assertRaises(TypeError):
                self.sl.proptodm(e0=e0, dmset_list=dmset_list, nxfresnel=perr)
                pass
            pass
        pass

    def test_dmset_real(self):
        """Check DM setting on input is real."""
        dmset_list = []
        for dm in self.sl.dmlist:
            dmset_list.append(np.zeros((dm.registration['nact'],
                                        dm.registration['nact'])))
            pass
        e0 = np.ones((128, 128))

        for perr in [1j, (1j,), 1j*np.ones((48, 48))]:
            dmset_list[0] = perr
            with self.assertRaises(TypeError):
                self.sl.proptodm(e0=e0, dmset_list=dmset_list)
                pass
            pass
        pass

    def test_emult_2darray(self):
        """Check optional input DM-plane field type valid when present."""
        e0 = np.ones((128, 128))

        dmset_list = []
        emult_list = []
        for dm in self.sl.dmlist:
            dmset_list.append(np.zeros((dm.registration['nact'],
                                        dm.registration['nact'])))
            emult_list.append(None)
            pass

        for perr in [np.ones((20,)), np.ones((10, 10, 10)), [], 'txt']:
            emult_list[0] = perr
            with self.assertRaises(TypeError):
                self.sl.proptodm(e0=e0, dmset_list=dmset_list,
                                 emult_list=emult_list)
                pass
            pass
        pass

    def test_emult_none_handled_as_no_op(self):
        """
        Check optional input DM-plane field handles None input correctly.

        Assumes 2 DMs in config
        """
        e0 = np.ones((128, 128))
        nxf = self.sl.nxfresnel_dm

        dmset_list = []
        for dm in self.sl.dmlist:
            dmset_list.append(np.zeros((dm.registration['nact'],
                                        dm.registration['nact'])))
            pass

        e1 = self.sl.proptodm(e0=e0, dmset_list=dmset_list,
                              emult_list=[None, None])
        # ones are identity for multiplication
        e2 = self.sl.proptodm(e0=e0, dmset_list=dmset_list,
                              emult_list=[np.ones((nxf, nxf)), None])
        e3 = self.sl.proptodm(e0=e0, dmset_list=dmset_list,
                              emult_list=[None, np.ones((nxf, nxf))])
        e4 = self.sl.proptodm(e0=e0, dmset_list=dmset_list,
                              emult_list=[np.ones((nxf, nxf)),
                                          np.ones((nxf, nxf))])

        self.assertTrue((e1 == e2).all())
        self.assertTrue((e1 == e3).all())
        self.assertTrue((e1 == e4).all())
        pass

    def test_functional_proptodm(self):
        """
        TODO: write a functional test, if possible (may be too complex).

        One idea: use pistons with inffix-normalized IF
        """
        #tol = 1e-13
        #radshift = 0.1 # radians to piston
        #
        #e0 = np.ones_like(self.sl.epup.e)
        #for dm in self.sl.dmlist:
        #    nact = dm.registration['nact']
        #    dmset = radshift*np.ones((nact, nact))
        #    nxfresnel = self.sl.nxfresnel_dm
        #    e1 = self.sl.proptodm(e0, dm, dmset, nxfresnel)
        #    self.assertTrue(np.max(np.abs(e1 - e0*np.exp(1j*radshift))) < tol)
        #    pass
        #pass


class TestPropToDMJac(unittest.TestCase):
    """Unit test suite for proptodm_jac()."""

    def setUp(self):
        self.sl = cfg.sl_list[0]

        nactlist = [dm.registration['nact'] for dm in self.sl.dmlist]
        self.dmset_list = [np.zeros((nact, nact)) for nact in nactlist]
        self.sl.get_jac_precomp(self.dmset_list)
        pass

    def test_emult_2darray(self):
        """Check optional input DM-plane field type valid when present"""

        emult_list = [None]*len(self.sl.dmlist)

        for perr in [np.ones((20,)), np.ones((10, 10, 10)), [], 'txt']:
            emult_list[0] = perr
            with self.assertRaises(TypeError):
                self.sl.proptodm_jac(emult_list=emult_list)
                pass
            pass
        pass

    def test_emult_none_handled_as_no_op(self):
        """
        Check optional input DM-plane field handles None input correctly

        Assumes 2 DMs in config
        """
        nxf = self.sl.nxfresnel_dm

        e1 = self.sl.proptodm_jac(emult_list=[None, None])
        # ones are identity for multiplication
        e2 = self.sl.proptodm_jac(emult_list=[np.ones((nxf, nxf)), None])
        e3 = self.sl.proptodm_jac(emult_list=[None, np.ones((nxf, nxf))])
        e4 = self.sl.proptodm_jac(emult_list=[np.ones((nxf, nxf)),
                                          np.ones((nxf, nxf))])

        self.assertTrue((e1 == e2).all())
        self.assertTrue((e1 == e3).all())
        self.assertTrue((e1 == e4).all())
        pass

    def test_catch_missing_sl_data_element(self):
        """Check fails as expected if precomputation was not done first."""
        lam = self.sl.lam
        epup = self.sl.epup
        pupil = self.sl.pupil
        fpm = self.sl.fpm
        lyot = self.sl.lyot
        fs = self.sl.fs
        dh = self.sl.dh
        initmaps = self.sl.initmaps
        dmlist = [dm for dm in self.sl.dmlist]
        ft_dir = self.sl.ft_dir

        SL_0 = SingleLambda(lam=lam,
                            epup=epup,
                            dmlist=dmlist,
                            pupil=pupil,
                            fpm=fpm,
                            lyot=lyot,
                            fs=fs,
                            dh=dh,
                            initmaps=initmaps,
                            ft_dir=ft_dir,
        )
        # precomp would go here normally

        with self.assertRaises(SingleLambdaException):
            SL_0.proptodm_jac()
            pass
        pass

    def test_proptodm_matches_proptodmjac(self):
        """
        Verify the two proptodm functions produce the same results even though
        they take different paths, with and without emult.
        """
        nxf = self.sl.nxfresnel_dm

        # no emult
        e1a = self.sl.proptodm(e0=self.sl.e_start, dmset_list=self.dmset_list,
                               nxfresnel=nxf)
        e1b = self.sl.proptodm_jac()
        self.assertTrue((e1a == e1b).all())  # use equality as should be doing
                                             # identical math (same algorithm)

        # with emult
        dmn = 0  # pick an expected visible actuator
        j = 16
        k = 16

        nactlist = [dm.registration['nact'] for dm in self.sl.dmlist]
        nact = nactlist[0]
        pokerad = np.zeros((nact, nact))
        pokerad[j, k] = 1  # rad
        nrow, ncol = self.sl.e_start.shape
        emult = 2.*1j*dmhtoph(nrow, ncol,
                              pokerad, **self.sl.dmlist[dmn].registration)

        e2a = self.sl.proptodm(e0=self.sl.e_start, dmset_list=self.dmset_list,
                               nxfresnel=nxf, emult_list=[emult, None])
        e2b = self.sl.proptodm_jac(emult_list=[emult, None])
        self.assertTrue((e2a == e2b).all())

    def test_functional_proptodm_jac(self):
        """
        TODO: write a functional test, if possible (may be too complex).

        One idea: use pistons with inffix-normalized IF
        """
        # tol = 1e-13
        # radshift = 0.1 # radians to piston

        # e0 = np.ones_like(self.sl.epup.e)
        # for dm in self.sl.dmlist:
        #     nact = dm.registration['nact']
        #     dmset = radshift*np.ones((nact, nact))
        #     nxfresnel = self.sl.nxfresnel_dm
        #     e1 = self.sl.proptodm(e0, dm, dmset, nxfresnel)
        #     self.assertTrue(np.max(np.abs(e1 - e0*np.exp(1j*radshift))) <
        #                     tol)
        #     pass
        # pass


class TestPropToDMFastJac(unittest.TestCase):
    """Unit test suite for proptodm_fast_jac()."""

    def setUp(self):
        self.sl = cfg.sl_list[0]

        nactlist = [dm.registration['nact'] for dm in self.sl.dmlist]
        self.dmset_list = [np.zeros((nact, nact)) for nact in nactlist]
        self.sl.get_jac_precomp(self.dmset_list)
        self.sl.get_fast_jac_precomp()

        self.yxLowerLeft = (4, 11)
        self.nSubarray = 100
        self.e0full = self.sl.epup.e.copy()
        self.nSurf = self.e0full.shape[0]
        self.e0crop = self.e0full[
            self.yxLowerLeft[0]:self.yxLowerLeft[0] + self.nSubarray,
            self.yxLowerLeft[1]:self.yxLowerLeft[1] + self.nSubarray]
        self.sl.fp_crop_list = self.sl.fp_list

    def test_emult_2darray(self):
        """Check optional input DM-plane field type valid when present."""
        emult_list = [None]*len(self.sl.dmlist)

        for perr in [np.ones((20,)), np.ones((10, 10, 10)), [], 'txt']:
            emult_list[0] = perr
            with self.assertRaises(TypeError):
                # self.sl.proptodm_fast_jac(emult_list=emult_list)
                self.sl.proptodm_fast_jac(self.yxLowerLeft, self.nSubarray,
                                          self.nSurf, emult_list=emult_list)

    def test_emult_none_handled_as_no_op(self):
        """
        Check optional input DM-plane field handles None input correctly.

        Assumes 2 DMs in config.

        The subarray is the same size as the full-sized array for this test
        because the cropped e-field propagation otherwise gives different
        results.
        """
        nxf = self.sl.nxfresnel_dm
        yxLowerLeft = [0, 0]
        nSubarray = nxf
        nSurf = nxf
        nxfresnel_fast_jac_orig = self.sl.nxfresnel_fast_jac.copy()
        self.sl.nxfresnel_fast_jac = nxf

        e1 = self.sl.proptodm_fast_jac(yxLowerLeft, nSubarray,
                                       nSurf, emult_list=[None, None])
        # ones are identity for multiplication
        e2 = self.sl.proptodm_fast_jac(yxLowerLeft, nSubarray, nSurf,
                                       emult_list=[np.ones((nxf, nxf)), None])
        e3 = self.sl.proptodm_fast_jac(yxLowerLeft, nSubarray, nSurf,
                                       emult_list=[None, np.ones((nxf, nxf))])
        e4 = self.sl.proptodm_fast_jac(yxLowerLeft, nSubarray, nSurf,
                                       emult_list=[np.ones((nxf, nxf)),
                                                   np.ones((nxf, nxf))])

        self.assertTrue((e1 == e2).all())
        self.assertTrue((e1 == e3).all())
        self.assertTrue((e1 == e4).all())

        self.sl.nxfresnel_fast_jac = nxfresnel_fast_jac_orig
        pass

    def test_catch_missing_sl_data_element(self):
        """Check fails as expected if precomputation was not done first."""
        lam = self.sl.lam
        epup = self.sl.epup
        pupil = self.sl.pupil
        fpm = self.sl.fpm
        lyot = self.sl.lyot
        fs = self.sl.fs
        dh = self.sl.dh
        initmaps = self.sl.initmaps
        dmlist = [dm for dm in self.sl.dmlist]
        ft_dir = self.sl.ft_dir

        SL_0 = SingleLambda(lam=lam,
                            epup=epup,
                            dmlist=dmlist,
                            pupil=pupil,
                            fpm=fpm,
                            lyot=lyot,
                            fs=fs,
                            dh=dh,
                            initmaps=initmaps,
                            ft_dir=ft_dir,
        )
        # precomp would go here normally

        with self.assertRaises(SingleLambdaException):
            SL_0.proptodm_fast_jac(self.yxLowerLeft, self.nSubarray, self.nSurf)


class TestGetDmphList(unittest.TestCase):
    """Unit test suite for get_dmph_list()."""

    def setUp(self):
        self.sl = cfg.sl_list[0]

    def test_dmset_list_2darrays(self):
        """Check DM setting type valid."""
        dmset_list = []
        for dm in self.sl.dmlist:
            dmset_list.append(np.zeros((dm.registration['nact'],
                                        dm.registration['nact'])))
            pass
        nxfresnel = self.sl.nxfresnel_dm

        for perr in [np.ones((20,)), np.ones((10, 10, 10)), [], 'txt']:
            dmset_list[0] = perr
            with self.assertRaises(TypeError):
                self.sl.get_dmph_list(dmset_list=dmset_list,
                                      nxfresnel=nxfresnel)
                pass
            pass
        pass

    def test_nxfresnel_positive_scalar_integer(self):
        """Check array size type valid."""
        dmset_list = []
        for dm in self.sl.dmlist:
            dmset_list.append(np.zeros((dm.registration['nact'],
                                        dm.registration['nact'])))
            pass

        for perr in [np.ones((20,)), np.ones((10, 10, 10)), (10,), [], 'txt']:
            with self.assertRaises(TypeError):
                self.sl.get_dmph_list(dmset_list=dmset_list, nxfresnel=perr)
                pass
            pass
        pass

    def test_dmph_list_contents(self):
        """Test outputs of dmph_list against analytic expectation."""
        # if dmset_list = initmaps, the *differential* phase from the nominal
        # dm point will be 0
        tol = 1e-13

        nxfresnel = self.sl.nxfresnel_dm
        flatdmlist = self.sl.get_dmph_list(dmset_list=self.sl.initmaps,
                              nxfresnel=nxfresnel)
        for flatdm in flatdmlist:
            self.assertTrue(np.max(np.abs(flatdm - np.ones_like(flatdm)))
                            < tol)
            self.assertTrue(flatdm.shape == (nxfresnel, nxfresnel))
            pass
        pass


class TestUpdateInorm(unittest.TestCase):
    """Unit test suite for update_inorm()."""
    def setUp(self):
        self.sl = cfg.sl_list[0]
        rng = np.random.default_rng(5551212)
        self.dm1 = 50 + rng.random(self.sl.initmaps[0].shape)
        self.dm2 = 50 + rng.random(self.sl.initmaps[0].shape)

    def test_inorm_same_value(self):
        """Check inorm is same as from get_inorm."""

        self.sl.update_inorm(dmset_list=[self.dm1, self.dm2])
        inorm0a = self.sl.inorm.copy()
        inorm0b = self.sl.get_inorm(dmset_list=[self.dm1, self.dm2])

        inorm1a = self.sl.get_inorm(dmset_list=self.sl.initmaps)
        self.sl.update_inorm(dmset_list=self.sl.initmaps)
        inorm1b = self.sl.inorm.copy()

        self.assertTrue(inorm0a == inorm0b)
        self.assertTrue(inorm1a == inorm1b)

    def test_inorm_updates(self):
        """Check inorm changes with new DM settings."""
        self.sl.update_inorm(dmset_list=self.sl.initmaps)
        inorm0 = self.sl.inorm.copy()

        self.sl.update_inorm(dmset_list=[self.dm1, self.dm2])
        inorm1 = self.sl.inorm.copy()

        self.sl.update_inorm(dmset_list=self.sl.initmaps)
        inorm2 = self.sl.inorm.copy()

        self.assertTrue(inorm2 == inorm0)
        self.assertTrue(inorm1 != inorm0)


class TestGetInorm(unittest.TestCase):
    """Unit test suite for get_inorm()."""
    def setUp(self):
        self.sl = cfg.sl_list[0]
        pass

    def test_inorm_geometric_throughput(self):
        """
        Check geometric throughput matches.

        """
        inorm0 = self.sl.get_inorm(dmset_list=self.sl.initmaps)

        # Get e-field directly from model for this DM setting
        edm0 = self.sl.eprop(self.sl.initmaps)
        ely = self.sl.proptolyot_nofpm(edm0)
        edh0 = self.sl.proptodh(ely, normalize=False)
        inorm_regular = np.max(np.abs(edh0)**2)
        tol = 1e-12
        self.assertTrue(np.abs(inorm0 - inorm_regular) < tol)
        pass

    def test_dmset_list_has_same_sizes_as_DMs_in_dmlist(self):
        """Check input DM sizes match SingleLambda DMs."""
        nactlist = [dm.registration['nact'] for dm in self.sl.dmlist]

        # list of lists, each sublist an invalid input
        dmset_list_list = []
        dmset_list_list.append([np.zeros((nact+1, nact+1))
                                for nact in nactlist])
        dmset_list_list.append([np.zeros((nact, nact))
                                for nact in nactlist[:-1]])
        dmset_list_list.append([np.zeros((nact, nact)) for nact in nactlist] +
                          [np.zeros((nactlist[-1], nactlist[-1]))])
        dmset_list_list.append([None for nact in nactlist])
        dmset_list_list.append([np.zeros((nact,)) for nact in nactlist])
        dmset_list_list.append([np.zeros((nact, nact, nact))
                                for nact in nactlist])

        for dmset_list in dmset_list_list:
            with self.assertRaises(TypeError):
                self.sl.get_inorm(dmset_list=dmset_list)
                pass
            pass
        pass

    def test_dmset_list_is_listlike(self):
        """Check the list of DMs is actually list-like."""
        for dmset_list in [5., 'txt', 1j]:
            with self.assertRaises(TypeError):
                self.sl.get_inorm(dmset_list=dmset_list)
                pass
            pass
        pass

    def test_dmset_list_is_None(self):
        """
        Check default case for DM settings (None)
        should handle this case without errors and produce 0 DM.
        """

        inorm_None = self.sl.get_inorm(dmset_list=None)
        inorm = self.sl.get_inorm(dmset_list=self.sl.initmaps)
        self.assertTrue(inorm == inorm_None)
        pass

    def test_unobscured_peak_is_one(self):
        """
        inorm with no occulter should produce a unit peak under normalization
        conventions.
        """
        edm = self.sl.eprop(self.sl.initmaps)
        elyn = self.sl.proptolyot_nofpm(edm)
        edhf = self.sl.proptodh(elyn)
        I00 = np.max(np.abs(edhf)**2)
        tol = 1e-12
        self.assertTrue(np.abs(I00-1) < tol)
        pass

    def test_tilt_does_not_change_inorm(self):
        """
        inorm should not be affected if we have a bulk upstream tip/tilt.
        """
        E0 = self.sl.epup
        etip0 = self.sl.epup.tip
        etilt0 = self.sl.epup.tilt
        L0 = self.sl.lyot
        tip0 = self.sl.lyot.tip
        tilt0 = self.sl.lyot.tilt
        inorm_flat = self.sl.get_inorm()

        L1 = LyotStop(L0.lam, L0.e, L0.pixperpupil, tip=tip0+1, tilt=tilt0+1)
        E1 = Epup(E0.lam, E0.e, E0.pixperpupil, tip=etip0+1, tilt=etilt0+1)

        self.sl.lyot = L1
        inorm_tilt = self.sl.get_inorm()
        # don't throw off other tests
        self.sl.lyot = L0

        self.sl.epup = E1
        inorm_tilt2 = self.sl.get_inorm()
        # don't throw off other tests
        self.sl.epup = E0

        tol = 1e-13

        self.assertFalse((L0.ttgrid == L1.ttgrid).all())
        self.assertFalse((E0.ttgrid == E1.ttgrid).all())
        self.assertTrue(np.abs(inorm_flat - inorm_tilt) < tol)
        self.assertTrue(np.abs(inorm_flat - inorm_tilt2) < tol)
        pass


class TestDM1D2D(unittest.TestCase):
    """Unit test suite for get_dmind1d() and get_dmind2d()."""

    def setUp(self):
        self.sl = cfg.sl_list[0]
        pass

    def test_dmind1d_success(self):
        """Verify get_dmind1d works on various edge cases."""
        nact = self.sl.dmlist[0].registration['nact']
        djklist = [(0, 0, 0),
               (0, 0, 1),
               (0, 1, 0),
               (0, nact-1, nact-1),
               (1, 0, 0),
               (1, 0, 1),
               (1, 1, 0),
               (1, nact-1, nact-1)]
        outlist = [0, 1, nact, nact**2-1, nact**2, nact**2+1,
                   nact**2 + nact, 2*(nact**2) - 1]

        for index, djk in enumerate(djklist):
            self.assertTrue(outlist[index] == self.sl.get_dmind1d(*djk))
            pass
        pass

    def test_dmind2d_success(self):
        """Verify get_dmind2d works on various edge cases."""
        nact = self.sl.dmlist[0].registration['nact']
        djklist = [(0, 0, 0),
               (0, 0, 1),
               (0, 1, 0),
               (0, nact-1, nact-1),
               (1, 0, 0),
               (1, 0, 1),
               (1, 1, 0),
               (1, nact-1, nact-1)]
        outlist = [0, 1, nact, nact**2-1, nact**2, nact**2+1,
                   nact**2 + nact, 2*(nact**2) - 1]

        for index, out in enumerate(outlist):
            self.assertTrue(djklist[index] == self.sl.get_dmind2d(out))
            pass
        pass

    def test_forward_and_back_one(self):
        """
        Verify starting with an index and going through both get_dmind2d and
        get_dmind1d returns the original index.
        """
        nact = self.sl.dmlist[0].registration['nact']
        outlist = [0, 1, nact, nact**2-1, nact**2, nact**2+1,
                   nact**2 + nact, 2*(nact**2) - 1]
        for out in outlist:
            self.assertTrue(out ==
                            self.sl.get_dmind1d(*self.sl.get_dmind2d(out)))
            pass

    def test_forward_and_back_three(self):
        """
        Verify starting with an DM number and (j,k) index coordinates, and
        going through both get_dmind2d and get_dmind1d, returns the original
        DM number and coordinates.
        """
        nact = self.sl.dmlist[0].registration['nact']
        djklist = [(0, 0, 0),
               (0, 0, 1),
               (0, 1, 0),
               (0, nact-1, nact-1),
               (1, 0, 0),
               (1, 0, 1),
               (1, 1, 0),
               (1, nact-1, nact-1)]

        for djk in djklist:
            self.assertTrue(djk ==
                            self.sl.get_dmind2d(self.sl.get_dmind1d(*djk)))
            pass
        pass

    def test_dmn_nonnegative_scalar_integer(self):
        """Check DM number type valid."""
        j = 0
        k = 0
        for perr in [-1.5, -1, 3.5, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                self.sl.get_dmind1d(dmn=perr, j=j, k=k)
                pass
            pass
        pass

    def test_j_nonnegative_scalar_integer(self):
        """Check DM row coordinate type valid."""
        dmn = 0
        k = 0
        for perr in [-1.5, -1, 3.5, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                self.sl.get_dmind1d(dmn=dmn, j=perr, k=k)
                pass
            pass
        pass

    def test_k_nonnegative_scalar_integer(self):
        """Check DM column coordinate type valid."""
        dmn = 0
        j = 0
        for perr in [-1.5, -1, 3.5, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                self.sl.get_dmind1d(dmn=dmn, j=j, k=perr)
                pass
            pass
        pass

    def test_dmn_too_large(self):
        """
        Check DM number is within range of the actual number of DMs in
        SingleLambda.
        """
        j = 0
        k = 0
        for perr in [len(self.sl.dmlist), len(self.sl.dmlist)+1]:
            with self.assertRaises(ValueError):
                self.sl.get_dmind1d(dmn=perr, j=j, k=k)
                pass
            pass
        pass

    def test_j_too_large(self):
        """
        Check DM row coordinate is within range of size of appropriate DM.
        """
        dmn = 0
        k = 0
        nact = self.sl.dmlist[dmn].registration['nact']
        for perr in [nact, nact+1]:
            with self.assertRaises(ValueError):
                self.sl.get_dmind1d(dmn=dmn, j=perr, k=k)
                pass
            pass
        pass

    def test_k_too_large(self):
        """
        Check DM column coordinate is within range of size of appropriate DM.
        """
        dmn = 0
        j = 0
        nact = self.sl.dmlist[dmn].registration['nact']
        for perr in [nact, nact+1]:
            with self.assertRaises(ValueError):
                self.sl.get_dmind1d(dmn=dmn, j=j, k=perr)
                pass
            pass
        pass

    def test_dmind_nonnegative_scalar_integer(self):
        """Check 1D DM index tpye valid."""
        for perr in [-1.5, -1, 3.5, 1j, [], 'perr', (5,)]:
            with self.assertRaises(TypeError):
                self.sl.get_dmind2d(dmind=perr)
                pass
            pass
        pass

    def test_dmind_too_large(self):
        """Check 1D DM index within range of total number of actuators."""
        allact = self.sl.ndmact[-1]
        for perr in [allact, allact+1]:
            with self.assertRaises(ValueError):
                self.sl.get_dmind2d(dmind=perr)
                pass
            pass
        pass


if __name__ == '__main__':
    unittest.main()
