# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Class file for a single-wavelength coronagraph model and associated propagation
functions.
"""
from types import SimpleNamespace
import logging
import numpy as np

from howfsc.util.ampthresh import ampthresh
from howfsc.util.fresnelprop import do_fft, do_ifft, fresnelprop
from howfsc.util.fresnelprop import get_fp, fresnelprop_fp
from howfsc.util.insertinto import insertinto
import howfsc.util.check as check
from howfsc.util.dmhtoph import dmhtoph
from howfsc.util.dmhtoph_jac import (
    dmhtoph_jac, dmhtoph_cropped_poke, compute_master_inf_func
)
from howfsc.util.math import ceil_even, ceil_odd
from howfsc.util.mft import do_mft, do_imft, do_offcenter_mft
from howfsc.util.remove_ptt import fit_and_remove_ptt_directly
from howfsc.util.unwrap import unwrap_segments
from howfsc.model.mask import (PupilMask, FocalPlaneMask, Epup, DMFace,
                               LyotStop, FieldStop)


class SingleLambdaException(Exception):
    """Thin class for exception handling """
    pass

def _checkclass(var, vclass, vname):
    """
    Input checker that input class type is valid

    returns var (1st arg)
    """
    if not isinstance(var, vclass):
        raise TypeError(str(vname)
                        + ' is not descended from expected class')


class SingleLambda:
    """
    Single-wavelength model class

    This class creates a model of a Lyot-type coronagraphic system at a single
    wavelength.  This model contains
     - an input field (pre-coronagraph wavefront)
     - one or more sequential DMs at user-specified distances from pupil
     - an initial pupil-plane mask or apodizer
     - a focal-plane occulter
     - a Lyot stop
     - a field stop
     - a 'dark hole' mask which specifies which pixels in the output image are
       to be used for wavefron control
     - the DM settings at which the input field was measured
    Other than the DMs, all optics are in pupil or focal planes.  These models
    are specified per-wavelength to allow the effect of chromatic masks (such
    as focal-plane dielectric with varying refractive index) to be captured, as
    well as the natural linear spreading of focal-plane PSF with wavelength.

    Pupil-plane masks must be defined right up to their edges, to permit the
    use of MFTs for propagation.  Focal-plane masks should be defined close to
    their edges if possible for efficiency, but it is not required.

    If a mask is not present in a coronagraph, TODO

    Arguments:
     lam: wavelength in meters
     epup: an Epup object representing the input field at this wavelength
     dmlist: a list of DMFace objects, one for each DM
     pupil: a PupilMask object representing the initial-plane optic
     fpm: a FocalPlaneMask object representing the occulter
     lyot: a PupilMask object representing the Lyot stop
     fs: a FocalPlaneMask object representing the field stop
     dh: a FocalPlaneMask object representing the set of pixels to be used for
      wavefront control.  Any nonzero pixels will be included in wavefront
      control.
     initmaps: a list of 2D arrays with DM maps with sizes that correspond
      to the DM sizes in dmlist
     ft_dir: one of 'forward' or 'reverse', to describe the direction of
      propagation from pupil to focus (forward -> focus is downstream of pupil,
      reverse -> focus is upstream of pupil)

    """

    def __init__(self, lam, epup, dmlist, pupil, fpm, lyot, fs, dh, initmaps,
                 ft_dir):
        # Input checking
        check.real_positive_scalar(lam, 'lam', TypeError)
        _checkclass(epup, Epup, 'epup')
        _checkclass(pupil, PupilMask, 'pupil')
        _checkclass(fpm, FocalPlaneMask, 'fpm')
        _checkclass(lyot, LyotStop, 'lyot')
        _checkclass(fs, FieldStop, 'fs')
        _checkclass(dh, FieldStop, 'dh')
        try:
            for index, dm in enumerate(dmlist):
                _checkclass(dm, DMFace, 'dm ' + str(index+1) \
                            + ' of ' + str(len(dmlist)))

        except TypeError:
            raise TypeError('dmlist must be an iterable')

        self.dmlist = dmlist
        self._check_dmset_list(initmaps)

        self.lam = lam
        self.epup = epup
        self.pupil = pupil
        self.fpm = fpm
        self.lyot = lyot
        self.fs = fs
        self.dh = dh
        self.initmaps = initmaps

        if ft_dir not in ['forward', 'reverse']:
            raise SingleLambdaException('ft_dir must be forward or reverse')

        self.ft_dir = ft_dir


        #--------------------
        # Derived parameters
        #--------------------

        # Round the Fresnel propagation size for angular spectrum up to the
        # nearest factor of two, so the FFTs don't take forever if you get e.g.
        # a prime size.  (Don't do it elsewhere, FFT size is directly linked
        # to focal-plane sampling when propagating between masks.)

        # zlist: distance to Fresnel-propagate to each DM + back to pupil
        self.zlist = []
        zprev = 0.
        for index, dm in enumerate(self.dmlist):
            self.zlist.append(dm.z-zprev)
            zprev = dm.z

        self.zlist.append(-zprev)

        # Compute nxfresnel_dm. Array sized to prevent aliasing during prop.
        maxe = np.max(self.epup.e.shape)
        length_list = [maxe, ]  # initialize
        for index, _ in enumerate(self.dmlist):
            nMin = ceil_even(self.lam * np.max(np.abs(self.zlist)) *
                             self.dmlist[index].pixpermeter**2)
            length_list.append(nMin)

        self.nxfresnel_dm = int(2**(np.ceil(np.log2(np.max(length_list)))))

        # Store cumulative number of actuators per DM so we don't have to
        # always recalculate when going 2D -> 1D.  DM n has indices
        # ndmact[n]:ndmact[n+1]
        self.ndmact = np.cumsum([0]
                                +[dm.registration['nact']**2
                                  for dm in self.dmlist])
        # Normalize star intensity to peak of 1
        self.inorm = self.get_inorm()
        # Get dark-hole indices to be extracted for wavefront control and
        # Jacobians
        self.dh_inds = np.flatnonzero(self.dh.e)

        # Handle directional ambiguity at final focus
        if self.ft_dir == 'forward':
            self.ft_dir_sign = 1
            pass
        elif self.ft_dir == 'reverse':
            self.ft_dir_sign = -1
            pass
        else:
            raise SingleLambdaException('Invalid direction to final focus')

        # Tip/tilt phase map to be applied upstream of the focal-plane mask
        # 2 pi/(ppl * ppp), combined with a grid of pixel numbers, gives
        # offset in pixels
        self.ttph_up = np.exp(1j*2*np.pi
                              /(self.epup.pixperpupil*self.fs.pixperlod)
                              *self.epup.ttgrid*self.ft_dir_sign)

        # Tip/tilt phase map to be applied downstream of the focal-plane mask
        # 2 pi/(ppl * ppp), combined with a grid of pixel numbers, gives
        # offset in pixels
        self.ttph_down = np.exp(1j*2*np.pi
                                /(self.lyot.pixperpupil*self.fs.pixperlod)
                                *self.lyot.ttgrid*self.ft_dir_sign)
        #---------------------
        # Jacobian parameters
        #---------------------

        # Don't fill this yet; some depend on DM state.  Use get_jac_precomp()
        # to fill them just before using.  calcjacs() will do this for you.
        self.dmph_list = None
        self.fp_list = None
        self.e_start = None
        self.e_list = None
        # Similar, but from get_fast_jac_precomp()
        self.dm1 = SimpleNamespace()
        self.dm2 = SimpleNamespace()
        self.nxfresnel_fast_jac = None
        self.fp_crop_list = None

    def update_inorm(self, dmset_list=None):
        """
        Recompute the stored value of inorm.
        
        Keyword Arguments:
         dmset_list: a list of DM settings, in voltage, currently applied to
          the DMs, or ``None``.  The DM sizes and ordering must match
          ``self.dmlist``.  If ``None``, then a uniform DM setting of 0 is
          assumed.
        
        Returns:
         nothing
        """
        self.inorm = self.get_inorm(dmset_list=dmset_list)

        return None


    def get_jac_precomp(self, dmset_list):
        """
        Given an initial DM setting, compute repeated terms to save Jacobian
        time.

        Sets:
         - self.dmph_list
         - self.zlist
         - self.fp_list
         - self.e_start
         - self.e_list
        Running this function will do nothing if you will not be using
        self.proptodm_jac() later.  If you run self.proptodm_jac() without
        running this first, you will throw an exception.  (If you are using
        calcjacs() to compute the Jacobian, this will all be handled
        automatically.)

        Arguments:
         dmset_list: a list of DM settings, in voltage, currently applied to
          the DMs.  The DM sizes and ordering must match ``self.dmlist``.

        Returns:
         nothing

        """
        self._check_dmset_list(dmset_list)

        # dmph_list: the complex-valued e^(i phase) term induced by each
        # baseline DM setting in dmset_list.  Also depends on wavelength (so
        # this needs to be per SingleLambda)
        self.dmph_list = self.get_dmph_list(dmset_list, self.nxfresnel_dm)

        # fp_list: quadratic phase terms for each propagation distance.  Also
        # depends on wavelength (so this needs to be per SingleLambda)
        self.fp_list = []  # uses zlist so run second
        for index, dm in enumerate(self.dmlist):
            fp = get_fp(self.lam, self.zlist[index], self.nxfresnel_dm,
                        dm.pixpermeter)
            self.fp_list.append(fp)

        # final propagation back to previous
        fp = get_fp(self.lam, self.zlist[-1], self.nxfresnel_dm,
                    self.dmlist[-1].pixpermeter)
        self.fp_list.append(fp)

        # e_start: starting front-end wavefront, including any tip-tilt and
        # resizing necessary
        self.e_start = self.epup.e*self.ttph_up

        # e_list: intermediate propagation fields for DM1 and DM2.  Mostly
        # expected to speed up DM2 calcs as DM1 already has a shortcut to skip
        # a zero-length propagation.  Uses e_start, zlist, fp_list, dmph_list
        self.e_list = []
        for index, dm in enumerate(self.dmlist):
            if index == 0:
                e = fresnelprop_fp(insertinto(self.e_start,
                                              (self.nxfresnel_dm,
                                               self.nxfresnel_dm)),
                                   self.zlist[index],
                                   self.nxfresnel_dm,
                                   self.fp_list[index]).astype('complex128')

            else:
                e = fresnelprop_fp(self.e_list[index-1],
                                   self.zlist[index],
                                   self.nxfresnel_dm,
                                   self.fp_list[index]).astype('complex128')

            e *= self.dmph_list[index]

            self.e_list.append(e)

    def get_fast_jac_precomp(self):
        """
        Pre-compute influence function related terms for fast Jac calculation.

        DM -pecific outputs are stored in self.dm1 and self.dm2.
        The same value of nxfresnel_fast_jac is used for both DMs.

        No arguments. No returns.

        """
        length_list = []  # initialize
        for dmn, dmobj in enumerate([self.dm1, self.dm2]):

            dmobj.pupil_shape = self.pupil.e.shape

            compute_master_inf_func(dmobj, self.dmlist[dmn].registration)

            # Compute minimum array width to prevent aliasing during
            # angular spectrum propagation.
            nOrig = dmobj.infMaster.shape[0]
            nMin = ceil_even(self.lam * np.max(np.abs(self.zlist)) *
                             self.dmlist[dmn].pixpermeter**2)
            length_list.append(int(np.max([nMin, nOrig])))

        self.nxfresnel_fast_jac = np.max(length_list)

        # fp_crop_list: quadratic phase terms for each propagation distance.
        # Used only for cropped poke propagation for the fast Jacobian.
        # Also depends on wavelength (so this needs to be per SingleLambda)
        self.fp_crop_list = []
        for index, dm in enumerate(self.dmlist):
            fp = get_fp(self.lam, self.zlist[index], self.nxfresnel_fast_jac,
                        dm.pixpermeter)
            self.fp_crop_list.append(fp)

        # last propagation back to first dm
        fp = get_fp(self.lam, self.zlist[-1], self.nxfresnel_fast_jac,
                    self.dmlist[-1].pixpermeter)
        self.fp_crop_list.append(fp)

    #------------------
    # Mask propagation
    #------------------

    def proptolyot(self, e0):
        """
        Propagate a function from the initial pupil/apodizer plane to the
        Lyot plane, inclusive.

        This function does not normalize with ``inorm``.

        Arguments:
         e0: 2D complex array with electric field directly before
          pupil/apodizer mask. Array should be the same size as the pupil.

        Returns:
         a 2D complex array of the same size as the Lyot mask

        """

        check.twoD_array(e0, 'e0', TypeError)

        e0 = self.pupil.applymask(e0)

        if self.fpm.isopen:  # Case 1: FPM is open at edge (HLC-like)
            # Use Babinet's principle
            be0 = e0.copy()
            e0 = do_mft(e0, self.fpm.e.shape, self.fpm.pixperlod,
                        self.pupil.pixperpupil, direction=self.ft_dir)
            e0 = self.fpm.applymask(e0)

            # imft to avoid image rotation
            e0 = do_imft(e0, self.lyot.e.shape, self.fpm.pixperlod,
                         self.lyot.pixperpupil, direction=self.ft_dir)
            e0 = be0 - e0  # Babinet
            e0 = self.lyot.applymask(e0)

        else:  # Case 2: FPM is closed at edge (SPC-like)
            e0 = do_mft(e0, self.fpm.e.shape, self.fpm.pixperlod,
                        self.pupil.pixperpupil, direction=self.ft_dir)
            e0 = self.fpm.applymask(e0)

            # imft to avoid image rotation
            e0 = do_imft(e0, self.lyot.e.shape, self.fpm.pixperlod,
                         self.lyot.pixperpupil, direction=self.ft_dir)
            e0 = self.lyot.applymask(e0)

        return e0

    def croppedproptolyot(self, e0, yxLowerLeft, nSurf):
        """
        Propagate an E-field subarray from the initial pupil/apodizer plane
        to the full-sized Lyot plane, inclusive.

        This function does not normalize with ``inorm``.

        Arguments:
         e0: Square, 2-D array. Sub-array of the E-field at pupil after DMs.
         yxLowerLeft: Tuple containing the (y, x) array coordinates at which
          to insert the surf_crop sub-array into a larger array having a shape
          of (nSurf, nSurf).
        nSurf: int; e0 goes back into a 2-D square array of shape (nSurf,
         nSurf) at the position given by yxLowerLeft.

        Returns:
         2-D complex E-field at the Lyot plane. Same size as the Lyot mask.

        """
        # Check inputs
        check.twoD_square_array(e0, 'e0', TypeError)
        check.positive_scalar_integer(nSurf, 'nSurf', TypeError)
        nSubarray = e0.shape[0]
        try:
            if len(yxLowerLeft) != 2:
                raise TypeError('yxLowerLeft must have 2 elements')
            for index, coord in enumerate(yxLowerLeft):
                check.nonnegative_scalar_integer(
                    coord, 'yxLowerLeft['+str(index)+']', TypeError)
                if (coord < 0) or (coord + nSubarray > nSurf):
                    raise TypeError('Array coordinates in yxLowerLeft must be'
                                    'within an (nSurf, nSurf) shaped array.')
        except TypeError:
            raise TypeError('yxLowerLeft must be an iterable')

        # Repopulate full pupil plane
        e0full = np.zeros((nSurf, nSurf), dtype='complex128')
        e0full[yxLowerLeft[0]:yxLowerLeft[0] + nSubarray,
               yxLowerLeft[1]:yxLowerLeft[1] + nSubarray] = e0.copy()

        # Apply pupil mask
        e0full = self.pupil.applymask(e0full)

        # Recrop pupil plane
        e0 = e0full[yxLowerLeft[0]:yxLowerLeft[0] + nSubarray,
                    yxLowerLeft[1]:yxLowerLeft[1] + nSubarray]

        if self.fpm.isopen:  # Case 1: FPM is open at edge (HLC-like)
            # Use Babinet's principle
            e0 = do_offcenter_mft(e0, self.fpm.e.shape, self.fpm.pixperlod,
                                  self.pupil.pixperpupil, (nSurf, nSurf),
                                  yxLowerLeft, direction=self.ft_dir)
            e0 = self.fpm.applymask(e0)

            # imft to avoid image rotation
            e0 = do_imft(e0, self.lyot.e.shape, self.fpm.pixperlod,
                         self.lyot.pixperpupil, direction=self.ft_dir)
            e0 = insertinto(e0full, self.lyot.e.shape) - e0  # Babinet
            e0 = self.lyot.applymask(e0)

            # Error if pupil plane samplings don't match because that is
            # needed for the Babinet's principle trick to work correctly.
            tol = np.finfo(float).eps
            if np.abs(self.pupil.pixperpupil - self.lyot.pixperpupil) > tol:
                raise ValueError("For fast Jacobian with open-extent FPM,"
                                 "Babinet's principle is used and requires"
                                 "pixperpupil to be equal at the pupil planes"
                                 "before and after the FPM.")

        else:  # Case 2: FPM is closed at edge (SPC-like)
            e0 = do_offcenter_mft(e0, self.fpm.e.shape, self.fpm.pixperlod,
                                  self.pupil.pixperpupil, (nSurf, nSurf),
                                  yxLowerLeft, direction=self.ft_dir)
            e0 = self.fpm.applymask(e0)

            # imft to avoid image rotation
            e0 = do_imft(e0, self.lyot.e.shape, self.fpm.pixperlod,
                         self.lyot.pixperpupil, direction=self.ft_dir)
            e0 = self.lyot.applymask(e0)

        return e0

    def proptolyot_nofpm(self, e0):
        """Short wrapper to handle the no-FPM case efficiently

        If the pupil and Lyot have the same sampling, just multiply them.
        Otherwise, we'll use FFT machinery to get the resampling done
        correctly.  FFT is significantly more efficient than MFT for this.

        Arguments:
         e0: 2D complex array with electric field directly before
          pupil/apodizer mask. Array should be the same size as the pupil.

        Returns:
         a 2D complex array of the same size as the Lyot mask

        """

        check.twoD_array(e0, 'e0', TypeError)

        if self.pupil.pixperpupil == self.lyot.pixperpupil:
            return self.lyot.applymask(self.pupil.applymask(e0))

        # resize epup to pupil if need
        nfft_pf = self._fftsize(self.pupil, self.fpm)
        e0 = insertinto(e0, (nfft_pf, nfft_pf))
        e0 = self.pupil.applymask(e0)

        e0 = do_fft(e0)

        nfft_fl = self._fftsize(self.lyot, self.fpm)
        e0 = insertinto(e0, (nfft_fl, nfft_fl))

        e0 = do_ifft(e0) # ifft to avoid image rotation

        # Reduce to match Lyot for mft consistency
        e0 = insertinto(e0, self.lyot.e.shape)
        return self.lyot.applymask(e0)

    def proptodh(self, e0, normalize=True):
        """
        Propagate from the plane directly following the pupil plane to the
        final image plane, including the field stop

        This function normalizes with ``inorm`` if normalize == True.

        Arguments:
         e0: 2D complex array with electric field directly after the Lyot stop.
          Array should be the same size as the pupil.

        Keyword Arguments:
         normalize: boolean for whether to normalize the output by inorm.

        Returns:
         a 2D complex array of the same size as field stop

        """
        check.twoD_array(e0, 'e0', TypeError)
        check.boolean(normalize, 'normalize', TypeError)

        # add tip/tilt at Lyot stop plane
        e0 = e0*self.ttph_down

        e0 = do_mft(e0, self.fs.e.shape, self.fs.pixperlod,
                    self.lyot.pixperpupil, direction=self.ft_dir)
        e0 = self.fs.applymask(e0)

        if normalize:
            e0 = e0/np.sqrt(self.inorm)

        return e0
    
    def proptodhpeak(self, e0):
        """
        Propagate from the plane directly following the pupil plane to the
        peak of the final image plane.
         
        Removes any piston/tip/tilt before propagating.

        Does not include the field stop.

        This function does not normalize with ``inorm``.

        Arguments:
         e0: 2D complex array with electric field directly after the Lyot stop.
          Array should be the same size as the pupil.

        Returns:
         the complex scalar E-field at the center of the focal plane

        """
        check.twoD_array(e0, 'e0', TypeError)
        
        # Remove all piston/tip/tilt at Lyot stop plane
        amp = np.abs(e0)
        sw_mask = ampthresh(amp).astype(bool)
        phase_to_fit = np.angle(e0)
        phase_to_fit_unwrapped, _ = unwrap_segments(phase_to_fit, amp)
        phase_no_ptt, ptt_fitted = fit_and_remove_ptt_directly(
            phase_to_fit_unwrapped, sw_mask)

        # Use ptt_fitted instead of phase_no_ptt because phase_no_ptt has
        # phase values zeroed out at low amplitude areas, which throws off
        # the accuracy of unit tests and cases when the amplitude isn't
        # exactly zero behind a mask like in a phase retrieval.
        e1 = e0 * np.exp(-1j*ptt_fitted)
        epeakarray = do_mft(
            e1, (1, 1), self.fs.pixperlod, self.lyot.pixperpupil,
            direction=self.ft_dir)

        return epeakarray[0, 0]

    def _fftsize(self, pupil, focal):
        """
        Given pupil and focal sampling, choose correct FFT size on the fly

        This is a function for consistency/refactor purposes

        Arguments:
         pupil: a Pupil object
         focal: a FocalPlaneMask object

        Returns
         integer size

        """
        _checkclass(pupil, PupilMask, 'pupil')
        _checkclass(focal, FocalPlaneMask, 'focal')
        return round(pupil.pixperpupil*focal.pixperlod)


    def eprop(self, dmset_list):
        """
        Propagate a DM setting through a pair of model DMs.

        Input DMs are expected in raw units; the model DM starting point will
        be subtracted off internally.

        Compare to ``pokeprop()``, which is a similar function with a DM
        poke.  This would be used for non-Jacobian propagation, e.g. with
        pairwise probing.

        Arguments:
         dmset_list: a list of DM settings, in voltage, currently applied to
          the DMs.  The DM sizes and ordering must match ``self.dmlist``.

        Returns:
         a electric field of the same size as ``self.epup.e``

        """

        self._check_dmset_list(dmset_list)

        e0 = self.epup.e*self.ttph_up
        e0 = self.proptodm(e0, dmset_list)

        return e0


    def pokeprop(self, dmind, dmset_list):
        """
        Propagate a 1 rad actuator surface poke as a differential field
        through the DMs for Jacobian calculation

        Input DMs are expected in raw units; the model DM starting point will
        be subtracted off internally.

        This function is streamlined for use with a Jacobian calculation, and
        expects the following precomputed data elements to be present:
         - self.e_start
        Running ``self.get_jac_precomp(dmset_list)`` prior to invoking this
        function will populate these values correctly.

        Arguments:
         dmind: 1D nonnegative scalar integer less than the total number of
          actuators.
         dmset_list: a list of DM settings, in voltage, currently applied to
          the DMs.  The DM sizes and ordering must match ``self.dmlist``.

        Returns:
         a electric field of the same size as ``self.epup.e``

        """

        # get_dmind2d will do input checking on dmind
        self._check_dmset_list(dmset_list)
        if self.e_start is None:
            raise SingleLambdaException('self.e_start undefined prior to ' +
                                        'pokeprop() call.')

        dmn, j, k = self.get_dmind2d(dmind)

        pokerad = 1 # rad
        nrow, ncol = self.e_start.shape
        emult = 2.*1j*dmhtoph_jac(nrow, ncol, pokerad, j, k,
                                  **self.dmlist[dmn].registration)

        emult_list = []
        for index in range(len(self.dmlist)):
            if dmn == index:
                emult_list.append(emult)

            else:
                emult_list.append(None)

        e0 = self.proptodm_jac(emult_list=emult_list)
        return insertinto(e0, self.e_start.shape)


    def croppedpokeprop(self, dmind, dmset_list):
        """
        Propagate a cropped 1 rad actuator surface poke as a differential
        field through the DMs for Jacobian calculation

        Input DMs are expected in raw units; the model DM starting point will
        be subtracted off internally.

        This function is streamlined for use with a Jacobian calculation, and
        expects the following precomputed data elements to be present:
         - self.e_start
        Running ``self.get_jac_precomp(dmset_list)`` prior to invoking this
        function will populate these values correctly.

        Arguments:
         dmind: 1D nonnegative scalar integer less than the total number of
          actuators.
         dmset_list: a list of DM settings, in voltage, currently applied to
          the DMs.  The DM sizes and ordering must match ``self.dmlist``.

        Returns:
         e0full : array_like
          2-D array of E-field at a pupil after propagating through the DMs.
         e0crop : array_like
          An offcenter-cropped subarray of e0full.
         yxLowerLeft : tuple
          Tuple containing the (y, x) array coordinates at which to insert
          the surf_crop sub-array into a larger array of shape dmArrayShape.
         nSurf : int
          surfCrop goes back into a 2-D array of shape (nSurf, nSurf) at the
          position given by yxLowerLeft.

        """
        # get_dmind2d will do input checking on dmind
        self._check_dmset_list(dmset_list)
        if self.e_start is None:
            raise SingleLambdaException('self.e_start undefined prior to '
                                        'pokeprop() call.')

        dmn, j, k = self.get_dmind2d(dmind)

        pokerad = 1  # rad
        if dmn == 0:
            dm = self.dm1
        elif dmn == 1:
            dm = self.dm2
        surfCrop, yxLowerLeft, nSurf = dmhtoph_cropped_poke(
            pokerad, j, k, dm)
        emult = 2.*1j*surfCrop
        nSubarray = surfCrop.shape[0]

        emult_list = []
        for index in range(len(self.dmlist)):
            if dmn == index:
                emult_list.append(emult)

            else:
                emult_list.append(None)

        e0crop = self.proptodm_fast_jac(
            yxLowerLeft, nSubarray, nSurf, emult_list=emult_list)

        # Create the full E-field array for peak Jac, no-FPM propagation part
        e0full = np.zeros((nSurf, nSurf), dtype='complex128')
        e0full[yxLowerLeft[0]:yxLowerLeft[0] + nSubarray,
               yxLowerLeft[1]:yxLowerLeft[1] + nSubarray] = e0crop

        return e0full, e0crop, yxLowerLeft, nSurf


    def proptodm(self, e0, dmset_list, nxfresnel=None, emult_list=None):
        """
        Include the effects of propagating to a DM surface

        This function will  Fresnel propagate ``e0`` forward by ``dmface.z``,
        create and multiply by DM phase, and Fresnel propagate back by
        ``dmface.z`` to the original plane.

        Arguments:
         e0: 2D complex array with starting e-field
         dmset_list: a list of DM settings, in voltage, currently applied to
          the DMs.  The DM sizes and ordering must match ``self.dmlist``.

        Keyword Arguments:
         nxfresnel: the number of pixels to use in the Fresnel propagation,
          or None.  If None, uses the default chosen in ``self.nxfresnel_dm``.
          ``self.nxfresnel_dm`` has a good estimate, but may be chosen
          otherwise by user.  Should be a positive integer if not None.
         emult_list: list of 2D arrays with the same center as `e0`` with any
          additional field multiplier at the DM surface, or ``None`` if you
          don't want to apply anything other that the phase effect of the DM.
          Primary expected use is the introduction of the differential field
          in Jacobian calculation. Defaults to ``None``.  Should be the same
          length as the number of DMs, and this will be checked.  Individual
          list elements may be None as well, if that DM is not going to get
          an additional multiplier.  It is recommended to oversize these with
          respect to e0 if they are open (ones) at the edges, as they will
          truncate out-of-pupil Fresnel terms otherwise.

        Returns:
         a 2D complex array of the same size as ``e0`` and located optically
          in the same plane

        """

        # Check inputs
        check.twoD_array(e0, 'e0', TypeError)
        self._check_dmset_list(dmset_list)

        if nxfresnel is not None:
            check.positive_scalar_integer(nxfresnel, 'nxfresnel', TypeError)

        else:
            nxfresnel = self.nxfresnel_dm

        if emult_list is None:
            emult_list = [None for dm in self.dmlist]

        # Check list elements before any propagation begins
        for index, dm in enumerate(self.dmlist):
            emult = emult_list[index]
            if emult is not None:
                check.twoD_array(emult, 'emult', TypeError)

        zprev = 0. # Track z so we go plane to plane, rather than back to 0
        e0shape = e0.shape # Use Fresnel size on this end, too, so we don't
                           # truncate dmsurf
        e0 = insertinto(e0, (nxfresnel, nxfresnel))

        # Actually propagate now
        dmph_list = self.get_dmph_list(dmset_list, nxfresnel)

        for index, dm in enumerate(self.dmlist):
            emult = emult_list[index]

            # Start ang spec
            e0 = fresnelprop(e0, self.lam, dm.z-zprev,
                             nxfresnel, dm.pixpermeter).astype('complex128')
            zprev = dm.z

            # 2x for surface --> wavefront
            e0 *= dmph_list[index]
            if emult is not None:
                e0 *= insertinto(emult, (nxfresnel, nxfresnel))

        # propagate back to previous dm
        e0 = fresnelprop(e0, self.lam, -zprev, nxfresnel,
                         self.dmlist[-1].pixpermeter)
        return insertinto(e0, e0shape)


    def proptodm_jac(self, emult_list=None):
        """
        Include the effects of propagating to a DM surface for Jacobian
        propagation

        This function will Fresnel propagate ``e0`` forward by ``dmface.z``,
        create and multiply by DM phase, and Fresnel propagate back by
        ``dmface.z`` to the original plane.

        This function is streamlined for use with a Jacobian calculation, and
        expects the following precomputed data elements to be present:
         - self.dmph_list
         - self.zlist
         - self.fp_list
         - self.e_start
         - self.e_list
        Running ``self.get_jac_precomp(dmset_list)`` prior to invoking this
        function will populate these values correctly.

        Keyword Arguments:
         emult_list: list of 2D arrays with the same center as `e0`` with any
          additional field multiplier at the DM surface, or ``None`` if you
          don't want to apply anything other that the phase effect of the DM.
          Primary expected use is the introduction of the differential field
          in Jacobian calculation. Defaults to ``None``.  Should be the same
          length as the number of DMs, and this will be checked.  Individual
          list elements may be None as well, if that DM is not going to get
          an additional multiplier.  It is recommended to oversize these with
          respect to e0 if they are open (ones) at the edges, as they will
          truncate out-of-pupil Fresnel terms otherwise.

        Returns:
         a 2D complex array of the same size as ``e0`` and located optically
          in the same plane

        """
        # check that pre-computed arrays have actually been computed
        if self.e_start is None:
            raise SingleLambdaException('self.e_start undefined prior to ' +
                                        'proptodm_jac() call.')
        if self.e_list is None:
            raise SingleLambdaException('self.e_list undefined prior to ' +
                                        'proptodm_jac() call.')
        if self.fp_list is None:
            raise SingleLambdaException('self.fp_list undefined prior to ' +
                                        'proptodm_jac() call.')
        if self.zlist is None:
            raise SingleLambdaException('self.zlist undefined prior to ' +
                                        'proptodm_jac() call.')
        if self.dmph_list is None:
            raise SingleLambdaException('self.dmph_list undefined prior to ' +
                                        'proptodm_jac() call.')

        e0 = self.e_start.copy()
        nxfresnel = self.nxfresnel_dm

        # Check inputs
        if emult_list is None:
            emult_list = [None]*len(self.dmlist)

        # Check list elements before any propagation begins
        for index in range(len(self.dmlist)):
            emult = emult_list[index]
            if emult is not None:
                check.twoD_array(emult, 'emult', TypeError)

        # Use Fresnel size on this end, too, so we don't truncate dmsurf
        e0shape = e0.shape
        #e0 = insertinto(e0, (nxfresnel, nxfresnel))

        # Actually propagate now
        emultyet = False
        for index in range(len(self.dmlist)):
            emult = emult_list[index]

            if emult is None and not emultyet:
                e0 = self.e_list[index]
                continue

            elif emult is not None and not emultyet:
                e0 = self.e_list[index]*insertinto(emult,
                                                   (nxfresnel, nxfresnel))
                emultyet = True
                continue

            elif emultyet:
                # Start ang spec
                e0 = fresnelprop_fp(e0, self.zlist[index], nxfresnel,
                                    self.fp_list[index]).astype('complex128')

                # 2x for surface --> wavefront
                e0 *= self.dmph_list[index]
                if emult is not None:
                    # In case of somehow more than one nonzero emult
                    e0 *= insertinto(emult, (nxfresnel, nxfresnel))

            else:
                raise SingleLambdaException('emult logic fell through somehow')

        e0 = fresnelprop_fp(e0, self.zlist[-1], nxfresnel,
                            self.fp_list[-1])

        return insertinto(e0, e0shape)

    def proptodm_fast_jac(self, yxLowerLeft, nSubarray, nSurf,
                          emult_list=None):
        """
        Propagate from the pupil before DMs to the one after.

        Only an offcenter sub-array of the full E-field is propagated in order
        to speed up the Jacobian.

        This function will Fresnel propagate ``e0`` forward by ``dmface.z``,
        create and multiply by DM phase, and Fresnel propagate back by
        ``dmface.z`` to the original plane.

        This function is streamlined for use with a Jacobian calculation, and
        expects the following precomputed data elements to be present:
         - self.dmph_list
         - self.zlist
         - self.fp_list
         - self.e_start
         - self.e_list
        Running ``self.get_jac_precomp(dmset_list)`` prior to invoking this
        function will populate these values correctly.

        Arguments:
         yxLowerLeft: 2-tuple containing the (y, x) array coordinates
          at which to insert the offcenter sub-array into a larger array
          having a shape of (nSurf, nSurf).
         nSubarray: width and height of the 2-D sub-arrays in emult_list
         nSurf: int. e0 goes back into a 2-D square array of shape
          (nSurf, nSurf) at the position given by yxLowerLeft.

        Keyword Arguments:
         emult_list: list of 2D arrays with the same center as `e0`` with any
          additional field multiplier at the DM surface, or ``None`` if you
          don't want to apply anything other that the phase effect of the DM.
          Primary expected use is the introduction of the differential field
          in Jacobian calculation. Defaults to ``None``.  Should be the same
          length as the number of DMs, and this will be checked.  Individual
          list elements may be None as well, if that DM is not going to get
          an additional multiplier.  It is recommended to oversize these with
          respect to e0 if they are open (ones) at the edges, as they will
          truncate out-of-pupil Fresnel terms otherwise.

        Returns:
         a 2D complex array of the same size as ``e0`` and located optically
          in the same plane

        """
        # check that pre-computed arrays have actually been computed
        if self.e_start is None:
            raise SingleLambdaException('self.e_start undefined prior to ' +
                                        'proptodm_fast_jac() call.')
        if self.e_list is None:
            raise SingleLambdaException('self.e_list undefined prior to ' +
                                        'proptodm_fast_jac() call.')
        if self.fp_list is None:
            raise SingleLambdaException('self.fp_list undefined prior to ' +
                                        'proptodm_fast_jac() call.')
        if self.zlist is None:
            raise SingleLambdaException('self.zlist undefined prior to ' +
                                        'proptodm_fast_jac() call.')
        if self.dmph_list is None:
            raise SingleLambdaException('self.dmph_list undefined prior to ' +
                                        'proptodm_fast_jac() call.')

        check.positive_scalar_integer(nSubarray, 'nSubarray', TypeError)
        check.positive_scalar_integer(nSurf, 'nSurf', TypeError)

        # Check inputs
        if emult_list is None:
            emult_list = [None]*len(self.dmlist)

        # Check list elements before any propagation begins
        for index in range(len(self.dmlist)):
            emult = emult_list[index]
            if emult is not None:
                check.twoD_square_array(emult, 'emult', TypeError)

        # Actually propagate now
        nxfresnel = self.nxfresnel_fast_jac
        emultyet = False
        for index in range(len(self.dmlist)):
            emult = emult_list[index]

            if emult is None and not emultyet:
                e0full = insertinto(self.e_list[index], (nSurf, nSurf))
                e0 = e0full[yxLowerLeft[0]:yxLowerLeft[0] + nSubarray,
                            yxLowerLeft[1]:yxLowerLeft[1] + nSubarray]
                continue
            elif emult is not None and not emultyet:
                e0full = insertinto(self.e_list[index], (nSurf, nSurf))
                e0 = e0full[yxLowerLeft[0]:yxLowerLeft[0] + nSubarray,
                            yxLowerLeft[1]:yxLowerLeft[1] + nSubarray]
                e0 *= insertinto(emult, e0.shape)
                emultyet = True
                continue
            elif emultyet:
                # Start ang spec
                e0 = fresnelprop_fp(e0, self.zlist[index], nxfresnel,
                                    self.fp_crop_list[index]
                                    ).astype('complex128')

                # 2x for surface --> wavefront
                dmphFull = insertinto(self.dmph_list[index], (nSurf, nSurf))
                e0 = insertinto(e0, (nSubarray, nSubarray))
                e0 *= dmphFull[yxLowerLeft[0]:yxLowerLeft[0] + nSubarray,
                               yxLowerLeft[1]:yxLowerLeft[1] + nSubarray]
                if emult is not None:
                    # In case of somehow more than one nonzero emult
                    e0 *= insertinto(emult, e0.shape)

            else:
                raise SingleLambdaException('emult logic fell through somehow')

        e0 = fresnelprop_fp(e0, self.zlist[-1], nxfresnel,
                            self.fp_crop_list[-1])

        return insertinto(e0, (nSubarray, nSubarray))

    def get_dmph_list(self, dmset_list, nxfresnel):
        """
        Get complex-exponential for each DM setting to save recalculation.

        Arguments:
         dmset_list: a list of DM settings, in voltage, currently applied to
          the DMs.  The DM sizes and ordering must match ``self.dmlist``.
         nxfresnel: the number of pixels to use in the Fresnel propagation,
          or None.  If None, uses the default chosen in ``self.nxfresnel_dm``.
          ``self.nxfresnel_dm`` has a good estimate, but may be chosen
          otherwise by user.  Should be a positive integer if not None.

        Returns:
         a list of complex-valued arrays which can be multiplied by a wavefront
          to produce a field representative of the field following a DM.

        """
        self._check_dmset_list(dmset_list)

        if nxfresnel is not None:
            check.positive_scalar_integer(nxfresnel, 'nxfresnel', TypeError)

        else:
            nxfresnel = self.nxfresnel_dm

        # We only want to propagate the *difference* between where DM was when
        # model data was taken and where we are now
        differential = [dmset_list[j]-self.initmaps[j]
                        for j in range(len(self.dmlist))]

        # Check list elements before any propagation begins
        for index, dm in enumerate(self.dmlist):
            dmset = differential[index]

            if not np.isrealobj(dmset):
                if (dmset.imag == 0).all():
                    # Let's not throw up if it's actually real in a complex
                    # type
                    dmset = dmset.real

                else:
                    raise TypeError('dmset must be real')

        temp = []
        for index, dm in enumerate(self.dmlist):
            dmset = differential[index]
            dmact_rad = dm.dmvobj.volts_to_dmh(dmset, self.lam)
            dmsurf = dmhtoph(nxfresnel, nxfresnel,
                             dmact_rad, **dm.registration)
            # 2x for surface --> wavefront
            temp.append(np.exp(1j*2.*dmsurf))

        return temp


    def get_inorm(self, dmset_list=None):
        """
        Get normalization factor with occulter out

        Normalized to a peak intensity of 1 in the final focal plane with the
        the occulter out (equivalent to off-axis PSF, or what a planet would
        look like)

        Normalization factor is for intensity; divide by sqrt of this to
        normalize amplitude.

        Keyword Arguments:
         dmset_list: a list of DM settings, in voltage, currently applied to
          the DMs, or ``None``.  The DM sizes and ordering must match
          ``self.dmlist``.  If ``None``, then a uniform DM setting of 0 is
          assumed.

        Returns
         intensity normalization factor as real scalar

        """

        # Check inputs
        if dmset_list is not None:
            self._check_dmset_list(dmset_list)

        else:
            # Build to match nominal model DM unless we want something specific
            dmset_list = self.initmaps

        # Copy as we don't want to modify original
        e0 = self.epup.e.copy()
        # Note re issue 52: do NOT multiply by ttph here!  This will shift the
        # star away from the DC component.  Courtesy @ajriggs

        e0 = self.proptodm(e0, dmset_list)
        e0 = self.proptolyot_nofpm(e0)
        e0 = self.proptodhpeak(e0)

        return np.abs(e0)**2

    def get_dmind1d(self, dmn, j, k):
        """
        Return a 1D index from a 2D DM specification

        Arguments:
         dmn: index of DM
         j: row index
         k: column index

        Returns
         index in 1D representation of all DMs

        """

        # check inputs
        check.nonnegative_scalar_integer(dmn, 'dmn', TypeError)
        check.nonnegative_scalar_integer(j, 'j', TypeError)
        check.nonnegative_scalar_integer(k, 'k', TypeError)

        if dmn >= len(self.dmlist):
            raise ValueError('DM number out of range')

        nact = self.dmlist[dmn].registration['nact']
        if j >= nact:
            raise ValueError('DM row index out of range')
        if k >= nact:
            raise ValueError('DM col index out of range')

        return self.ndmact[dmn] + j*nact + k


    def get_dmind2d(self, dmind):
        """
        Return a DM number and a row and column index given an index from a
        1D array.

        Used to go from Jacobian indexing (which spreads all DM actuators out
        along one edge of a matrix) to DM indexing (which has a 2D array for
        each DM corresponding to its physical location).

        Arguments:
         dmind: 1D nonnegative scalar integer less than the total number of
          actuators.

        Returns:
         3-tuple with (DM number [starts at 0], row index, column index)

        """

        check.nonnegative_scalar_integer(dmind, 'dmind', TypeError)
        if dmind >= self.ndmact[-1]:
            raise ValueError('1D DM index out of range')

        # ndmact are fenceposts, skip first one (=0)
        for index, dm in enumerate(self.dmlist):
            nact = dm.registration['nact']
            if dmind < nact**2:
                return (index, dmind // nact, dmind % nact)

            else:
                dmind -= nact**2

        raise SingleLambdaException('1D index unassigned, this should ' + \
                                    'never happen')


    def _check_dmset_list(self, dmset_list):
        """
        Check that a list of input DM settings is consistent with the model
        definition.

        This is a function for consistency/refactor purposes.

        Arguments:
         dmset_list: a list of DM settings, in voltage, currently applied to
          the DMs.  The DM sizes and ordering must match ``self.dmlist``.

        Returns:
         nothing

        """
        nactlist = [dm.registration['nact'] for dm in self.dmlist]
        try:
            if len(dmset_list) != len(nactlist):
                raise TypeError('dmset_list must be None ' + \
                                'or an iterable of the same length as dmlist')

        except TypeError: # not iterable
            raise TypeError('dmset_list must an iterable if it is not None')

        for index, dmset in enumerate(dmset_list):
            check.twoD_array(dmset, 'dmset', TypeError)
            if dmset.shape != (nactlist[index], nactlist[index]):
                raise TypeError('Item ' + str(index) +
                                ' of dmset_list must be the same size ' + \
                                'as the corresponding DM')

    def check_nsurf_pupildim(self):
        """
        Check DM is smaller than pupil

        Soft warning message to alert the user when this situation arises.
        Not fatal to any calculation

        Arguments:
         nothing

        Returns:
         nothing

        """

        reg_dict = self.dmlist[0].registration
        ppact_cx = reg_dict['ppact_cx']
        ppact_cy = reg_dict['ppact_cy']
        ppact_d = reg_dict['ppact_d']
        dx = reg_dict['dx']
        dy = reg_dict['dy']
        inf_func = reg_dict['inf_func']
        nact = reg_dict['nact']
        thact = reg_dict['thact']

        thactRad = thact*np.pi/180

        # Compute minimum array width to contain full DM surface.
        # Forced to be odd.
        nInf0 = inf_func.shape[0]
        maxppa = max(ppact_cx, ppact_cy)
        maxshift = max(np.abs(dx), np.abs(dy))
        # min width factor of square needed to contain rotated inf func
        wsquare = np.abs(np.cos(thactRad)) + np.abs(np.sin(thactRad))
        nSurf = ceil_odd(wsquare*(nact + nInf0/ppact_d)*maxppa + 2*maxshift)

        # pupil_dim:
        dmobj = self.dm1
        dmobj.pupil_shape = self.pupil.e.shape
        pupil_dim = np.max(dmobj.pupil_shape)

        if nSurf < pupil_dim:
            logging.warning(
                'The DM surface (nsurf=%d) is smaller than the pupil mask %d.',
                nSurf, pupil_dim
            )
            # dmhtoph will automatically use nSurf = ceil_odd(pupil_dim)
