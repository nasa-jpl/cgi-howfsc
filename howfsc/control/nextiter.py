# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Functions to set up data for use during the next iteration

Includes:
 - Compute expected mean total contrast for next iteration
 - Compute expected time to complete next iteration
 - Compute probe scale factors

May eventually include:
 - Compute absolute DM settings
 - Compute camera settings

"""

import numpy as np

from howfsc.model.mode import CoronagraphMode
from howfsc.util.insertinto import insertinto
import howfsc.util.check as check

valid_method_list = ['mean', 'percentile']

def get_next_c(cfg, dmlist, croplist, fixedbp, n2clist, destlist,
               cleanrow=1024, cleancol=1024, method='mean', percentile=50,
               index_list=None, fullframe=False):
    """
    Compute expected mean total contrast for next iteration, or optionally a
    a different metric (for use in selecting camera settings)

    This computation takes in the (measured - prev model estimate) for each
    wavelength, and adds the new model estimate to it.  (This should also
    capture any nonlinearities incorporated into the model.)

    This function will evaluate contrast over the pixels in the dark hole
    which are not included in the fixed bad pixel map and which are not
    flagged as bad electric-field estimates in the measured data.  (Bad
    electric-field estimates will appear as NaNs in destlist.)

    Note: when using this tool to compute the 'max', it is recommended to use
    method=percentile with a high percentile value (e.g. 99) but not all the
    way to 100th to avoid getting fooled by any outlier cosmic rays that might
    escape earlier processing.  These will be read from the hconf file when
    running this from GITL.

    Arguments:
     cfg: CoronagraphMode object
     dmlist: list of ndarrays, of the same size as the arrays expected by
      cfg.dmlist objects. These are DM voltages.  This should have the same
      number of DMs as the model.
     croplist: list of 4-tuples of (lower row, lower col,
      number of rows, number of columns), indicating where in a clean frame
      each PSF is taken.  All are integers; the first two must be >= 0 and the
      second two must be > 0.  This should have the same number of elements as
      the model has wavelengths.
     fixedbp: this is a fixed bad pixel map for a clean frame, and will be a 2D
      boolean array with the same size as a cleaned frame.
     n2clist: list of 2D floating-point arrays giving the scale factor to
      convert from normalized intensity to contrast as a multiplier.  These
      correspond to the relative drop-off in flux from unity due to the
      presence of nearby coronagraphic masks.

      As an example, if the measured normalized intensity is 5e-9, but the
      presence of a mask edge is causing the flux to drop to 0.5x of usual,
      then the value in an array of n2clist will be 2, and the actual contrast
      will be 1e-8.

      Elements of n2clist must be at least as large as the number of rows and
      columns in a cropped region (elements 2 and 3 of each croplist tuple) to
      ensure that every valid pixel has a conversion factor to contrast.  This
      should have the same number of elements as the model has wavelengths.
     destlist: complex-valued 2D array of the same shape as
      cfg.sl_list[lind].dh.e for each lind (wavelength) in the CoronagraphMode.
      These will be (measured - previous-model).

    Keyword Arguments:
     cleanrow: Number of rows in a clean frame.  Integer > 0.  Defaults to
      1024, the number of active area rows on the EXCAM detector; under nominal
      conditions, there should be no reason to use anything else.
     cleancol: Number of columns in clean frame. Integer > 0.  Defaults to
      1024, the number of active area cols on the EXCAM detector; under nominal
      conditions, there should be no reason to use anything else.
     method: string containing one of the names in valid_method_list. Currently
      only 'mean' and 'percentile' are supported.  mean will return the mean
      over all good pixels, and is the default.
     percentile: Float between 0 and 100 inclusive, indicating which percentile
      of flux across all good pixels to return.  50 is the median over all good
      pixels, and is the default.  This argument is unused if the 'method'
      keyword argument is not 'percentile'.
     index_list: list of integer indices, or None, indicating which of the
      wavelengths in cfg will be used for the calculation.  If None, all
      wavelengths will be used.
     fullframe: Boolean.  If True, will use the entire cropped frame to do the
      calculation instead constraining to the dark-hole region.  If False, will
      use the dark hole only.  Either way, bad pixels are still ignored.
      Defaults to False.  Expected use case is to get the bright pixel level in
      the image associated with scale_bright for saturation prevention.

    Returns:
     a single scalar contrast value

    """

    # Check inputs
    if not isinstance(cfg, CoronagraphMode):
        raise TypeError('Input model must be a CoronagraphMode object')

    if not isinstance(dmlist, list):
        raise TypeError('dmlist must be a list')
    if len(dmlist) != len(cfg.dmlist):
        raise TypeError('Number of DMs does not match model')
    for index, dm in enumerate(dmlist):
        check.twoD_array(dm, 'dm', TypeError)
        nact = cfg.dmlist[index].registration['nact']
        if dm.shape != (nact, nact):
            raise TypeError('DM dimensions do not match model')
        pass

    if not isinstance(croplist, list):
        raise TypeError('croplist must be a list')
    if len(croplist) != len(cfg.sl_list):
        raise TypeError('Number of crop regions does not match model')

    for index, crop in enumerate(croplist):
        if not isinstance(crop, tuple):
            raise TypeError('croplist[' + str(index) + '] must be a tuple')
        if len(crop) != 4:
            raise TypeError('Each element of croplist must be a 4-tuple')
        check.nonnegative_scalar_integer(crop[0], 'croplist[' +
                                         str(index) + '][0]', TypeError)
        check.nonnegative_scalar_integer(crop[1], 'croplist[' +
                                         str(index) + '][1]', TypeError)
        check.positive_scalar_integer(crop[2], 'croplist[' +
                                      str(index) + '][2]', TypeError)
        check.positive_scalar_integer(crop[3], 'croplist[' +
                                      str(index) + '][3]', TypeError)
        pass

    check.positive_scalar_integer(cleanrow, 'cleanrow', TypeError)
    check.positive_scalar_integer(cleancol, 'cleancol', TypeError)
    check.twoD_array(fixedbp, 'fixedbp', TypeError)
    if fixedbp.shape != (cleanrow, cleancol):
        raise TypeError('fixedbp must be the same size as a cleaned frame')
    if fixedbp.dtype != bool:
        raise TypeError('fixedbp must be boolean')

    if not isinstance(n2clist, list):
        raise TypeError('n2clist must be a list')
    if len(n2clist) != len(cfg.sl_list):
        raise TypeError('Number of NI-to-contrast conversion matrices '+
                        'does not match model')
    for index, n2c in enumerate(n2clist):
        check.twoD_array(n2c, 'n2clist[' + str(index) + ']', TypeError)
        if n2c.shape[0] < croplist[index][2]:
            raise TypeError('Number of rows in ' +
                            'n2clist[' + str(index) + '] must be at least ' +
                            'as large as croplist[' + str(index) + '][2]')
        if n2c.shape[1] < croplist[index][3]:
            raise TypeError('Number of columns in ' +
                            'n2clist[' + str(index) + '] must be at least ' +
                            'as large as croplist[' + str(index) + '][3]')
        pass

    if not isinstance(destlist, list):
        raise TypeError('destlist must be a list')
    if len(destlist) != len(cfg.sl_list):
        raise TypeError('destlist must have the same number of elements as ' +
                        'cfg has wavelengths')
    for index, dest in enumerate(destlist):
        check.twoD_array(dest, 'destlist[' + str(index) + ']', TypeError)
        if dest.shape != cfg.sl_list[index].dh.e.shape:
            raise TypeError('destlist sizes must match cfg')
        pass

    if method not in valid_method_list:
        raise TypeError('method must be in the following list: ' +
                        repr(valid_method_list))
    check.real_scalar(percentile, 'percentile', TypeError)
    if percentile < 0 or percentile > 100:
        raise ValueError('percentile must be in the range [0, 100] inclusive')

    if index_list is None:
        index_list = [j for j in range(len(cfg.sl_list))]

    if not isinstance(index_list, list):
        raise TypeError('index_list must be a list')
    if len(index_list) == 0:
        raise TypeError('index_list must not be an empty list')
    for index in index_list:
        check.nonnegative_scalar_integer(index,
                                         'index_list[' + str(index) + ']',
                                         TypeError)
        if index >= len(cfg.sl_list):
            raise TypeError('indices in index_list cannot be larger ' +
                            'than the number of wavelengths in cfg')
        pass

    check.boolean(fullframe, 'fullframe', TypeError)

    #----------------------
    # Contrast computation
    #----------------------

    dharray = np.array([])

    for index in index_list:
        sl = cfg.sl_list[index]
        nrow = croplist[index][2]
        ncol = croplist[index][3]

        # use croplist to get fixed bp map for that frame
        # handle case where crop region falls off detector by making those
        # "pixels" bad (i.e. True)
        slbp = np.ones((nrow, ncol)).astype('bool')
        chunk = fixedbp[croplist[index][0]:croplist[index][0]+nrow,
                        croplist[index][1]:croplist[index][1]+ncol]
        slbp[:chunk.shape[0], :chunk.shape[1]] = chunk

        # Compute e-field
        edm = sl.eprop(dmlist)
        ely = sl.proptolyot(edm)
        edh = sl.proptodh(ely)

        # Put everything else in a crop-sized array
        edh_crop = insertinto(edh + destlist[index], (nrow, ncol))
        if fullframe:
            dh_crop = np.ones((nrow, ncol), dtype=sl.dh.e.dtype)
        else:
            dh_crop = insertinto(sl.dh.e, (nrow, ncol))
        n2c = insertinto(n2clist[index], (nrow, ncol))

        # convert to NI
        nim = np.abs(edh_crop)**2

        # accumulate for mean, only over good pixels
        control = np.logical_and(np.logical_and(dh_crop, ~slbp),
                                 ~np.isnan(nim))
        dharray = np.append(dharray, nim[control]*n2c[control].ravel())

        pass

    if dharray.size == 0:
        raise ZeroDivisionError('No valid pixels in any frame')

    if method == 'mean':
        return np.mean(dharray)
    elif method == 'percentile':
        return np.percentile(dharray, percentile)
    else: # should never reach here
        return Exception('reached end with an invalid method despite checks')


def expected_time(ndm, nfilt, exptime, nframes, overdm, overfilt, overboth,
                  overfixed, overframe):
    """
    Utility function to estimate how long a iteration is expected to take

    This function is a broad parametric estimate, to be provided to SSC to
    determine whether to continue iterating HOWFSC.

    Note: this estimator might be revisited and refined once the iteration
    procedure is better known, if some better degree of accuracy is necessary.

    This function assumes that each element of exptime and of nframes with the
    same index are associated with the same exposure.

    total time is:
     fixed overhead + ndm*(overhead per DM) + nfilt*(overhead per filter)
     + ndm*nfilt*(overhead per combo) + elementwise_sum([exposure time +
      overhead per frame]*[number of frames])

    Arguments:
     ndm: number of total DM positions to use in an iteration. Generally will
      be odd (e.g. one unprobed, 6 probes implemented as 3 pairs for 7 total).
      In any event, must be an integer > 0.
     nfilt: number of total CFAM filters to use in an iteration. Generally will
      be 3 (NFOV, WFOV) or 5 (SPEC).  In any event, must be an integer > 0.
     exptime: 1D array with a list of floating point values > 0.  These are
      exposure times at each DM setting/CFAM filter setting combination.  Must
      be length ndm*nfilt.  Units of seconds.
     nframes: 1D array with a list of integer values > 0.  There are the
      numbers of frames to be collected at each DM setting/CFAM filter setting
      combination.  Must be length ndm*nfilt.
     overdm: overhead associated with a DM move, such as the time to command
      DME1.  Must be a floating-point scalar >= 0.  Units of seconds.
     overfilt: overhead associated with a CFAM filter move, such as the time
      to translate CFAM.  Must be a floating-point scalar >= 0.  Units of
      seconds.
     overboth: overhead associated with each DM setting/CFAM filter setting
      combination, such as the time to apply new camera settings.  Must be a
      floating-point scalar >= 0.  Units of seconds.  This should *not* include
      any times already covered by overdm or overfilt, or else they will be
      double-counted.
     overfixed: fixed overhead time for doing an iteration, for tasks such as
      setting DM2.  Must be a floating-point scalar >= 0.  Units of seconds.
     overframe: overhead per individual frame for readout.  Must be a floating-
      point scalar >= 0.  Units of seconds.

    Returns:
     single scalar value in seconds

    """

    # Check inputs
    check.positive_scalar_integer(ndm, 'ndm', TypeError)
    check.positive_scalar_integer(nfilt, 'nfilt', TypeError)
    check.real_nonnegative_scalar(overdm, 'overdm', TypeError)
    check.real_nonnegative_scalar(overfilt, 'overfilt', TypeError)
    check.real_nonnegative_scalar(overboth, 'overboth', TypeError)
    check.real_nonnegative_scalar(overfixed, 'overfixed', TypeError)
    check.real_nonnegative_scalar(overframe, 'overframe', TypeError)

    try:
        llen = len(exptime)
    except TypeError: # not iterable
        raise TypeError('exptime must be iterable')
    if llen != ndm*nfilt:
        raise TypeError('exptime must be of length ndm*nfilt')

    try:
        llen = len(nframes)
    except TypeError: # not iterable
        raise TypeError('nframes must be iterable')
    if llen != ndm*nfilt:
        raise TypeError('nframes must be of length ndm*nfilt')

    for index, elem in enumerate(exptime):
        check.real_positive_scalar(elem, 'exptime[' + str(index) + ']',
                                      TypeError)
        pass
    for index, elem in enumerate(nframes):
        check.positive_scalar_integer(elem, 'nframes[' + str(index) + ']',
                                      TypeError)
        pass

    # actual calc
    camtotal = np.sum((np.asarray(exptime) + overframe)*np.asarray(nframes))
    overtotal = overfixed + ndm*overdm + nfilt*overfilt + ndm*nfilt*overboth
    return camtotal + overtotal


def get_scale_factor_list(dmrel_ph_list, current_ph):
    """
    Given a characteristic probe height for a list of settings, return a scale
    factor to use for the next iteration.

    What we actually NEED is the probe height P0 associated with a relative DM
    setting at scale = 1.  The scale factor for the next iteration is derived
    from the control-strategy probe height Pc as scale = sqrt(PC/P0), repeated
    across the three relative DM settings (which each has their own probe
    height).

    Scale factors scale the DM's probe commands linearly, and the probe
    intensity contributes quadratically.

    We use two different variables as the probe height is the right value for
    a user to target a specific illumination level, but the scale is necessary
    to be used with an FSW parameter.

    Arguments:
     dmrel_ph_list: a list of probe heights, each a floating-point value > 0
      representing the probe height equivalent of a relative DM setting with
      scale = 1.
     current_ph: requested probe height from control strategy, floating-point
      value > 0

    Returns:
     list of scale factors (floating-point scalars > 0) twice the length of
      dmrel_ph_list.  Second half will be the negatives of the first half; we
      are making these explicitly so that it is easy to fill parameters later.
      CGI has separate FSW parameters for positive and negative scales.

    """

    # Check inputs
    if not isinstance(dmrel_ph_list, list):
        raise TypeError('dmrel_ph_list must be a list')
    for index, dmrel_ph in enumerate(dmrel_ph_list):
        check.real_positive_scalar(dmrel_ph, 'dmrel_ph_list[' + str(index) +
                                   ']', TypeError)
        pass
    check.real_positive_scalar(current_ph, 'current_ph', TypeError)

    #--------
    # Positives
    scale_factor_list = []
    for dmrel_ph in dmrel_ph_list:
        scale = np.sqrt(current_ph/dmrel_ph)
        scale_factor_list.append(scale)
        pass
    # Negatives
    for dmrel_ph in dmrel_ph_list:
        scale = np.sqrt(current_ph/dmrel_ph)
        scale_factor_list.append(-scale)
        pass

    return scale_factor_list
