# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
# pylint: disable=line-too-long
"""
Functions to build a relative DM probe and other propagation manipulation
"""
import copy
import numpy as np

from howfsc.model.mode import CoronagraphMode
from howfsc.util.constrain_dm import tie_with_matrix
from howfsc.util.dmshapes import probe, probe_gaussian
from howfsc.util.dmhtoph import dmhtoph
from howfsc.util.insertinto import insertinto
import howfsc.util.check as check

def efield(cfg, dmlist, ind):
    """
    Electric field at the focal plane in channel with index ind

    Arguments:
     cfg: CoronagraphMode object
     dmlist: list of DMs for a current DM setting.  2 ndarrays; should be DM1
      and DM2, in that order.  Size should match the number of actuators
      defined in cfg.
     ind: index of cfg wavelength to use for model.  Must be a nonnegative
      integer less than the number of wavelengths in cfg.

    Returns
     2D electric field at focal plane

    """
    # Check inputs
    # cfg
    if not isinstance(cfg, CoronagraphMode):
        raise TypeError('cfg must be a CoronagraphMode object')

    # dmlist
    try:
        lendmlist = len(dmlist)
    except TypeError: # not iterable
        raise TypeError('dmlist must be an iterable') # reraise
    if lendmlist != 2:
        raise TypeError('dmlist must contain 2 DM arrays')
    for index, dm in enumerate(dmlist):
        check.twoD_array(dm, f'dmlist[{index}]', TypeError)
        pass
    dm1nact = cfg.dmlist[0].registration['nact']
    if dmlist[0].shape != (dm1nact, dm1nact):
        raise TypeError('First element of dmlist does not match cfg')
    dm2nact = cfg.dmlist[1].registration['nact']
    if dmlist[1].shape != (dm2nact, dm2nact):
        raise TypeError('Second element of dmlist does not match cfg')

    # ind
    check.nonnegative_scalar_integer(ind, 'ind', TypeError)
    if ind >= len(cfg.sl_list):
        raise TypeError('ind must be less than the number of channels in cfg')

    sl = cfg.sl_list[ind]
    edm0 = sl.eprop(dmlist)
    ely = sl.proptolyot(edm0)
    edh0 = sl.proptodh(ely)
    return edh0


def open_efield(cfg, dmlist, ind):
    """
    Unocculted electric field at the focal plane in channel with index ind

    Arguments:
     cfg: CoronagraphMode object
     dmlist: list of DMs for a current DM setting.  2 ndarrays; should be DM1
      and DM2, in that order.  Size should match the number of actuators
      defined in cfg.
     ind: index of cfg wavelength to use for model.  Must be a nonnegative
      integer less than the number of wavelengths in cfg.

    Returns
     2D electric field at focal plane (no FPM)

    """
    # Check inputs
    # cfg
    if not isinstance(cfg, CoronagraphMode):
        raise TypeError('cfg must be a CoronagraphMode object')

    # dmlist
    try:
        lendmlist = len(dmlist)
    except TypeError: # not iterable
        raise TypeError('dmlist must be an iterable') # reraise
    if lendmlist != 2:
        raise TypeError('dmlist must contain 2 DM arrays')
    for index, dm in enumerate(dmlist):
        check.twoD_array(dm, f'dmlist[{index}]', TypeError)
        pass
    dm1nact = cfg.dmlist[0].registration['nact']
    if dmlist[0].shape != (dm1nact, dm1nact):
        raise TypeError('First element of dmlist does not match cfg')
    dm2nact = cfg.dmlist[1].registration['nact']
    if dmlist[1].shape != (dm2nact, dm2nact):
        raise TypeError('Second element of dmlist does not match cfg')

    # ind
    check.nonnegative_scalar_integer(ind, 'ind', TypeError)
    if ind >= len(cfg.sl_list):
        raise TypeError('ind must be less than the number of channels in cfg')

    # Make the field stop all ones so that the stellar peak is not blocked.
    # Do in a copy so that no other usage is affected.
    cfgNoFS = copy.deepcopy(cfg)
    sl = cfgNoFS.sl_list[ind]
    sl.fs.e = np.ones_like(sl.fs.e)
    edm0 = sl.eprop(dmlist)
    ely = sl.proptolyot_nofpm(edm0)
    edh0 = sl.proptodh(ely)

    return edh0


def model_pm0(cfg, dm0, dmplus, dmminus, otherdm, ind, swap_dms=False):
    """
    Given DM settings corresponding to +/-/0 pokes, compute the model-based
    intensity residual (I+ + I-)/2 - I0

    Can be used for satellite spots, but also for DM probes or any other
    setting applied with a positive relative motion, negative relative motion
    of the same shape, and zero relative motion.

    Arguments:
     cfg: CoronagraphMode object
     dm0: Absolute DM setting for zero relative motion, nominally on DM1.
      2D ndarray whose size is consistent with cfg.
     dmplus: Absolute DM setting for positive relative motion, nominally on
      DM1.  2D ndarray whose size is consistent with cfg.
     dmminus: Absolute DM setting for negative relative motion, nominally on
      DM1.  2D ndarray whose size is consistent with cfg.
     otherdm: Absolute DM setting for the other DM which doesn't move,
      nominally on DM2.  ndarray whose size is consistent with cfg.
     ind: index of cfg wavelength to use for model.  Must be a nonnegative
      integer less than the number of wavelengths in cfg.

    Keyword Arguments:
     swap_dms: Boolean.  If True, models the motion on DM2 instead of DM1.
      Defaults to False (i.e. relative motions on DM1)

    Returns:
     2D ndarray with a model-based intensity residual from calculating
       (I+ + I-)/2 - I0

    """
    # Check inputs
    if not isinstance(cfg, CoronagraphMode):
        raise TypeError('cfg must be a CoronagraphMode object')
    check.twoD_array(dm0, 'dm0', TypeError)
    check.twoD_array(dmplus, 'dmplus', TypeError)
    check.twoD_array(dmminus, 'dmminus', TypeError)
    check.twoD_array(otherdm, 'otherdm', TypeError)

    if swap_dms:
        move = 1
        fixed = 0
    else:
        move = 0
        fixed = 1

    movenact = cfg.dmlist[move].registration['nact']
    if dm0.shape != (movenact, movenact):
        raise TypeError('dm0 does not match cfg')
    if dmplus.shape != (movenact, movenact):
        raise TypeError('dmplus does not match cfg')
    if dmminus.shape != (movenact, movenact):
        raise TypeError('dm0 does not match cfg')

    fixednact = cfg.dmlist[fixed].registration['nact']
    if otherdm.shape != (fixednact, fixednact):
        raise TypeError('otherdm does not match cfg')

    check.nonnegative_scalar_integer(ind, 'ind', TypeError)
    if ind >= len(cfg.sl_list):
        raise TypeError('ind must be less than the number of channels in cfg')

    # Prep common inputs
    dmlist = [None, None]
    dmlist[fixed] = otherdm
    sl = cfg.sl_list[ind]

    # Note to future code maintainers: the math-checking part of the test
    # expects the order 0, +, - for proptodh calls.  If you need to change
    # this, be sure to change the tests as well.

    # 0
    dmlist[move] = dm0
    edh0 = sl.proptodh(sl.proptolyot(sl.eprop(dmlist)))

    # +
    dmlist[move] = dmplus
    edhp = sl.proptodh(sl.proptolyot(sl.eprop(dmlist)))

    # -
    dmlist[move] = dmminus
    edhm = sl.proptodh(sl.proptolyot(sl.eprop(dmlist)))

    return (np.abs(edhp)**2 + np.abs(edhm)**2)/2 - np.abs(edh0)**2


def make_dmrel_probe(cfg, dmlist, dact, xcenter, ycenter, clock, ximin, ximax,
                     etamin, etamax, phase, target, lod_min, lod_max,
                     ind, maxiter=5, verbose=True):
    """
    Make a relative DM probe setting whose probe height is equal to an input

    This function creates a sinc*sinc*sin probe, and adjusts the height of that
    probe iteratively until |probe amplitude|**2 within a user-specified region
    approaches a user-specified target.

    User region is annular, from lod_min to lod_max

    If using verbose: the first iteration should be off the 'target' input by
    no more than a factor of low O(1).  If off by 1-2 orders of magnitude from
    target, check the xcenter and ycenter and verify that the probe peak is
    not blocked by a pupil obscuration.

    For the three 'typical' probes:
     - use clock = 0, phase = 90 for cos
     - use clock = 0, phase = 0 for sinlr
     - use clock = 90, phase = 0 for sinud
    Clock all three together by the same amount for rotated probes.

    To probe a square region, it is recommended to make one of xi/eta go from
    0 to X and the other from -X to X.

    Arguments:
     cfg: CoronagraphMode object
     dmlist: list of DMs for a current DM setting.
     dact: diameter of pupil, in actuators. > 0.
     xcenter: number of actuators to move the center of the DM pattern along
      the positive x-axis, as seen from the camera.  Negative and fractional
      inputs are acceptable.
     ycenter: number of actuators to move the center of the DM pattern along
      the positive y-axis, as seen from the camera.  Negative and fractional
      inputs are acceptable.
     clock: angle in degrees to rotate the DM pattern about its center.  This
      rotation is clockwise and is applied before the xcenter/ycenter
      shifts are applied.
     ximin: min lambda/D along the x-axis in the focal plane for the probe
      rectangle, must be less than ximax
     ximax: max lambda/D along the x-axis in the focal plane for the probe
      rectangle
     etamin: min lambda/D along the y-axis in the focal plane for the probe
      rectangle, must be less than etamax
     etamax: max lambda/D along the y-axis in the focal plane for the probe
      rectangle
     phase: phase angle in degrees to shift this particular probe; at phase = 0
      the modulation will be a sine and at phase = 90 it will be a cosine.
     target: desired probe intensity (i.e. |probe amplitude|**2) within the
      focal plane region of interest).  > 0.
     lod_min: minimum L/D for region of interest, must be less than lod_max.
      >= 0.
     lod_max: maximum L/D for region of interest.  > 0.
     ind: index of cfg wavelength to use for model

    Keyword Arguments:
     maxiter: number of times to iterate on DM setting.  integer > 0.
      Defaults to 5.
     verbose: if True, prints status to command line.  Defaults to True.

    Returns:
     tuple with:
      - DM1 relative DM setting, in volts
      - probe normalized-intensity 2D array in the focal plane
      - boolean mask used to select pixels to evaluate in the focal plane
      - a map of the DM surface in radians in the pupil plane
      - the product of the amplitudes of 'epup', 'pupil', and 'lyot', which
        show the region of physical obscuration in the pupil plane.  Note if
        these masks are different sizes, the output along each dimension will
        be the largest of the three on that dimension.
      The DM surface map in radians and the product of amplitudes will have the
      same array size.

    """
    # Check inputs
    if not isinstance(cfg, CoronagraphMode):
        raise TypeError('cfg must be a CoronagraphMode object')

    try:
        lendmlist = len(dmlist)
    except TypeError: # not iterable
        raise TypeError('dmlist must be an iterable') # reraise
    if lendmlist != 2:
        raise TypeError('dmlist must contain 2 DM arrays')
    for index, dm in enumerate(dmlist):
        check.twoD_array(dm, f'dmlist[{index}]', TypeError)
        pass
    dm1nact = cfg.dmlist[0].registration['nact']
    if dmlist[0].shape != (dm1nact, dm1nact):
        raise TypeError('First element of dmlist does not match cfg')
    dm2nact = cfg.dmlist[1].registration['nact']
    if dmlist[1].shape != (dm2nact, dm2nact):
        raise TypeError('Second element of dmlist does not match cfg')

    check.real_positive_scalar(dact, 'dact', TypeError)
    check.real_scalar(xcenter, 'xcenter', TypeError)
    check.real_scalar(ycenter, 'ycenter', TypeError)
    check.real_scalar(clock, 'clock', TypeError)
    check.real_scalar(ximin, 'ximin', TypeError)
    check.real_scalar(ximax, 'ximax', TypeError)
    check.real_scalar(etamin, 'etamin', TypeError)
    check.real_scalar(etamax, 'etamax', TypeError)
    check.real_scalar(phase, 'phase', TypeError)
    check.real_positive_scalar(target, 'target', TypeError)
    check.real_nonnegative_scalar(lod_min, 'lod_min', TypeError)
    check.real_positive_scalar(lod_max, 'lod_max', TypeError)
    check.nonnegative_scalar_integer(ind, 'ind', TypeError)
    if ximin >= ximax:
        raise ValueError('ximin must be strictly less than ximax')
    if etamin >= etamax:
        raise ValueError('etamin must be strictly less than etamax')
    if lod_min >= lod_max:
        raise ValueError('lod_min must be strictly less than lod_max')
    if ind >= len(cfg.sl_list):
        raise TypeError('ind must be less than the number of channels in cfg')

    check.positive_scalar_integer(maxiter, 'maxiter', TypeError)
    if not isinstance(verbose, bool):
        raise TypeError('verbose must be a boolean')


    scale = 1 # first guess: assume we got the amplitude right
    dind = 0 # only using DM1 to probe

    iopen = np.abs(open_efield(cfg, dmlist, ind))**2
    ipeak = np.max(iopen)

    ppl = cfg.sl_list[ind].dh.pixperlod
    nrow, ncol = cfg.sl_list[ind].dh.e.shape
    rld, cld = np.meshgrid((np.arange(nrow) - nrow//2)/ppl,
                           (np.arange(ncol) - ncol//2)/ppl,
                           indexing='ij')
    lod_mask = np.logical_and(np.sqrt(rld**2 + cld**2) >= lod_min,
                          np.sqrt(rld**2 + cld**2) <= lod_max)
    lod_mask = np.logical_and(lod_mask,
                              cfg.sl_list[ind].dh.e) # only keep valid pix

    j = 0
    measph = target # no effect on first iteration
    while j < maxiter:
        scale = scale/measph*target

        # Since we're iterating to get the probe amplitude right, don't bother
        # trying to account for any other scalar factors
        dp0 = probe(cfg.dmlist[dind].registration['nact'],
                    dact,
                    xcenter,
                    ycenter,
                    clock,
                    np.sqrt(scale),
                    ximin,
                    ximax,
                    etamin,
                    etamax,
                    phase, #90, # always do cosine first, since there's no nulls
                    )
        dp1 = tie_with_matrix(dp0, cfg.dmlist[dind].dmvobj.tiemap)
        dpv = cfg.dmlist[dind].dmvobj.dmh_to_volts(dp1, cfg.sl_list[ind].lam)

        eplus = efield(cfg, [dmlist[0]+dpv, dmlist[1]], ind)
        eminus = efield(cfg, [dmlist[0]-dpv, dmlist[1]], ind)

        pampe = np.abs((eplus - eminus)/2j)
        probe_int = pampe**2/ipeak

        measph = np.mean(probe_int[lod_mask])
        if verbose:
            print("iteration " + str(j) + ": measured = " + str(measph) +
                  ", fractional error: " + str((measph-target)/target))

        j += 1
        pass

    # Redo with final scale, actual input phase
    dp0 = probe(cfg.dmlist[dind].registration['nact'],
                dact,
                xcenter,
                ycenter,
                clock,
                np.sqrt(scale),
                ximin,
                ximax,
                etamin,
                etamax,
                phase,
                )
    dp1 = tie_with_matrix(dp0, cfg.dmlist[dind].dmvobj.tiemap)
    dpv = cfg.dmlist[dind].dmvobj.dmh_to_volts(dp1, cfg.sl_list[ind].lam)

    # create additional data products to help users
    epups = cfg.sl_list[ind].epup.e.shape
    sps = cfg.sl_list[ind].pupil.e.shape
    lys = cfg.sl_list[ind].lyot.e.shape
    nrow = max(epups[0], sps[0], lys[0])
    ncol = max(epups[1], sps[1], lys[1])

    dm_surface = dmhtoph(
        nrow=nrow,
        ncol=ncol,
        dmin=dpv,
        nact=cfg.dmlist[dind].registration['nact'],
        inf_func=cfg.dmlist[dind].registration['inf_func'],
        ppact_d=cfg.dmlist[dind].registration['ppact_d'],
        ppact_cx=cfg.dmlist[dind].registration['ppact_cx'],
        ppact_cy=cfg.dmlist[dind].registration['ppact_cy'],
        dx=cfg.dmlist[dind].registration['dx'],
        dy=cfg.dmlist[dind].registration['dy'],
        thact=cfg.dmlist[dind].registration['thact'],
        flipx=cfg.dmlist[dind].registration['flipx'],
    )

    pupil_mask = (
        insertinto(np.abs(cfg.sl_list[ind].epup.e), (nrow, ncol))*
        insertinto(np.abs(cfg.sl_list[ind].pupil.e), (nrow, ncol))*
        insertinto(np.abs(cfg.sl_list[ind].lyot.e), (nrow, ncol))
    )

    return dpv, probe_int, lod_mask, dm_surface, pupil_mask


def make_dmrel_probe_gaussian(cfg, dmlist, xcenter, ycenter, sigma, target, lod_min, lod_max,
                              ind, maxiter=5, verbose=True):
    """
    Make a relative DM probe setting whose probe height is equal to an input.

    This function creates a Gaussian probe, and adjusts the height of that
    probe iteratively until |probe amplitude|**2 within a user-specified region
    approaches a user-specified target.

    User region is annular, from lod_min to lod_max

    If using verbose: the first iteration should be off the 'target' input by
    no more than a factor of low O(1).  If off by 1-2 orders of magnitude from
    target, check the xcenter and ycenter and verify that the probe peak is
    not blocked by a pupil obscuration.

    Arguments:
     cfg: CoronagraphMode object
     dmlist: list of DMs for a current DM setting.
     xcenter: number of actuators to move the center of the DM pattern along
      the positive x-axis, as seen from the camera.  Negative and fractional
      inputs are acceptable.
     ycenter: number of actuators to move the center of the DM pattern along
      the positive y-axis, as seen from the camera.  Negative and fractional
      inputs are acceptable.
     sigma: width of Gaussian probe, in actuators.  > 0.
     target: desired probe intensity (i.e. |probe amplitude|**2) within the
      focal plane region of interest).  > 0.
     lod_min: minimum L/D for region of interest, must be less than lod_max.
      >= 0.
     lod_max: maximum L/D for region of interest.  > 0.
     ind: index of cfg wavelength to use for model

    Keyword Arguments:
     maxiter: number of times to iterate on DM setting.  integer > 0.
      Defaults to 5.
     verbose: if True, prints status to command line.  Defaults to True.

    Returns:
     tuple with:
      - DM1 relative DM setting, in volts
      - probe normalized-intensity 2D array in the focal plane
      - boolean mask used to select pixels to evaluate in the focal plane
      - a map of the DM surface in radians in the pupil plane
      - the product of the amplitudes of 'epup', 'pupil', and 'lyot', which
        show the region of physical obscuration in the pupil plane.  Note if
        these masks are different sizes, the output along each dimension will
        be the largest of the three on that dimension.
      The DM surface map in radians and the product of amplitudes will have the
      same array size.

    """
    # Check inputs
    if not isinstance(cfg, CoronagraphMode):
        raise TypeError('cfg must be a CoronagraphMode object')

    try:
        lendmlist = len(dmlist)
    except TypeError: # not iterable
        raise TypeError('dmlist must be an iterable') # reraise
    if lendmlist != 2:
        raise TypeError('dmlist must contain 2 DM arrays')
    for index, dm in enumerate(dmlist):
        check.twoD_array(dm, f'dmlist[{index}]', TypeError)
        pass
    dm1nact = cfg.dmlist[0].registration['nact']
    if dmlist[0].shape != (dm1nact, dm1nact):
        raise TypeError('First element of dmlist does not match cfg')
    dm2nact = cfg.dmlist[1].registration['nact']
    if dmlist[1].shape != (dm2nact, dm2nact):
        raise TypeError('Second element of dmlist does not match cfg')

    check.real_scalar(xcenter, 'xcenter', TypeError)
    check.real_scalar(ycenter, 'ycenter', TypeError)
    check.real_positive_scalar(sigma, 'sigma', TypeError)
    check.real_positive_scalar(target, 'target', TypeError)
    check.real_nonnegative_scalar(lod_min, 'lod_min', TypeError)
    check.real_positive_scalar(lod_max, 'lod_max', TypeError)
    check.nonnegative_scalar_integer(ind, 'ind', TypeError)
    if lod_min >= lod_max:
        raise ValueError('lod_min must be strictly less than lod_max')
    if ind >= len(cfg.sl_list):
        raise TypeError('ind must be less than the number of channels in cfg')

    check.positive_scalar_integer(maxiter, 'maxiter', TypeError)
    if not isinstance(verbose, bool):
        raise TypeError('verbose must be a boolean')

    scale = 1 # first guess: assume we got the amplitude right
    dind = 0 # only using DM1 to probe

    iopen = np.abs(open_efield(cfg, dmlist, ind))**2
    ipeak = np.max(iopen)

    ppl = cfg.sl_list[ind].dh.pixperlod
    nrow, ncol = cfg.sl_list[ind].dh.e.shape
    rld, cld = np.meshgrid((np.arange(nrow) - nrow//2)/ppl,
                           (np.arange(ncol) - ncol//2)/ppl,
                           indexing='ij')
    lod_mask = np.logical_and(np.sqrt(rld**2 + cld**2) >= lod_min,
                          np.sqrt(rld**2 + cld**2) <= lod_max)
    lod_mask = np.logical_and(lod_mask,
                              cfg.sl_list[ind].dh.e) # only keep valid pix

    j = 0
    measph = target # no effect on first iteration
    while j < maxiter:
        scale = scale / measph * target

        # Since we're iterating to get the probe amplitude right, don't bother
        # trying to account for any other scalar factors
        dp0 = probe_gaussian(cfg.dmlist[dind].registration['nact'],
                             xcenter,
                             ycenter,
                             sigma,
                             np.sqrt(scale)
                             )
        dp1 = tie_with_matrix(dp0, cfg.dmlist[dind].dmvobj.tiemap)
        dpv = cfg.dmlist[dind].dmvobj.dmh_to_volts(dp1, cfg.sl_list[ind].lam)

        eplus = efield(cfg, [dmlist[0]+dpv, dmlist[1]], ind)
        eminus = efield(cfg, [dmlist[0]-dpv, dmlist[1]], ind)

        pampe = np.abs((eplus - eminus) / 2j)
        probe_int = pampe**2 / ipeak

        measph = np.mean(probe_int[lod_mask])
        if verbose:
            print("iteration " + str(j) + ": measured = " + str(measph) +
                  ", fractional error: " + str((measph - target) / target))

        j += 1
        pass

    # Redo with final scale
    dp0 = probe_gaussian(cfg.dmlist[dind].registration['nact'],
                         xcenter,
                         ycenter,
                         sigma,
                         np.sqrt(scale)
                         )

    dp1 = tie_with_matrix(dp0, cfg.dmlist[dind].dmvobj.tiemap)
    dpv = cfg.dmlist[dind].dmvobj.dmh_to_volts(dp1, cfg.sl_list[ind].lam)

    # create additional data products to help users
    epups = cfg.sl_list[ind].epup.e.shape
    sps = cfg.sl_list[ind].pupil.e.shape
    lys = cfg.sl_list[ind].lyot.e.shape
    nrow = max(epups[0], sps[0], lys[0])
    ncol = max(epups[1], sps[1], lys[1])

    dm_surface = dmhtoph(
        nrow=nrow,
        ncol=ncol,
        dmin=dpv,
        nact=cfg.dmlist[dind].registration['nact'],
        inf_func=cfg.dmlist[dind].registration['inf_func'],
        ppact_d=cfg.dmlist[dind].registration['ppact_d'],
        ppact_cx=cfg.dmlist[dind].registration['ppact_cx'],
        ppact_cy=cfg.dmlist[dind].registration['ppact_cy'],
        dx=cfg.dmlist[dind].registration['dx'],
        dy=cfg.dmlist[dind].registration['dy'],
        thact=cfg.dmlist[dind].registration['thact'],
        flipx=cfg.dmlist[dind].registration['flipx'],
    )

    pupil_mask = (
        insertinto(np.abs(cfg.sl_list[ind].epup.e), (nrow, ncol))*
        insertinto(np.abs(cfg.sl_list[ind].pupil.e), (nrow, ncol))*
        insertinto(np.abs(cfg.sl_list[ind].lyot.e), (nrow, ncol))
    )

    return dpv, probe_int, lod_mask, dm_surface, pupil_mask

