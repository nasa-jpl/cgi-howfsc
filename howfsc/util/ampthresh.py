# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Module for ampthresh, which decides where the pupil is illuminated."""
import numpy as np

from . import check


def ampthresh(pupilMap, nBin=21):
    """
    Threshold a pupil map to give pixels in the open area.

    Given an amplitude, complex amplitude, or intensity map for a pupil plane,
    threshold it to select pixels which are inside the open area. Used
    to create masks to eliminate low SNR regions.

    Parameters
    ----------
    pupilMap : array_like
        2-D pupil map. Can be an intensity image, pupil amplitude, or a
        complex-valued phase retrieval
    nBin : int
        Number of bins used when making a histogram of pupil values.
        Default is 21.

    Returns
    -------
    boolMask : numpy ndarray
        2-D boolean map of pixels above the threshold value in the pupil
        map. Same size as the input pupilMap.
    """
    check.positive_scalar_integer(nBin, 'nBin', ValueError)
    pupilMap = check.twoD_array(pupilMap, 'pupilMap', ValueError)
    pupilMap = np.abs(pupilMap)  # Force to be real

    # Don't fail if all the same value.
    if np.min(pupilMap) == np.max(pupilMap):
        boolMask = pupilMap.astype(bool)
        return boolMask

    # use histogram of intensities to choose threshold
    Icount, IbinEdges = np.histogram(pupilMap, bins=nBin)

    # find the minima in the histogram
    bV = np.logical_and(Icount[1:-1] <= Icount[:-2],
                        Icount[1:-1] < Icount[2:])

    # vector of intensity values at the center of each bin
    binCenter = 0.5*(IbinEdges[:-1]+IbinEdges[1:])

    # List of bin values where the histogram has a miminum
    binVal = binCenter[1:-1][bV]

    # Choose the first minimum as the threshold. this will be the lowest
    # intensity value above the background intensity level. All pixels
    # with greater intensity must be "signal". If binVal is empty, the
    # histogram has no o minima except at an endpoint. This is a
    # failure, so return empty.
    if np.size(binVal) > 0:
        thresh = binVal[0]
    else:
        raise ValueError('PupilType failed to find a histogram minimum')

    # Apply theshold to create boolean mask
    boolMask = pupilMap > thresh

    return boolMask
