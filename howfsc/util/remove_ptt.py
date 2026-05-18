# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""Functions to remove piston, tip, and tilt from a beam."""

import numpy as np

from . import check


def fit_and_remove_ptt_directly(arrayToFit, mask):
    """
    Remove piston and x- and y-ramp terms from an array directly.

    Parameters
    ----------
    arrayToFit : array_like
        2-D array of values to fit.
    mask : array_like
        2-D boolean mask of which pixels to use in "arrayToFit".
        Must be same size as "arrayToFit".

    Returns
    -------
    arrayOut : array_like
        Input array with piston, tip, and tilt subtracted. Masked by the
        input array named mask.
    arrayFitted : array_like
        The array with the best fit piston, tip, and tilt included.
        Does not use the mask variable.
    """
    check.twoD_array(arrayToFit, 'arrayToFit', ValueError)
    check.twoD_array(mask, 'mask', ValueError)
    if arrayToFit.shape != mask.shape:
        raise ValueError('arrayToFit and mask must have same shape')

    maskBool = np.asarray(mask).astype(bool)
    nPix = int(np.sum(mask))
    ny = arrayToFit.shape[0]
    nx = arrayToFit.shape[1]
    xVec = np.arange(-nx/2., nx/2.)/nx
    yVec = np.arange(-ny/2., ny/2.)/ny

    #  Set the basis functions
    ONES = np.ones((ny, nx))
    [X, Y] = np.meshgrid(xVec, yVec)
    f0 = X
    f1 = Y
    f2 = ONES

    # Write as matrix equation and solve for coefficients
    A = np.concatenate((f0[maskBool].reshape((nPix, 1)),
                        f1[maskBool].reshape((nPix, 1)),
                        f2[maskBool].reshape((nPix, 1))), axis=1)
    y = arrayToFit[maskBool].flatten()
    temp = np.linalg.lstsq(A, y, rcond=None)
    coeffs = temp[0]
    a, b, c = coeffs[0:3]
    arrayFitted = (a*X + b*Y + c*ONES)
    arrayOut = maskBool * (arrayToFit - arrayFitted)

    return arrayOut, arrayFitted
