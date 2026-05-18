# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Functions to load data from FITS files and create complex-valued objects if
necessary
"""
import os

import numpy as np
import astropy.io.fits as pyfits

import howfsc.util.check as check

def load_ri(fnreal, fnimag):
    """
    Load a mask by real and imaginary parts

    Assumes all masks are stored in the primary element of 2 separate FITS
    files as a 2D array, one for real and one for imaginary parts.  If either
    is missing, an IOError will be thrown (by pyfits).

    Assumes the two arrays are the same size.  If not, an error will be thrown.

    Output array is a numpy type.

    Arguments:
     fnreal: string with filename for real part of mask.
     fnimag: string with filename for imaginary part of mask.

    Returns:
     complex-valued ndarray (numpy type)

    """
    fnreal = os.path.expandvars(fnreal)
    fnimag = os.path.expandvars(fnimag)

    # FITS can only store real data, hence the two-file split
    maskreal = pyfits.getdata(fnreal)
    maskimag = pyfits.getdata(fnimag)

    # Check loaded arrays
    check.twoD_array(maskreal, 'real', TypeError)
    check.twoD_array(maskimag, 'imag', TypeError)

    if maskreal.shape != maskimag.shape: # pylint: disable=maybe-no-member
        raise TypeError('The files loaded from ' + str(fnreal) +
                        ' and ' + str(fnimag) + ' must be the same' +
                        ' dimensions.')

    return maskreal + 1j*maskimag


def load_ap(fnamp, fnph):
    """
    Load a mask by amplitude and phase

    Assumes all masks are stored in the primary element of 2 separate FITS
    files as a 2D array, one for amplitude and one for phase.  If either is
    missing, an IOError will be thrown (by pyfits).

    Assumes the two arrays are the same size.  If not, an error will be thrown.

    Output array is a numpy type.

    Arguments:
     fnamp: string with filename for amplitude of mask.
     fnph: string with filename for phase of mask.

    Returns:
     complex-valued ndarray (numpy type)

    """
    fnamp = os.path.expandvars(fnamp)
    fnph = os.path.expandvars(fnph)

    # FITS can only store real data, hence the two-file split
    maskamp = pyfits.getdata(fnamp)
    maskph = pyfits.getdata(fnph)

    # Check loaded arrays
    check.twoD_array(maskamp, 'amp', TypeError)
    check.twoD_array(maskph, 'ph', TypeError)

    if maskamp.shape != maskph.shape: # pylint: disable=maybe-no-member
        raise TypeError('The files loaded from ' + str(fnamp) +
                        ' and ' + str(fnph) + ' must be the same' +
                        ' dimensions.')

    return maskamp*np.exp(1j*maskph)


def load(fn):
    """
    Load a real-valued mask from a FITS file

    Assumes mask is stored in the primary element a FITS file.  Output array
    is a numpy type.  If missing, an IOError will be thrown (by pyfits)

    Arguments:
     fn: string with filename for mask.

    Returns:
     real-valued ndarray (numpy type)

    """

    mask = pyfits.getdata(os.path.expandvars(fn))
    check.twoD_array(mask, 'mask', TypeError)
    return mask
