# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Functions to parse and validate HOWFSC optical model definition file
specifications, and to create a CoronagraphMode object
"""
import os

import numpy as np
import astropy.io.fits as pyfits
import yaml

from howfsc.util.loadyaml import loadyaml
import howfsc.util.check as check

#----------------------------
# Validation utility classes
#----------------------------

class MDFException(Exception):
    """
    Thin exception class for HOWFSC optical model definition file specification
    noncompliances
    """
    pass


#----------------------
# Validation functions
#----------------------

def validate_model_file(fn, usefiles=False, verbose=True):
    """
    Utility function which will evaluate a HOWFSC optical model definiton file
    for spec compliance

    This function will print to the command line if verbose=True, and return
    True if it complies with the specification.

    Arguments:
     fn: string containing path to a filename with a HOWFSC optical model YAML
      file

    Keyword Arguments:
     usefiles: boolean indicating whether to actually load files (True) or
      just to check whether the YAML file contains strings that could be files
      (False).  Use True when you want to create and use an optical model; use
      False when you want to check the format validity but don't have all of
      the files in the right locations at the moment.  Defaults to False, on
      the assumption that the primary use case is to validate the spec and file
      checkout is outside the spec.

    Returns:
     True if meets spec, False if does not meet spec, None if could not be
      validated under the given conditions

    """

    if not isinstance(fn, str):
        raise TypeError('fn input must be a string')
    if not isinstance(usefiles, bool):
        raise TypeError('usefiles must be a boolean')
    if not isinstance(verbose, bool):
        raise TypeError('verbose must be a boolean')

    try:
        check_mode_lists(fn, usefiles) # ignore return values
        pass
    except MDFException as e:
        if verbose:
            print('HOWFSC optical model definition file DOES NOT MEET ' +
                  'specification.')
            print('Error message: ' + str(e.args[0]))
            pass
        return False
    except TypeError as e:
        if verbose:
            print('Model definition file cannot be evaluated due to invalid ' +
                  'inputs.')
            print('Error message: ' + str(e.args[0]))
            pass
        return None
    except IOError as e:
        if verbose:
            print('Model definition file cannot be evaluated due to ' +
                  'invalid data file name or contents when running with ' +
                  'usefiles=True.')
            print('Error message: ' + str(e.args[0]))
            pass
        return None
    except Exception as e: # pylint: disable=broad-except
        if verbose:
            print('Unknown software error during validation')
            print(repr(e))
            pass
        return None

    if verbose:
        print('HOWFSC optical model definition file MEETS specification.')
        pass
    return True



def check_mode_lists(fn, usefiles):
    """
    Read in a model definition file, check that it meet specifications, and
    return the data in a format usable by a CoronagraphMode object.

    If this function returns, the spec was met.  An MDFException will be raised
    (see below) if the spec was not met, and other exceptions may be raised if
    there are other problems besides specification compliance.  This function
    does not return any data when it returns.

    Arguments:
     fn: string containing path to a filename with a HOWFSC optical model
      defintion YAML file
     usefiles: boolean indicating whether to actually load files (True) or
      just to check whether the YAML file contains strings that could be files
      (False).  Use True when you want to create and use an optical model; use
      False when you want to check the format validity but don't have all of
      the files in the right locations at the moment.

      Paths can be absolute or relative; absolute is preferred strongly for
      implementation, but relative is necessary for test purposes.  Relative
      paths will be relative to the directory fn is located in.

    Raises:
     MDFException: custom Exception raised if the file does not meet the model
      definition file specification (in D-107617).
     TypeError: Raised if the input types are incorrect; this does not say
      anything about compliance with the model definition file specification.
     OSError: Raised if usefiles=True and one of the filenames in the
      YAML file cannot be found.

    """
    # Check inputs
    if not isinstance(fn, str):
        raise TypeError('fn input must be a string')
    if not isinstance(usefiles, bool):
        raise TypeError('usefiles must be a boolean')

    # 1. Each model definition file shall be a YAML 1.1 file.
    y0 = loadyaml(fn, custom_exception=MDFException)

    # 2. Each model definition file shall have exactly 3 top-level collections
    # with the following keys: 'dms', 'init', and 'sls'
    keys2 = {'dms', 'sls', 'init'}
    _exact_keys(y0, keys2, errstr='check 2')

    # 2a. The contents of the 'dms' collection shall have exactly 2
    # collections, with the following keys: 'DM1' and 'DM2'
    keys2a = {'DM1', 'DM2'}
    _exact_keys(y0['dms'], keys2a, errstr='check 2a')

    # 2ai. Each of these two collections shall have exactly 4 collections,
    # with the following keys: 'pitch', 'z', 'registration', and 'voltages'
    keys2ai = {'pitch', 'z', 'registration', 'voltages'}
    for k in keys2a:
        _exact_keys(y0['dms'][k], keys2ai, errstr='check 2ai')

        # 2ai1. The key 'pitch' shall have a single floating-point value > 0.
        # It will be a value in meters denoting the spacing between actuators
        # on the DM facesheet.
        check.real_positive_scalar(
            _tofloat(y0['dms'][k]['pitch'], 'pitch'),
            "y0['dms'][k]['pitch']",
            MDFException,
        )

        # 2ai2. The key 'z' shall have a single floating-point value.  It will
        # be a value in meters denoting the distance of the DM surface from the
        # pupil along the beam, as measured in collimated space.
        check.real_scalar(
            _tofloat(y0['dms'][k]['z'], 'z'),
            "y0['dms'][k]['z']",
            MDFException,
        )

        # 2ai3. The contents of the 'registration' collection shall have
        # exactly 9 collections, with the following keys: 'dx', 'dy', 'inffn',
        # 'nact', 'ppact_cx', 'ppact_cy', 'ppact_d', 'thact', 'flipx'.
        keys2ai3 = {'dx', 'dy', 'inffn', 'nact', 'ppact_cx', 'ppact_cy',
                    'ppact_d', 'thact', 'flipx'}
        _exact_keys(y0['dms'][k]['registration'], keys2ai3,
                    errstr='check 2ai3')

        # 2ai3a. 'dx' shall be a single floating-point value.  It will be a
        # value in EXCAM pupil-plane pixels denoting the number of pixels to
        # shift the DM grid off center in the x-direction as seen on EXCAM.
        check.real_scalar(
            _tofloat(y0['dms'][k]['registration']['dx'], 'dx'),
           "y0['dms'][k]['registration']['dx']",
            MDFException,
        )

        # 2ai3b. 'dy' shall be a single floating-point value.  It will be a
        # value in EXCAM pupil-plane pixels denoting the number of pixels to
        # shift the DM grid off center in the y-direction as seen on EXCAM.
        check.real_scalar(
            _tofloat(y0['dms'][k]['registration']['dy'], 'dy'),
            "y0['dms'][k]['registration']['dy']",
            MDFException,
        )

        # 2ai3c.'inffn' shall be a string.  This string should contain a
        # path to a FITS file with a master influence function for the DM used
        # to model the change in the facesheet for each actuator.  Each
        # actuator is modeled as creating the same facesheet profile for a
        # single DM, but the profiles between 'DM1' and 'DM2' may differ due
        # to manufacturing difference between the hardware (facesheet thickness
        # in particular).  However, validation of this file is out of scope for
        # the specification.
        if not isinstance(y0['dms'][k]['registration']['inffn'], str):
            raise MDFException('inffn must be a string')

        if usefiles:
            path = y0['dms'][k]['registration']['inffn']
            path = _absrelpaths(path, 'inffn', fn)

            try:
                inf_func = pyfits.getdata(path)
            except OSError: # could not be read as a FITS
                raise MDFException('inffn should point at a FITS file')

            check.twoD_array(inf_func, 'inf_func', MDFException)
            pass

        # 2ai3d. 'nact' shall be an single integer > 0.   It will be the
        # number of actuators along one edge of the DM, assumed square.  For
        # CGI, this value will be 48, and no other value is expected to be
        # used for any purpose other than software testing.
        check.positive_scalar_integer(y0['dms'][k]['registration']['nact'],
                                      "y0['dms'][k]['registration']['nact']",
                                      MDFException)

        # 2ai3e. 'ppact_cx' shall be a single floating-point value > 0.  It
        # will be the number of pixels per actuator in the x-direction on
        # EXCAM.
        check.real_positive_scalar(
            _tofloat(y0['dms'][k]['registration']['ppact_cx'], 'ppact_cx'),
            "y0['dms'][k]['registration']['ppact_cx']",
            MDFException,
        )

        # 2ai3f. 'ppact_cy' shall be a single floating-point value > 0.  It
        # will be the number of pixels per actuator in the y-direction on
        # EXCAM.
        check.real_positive_scalar(
            _tofloat(y0['dms'][k]['registration']['ppact_cy'], 'ppact_cy'),
            "y0['dms'][k]['registration']['ppact_cy']",
            MDFException,
        )

        # 2ai3g. 'ppact_d' shall be a single floating-point value > 0.  It
        # will be the number of pixels per actuator in the master influence
        # function pointed to by inffn.
        check.real_positive_scalar(
            _tofloat(y0['dms'][k]['registration']['ppact_d'], 'ppact_d'),
            "y0['dms'][k]['registration']['ppact_d']",
             MDFException,
        )

        # 2a13h. 'thact' shall be a single floating-point value.  It will be
        # the number of degrees to rotate the grid in a counterclockwise
        # direction about the center of the array.
        check.real_scalar(
            _tofloat(y0['dms'][k]['registration']['thact'], 'thact'),
            "y0['dms'][k]['registration']['thact']",
            MDFException,
        )

        # 2ai3i. 'flipx' shall be a single Boolean value (True or False).  It
        # will indicate whether to flip the output in the x-direction, leaving
        # the y-direction unchanged.
        if not isinstance(y0['dms'][k]['registration']['flipx'], bool):
            raise MDFException('flipx must be a boolean')

        # 2ai4. The contents of the 'voltages' collection shall have exactly 8
        # collections, with the following keys: 'vmin', 'vmax', 'vneighbor',
        # 'vcorner', 'vquant', 'gainfn', 'flatfn', 'tiefn'.
        keys2ai4 = {'vmin', 'vmax', 'vneighbor', 'vcorner', 'vquant', 'gainfn',
                    'flatfn', 'tiefn', 'crosstalkfn'}
        _exact_keys(y0['dms'][k]['voltages'], keys2ai4,
                    errstr='check 2ai4')

        # 2ai4a. 'vmin' shall be a single floating-point value.  It will the
        # smallest permitted DM actuator voltage the model can use, in volts.
        vmin = _tofloat(y0['dms'][k]['voltages']['vmin'], 'vmin')
        check.real_scalar(
            vmin,
            "y0['dms'][k]['voltages']['vmin']",
            MDFException,
        )

        # 2ai4b. 'vmax' shall be a single floating-point value > 'vmin'.  It
        # will the largest permitted value the model can use, in volts.
        vmax = _tofloat(y0['dms'][k]['voltages']['vmax'], 'vmax')
        check.real_scalar(
            vmax,
            "y0['dms'][k]['voltages']['vmax']",
            MDFException,
        )

        if vmax <= vmin:
            raise MDFException('File must have vmax > vmin')

        # 2ai4c. 'vneighbor' shall be a single floating-point value > 0.  It
        # will be the largest permissible difference between the voltages of
        # actuators which are laterally-adjacent (i.e. connected as the arms
        # of a '+').
        check.real_positive_scalar(
            _tofloat(y0['dms'][k]['voltages']['vneighbor'], 'vneighbor'),
            "y0['dms'][k]['voltages']['vneighbor']",
            MDFException,
        )

        # 2ai4d. 'vcorner' shall be a single floating-point value > 0.  It
        # will be the largest permissible difference between the voltages of
        # actuators which are diagonally-adjacent (i.e. connected as the arms
        # of an 'x').
        check.real_positive_scalar(
            _tofloat(y0['dms'][k]['voltages']['vcorner'], 'vcorner'),
            "y0['dms'][k]['voltages']['vcorner']",
            MDFException,
        )

        # 2a14e. 'vquant' shall be a single floating-point value > 0.  It will
        # be the magnitude of the least significant bit (LSB) of the deformable
        # mirror electronics, in volts.  For CGI, this value will be
        # 0.001678466796875 (=110/2^16) and no other value is expected to be
        # used for any purpose other than software testing.
        check.real_positive_scalar(
            _tofloat(y0['dms'][k]['voltages']['vquant'], 'vquant'),
            "y0['dms'][k]['voltages']['vquant']",
            MDFException,
        )

        # 2ai4f. 'gainfn' shall be a string.  This string should contain a
        # path to a FITS file with a 48x48 gain map.  Gain map specifications
        # are defined in the Roman CGI Deformable Mirror FDD (D-105741).
        # However, validation of this file is out of scope for the
        # specification.
        if not isinstance(y0['dms'][k]['voltages']['gainfn'], str):
            raise MDFException('gainfn must be a string')

        if usefiles:
            path = y0['dms'][k]['voltages']['gainfn']
            path = _absrelpaths(path, 'gainfn', fn)

            try:
                gain = pyfits.getdata(path)
            except OSError: # could not be read as a FITS
                raise MDFException('gainfn should point at a FITS file')

            check.twoD_array(gain, 'gain', MDFException)
            pass

        # 2ai4g. 'flatfn' shall be a string.  This string should contain a
        # path to a FITS file with a 48x48 flat map.  Flat map specifications
        # are defined in the Roman CGI Deformable Mirror FDD (D-105741).
        # However, validation of this file is out of scope for the
        # specification.
        if not isinstance(y0['dms'][k]['voltages']['flatfn'], str):
            raise MDFException('flatfn must be a string')

        if usefiles:
            path = y0['dms'][k]['voltages']['flatfn']
            path = _absrelpaths(path, 'flatfn', fn)

            try:
                flat = pyfits.getdata(path)
            except OSError: # could not be read as a FITS
                raise MDFException('flatfn should point at a FITS file')

            check.twoD_array(flat, 'flat', MDFException)
            pass

        # 2ai4h. 'tiefn' shall be a string.  This string should contain a path
        # to a FITS file with a 48x48 tie map.  Tie map specifications are
        # defined in the Roman CGI Deformable Mirror FDD (D-105741). However,
        # validation of this file is out of scope for the specification.
        if not isinstance(y0['dms'][k]['voltages']['tiefn'], str):
            raise MDFException('tiefn must be a string')

        if usefiles:
            path = y0['dms'][k]['voltages']['tiefn']
            path = _absrelpaths(path, 'tiefn', fn)

            try:
                tie = pyfits.getdata(path)
            except OSError: # could not be read as a FITS
                raise MDFException('tiefn should point at a FITS file')

            check.twoD_array(tie, 'tie', MDFException)
            pass

        # 2ai4i. 'crosstalkfn' shall either be a string OR be empty (that is,
        # a line with 'crosstalkfn: ' and nothing else). If present, this
        # string should contain a path to a YAML file with crosstalk
        # information for the DM. Validation of this file is out of scope for
        # the specification. If there is no string, then the model for that DM
        # will not use any actuator crosstalk.
        if not isinstance(y0['dms'][k]['voltages']['crosstalkfn'], str):
            if y0['dms'][k]['voltages']['crosstalkfn'] is not None:
                raise MDFException('crosstalkfn must be a string or empty')

        if usefiles and (y0['dms'][k]['voltages']['crosstalkfn'] is not None):
            path = y0['dms'][k]['voltages']['crosstalkfn']
            path = _absrelpaths(path, 'crosstalkfn', fn)

            # check it's YAML, don't parse actual contents
            # - need to use safe_load_all as diagonals are separate files
            #   internal to YAML file uniquely to this filetype
            #   (see issue #269)
            try:
                with open(path) as f:
                    ctalk = yaml.safe_load_all(f)
                    pass
                pass
            except IOError:
                raise MDFException('Config file does not exist.')
            except yaml.YAMLError: # this is base class for all YAML errors
                raise MDFException('File is not valid YAML.')
            except UnicodeDecodeError:
                raise MDFException('File is not valid YAML.')

            pass

        pass

    # 2b. The contents of the 'init' collection shall have exactly 2
    # collections, with the following keys: 'DM1' and 'DM2'
    keys2b = {'DM1', 'DM2'}
    _exact_keys(y0['init'], keys2b, errstr='check 2b')

    # 2bi. Each of these two collections shall have exactly 1 collection, with
    # the following key: 'dminit'
    keys2bi = {'dminit'}
    for k in keys2b:
        _exact_keys(y0['init'][k], keys2bi, errstr='check 2bi')

        # 2bi1. The key “dminit” shall be a string.  This string should contain
        # a path to a FITS file with a 48x48 DM setting in floating-point
        # voltage with the absolute voltage on DM1 or DM2 at the time of
        # collection of input-wavefront data ('epup').  However, validation of
        # this file is out of scope for the specification.
        if not isinstance(y0['init'][k]['dminit'], str):
            raise MDFException('dminit must be a string')

        if usefiles:
            path = y0['init'][k]['dminit']
            path = _absrelpaths(path, 'dminit', fn)

            try:
                dminit = pyfits.getdata(path)
            except OSError: # could not be read as a FITS
                raise MDFException('dminit should point at a FITS file')

            check.twoD_array(dminit, 'dminit', MDFException)
            pass

        pass

    # 2c. The contents of the 'sls' collection shall have one or more
    # collections, with integer keys numbered from 0 to N-1 for N collections.
    # These are used to define a coronagraph monochromatic optical model for a
    # single representative wavelength, and the set of them together is used to
    # understand the performance of the coronagraph across several bands.
    try:
        N = len(y0['sls'])
    except TypeError: # not iterable
        raise MDFException('sls is not an iterable')
    if N < 1:
        raise MDFException('sls must have at least one collection')

    keys2c = set(range(N))
    _exact_keys(y0['sls'], keys2c, errstr='check 2c')

    # 2ci. Each of these collections shall have exactly 8 collections, with
    # the following keys: 'lam', 'epup', 'sp', 'fpm', 'lyot', 'fs', 'dh',
    # 'ft_dir'.
    keys2ci = {'lam', 'epup', 'sp', 'fpm', 'lyot', 'fs', 'dh', 'ft_dir'}
    for k in keys2c:
        _exact_keys(y0['sls'][k], keys2ci, errstr='check 2ci')

        # 2ci1. 'lam' shall be a single floating-point value > 0, representing
        # the effective wavelength in meters.
        check.real_positive_scalar(
            _tofloat(y0['sls'][k]['lam'], 'lam'),
            "y0['sls'][k]['lam']",
            MDFException,
        )

        # 2ci2. 'epup' shall be a collection with exactly five collections,
        # which are drawn from one of the following lists: ['afn', 'pfn',
        # 'pdp', 'tip', 'tilt'] OR ['rfn', 'ifn', 'pdp', 'tip', 'tilt']
        keys2ci2_1 = {'afn', 'pfn', 'pdp', 'tip', 'tilt'}
        keys2ci2_2 = {'rfn', 'ifn', 'pdp', 'tip', 'tilt'}
        keys2ci2 = _exact_two_keys(y0['sls'][k]['epup'],
                                   keys2ci2_1,
                                   keys2ci2_2,
                                   errstr='check 2ci2')

        # 2ci2a. Any of 'afn', 'pfn', 'rfn', and 'ifn' which are present shall
        # be strings.  These strings should contain paths to FITS files with
        # partial representations of the wavefront through the coronagraph
        # with all masks out and the back-end phase removed.  They are provided
        # in pairs: either 'afn' and 'pfn' for amplitude and phase of
        # wavefront, respectively, or 'rfn' and 'ifn' for real and imaginary
        # part of wavefront, respectively.  The 2D arrays contained in these
        # files should be the same dimensions.  However, validation of the
        # contents of these files is out of scope for the specification.
        dim = None
        for fnstr in {'afn', 'pfn', 'rfn', 'ifn'}:
            if fnstr in keys2ci2:
                if not isinstance(y0['sls'][k]['epup'][fnstr], str):
                    raise MDFException(fnstr + ' must be a string')

                if usefiles:
                    path = y0['sls'][k]['epup'][fnstr]
                    path = _absrelpaths(path, 'epup', fn)

                    try:
                        data = pyfits.getdata(path)
                    except OSError: # could not be read as a FITS
                        raise MDFException(fnstr +
                                           ' should point at a FITS file')

                    # Expect 2D arrays of same size
                    check.twoD_array(data, fnstr, MDFException)
                    if dim is None:
                        dim = data.shape
                        pass
                    else:
                        if data.shape != dim:
                            raise MDFException('arrays in afn, pfn, rfn, ' +
                                               'and ifn must be the same size')
                        pass
                    pass
                pass
            pass

        # 2ci2b. 'pdp' shall be an floating-point value > 0 which gives the
        # number of pixels across the pupil at this plane in the model
        # representation.  This is used to make the scaling appropriate in the
        # Fourier transforms.  Each pupil is asserted to have the same sampling
        # in X and Y and masks are adjusted in aspect to represent any true
        # pupil ellipticity.
        check.real_positive_scalar(
            _tofloat(y0['sls'][k]['epup']['pdp'], 'pdp'),
            "y0['sls'][k]['epup']['pdp']",
            MDFException,
        )

        # 2ci2c. 'tip' shall be a floating-point value.  This represents the
        # decenter of the star along a row in units of EXCAM pixels before the
        # focal-plane mask.
        check.real_scalar(
            _tofloat(y0['sls'][k]['epup']['tip'], 'tip'),
            "y0['sls'][k]['epup']['tip']",
            MDFException,
        )

        # 2ci2d. 'tilt' shall be a floating-point value.  This represents the
        # decenter of the star along a column in units of EXCAM pixels before
        # the focal-plane mask.
        check.real_scalar(
            _tofloat(y0['sls'][k]['epup']['tilt'], 'tilt'),
            "y0['sls'][k]['epup']['tilt']",
            MDFException,
        )

        # 2ci3. 'sp' shall be a collection with exactly three collections,
        # which are drawn from one of the following lists: ['afn', 'pfn',
        # 'pdp'] OR ['rfn', 'ifn', 'pdp']
        keys2ci3_1 = {'afn', 'pfn', 'pdp'}
        keys2ci3_2 = {'rfn', 'ifn', 'pdp'}
        keys2ci3 = _exact_two_keys(y0['sls'][k]['sp'],
                                   keys2ci3_1,
                                   keys2ci3_2,
                                   errstr='check 2ci3')

        # 2ci3a. Any of 'afn', 'pfn', 'rfn', and 'ifn' which are present shall
        # be strings.  These strings should contain paths to FITS files with
        # partial representations of the wavefront change introduced when
        # hitting the mask in the SPAM mechanism.  They are provided in pairs:
        # either 'afn' and 'pfn' for amplitude and phase of wavefront,
        # respectively, or 'rfn' and 'ifn' for real and imaginary part of
        # wavefront, respectively.  The 2D arrays contained in these files
        # should be the same dimensions.  However, validation of the contents
        # of these files is out of scope for the specification.
        dim = None
        for fnstr in {'afn', 'pfn', 'rfn', 'ifn'}:
            if fnstr in keys2ci3:
                if not isinstance(y0['sls'][k]['sp'][fnstr], str):
                    raise MDFException(fnstr + ' must be a string')

                if usefiles:
                    path = y0['sls'][k]['sp'][fnstr]
                    path = _absrelpaths(path, 'sp', fn)

                    try:
                        data = pyfits.getdata(path)
                    except OSError: # could not be read as a FITS
                        raise MDFException(fnstr +
                                           ' should point at a FITS file')

                    # Expect 2D arrays of same size
                    check.twoD_array(data, fnstr, MDFException)
                    if dim is None:
                        dim = data.shape
                        pass
                    else:
                        if data.shape != dim:
                            raise MDFException('arrays in afn, pfn, rfn, ' +
                                               'and ifn must be the same size')
                        pass
                    pass
                pass
            pass

        # 2ci3b. 'pdp' shall be an floating-point value > 0 which gives the
        # number of pixels across the pupil at this plane in the model
        # representation.  This is used to make the scaling appropriate in the
        # Fourier transforms.  Each pupil is asserted to have the same
        # sampling in X and Y and masks are adjusted in aspect to represent
        # any true pupil ellipticity.
        check.real_positive_scalar(
            _tofloat(y0['sls'][k]['sp']['pdp'], 'pdp'),
            "y0['sls'][k]['sp']['pdp']",
            MDFException,
        )

        # 2ci4. 'fpm' shall be a collection with exactly four collections,
        # which are drawn from one of the following lists: ['afn', 'pfn',
        # 'ppl', 'isopen'] OR ['rfn', 'ifn', 'ppl', 'isopen']
        keys2ci4_1 = {'afn', 'pfn', 'ppl', 'isopen'}
        keys2ci4_2 = {'rfn', 'ifn', 'ppl', 'isopen'}
        keys2ci4 = _exact_two_keys(y0['sls'][k]['fpm'],
                                   keys2ci4_1,
                                   keys2ci4_2,
                                   errstr='check 2ci4')

        # 2ci4a. Any of 'afn', 'pfn', 'rfn', and 'ifn' which are present shall
        # be strings.  These strings should contain paths to FITS files with
        # partial representations of the wavefront change introduced when
        # hitting the mask in the FPAM mechanism.  They are provided in pairs:
        # either 'afn' and 'pfn' for amplitude and phase of wavefront,
        # respectively, or 'rfn' and 'ifn' for real and imaginary part of
        # wavefront, respectively.  The 2D arrays contained in these files
        # should be the same dimensions.  However, validation of the contents
        # of these files is out of scope for the specification.
        dim = None
        for fnstr in {'afn', 'pfn', 'rfn', 'ifn'}:
            if fnstr in keys2ci4:
                if not isinstance(y0['sls'][k]['fpm'][fnstr], str):
                    raise MDFException(fnstr + ' must be a string')

                if usefiles:
                    path = y0['sls'][k]['fpm'][fnstr]
                    path = _absrelpaths(path, 'fpm', fn)

                    try:
                        data = pyfits.getdata(path)
                    except OSError: # could not be read as a FITS
                        raise MDFException(fnstr +
                                           ' should point at a FITS file')

                    # Expect 2D arrays of same size
                    check.twoD_array(data, fnstr, MDFException)
                    if dim is None:
                        dim = data.shape
                        pass
                    else:
                        if data.shape != dim:
                            raise MDFException('arrays in afn, pfn, rfn, ' +
                                               'and ifn must be the same size')
                        pass
                    pass
                pass
            pass

        # 2ci4b. 'ppl' shall be an floating-point value > 0 which gives the
        # number of pixels per lambda/D at this plane in the model
        # representation.  This is used to make the scaling appropriate in the
        # Fourier transforms.
        check.real_positive_scalar(
            _tofloat(y0['sls'][k]['fpm']['ppl'], 'ppl'),
            "y0['sls'][k]['fpm']['ppl']",
            MDFException,
        )

        # 2ci4c. 'isopen' shall be a Boolean which indicates whether the edge
        # of the mask representation is open or closed.  An open representation
        # means the open area of the mask has no outer boundary—at least not
        # in the model representation—while a closed representation does have
        # an outer edge beyond while no light is transmitted.  This is used by
        # the model to choose a propagation path which minimizes computation.
        # The value 'True' implies an open representation and 'False' implies
        # a closed one.
        if not isinstance(y0['sls'][k]['fpm']['isopen'], bool):
            raise MDFException('isopen must be a boolean')

        # 2ci5. 'lyot' shall be a collection with exactly five collections,
        # which are drawn from one of the following lists: ['afn', 'pfn',
        # 'pdp', 'tip', 'tilt'] OR ['rfn', 'ifn', 'pdp', 'tip', 'tilt']
        keys2ci5_1 = {'afn', 'pfn', 'pdp', 'tip', 'tilt'}
        keys2ci5_2 = {'rfn', 'ifn', 'pdp', 'tip', 'tilt'}
        keys2ci5 = _exact_two_keys(y0['sls'][k]['lyot'],
                                   keys2ci5_1,
                                   keys2ci5_2,
                                   errstr='check 2ci5')

        # 2ci5a. Any of 'afn', 'pfn', 'rfn', and 'ifn' which are present shall
        # be strings.  These strings should contain paths to FITS files with
        # partial representations of the wavefront change introduced when
        # hitting the mask in the LSAM mechanism.  They are provided in pairs:
        # either 'afn' and 'pfn' for amplitude and phase of wavefront,
        # respectively, or 'rfn' and 'ifn' for real and imaginary part of
        # wavefront, respectively.  The 2D arrays contained in these files
        # should be the same dimensions.  However, validation of the contents
        # of these files is out of scope for the specification.
        dim = None
        for fnstr in {'afn', 'pfn', 'rfn', 'ifn'}:
            if fnstr in keys2ci5:
                if not isinstance(y0['sls'][k]['lyot'][fnstr], str):
                    raise MDFException(fnstr + ' must be a string')

                if usefiles:
                    path = y0['sls'][k]['lyot'][fnstr]
                    path = _absrelpaths(path, 'lyot', fn)

                    try:
                        data = pyfits.getdata(path)
                    except OSError: # could not be read as a FITS
                        raise MDFException(fnstr +
                                           ' should point at a FITS file')

                    # Expect 2D arrays of same size
                    check.twoD_array(data, fnstr, MDFException)
                    if dim is None:
                        dim = data.shape
                        pass
                    else:
                        if data.shape != dim:
                            raise MDFException('arrays in afn, pfn, rfn, ' +
                                               'and ifn must be the same size')
                        pass
                    pass
                pass
            pass

        # 2ci5b. 'pdp' shall be an floating-point value > 0 which gives the
        # number of pixels across the pupil at this plane in the model
        # representation.  This is used to make the scaling appropriate in the
        # Fourier transforms.  Each pupil is asserted to have the same
        # sampling in X and Y and masks are adjusted in aspect to represent
        # any true pupil ellipticity.
        check.real_positive_scalar(
            _tofloat(y0['sls'][k]['lyot']['pdp'], 'pdp'),
            "y0['sls'][k]['lyot']['pdp']",
            MDFException,
        )

        # 2ci5bi. If 'isopen' is 'True' in 'fpm', this value shall be the same
        # as the value in 'sp'.  If this is not true, that propagation path
        # (using Babinet’s principle) will not function correctly.  This
        # constraint does not apply if 'isopen' is 'False' in 'fpm'.
        if y0['sls'][k]['fpm']['isopen']:
            sp_pdp = _tofloat(y0['sls'][k]['sp']['pdp'], 'sp pdp')
            lyot_pdp = _tofloat(y0['sls'][k]['lyot']['pdp'], 'lyot pdp')
            if sp_pdp != lyot_pdp:
                raise MDFException('If isopen=True in fpm, pdp must be the ' +
                                   'same in sp and lyot')
            pass

        # 2ci5c. 'tip' shall be a floating-point value.  This represents the
        # decenter of the star along a row in units of EXCAM pixels after the
        # focal-plane mask.
        check.real_scalar(
            _tofloat(y0['sls'][k]['lyot']['tip'], 'tip'),
            "y0['sls'][k]['lyot']['tip']",
            MDFException,
        )

        # 2ci5d. 'tilt' shall be a floating-point value.  This represents the
        # decenter of the star along a column in units of EXCAM pixels after
        # the focal-plane mask.
        check.real_scalar(
            _tofloat(y0['sls'][k]['lyot']['tilt'], 'tilt'),
            "y0['sls'][k]['lyot']['tilt']",
            MDFException,
        )

        # 2ci6. 'fs' shall be a collection with exactly three collections,
        # which are drawn from one of the following lists: ['afn', 'pfn',
        # 'ppl'] OR ['rfn', 'ifn', 'ppl']
        keys2ci6_1 = {'afn', 'pfn', 'ppl'}
        keys2ci6_2 = {'rfn', 'ifn', 'ppl'}
        keys2ci6 = _exact_two_keys(y0['sls'][k]['fs'],
                                   keys2ci6_1,
                                   keys2ci6_2,
                                   errstr='check 2ci6')

        # 2ci6a. Any of 'afn', 'pfn', 'rfn', and 'ifn' which are present shall
        # be strings.  These strings should contain paths to FITS files with
        # partial representations of the wavefront change introduced when
        # hitting the mask in the FSAM mechanism.  They are provided in pairs:
        # either 'afn' and 'pfn' for amplitude and phase of wavefront,
        # respectively, or 'rfn' and 'ifn' for real and imaginary part of
        # wavefront, respectively.  The 2D arrays contained in these files
        # should be the same dimensions.  However, validation of the contents
        # of these files is out of scope for the specification.
        dim = None
        for fnstr in {'afn', 'pfn', 'rfn', 'ifn'}:
            if fnstr in keys2ci6:
                if not isinstance(y0['sls'][k]['fs'][fnstr], str):
                    raise MDFException(fnstr + ' must be a string')

                if usefiles:
                    path = y0['sls'][k]['fs'][fnstr]
                    path = _absrelpaths(path, 'fs', fn)

                    try:
                        data = pyfits.getdata(path)
                    except OSError: # could not be read as a FITS
                        raise MDFException(fnstr +
                                           ' should point at a FITS file')

                    # Expect 2D arrays of same size
                    check.twoD_array(data, fnstr, MDFException)
                    if dim is None:
                        dim = data.shape
                        pass
                    else:
                        if data.shape != dim:
                            raise MDFException('arrays in afn, pfn, rfn, ' +
                                               'and ifn must be the same size')
                        pass
                    pass
                pass
            pass

        dim_fs = dim # for use in 2ci7

        # 2ci6b. 'ppl' shall be an floating-point value > 0 which gives the
        # number of pixels per lambda/D at this plane in the model
        # representation.  This is used to make the scaling appropriate in the
        # Fourier transforms.
        check.real_positive_scalar(
            _tofloat(y0['sls'][k]['fs']['ppl'], 'ppl'),
            "y0['sls'][k]['fs']['ppl']",
            MDFException,
        )

        # 2ci7. 'dh' shall be a string.  This string should contain the path
        # to a FITS file with a 2D 0/1-valued representation of which pixels
        # should be included in the CGI dark hole, as used by wavefront control
        # and control calculation.  1 indicates inclusion, 0 indicated
        # exclusion.  The 2D array should have the same dimensions as the 2D
        # arrays in the FITS files pointed to by the 'fs' collection, and
        # correspond to the same number of pixels per lambda/D, as it will be
        # used to extract pixels from an array directly after the application
        # of the mask in 'fs'.  However, validation of the contents or
        # properties of this file is out of scope for the specification.
        if not isinstance(y0['sls'][k]['dh'], str):
            raise MDFException('dh must be a string')

        if usefiles:
            path = y0['sls'][k]['dh']
            path = _absrelpaths(path, 'dh', fn)

            try:
                dh = pyfits.getdata(path)
            except OSError: # could not be read as a FITS
                raise MDFException('dh should point at a FITS file')

            # 2D
            check.twoD_array(dh, 'dh', MDFException)
            # 0/1-valued
            if np.logical_and((dh != 0), (dh != 1)).any():
                raise MDFException('dh should be 0/1-valued')
            # same size as fs
            if dh.shape != dim_fs:
                raise MDFException('dh should be the same size as fs')
            pass

        pass

        # 2ci8. "ft_dir" shall be a string containing one of the following:
        # "forward" or "reverse".  No other value is permitted.  If forward,
        # Fourier transforms from pupil to focus will be done as though the
        # focus is downstream of the pupil.  If reverse, Fourier transforms
        # from pupil to focus will be done as though the focus is upstream of
        # the pupil.
        if y0['sls'][k]['ft_dir'] not in ['forward', 'reverse']:
            raise MDFException('ft_dir should be "forward" or "reverse"')


    # 2cii. Each of the N collections in sls shall be arranged in wavelength
    # order, from shortest wavelength to longest.
    # 2ciii. There shall be no more than one collection per wavelength.  There
    # are at present no CGI-baseline use cases for an optical model definition
    # file which collects multiple data sets at the same wavelength, and we
    # defer any specification of that ordering to a future revision, to be
    # based on whichever other parameters are being changed
    for k in range(N-1):
        # > does 2cii, == does 2ciii.
        if y0['sls'][k]['lam'] > y0['sls'][k+1]['lam']:
            raise MDFException('sls channels must be ordered from shortest ' +
                               'wavelength to longest.')
        if y0['sls'][k]['lam'] == y0['sls'][k+1]['lam']:
            raise MDFException('No more than one sls channel per wavelength')
        pass

    pass


#--------------------------
# Utility validation tools
#--------------------------

def _exact_keys(checkme, keys, errstr):
    """
    Utility function to verify that the set of keys in a dictionary is
    identical to a pre-specified set.  No more, no less.

    No returns; this will raise an MDFException if there's something wrong

    Arguments:
     checkme: dictionary to check
     keys: iterable of items to check; will be cast to set()
     errstr: string with name for error message output

    """

    if not isinstance(checkme, dict):
        raise MDFException('input not a dict in ' + str(errstr))
    setkeys = set(keys)
    setcheck = set(checkme.keys())

    # Check missing and extra separately, so we can make an appropriately
    # informative message
    try:
        misskeys = setkeys - setcheck
        if misskeys != set():
            raise MDFException('Missing top-level keys in ' + str(errstr))
        pass

        extrakeys = setcheck - setkeys
        if extrakeys != set():
            raise MDFException('Extra top-level keys in ' + str(errstr))
    except AttributeError: # checkme is not a dict
        raise MDFException('Model definition file formatted incorrectly at ' +
                           str(errstr))
    pass


def _exact_two_keys(checkme, keys1, keys2, errstr):
    """
    Utility function to verify that the set of keys in a dictionary is
    identical to one of two pre-specified sets.

    Arguments:
     checkme: dictionary to check
     keys1: first iterable of items to check; will be cast to set()
     keys2: second iterable of items to check; will be cast to set()
     errstr: string with name for error message output

    Returns:
     either keys1 or keys2 cast to a set, whichever matches.  If neither
      matches, this will raise an MDFException

    """

    if not isinstance(checkme, dict):
        raise MDFException('input not a dict in ' + str(errstr))
    setkeys1 = set(keys1)
    setkeys2 = set(keys2)
    setcheck = set(checkme.keys())

    # Check missing and extra separately, so we can make an appropriately
    # informative message
    try:
        misskeys1 = setkeys1 - setcheck
        misskeys2 = setkeys2 - setcheck
        if not (misskeys1 == set() or misskeys2 == set()):
            raise MDFException('Missing top-level keys in ' + str(errstr))
        pass

        # Only check the one that passed, so we don't have a mismatch
        if misskeys1 == set():
            extrakeys = setcheck - setkeys1
            if extrakeys != set():
                raise MDFException('Extra top-level keys in ' + str(errstr))
            return setkeys1
        else:
            extrakeys = setcheck - setkeys2
            if extrakeys != set():
                raise MDFException('Extra top-level keys in ' + str(errstr))
            return setkeys2
        pass

    except AttributeError: # checkme is not a dict
        raise MDFException('Model definition file formatted incorrectly at ' +
                           str(errstr))
    pass



def _absrelpaths(path, filestr, fn):
    """
    Check if a path exists as an absolute or relative path.  Relative is in
    respect to the file being validated.

    This will raise an OSError if there's something wrong.

    Arguments:
     path: string with potential path name
     filestr: string with short name of variable stored in file.  Used to make
      an informative error message
     fn: YAML file to be relative to

    Returns:
     valid absolute path to the file indicated in path

    """
    # 3. All strings representing file paths in the YAML file shall either be
    # 1) absolute paths or 2) paths relative to the directory where the YAML
    # file is located.  For example, a relative file path with no directory
    # prepended should be in the same directory as the YAML file itself.

    path = os.path.expandvars(path)
    if os.path.isabs(path):
        if os.path.exists(path):
            return path
        else:
            raise OSError('Absolute path to ' + str(filestr) + ' file could ' +
                          'not be found')
        pass
    else:
        localpath = os.path.dirname(os.path.abspath(fn))
        path = os.path.join(localpath, path)
        if os.path.exists(path):
            return path
        else:
            raise OSError('Relative path to ' + str(filestr) + ' file could ' +
                          'not be found')
        pass

    pass


def _tofloat(x, errstr):
    """
    Returns input cast to a float.

    Raises an MDFException if it cannot be cast.  This is only used for floats
    and not ints, as YAML will parse scientific-notation floating-point values
    that do not have decimal points as strings.

    Arguments:
     x: input to cast
     errstr: string with information on what x is, for use by error messages

    Returns
     x cast to float

    """

    try: # handle mediocre YAML parsing (parses no decpt flt as string)
        fx = float(x)
        pass
    except (TypeError, ValueError): # Value if bad str, Type if not str
        raise MDFException(str(errstr) + ' not parseable as float')

    return fx
