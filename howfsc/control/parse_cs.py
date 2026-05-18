# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Functions to parse and validate control strategy specifications, and to create
a ControlStrategy object
"""
import os

import numpy as np
import astropy.io.fits as pyfits

from howfsc.util.loadyaml import loadyaml
import howfsc.util.check as check

#----------------------------
# Validation utility classes
#----------------------------

class Region(object):
    """
    Region object

    Stores bounds and value; no internal methods.
    """

    def __init__(self, first, last, low, high, value):
        """
        Set up internal state
        """
        self.first = first
        self.last = last
        self.low = low
        self.high = high
        self.value = value

        pass

class CSException(Exception):
    """
    Thin exception class for control strategy specification noncompliances
    """
    pass


#----------------------
# Validation functions
#----------------------

def validate_control_strategy(fn, usefiles=False, verbose=True):
    """
    Utility function which will evaluate a control strategy file for spec
    compliance

    This function will print to the command line if verbose=True, and return
    True if it complies with the specification.

    Arguments:
     fn: string containing path to a filename with a control strategy YAML
      file

    Keyword Arguments:
     usefiles: boolean indicating whether to actually load files in for
      pixelweights and fixedbp (True), or just to check whether the YAML file
      contains strings that could be files (False).  Use True when you want to
      create and use a control strategy; use False when you want to check the
      format validity but don't have the weighting/bad pixels files in the
      right locations at the moment.  Defaults to False, on the assumption that
      the primary use case is to validate the spec and file checkout is outside
      the spec.

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
        create_cs_lists(fn, usefiles) # ignore return values
        pass
    except CSException as e:
        if verbose:
            print('Control strategy DOES NOT MEET specification.')
            print('Error message: ' + str(e.args[0]))
            pass
        return False
    except TypeError as e:
        if verbose:
            print('Control strategy cannot be evaluated due to invalid ' +
                  'inputs.')
            print('Error message: ' + str(e.args[0]))
            pass
        return None
    except IOError as e:
        if verbose:
            print('Control strategy cannot be evaluated due to invalid data ' +
                  'file name or contents when running with usefiles=True.')
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
        print('Control strategy MEETS specification.')
        pass
    return True



def create_cs_lists(fn, usefiles):
    """
    Read in a control strategy file, check that it meet specifications, and
    return the data in a format usable by a ControlStrategy object.

    If this function returns, the spec was met.  A CSException will be raised
    (see below) if the spec was not met, and other exceptions may be raised if
    there are other problems besides specification compliance.

    Arguments:
     fn: string containing path to a filename with a control strategy YAML
      file
     usefiles: boolean indicating whether to actually load files in for
      pixelweights and fixedbp (True), or just to check whether the YAML file
      contains strings that could be files (False).  Use True when you want to
      create and use a control strategy; use False when you want to check the
      format validity but don't have the weighting/bad pixels files in the
      right locations at the moment.

      Paths can be absolute or relative; absolute is preferred strongly for
      implementation, but relative is necessary for test purposes.  Relative
      paths will be relative to the directory fn is located in.

    Returns: 8-tuple with the following, in order
     list of regularization regions
     list of pixelweights regions
     list of pixelweights regions, populated with pw filenames (this will be
      the same as the above if usefiles=False)
     list of dmmultgain regions
     list of probeheight regions
     list of unprobedsnr regions
     list of probedsnr regions
     fixedbp (either filename or data element, depending on usefiles)

    Raises:
     CSException: custom Exception raised if the file does not meet the control
      strategy specification (in D-106718).  If it returns, the spec was met.
     TypeError: Raised if the input types are incorrect; this does not say
      anything about compliance with the control strategy specification.
     FileNotFoundError: Raised if usefiles=True and one of the filenames in
      pixelweights or fixedbp cannot be found.

    """
    # Check inputs
    if not isinstance(fn, str):
        raise TypeError('fn input must be a string')
    if not isinstance(usefiles, bool):
        raise TypeError('usefiles must be a boolean')

    regularization = []
    pixelweights = []
    pixelweights_fn = []
    dmmultgain = []
    probeheight = []
    unprobedsnr = []
    probedsnr = []

    # Each control strategy shall be a YAML 1.1 file.
    csin = loadyaml(fn, custom_exception=CSException)

    # Each control strategy shall have exactly 57 top-level collections
    # with the following keys:
    # - regularization, pixelweights, dmmultgain, probeheight, unprobedsnr,
    #   probedsnr, fixedbp
    topkeys = {'regularization',
               'pixelweights',
               'dmmultgain',
               'probeheight',
               'unprobedsnr',
               'probedsnr',
               'fixedbp'}
    try:
        misskeys = topkeys - set(csin.keys())
        if misskeys != set():
            raise CSException('Missing top-level keys in input file')

        extrakeys = set(csin.keys()) - topkeys
        if extrakeys != set():
            raise CSException('Extra top-level keys in input file')
    except AttributeError: # csin is not a dict
        raise CSException('Control strategy file formatted incorrectly at ' +
                          'top level')

    # The contents of the "regularization", "pixelweights", "dmmultgain",
    # "unprobedsnr", "probedsnr", and "probeheight" collections shall be a
    # list of regions.
    #
    # A region is a collection with exactly five key/value pairs that
    # delineate the control strategy behavior over a rectangular region of
    # the contrast/iteration number parameter space.
    #
    # Each region shall have the following 5 keys:
    # - first, last, low, high, value
    regkeys = {'first', 'last', 'low', 'high', 'value'}
    for d in {'regularization', 'pixelweights', 'dmmultgain',
              'unprobedsnr', 'probedsnr', 'probeheight'}:
        if not isinstance(csin[d], list):
            raise CSException(d + ' does not contain a list')
        for elem in csin[d]:
            try:
                misskeys = regkeys - set(elem.keys())
                if misskeys != set():
                    raise CSException('Missing keys in region spec')

                extrakeys = set(elem.keys()) - regkeys
                if extrakeys != set():
                    raise CSException('Extra keys in region spec')
            except AttributeError: # elem is not a dict
                raise CSException('List element does not define a dict')

            # "first" shall be an integer giving the first iteration
            # number that this region applies to, inclusive.
            #
            # "first" shall be >= 1
            check.positive_scalar_integer(elem['first'],
                                          "elem['first']",
                                          CSException)

            # "last" shall be an integer giving the final iteration number
            # that this region applies to, inclusive, or None.
            #
            # "last" shall be >= "first" unless "last" is None.  If "last"
            # is None, the upper end of this region is unbounded

            if elem['last'] == 'None': # handle questionable yaml str parse
                elem['last'] = None
                pass

            if elem['last'] is not None:
                check.positive_scalar_integer(elem['last'],
                                              "elem['last']",
                                              CSException)
                if elem['last'] < elem['first']:
                    raise CSException('last precedes first in region')
                pass
            else: # set numerically unbounded
                elem['last'] = np.inf
                pass

            # "low" shall be a floating-point value giving the smallest
            # (i.e. closest to zero) mean total contrast over which this
            # region applies, inclusive.
            #
            # "low" shall be >= 0

            try: # handle mediocre YAML parsing (parses no decpt flt as string)
                elem['low'] = float(elem['low'])
                pass
            except (TypeError, ValueError): # Value if bad str, Type if not str
                raise CSException('low not parseable as float')

            check.real_nonnegative_scalar(elem['low'],
                                          "elem['low']",
                                          CSException)

            # "high" shall be a floating-point value giving the largest
            # (i.e. furthest from zero) mean total contrast over which
            # this region applies, exclusive, or None.
            #
            # "high" shall be > "low" unless "high" is None.  If "high" is
            # None, the upper end of this region is unbounded

            if elem['high'] == 'None': # handle questionable yaml str parse
                elem['high'] = None
                pass

            if elem['high'] is not None:
                try:
                    elem['high'] = float(elem['high'])
                    pass
                except (TypeError, ValueError):
                    raise CSException('high not parseable as float')
                pass

                check.real_nonnegative_scalar(elem['high'],
                                              "elem['high']",
                                              CSException)
                if elem['high'] <= elem['low']:
                    raise CSException('high precedes or matches low ' +
                                      'in region')
                pass
            else:
                elem['high'] = np.inf
                pass

            # "value" shall be a single value which applies to all
            # (iteration number, contrast) pairs which fall within "low"
            # <= contrast < "high" and "first" <= iteration number <=
            # "last"
            #
            # There shall be no overlapping regions within a collection.
            # Intent is that any (iteration number, contrast) tuple will
            # correspond to exactly one region.

            # For "regularization", "value" shall be a single
            # floating-point value equal to log10 of the value relative
            # to square of the largest singular value of the weighted
            # Jacobian.
            if d == 'regularization':
                try:
                    elem['value'] = float(elem['value'])
                    pass
                except (TypeError, ValueError):
                    raise CSException('value not parseable as float')
                pass

                check.real_scalar(elem['value'],
                                  "elem['value']",
                                  CSException)

                r = Region(elem['first'], elem['last'],
                           elem['low'], elem['high'], elem['value'])

                if _does_region_overlap_list(r, regularization):
                    raise CSException('regularization regions overlap')

                regularization.append(r)
                pass

            # For "pixelweights", "value" shall be a string
            #
            # This string should contain a path to a FITS file with a N 2D
            # weighting matrices, the first in its primary HDU and the
            # remaining N-1 2D weighting sequentially in image HDUs. There
            # should be one for each wavelength in an optical model,
            # running from shortest to longest wavelength in order.
            # However, validation of this file is out of scope for the
            # specification.
            elif d == 'pixelweights':
                if not isinstance(elem['value'], str):
                    raise CSException('pixelweights values must be strs')

                if usefiles:
                    elem['value'] = os.path.expandvars(elem['value'])
                    if os.path.isabs(elem['value']):
                        if not os.path.exists(elem['value']):
                            raise IOError('Absolute path to ' +
                                                  'pixelweights file could ' +
                                                  'not be found')
                        pass
                    else:
                        localpath = os.path.dirname(os.path.abspath(fn))
                        elem['value'] = os.path.join(localpath, elem['value'])
                        if not os.path.exists(elem['value']):
                            raise IOError('Relative path to ' +
                                                  'pixelweights file could ' +
                                                  'not be found')
                        pass
                    pass

                    # check FITS file with HDUs
                    try:
                        hdus = pyfits.open(elem['value']) # HDUList
                        pass
                    except FileNotFoundError:
                        raise IOError('pixelweights file could not ' +
                                              'be found')
                    if len(hdus) == 0:
                        raise IOError('pixelweights file contains ' +
                                              'no HDUs')

                    # check 2D matrices
                    pw = []
                    for index, h in enumerate(hdus):
                        check.twoD_array(h.data, 'hdus[' + str(index) +
                                         '] {pixelweights}', TypeError)
                        pw.append(h.data)
                        pass

                    # load in a list of arrays
                    r = Region(elem['first'], elem['last'],
                               elem['low'], elem['high'], pw)
                    pass
                else:
                    r = Region(elem['first'], elem['last'],
                               elem['low'], elem['high'], elem['value'])
                    pass

                if _does_region_overlap_list(r, pixelweights):
                    raise CSException('pixelweights regions overlap')

                pixelweights.append(r)

                # store the file name, too, so it's easier to look for repeats
                pwfn = Region(elem['first'], elem['last'],
                              elem['low'], elem['high'], elem['value'])
                pixelweights_fn.append(pwfn)
                pass

            # For "dmmultgain", "value" shall be a single floating-point
            # value > 0 giving the scale factor to multiply the calculated
            # DM.
            elif d == 'dmmultgain':
                try:
                    elem['value'] = float(elem['value'])
                    pass
                except (TypeError, ValueError):
                    raise CSException('value not parseable as float')
                pass

                check.real_positive_scalar(elem['value'],
                                           "elem['value']",
                                           CSException)

                r = Region(elem['first'], elem['last'],
                           elem['low'], elem['high'], elem['value'])

                if _does_region_overlap_list(r, dmmultgain):
                    raise CSException('dmmultgain regions overlap')

                dmmultgain.append(r)
                pass

            # For "probeheight", "value" shall be a single floating-point
            # value > 0 giving the desired mean probe amplitude across the
            # region of modulation.
            elif d == 'probeheight':
                try:
                    elem['value'] = float(elem['value'])
                    pass
                except (TypeError, ValueError):
                    raise CSException('value not parseable as float')
                pass

                check.real_positive_scalar(elem['value'],
                                           "elem['value']",
                                           CSException)

                r = Region(elem['first'], elem['last'],
                           elem['low'], elem['high'], elem['value'])

                if _does_region_overlap_list(r, probeheight):
                    raise CSException('probeheight regions overlap')

                probeheight.append(r)
                pass

            # For "unprobedsnr", "value" shall be a single floating-point
            # value > 0 giving the desired target SNR in the dark hole region
            # during data collection without DM probes applied.
            elif d == 'unprobedsnr':
                try:
                    elem['value'] = float(elem['value'])
                    pass
                except (TypeError, ValueError):
                    raise CSException('value not parseable as float')
                pass

                check.real_positive_scalar(elem['value'],
                                           "elem['value']",
                                           CSException)

                r = Region(elem['first'], elem['last'],
                           elem['low'], elem['high'], elem['value'])

                if _does_region_overlap_list(r, unprobedsnr):
                    raise CSException('unprobedsnr regions overlap')

                unprobedsnr.append(r)
                pass

            # For "probedsnr", "value" shall be a single floating-point
            # value > 0 giving the desired target SNR in the dark hole region
            # during data collection with DM probes applied.
            elif d == 'probedsnr':
                try:
                    elem['value'] = float(elem['value'])
                    pass
                except (TypeError, ValueError):
                    raise CSException('value not parseable as float')
                pass

                check.real_positive_scalar(elem['value'],
                                           "elem['value']",
                                           CSException)

                r = Region(elem['first'], elem['last'],
                           elem['low'], elem['high'], elem['value'])

                if _does_region_overlap_list(r, probedsnr):
                    raise CSException('probedsnr regions overlap')

                probedsnr.append(r)
                pass

            else:
                # should never get here given set of d values
                raise CSException('Internal error: iterate unknown key')

            pass
        pass

    # The list of regions in each collection, taken together, shall span
    # the entire quarter-plane with contrast >= 0 and iteration number
    # >= 1.
    for rlist in [regularization, pixelweights, dmmultgain,
                  unprobedsnr, probedsnr, probeheight]:
        lowhigh = set()
        firstlast = set()
        for r in rlist:
            lowhigh.add(r.low)
            lowhigh.add(r.high)
            firstlast.add(r.first)
            firstlast.add(r.last+1) # r.last is inclusive, so look at next
            pass

        # Expect each axis covers full range => bounds of each range were
        # contained in the bounds of at least one rectangle
        if 1 not in firstlast:
            raise CSException('iterations in region list do not extend ' +
                              'to lower bound')
        if np.inf not in firstlast:
            raise CSException('iterations in region list do not extend ' +
                              'to upper bound')
        if 0 not in lowhigh:
            raise CSException('contrasts in region list do not extend ' +
                              'to lower bound')
        if np.inf not in lowhigh:
            raise CSException('contrasts in region list do not extend ' +
                              'to upper bound')

        # collect the list of low and high values, and first and last+1
        # values, and cycle through every 2-tuple corner and be sure that
        # it is included in a box.  The set of all rectangles in the
        # region is built from the set of sub-rectangles that fit between
        # grid lines.  Ugly and scales very poorly, but effective.
        for contrast in lowhigh:
            for iteration in firstlast:
                if not _does_point_overlap_list(iteration,
                                                contrast,
                                                rlist):
                    raise CSException('Subregion of iteration/contrast ' +
                                      'space not covered in control ' +
                                      'strategy')
                pass
            pass
        pass

    # "fixedbp" shall be a key with a single value which is a string.
    # This string should contain a path to a FITS file with a 2D fixed bad
    # pixel map in its primary HDU. However, validation of this file is
    # out of scope for the specification.

    if not isinstance(csin['fixedbp'], str):
        raise CSException('fixedbp must be str')

    if usefiles:
        csin['fixedbp'] = os.path.expandvars(csin['fixedbp'])
        if os.path.isabs(csin['fixedbp']):
            if not os.path.exists(csin['fixedbp']):
                raise IOError('Absolute path to ' +
                                      'fixedbp file could ' +
                                      'not be found')
            pass
        else:
            localpath = os.path.dirname(os.path.abspath(fn))
            csin['fixedbp'] = os.path.join(localpath, csin['fixedbp'])
            if not os.path.exists(csin['fixedbp']):
                raise IOError('Relative path to ' +
                                      'fixedbp file could ' +
                                      'not be found')
            pass
        pass

        # check FITS file with one HDU
        try:
            hdus = pyfits.open(csin['fixedbp']) # HDUList
            pass
        except FileNotFoundError:
            raise IOError('fixedbp file could not be found')
        if len(hdus) != 1:
            raise IOError('fixedbp file should contain a single HDU')

        check.twoD_array(hdus[0].data, 'hdus[0].data {fixedbp}', TypeError)

        fixedbp = hdus[0].data.astype('bool')
        pass
    else:
        fixedbp = csin['fixedbp']
        pass

    return (regularization,
            pixelweights,
            pixelweights_fn,
            dmmultgain,
            probeheight,
            unprobedsnr,
            probedsnr,
            fixedbp)



#--------------------
# Box overlap checks
#--------------------

def _does_region_overlap_list(region, rlist):
    """
    Check if a region overlaps any of a list of regions

    Arguments:
     region: a Region object
     rlist: a list of Region objects

    Returns:
     True if any region in the list overlaps the input region, False otherwise

    """
    for r in rlist:
        if _do_boxes_overlap(region, r):
            return True
        pass
    return False


def _do_boxes_overlap(region1, region2):
    """
    Check if two boxes overlap

    Two boxes overlap if and only if their ranges overlap along each axis.

    Arguments:
     region1: a Region object
     region2: a Region object

    Returns:
     True if regions overlap, False otherwise

    """

    return ((region1.first <= region2.last) and
            (region2.first <= region1.last) and
            (region1.low < region2.high) and
            (region2.low < region1.high))

#----------------------
# Point overlap checks
#----------------------

def _does_point_overlap_list(iteration, contrast, rlist):
    """
    Check if an iteration and contrast point overlaps any of a list of regions

    Arguments:
     iteration: integer >= 1
     contrast: floating-point value >= 0
     rlist: a list of Region objects

    Returns:
     True if any region in the list overlaps the input point, False otherwise

    """
    for r in rlist:
        if is_point_in_box(iteration, contrast, r):
            return True
        pass
    return False


def is_point_in_box(iteration, contrast, region):
    """
    Check if an iteration and contrast point are in a particular region

    Arguments:
     iteration: integer >= 1
     contrast: floating-point value >= 0
     region: a Region object

    Returns:
     True if the point is in the region, False otherwise

    """
    # first and last are inclusive
    does_iteration_overlap = ((iteration >= region.first) and
                              (iteration <= region.last))

    # low is inclusive, high is exclusive (except infinity)
    does_contrast_overlap = (((contrast >= region.low) and
                              (contrast < region.high)) or
                             (np.isinf(contrast) and np.isinf(region.high)))

    return does_iteration_overlap and does_contrast_overlap
