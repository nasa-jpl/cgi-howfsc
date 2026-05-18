# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Standalone function to enforce all voltage constraints (voltage high/low
limits, DM neighbor rules, tied and dead actuators)
"""

import numpy as np

from . import check
from .dmsmooth import dmsmooth
from .flat_tie import checktie, checkflat

DME_LSB = 110.0/2**16 # DAC LSB (not same as required min accuracy)
VMIN = 0

class ConstrainDMException(Exception):
    """
    Thin exception class for errors not covered by exceptions in the Python
    standard library
    """
    pass

def constrain_dm(volts,
                 flatmap,
                 tie,
                 vmax=100.0,
                 vlat=50.0,
                 vdiag=75.0,
                 vquant=DME_LSB,
                 maxiter=10000):
    """
    Given a DM setting, return one consistent with physical and user
    mandated constraints.

    Constrains each individual voltage to be in 0 <= v <= `vmax`.  Use of
    defaults (min voltage=0V, `vmax`=100V) enforces clip requirements in DNG
    884740 and 884741.

    Constrains each pair of laterally-adjacent actuators to be <= `vlat` after
    subtraction of the DM flat map in `flatmap`. Use of default (`vlat`=50V)
    enforces neighbor-rule requirements in DNG 884742 and 884743.

    Constrains each pair of diagonally-adjacent actuators to be <= `vdiag`
    after subtraction of the DM flat map in `flatmap`. Use of default
    (`vdiag`=75V) enforces neighbor-rule requirements in DNG 1073291 and
    1073292.

    Constrains all tied actuators (groups in the `tie` matrix with value > 0)
    to have the same voltage.  Constrains all dead actuators (groups in the
    `tie` matrix with value = -1) to be 0V.

    Note that in practical use, the flatmap and tie matrices have certain
    expectations on their format:
     - flatmap should be >= 0, <= vmax, have all ties at the same voltage,
      and have all dead actuators at 0V voltage.
     - tie should be -1, 0, or the integer range 1-N for some N (with no gaps)
    These expectations will be enforced.

    NOTE: this function assumes that the minimum commandable voltage is equal
    to the dead actuator voltage == 0V, and this also corresponds to 0x0000 in
    the DAC (so exactly the bottom value).  All of these things are true for
    CGI in its current implementation, but if this were to be revisited in
    another use case where these assumptions are not valid, then the logic here
    needs to be revisited.  (In particular, dead actuator voltage < minimum
    commandable voltage makes strange edge effects, as does the case where the
    voltage corresponding to the DAC value 0x0000 at the lower edge is not 0V
    but a dead actuator is.)

    Arguments:
     volts: a 2D array of floating-point voltages.  This is the set of voltages
      which we are fixing before sending to the DM.
     flatmap: a 2D array of floating-point voltages, of the same size as
      `volts`.  This array represents a physically-flat DM surface.
     tie: a 2D array of integers, of the same size as `volts`, which can take
      on the values 0, -1, or consecutive integers 1 -> N.

    Keyword Arguments:
     vmax: maximum commandable voltage, in volts.  Floating-point scalar, must
      be > 0. Defaults to 100V, which is the max voltage
      for CGI.
     vlat: maximum allowable voltage differential between laterally-adjacent
      actuators, in volts.  Floating-point scalar > 0.  Defaults to 50V, which
      is the CGI requirement.
     vdiag: maximum allowable voltage differential between diagonally-adjacent
      actuators, in volts.  Floating-point scalar > 0.  Defaults to 75V, which
      is the CGI requirement.
     vquant: smallest voltage step (1 LSB) which the DME electronics can
      produce.  Used to keep the constraints from being broken after EU->DN
      conversion. Floating-point scalar > 0; using 0 is equivalent to not
      accounting for DAC discretization effects at all, and is not currently
      supported.  Defaults to 110/2^16, the CGI LSB.
     maxiter: number of times to iterate between smoothing and tying before
      giving up.  Smoothing and tying are both convergent and we do not
      expect to need this, but it seemed a reasonable safety measure
      against a having the 'while True' loop repeat indefinitely for some
      unforeseen corner case.  Defaults to 10000.

    Returns:
     a constrained 2D array of voltages

    """
    # Check argumemts
    check.twoD_array(volts, 'volts', TypeError)
    check.twoD_array(flatmap, 'flatmap', TypeError)
    check.twoD_array(tie, 'tie', TypeError)
    if volts.shape != flatmap.shape:
        raise TypeError('volts and flatmap must have the same shape')
    if volts.shape != tie.shape:
        raise TypeError('volts and tie must have the same shape')

    # Check keyword arguments
    check.real_positive_scalar(vmax, 'vmax', TypeError)
    check.real_positive_scalar(vlat, 'vlat', TypeError)
    check.real_positive_scalar(vdiag, 'vdiag', TypeError)
    check.real_positive_scalar(vquant, 'vquant', TypeError)
    check.positive_scalar_integer(maxiter, 'maxiter', TypeError)

    if vmax <= VMIN:
        raise ValueError('VMIN must be < vmax')

    # enforce tie and flat formatting
    if not checktie(tie):
        raise ValueError('tie must have values 0, -1, or consecutive ' +
                         'integers 1 -> N')
    if not checkflat(flatmap, VMIN, vmax, tie):
        raise ValueError('flatmap must be <= vmax, >= VMIN, have all tied ' +
                         'actuators tied already, and have all dead ' +
                         'actuators = 0V')


    # Run initial smoothing
    smoothed = dmsmooth(volts, vmax, vquant, vlat, vdiag, dmflat=flatmap)

    # Dummy array to initialize while loop
    tied = None

    # Loop tie and smooth unitl they converge
    i = 0
    while not (tied == smoothed).all() and i < maxiter:
        tied = tie_with_matrix(smoothed, tie)
        smoothed = dmsmooth(tied, vmax, vquant, vlat, vdiag, dmflat=flatmap)
        i += 1
    if i >= maxiter:
        raise ConstrainDMException('maxiter exceeded for constrain_dm')

    return smoothed



def tie_with_matrix(volts, tie):
    """
    Tie specified actuators to single value. The value is the mean value of
    all actuators.  This uses a matrix with to indicate which actuators are
    tied together, with a specific format:
     - 0 indicates no tie
     - -1 indicates dead (0V)
     - any other integer 1->N indicates a tie group; all actuators in that
      group will be assigned the mean voltage across that set of actuators.

    Arguments:
     volts: a 2D array of floating-point voltages.  This is the set of voltages
      which we are fixing before sending to the DM.
     tie: a 2D array of integers, of the same size as `volts`, which can take
      on the values 0, -1, or consecutive integers 1 -> N.

    Returns:
     Output DM array with tied values.

    """
    # dimensionality checks
    check.twoD_array(volts, 'volts', TypeError)
    check.twoD_array(tie, 'tie', TypeError)
    if volts.shape != tie.shape:
        raise TypeError('volts and tie must have the same shape')

    # enforce tie formatting
    if not checktie(tie):
        raise ValueError('tie must have values 0, -1, or consecutive ' +
                         'integers 1 -> N')

    # Loop through each DM mask in data cube
    dmtied = volts.copy()
    tienumset = set(tie.ravel())
    for tienum in tienumset:
        if tienum == 0:
            # not tied
            continue
        elif tienum == -1:
            # dead actuators
            dmtied[tie == tienum] = 0
            continue
        else:
            # Grab mean value of all actuators that are tied together
            # upcast to f128 for greater bit width (see PFR 218113)
            # use longdouble as no float128 on Windows (from CPP); no
            #  impact on linux
            mean_val = np.mean(dmtied[tie == tienum], dtype=np.longdouble)
            # Assign indices in DM mask to mean value, back as f64
            dmtied[tie == tienum] = mean_val.astype(np.float64)
            continue
        pass

    return dmtied
