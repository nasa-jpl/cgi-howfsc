# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
class library for creating and applying DM crosstalk
"""
import os

import numpy as np
import yaml
import scipy.sparse as sparse
import scipy.sparse.linalg as sp_linalg

import howfsc.util.check as check
from howfsc.util.load import load

DEFAULT_YAML_SAVE = 'DMCrosstalk_temporary_data.yaml'

class CDMCrosstalk:
    """
    DM crosstalk object stores the crosstalk data and includes methods for
    reading, storing, and applying DM crosstalk
    """
    def __init__(self, yaml_fn=None,
                 list_xtalk_arrays=None, list_off_row_col=None,
                 list_array_roll=None, list_flip_x=None):
        """
        class object definition for implementing DM crosstalk. DM crosstalk
        is defined by:
                dm_actual[i, j] = dm_command[i, j]
                          + crosstalk[i, j] * dm_command[i-irow, j-jcol]
        (Equation 1)
        
        Arguments:
         yaml_fn: path+filename of yaml file with crosstalk data

         list_xtalk_arrays: a list of nact x nact arrays representing DM
          crosstalk, requires list_off_row_col of same length

         list_off_row_col: a list of [irow, jcol] actuator offset for applying
           xtalk, not used if list_xtalk_arrays is None

         list_array_roll: a list of [roll_row, roll_col] to be applied to the 
           xtalk_arrays before creating the sparse matrix. Without this roll, the
           application of crosstalk is as defined in Equation 1 above. If array_roll is defined,
                dm_actual[i, j] = dm_command[i, j]
                          + crosstalk[i-roll_row, j-roll_col] * dm_command[i-irow, j-jcol]
           (Equation 2)
           
           roll is not used if list_xtalk_arrays is None

         list_flip_x: a list of bool, same length as list_xtalk_arrays. If True,
           then the xtalk_array is flip-X (columns) before inserted in the Crosstalk
           object sparse matrix. Order is: flip-x then roll (if any).
           not used if list_xtalk_arrays is None

         if no inputs, a crosstalk object of identity (no crosstalk) is created

         create a Cdm_crosstalk object by either reading from the yaml_fn
         and/or loading from a list_xtalk_array with list_off_row_col offsets.
         If both a yaml_fn and the list of xtalk arrays are input, then an
         object will be constructed aggregating both the data from the yaml and
         the arrays.

        """

        # first initialize to no crosstalk
        # the rest of HOWFS, such as registration, assumes square DM array,
        # same here
        self.nact = None
        self.HC_sparse = None
        self.list_k_diag = [] # only used for record and test

        # if given, load data from yaml file, the method does all the checking
        if not yaml_fn is None:
            self.read_crosstalk_yaml(yaml_fn)

        # now load crosstalk data if given
        if not list_xtalk_arrays is None:
            # consistency checks
            if not isinstance(list_xtalk_arrays, list):
                raise TypeError('list_xtalk_arrays must be list')
            if not isinstance(list_off_row_col, list):
                raise TypeError('list_off_row_col must be list')

            if len(list_off_row_col) != len(list_xtalk_arrays):
                raise ValueError('all list arguments must be same length')

            if list_array_roll and not isinstance(list_array_roll, list):
                raise TypeError('list_array_roll must be list or None')
            
            if list_flip_x and not isinstance(list_flip_x, list):
                raise TypeError('list_flip_x must be list or None')

            if isinstance(list_array_roll, list) or isinstance(list_flip_x, list):
                # apply roll and flip to xtalk arrays as a pre-processing step
                list_xtalk_arrays = apply_roll_flip(
                    list_xtalk_arrays, list_array_roll, list_flip_x
                )

            # the parameters of crosstalk in the list are checked when added

            # for each in the list, add the xtalk to
            # the appropriate diagonal of self.HC_sparse
            for xtalk, off_i_j in zip(list_xtalk_arrays, list_off_row_col):
                self.add_xtalk_array(xtalk, off_i_j)

    def init_HC_sparse(self, nact):
        """
        creates the sparse matrix initializing to identity matrix
        of size (nact*nact, nact*nact).

        this init must be called before adding crosstalk data. It
        is called by:
          add_xtalk_diagonal(), add_xtalk_array(), read_crosstalk_yaml()
        when necessary.

        Arguments:
         nact: number of dm actuators.

        Returns:

        """
        # if this is the first crosstalk data, initialize the sparse matrix
        if self.HC_sparse is None:

            # note: sparse.csc_matrix is best type for linear algebra
            self.nact = nact
            self.HC_sparse = sparse.csc_matrix(sparse.eye(nact*nact))

    def add_xtalk_diagonal(self, xtalk, k_diag):
        """
        Given 1-d array of crosstalk coefficients, add to the appropriate
        diagonal of the sparse array.

        Assumes self.HC_sparse, self.nact have already been initialized

        Arguments:
         xtalk: 1-d array of crosstalk coefficients. Must be length
          (nact*nact,),

          A flattened 2-d array of naxt x nact coeffients
          is length (nact*nact,). The length of the k-th diagonal of the
          sparse array is
          (nact*nact - k,). We automatically truncate the first or
          last k_diag elements, depending on the
          sign of k_diag.
         k_diag: scalar integer, in the range -(nact**2 - 1) to (nact**2 - 1),
          indicating the diagonal of the sparse matrix to insert the xtalk
          coefficients.

        Returns:
        """

        # Check inputs
        check.oneD_array(xtalk, 'xtalk', TypeError)
        check.real_array(xtalk, 'xtalk', TypeError)
        xtalk = np.array(xtalk) # force to numpy array
        check.scalar_integer(k_diag, 'k_diag', TypeError)

        # check already initialized
        if self.HC_sparse is None or self.nact is None:
            raise ValueError(
               'must initialize with init_HC_sparse() before adding diagonals')

        if k_diag < -(self.nact*self.nact - 1) or \
           k_diag > (self.nact*self.nact-1):
            raise ValueError('k_diag is out of limits')

        # create a sparse matrix version of the crosstalk operator
        # Hc = I + diag(xtalk, k=1)
        # then, (matlab notation)
        #       Hc * command = actual
        #       command = Hc \ actual
        Ccross_sparse = sparse.dia_matrix((self.nact*self.nact,
                                           self.nact*self.nact))

        # for offsets above the diagonal, need to offset the first actuator
        # otherwise the assignment to diagonal starts with wrong actuator
        if xtalk.shape != (self.nact*self.nact,):
            raise ValueError('xtalk length is not valid')

        if k_diag < 0:
            xtalk = xtalk[-k_diag:]
        elif k_diag > 0:
            xtalk = xtalk[:-k_diag]

        # set the appropriate diagonal to the crosstalk values
        Ccross_sparse.setdiag(xtalk, k=k_diag)

        # add the new diagonal to the object
        self.HC_sparse += Ccross_sparse

        # record this k_diag
        self.list_k_diag.append(k_diag)


    def add_xtalk_array(self, xtalk_array, off_i_j):
        """
        Add the crosstalk coefficients in the array xtalk_array to the sparse
        matrix

        Arguments:
         xtalk_array: 2-d array, nact x nact real doubles
         off_i_j: [irow, jcol] is the actuator offset for applying crosstalk
          for example, if off_i_j = [0, 1], then
          dm_actual = command + xtalk_array * np.roll(command, -1, axis=1)

        Returns:

        """

        # check this new xtalk_array
        check.twoD_array(xtalk_array, 'xtalk_array', TypeError)
        check.real_array(xtalk_array, 'xtalk_array', TypeError)
        nact, nactc = xtalk_array.shape
        if nact != nactc:
            raise TypeError('crosstalk arrays must be square')

        # init the HC_sparse if necessary, also sets self.nact
        if self.HC_sparse is None:
            self.init_HC_sparse(nact)

        # continue checking
        if xtalk_array.shape != (self.nact, self.nact):
            raise ValueError('crosstalk arrays not all the same shape')

        check.oneD_array(off_i_j, 'off_i_j', TypeError)
        if len(off_i_j) != 2:
            raise TypeError('off_i_j must be length 2')

        offset_i, offset_j = off_i_j # deals
        check.scalar_integer(offset_i, 'offset_i', TypeError)
        check.scalar_integer(offset_j, 'offset_j', TypeError)

        # force crosstalk of the 'wrap-around' row or column to zero
        if offset_i > 0:
            xtalk_array[-offset_i:, :] = 0.0
        if offset_i < 0:
            xtalk_array[:-offset_i, :] = 0.0
        if offset_j > 0:
            xtalk_array[:, -offset_j:] = 0.0
        if offset_j < 0:
            xtalk_array[:, :-offset_j] = 0.0

        # flatten and reshape
        xtalk_flat = xtalk_array.flatten()

        # add the flattened array as a diagonal
        self.add_xtalk_diagonal(xtalk_flat, self.k_diag(offset_i, offset_j))


    def write_crosstalk_yaml(self, yaml_fn):
        """
        Save the crosstalk sparse array to a yaml file

        Save only the off-diagonals. The main diagonal is always eye(),
        no need to save
        The yaml file will be a list of dict, one per offset diagonal

        Arguments:
         yaml_fn: filename, with path, to write the crosstalk data. If the
          files exists, it will be overwritten.

        Returns:

        """

        # convert HC_sparse into sparse.dia_matrix
        HC_sparse = self.HC_sparse.todia()

        # collect all the nonzero off-diagonals
        list_diag = [HC_sparse.diagonal(k) for k in self.list_k_diag]

        # debug check: reconstruct sparse matrix from diagonals
        tmp_sparse = sparse.dia_matrix(sparse.eye(self.nact*self.nact))
        for k, d in zip(self.list_k_diag, list_diag):
            tmp_sparse.setdiag(d, k)
        # True if the sparse matrices are ==
        if (tmp_sparse != HC_sparse).nnz != 0:
            raise Exception('list_k_diag did not capture all the diagonals')

        # for each nonzero off-diagonal,
        #    make a dict()
        #    add dict to list_Yaml

        # build list of dict for writing to yaml
        # for each nonzero off-diagonal
        list_Yaml = []
        for k, diag in zip(self.list_k_diag, list_diag):

            # yaml cannot handle numpy arrays, or numpy int or double
            diag = diag.tolist()

            # offset delta actuator row, col from k
            off_i, off_j = self.ij_from_k(k)

            # off diagonals have < nact*nact elements. prepend or postpend
            # zeros so that nact*nact values are written, making it easy to
            # reshape as an array of actuators for display, etc.
            # for offsets above the diagonal (k<0), need to prepend zeros
            # for offsets below the diagonal (k>0), need to postpend zeros
            if k < 0:
                diag = [0.0]*(-k) + diag
            if k > 0:
                diag = diag + [0.0]*k

            # add this diagonal as a dict to the list
            list_Yaml.append({
                'nact':self.nact,
                'off_i':off_i,
                'off_j':off_j,
                'k_diag':k,
                'crosstalk':diag, #list_crosstalk,
                # yaml cannot handle numpy array
            })

        # write list of dict to yaml
        with open(os.path.expandvars(yaml_fn), 'wt') as fid:
            yaml.safe_dump_all(list_Yaml, fid)


    def read_crosstalk_yaml(self, yaml_fn):
        """
        Read the crosstalk data from a yaml file.

        If this instance (self) already contains crosstalk data, this new
        crosstalk will be added if: nact agrees

        If the existing crosstalk already contains data on the same diagonal
        (offset) as the new crosstalk data, the data will be added to the old.

        If this instance (self) is empty (self.HC_sparse = None), then create
        a new sparse matrix, etc.

        Assumes the format of the yaml file is identical to the saved format in
        write_crosstalk_yaml() above.

        Arguments:
         yaml_fn: filename, with path, to read the crosstalk data.

        Returns:
         success: True if no error, or False
        """

        try:
            with open(os.path.expandvars(yaml_fn), 'rt') as fid:
                # gen_obj is a generator,
                gen_obj = yaml.safe_load_all(fid)

                # iterate to get each crosstalk offset (diagonal) saved in the
                # yaml
                for xtalk_k in gen_obj:

                    # xtalk_k is a dict, as written in write_crosstalk_yaml()
                    # check for the correct keys
                    if not all([akey in xtalk_k for akey in
                                ['nact', 'k_diag', 'off_i', 'off_j',
                                 'crosstalk']]):
                        raise ValueError('load yaml, missing key')

                    # init the HC_sparse if this is the very first data,
                    # sets self.nact
                    if self.HC_sparse is None:
                        self.init_HC_sparse(xtalk_k['nact'])

                    # check this crosstalk data is consistent with existing:
                    if xtalk_k['nact'] != self.nact:
                        raise ValueError('load yaml, mismatch nact')

                    # add to this object:
                    self.add_xtalk_diagonal(
                        np.array(xtalk_k['crosstalk']), xtalk_k['k_diag']
                    )

        except IOError:
            raise Exception('Error opening file, %s'%yaml_fn)
        except UnicodeDecodeError:
            raise Exception('File is not a valid YAML, %s'%yaml_fn)


    def crosstalk_forward(self, dm_command):
        """
        apply crosstalk to DM command
        return the "actual" DM displacement
        """

        # if no crosstalk, do nothing
        if self.HC_sparse is None:
            return dm_command

        if dm_command.shape != (self.nact, self.nact):
            raise ValueError('dm_command is not nact x nact')

        # dm_actual_flat = self.HC_sparse @ dm_command.flatten()
        dm_actual_flat = self.HC_sparse @ dm_command.flatten()

        # return dm_actual_flat.reshape((self.nact, self.nact))
        return dm_actual_flat.reshape((self.nact, self.nact))

    def crosstalk_backward(self, dm_actual):
        """
        determine the dm_command to create the dm_actual accounting for
        crosstalk

        return the "dm_command"

        crosstalk_backward() is the inverse of crosstalk_forward()
        crosstalk_backward(crosstalk_forward(dm_command)) should = dm_command
        """

        if self.HC_sparse is None:
            return dm_actual

        if dm_actual.shape != (self.nact, self.nact):
            raise ValueError('dm_actual is not nact x nact')

        # solve HC * dmh_command = dmh_actual
        # tmp = sparse.linalg.spsolve(self.HC_sparse, dm_actual.flatten())
        # return np.reshape(tmp, (self.nact, self.nact))
        tmp = sp_linalg.spsolve(self.HC_sparse, dm_actual.flatten())
        return np.reshape(tmp, (self.nact, self.nact))

    def ij_from_k(self, k):
        """
        return row, col offset (i, j) for a diagonal, k
        k can be positive or negative
        assumes row-major

        offset (i, j) are consistent with applying crosstalk:
        dm_actual = \
           command + \
           xtalk_array * np.roll( np.roll(command, -j, axis=1), -i, axis=0)

        Arguments:
         k: diagonal of HC_sparse

        Returns:
         off_i, off_j: integers
        """
        a = np.sign(k)
        k = abs(k)
        off_i = a*(k//self.nact)
        off_j = a*(k%self.nact)
        # return native int rather than numpy.int
        return int(off_i), int(off_j)

    def k_diag(self, i, j):
        """
        get diagonal k for row, col offset, positive and negative directions
        row major

        offset (i, j) are consistent with applying crosstalk:
        dm_actual = \
           command + \
           xtalk_array * np.roll( np.roll(command, -j, axis=1), -i, axis=0)

        Arguments:
         off_i, off_j: integers, row, col offset for applying crosstalk

        Returns:
         k: diagonal of HC_sparse
        """

        return self.nact*i + j

##### end class Cdm_crosstalk

def apply_roll_flip(list_xtalk_arrays, list_array_roll=None, list_flip_x=None):
    """
    Before adding an array (nact x nact) of coupling coefficients to the sparse
    crosstalk array, it might be necessary to shift (roll) the array and/or flip-x
    (A(x,y) => A(-x,y)) to match the conventions and the location of the DM mirror.

    The convention used in the CDCCrosstalk class is
                dm_actual[i, j] = dm_command[i, j]
                          + crosstalk[i, j] * dm_command[i-irow, j-jcol]

    If, for example, the crosstalk coefficients were determined using the convention:
            dm_actual[i, j] = dm_command[i, j]
                          + crosstalk[i-irow, j-jcol] * dm_command[i-irow, j-jcol]
    then the crosstalk array must translated (np.roll) by (irow, jcol) before adding
    to the CDMCrosstalk object.

    If the crosstalk coefficient array was determined using an "unfolded" optical
    model, but the physical DM is in a mirror-flipped position, then the crosstalk
    coefficient array must be mirrored, i.e. flip-x, before adding the the CDMCrosstalk
    object.

    This routine is a convenient utility for applying any necessary shift or flip-x
    to conform to the CDMCrosstalk convention. It is incorporated in 
    dm_crosstalk_fits_to_yaml() for creating a dm_crosstalk yaml, or it can be called
    separately.

    Order is: flip-x then roll (if any).

    Arguments:
     list_xtalk_arrays: list of numpy arrays of shape (nact, nact)

     list_array_roll: list of numpy arrays or lists of length 2. Each roll array is
       [row, col], and used directly in np.roll(A, rollij, axis=(0,1)), where A is a
       xtalk array and rollij is a [row, col]. Default=None (do nothing).

     list_flip_x: list of bool, True => A = fliplr(A), False => do nothing.
       Default=None (do nothing).
    

    """

    # validate inputs
    if not isinstance(list_xtalk_arrays, list):
        raise TypeError('list_xtalk_arrays must be a list')

    Narrays = len(list_xtalk_arrays)

    if list_flip_x:
        if isinstance(list_flip_x, list):
            if len(list_flip_x) != Narrays:
                raise ValueError('list_flip_x must be same length as list_xtalk_arrays')

        else:
            raise TypeError('list_flip_x must be list or None')

        if not all([isinstance(ff, bool) or ff is None for ff in list_flip_x]):
            raise TypeError('each value in flip_x must be bool')
        
        for ii, flip_x in enumerate(list_flip_x):
            if flip_x:
                list_xtalk_arrays[ii] = np.fliplr(list_xtalk_arrays[ii])

    if list_array_roll:
        if isinstance(list_array_roll, list):
            if len(list_array_roll) != Narrays:
                raise ValueError('list_array_roll must be same length as list_xtalk_arrays')

        else:
            raise TypeError('list_array_roll must be list or None')

        # established that list_array_roll is a list same length as list_xtalk_arrays
        # check that each in the list is a [roll_row, roll_col] integers
        for ii, rollij in enumerate(list_array_roll):
            if not len(rollij) == 2:
                raise ValueError('rollij is not length 2')
            # np.roll will check for proper type
            list_xtalk_arrays[ii] = np.roll(list_xtalk_arrays[ii], rollij, axis=(0, 1))


    return list_xtalk_arrays

################## interfaces to instantiate CDMCrosstalk objects
def dm_crosstalk_fits_to_yaml(list_xtalk_fits, list_off_row_col, yaml_save_fn,
                              list_array_roll=None, list_flip_x=None):
    """
    read a list of 2-d crosstalk arrays from fits files and create a
    dm_crosstalk yaml file to store with CoronagraphMode cfgfile

    Definition of crosstalk [irow, jcol]:
    dm_actual[i, j] = dm_command[i, j]
    + crosstalk[i, j] * dm_command[i-irow, j-jcol]

    Arguments:
     list_xtalk_fits: is a list of fits files, each containing a
      nact x nact real double array, representing DM crosstalk for one offset

     list_off_row_col: is a list of [irow, jcol] actuator offset for applying
      xtalk. The list must have one [irow, jcol] pair per fits file in
      list_xtalk_fits. If only one xtalk array, list_off_row_col should be
      like, [[ir, jc]], so that it passes the twoD_array() check.

     yaml_save_fn: path+filename of yaml file to store aggregate crosstalk data

     list_array_roll (optional): list of numpy arrays or lists of length 2. Each roll array is
       [row, col], and used directly in np.roll(A, rollij, axis=(0,1)), where A is a
       xtalk array and rollij is a [row, col]. Default=None (do nothing).

     list_flip_x (optional): list of bool, True => A = fliplr(A), False => do nothing.
       Default=None (do nothing).

    Returns:
     the CDMCrosstalk object. Use it wherever you want to apply DM crosstalk

    """

    # validate arguments
    # list_xtalk_fits is a list of existing fits files
    # and each contains arrays of the same size
    if not isinstance(list_xtalk_fits, list):
        raise TypeError('list_xtalk_fits is not a list')

    # load() and pyfits() do the error checking
    list_xtalk_arrays = [load(fn) for fn in list_xtalk_fits]

    # check list_off_row_col
    # even if only one xtalk is defined, e.g. [[1, 0]], it passes:
    check.twoD_array(np.array(list_off_row_col), 'list_off_row_col', TypeError)

    # apply optional roll and flip-x
    list_xtalk_arrays = apply_roll_flip(list_xtalk_arrays, list_array_roll, list_flip_x)
    
    # create the dm_crosstalk object, and call write to yaml
    # the object __init__() will do the rest of the checking
    objXtalk = CDMCrosstalk(list_xtalk_arrays=list_xtalk_arrays,
                            list_off_row_col=list_off_row_col)

    objXtalk.write_crosstalk_yaml(yaml_save_fn)

    return objXtalk

def dm_crosstalk(xtalk_fits, xtalk_yaml, yaml_save_fn=DEFAULT_YAML_SAVE):
    """
    Simplest interface to instantiate a CDMCrosstalk object from a single array
    of coupling coefficients

    Arguments:
     xtalk_fits: str, full filename of a fits file with a 2-D array of nact x nact
       coupling coefficients

     xtalk_yaml: str, full filename of a yaml file with the following parameters:
       required:
         off_row: int, apply the crosstalk coefficient to the actuator with this offset row
         off_col: int, apply the crosstalk coefficient to the actuator with this offset column
       optional:
         roll_row: int, np.roll(xtalk_array, roll_row, axis=0)
         roll_col: int, np.roll(xtalk_array, roll_col, axis=1)
         flip_x: bool, if True, xtalk_array = np.fliplr(xtalk_array)

     yaml_save_fn: str, save the CDMCrosstalk object to this yaml. After this yaml is
       created, the CDMCrosstalk object can be repeatedly instantiated by simply:
         objxtalk = CDMCrosstalk(yaml_save_fn)


    Returns:
     objxtalk, instance of class CDMCrosstalk
    """


    # input fits filename gets passed directly to dm_crosstalk_fits_to_yaml(),
    # and is validated there.

    # read parameter from yaml
    with open(os.path.expandvars(xtalk_yaml), 'rt') as fid:
        # params is a dict
        params = yaml.safe_load(fid)

    # required keys:
    if not all([kk in params for kk in ['off_row', 'off_col']]):
        raise ValueError('input yaml parameters must contain off_row and off_col')

    # parse params
    off_row_col = [params['off_row'], params['off_col']]

    # optional keys
    roll_row_col = [params.get('roll_row', 0), params.get('roll_col', 0)]
    # guard against None values
    roll_row_col = [0 if rr is None else rr for rr in roll_row_col]

    flip_x = params.get('flip_x', None)
    if not (isinstance(flip_x, bool) or flip_x is None):
        raise TypeError('parameter flip_x must be bool or None')

    objxtalk = dm_crosstalk_fits_to_yaml(
        [xtalk_fits,], [off_row_col], yaml_save_fn,
        list_array_roll=[roll_row_col], list_flip_x=[flip_x]
    )

    return objxtalk
