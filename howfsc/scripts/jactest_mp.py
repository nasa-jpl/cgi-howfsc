# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Compute a Jacobian for the mode defined in cfg

Use multiprocessing! (not good for translation, but great for testing)
"""

import logging
import time
import os
import argparse
import multiprocessing

import numpy as np
import astropy.io.fits as pyfits

import howfsc
from howfsc.control.calcjacs import calcjacs
from howfsc.model.mode import CoronagraphMode

import howfsc.util.check as check

JACMETHOD = 'fast'


def calculate_jacobian_multiprocessed(mode='narrowfov', howfsc_file=howfsc.__file__, nact=None, output=None, proc=None, num_threads=None):
    """
    Compute a full or partial Jacobian using multiple processors.

    Stores the output in a FITS file, if a name is provided, and prints the time spent to do the calculation.
    The filename must not be used already, as this script will not overwrite an existing file. The output file
    will be a 2 x (num actuators) x (num pixels) 3D array. The first axis with 2 dimensions will be the real
    and imaginary parts of the Jacobian, as FITS files cannot store complex data directly.

    Parameters:
    -----------
    mode : str, optional
        Coronagraph mode from test data; must be one of 'widefov', 'narrowfov' (default), 'nfov_dm', 'nfov_flat', or 'spectroscopy'.
    nact : int, optional
        Number of actuators in Jacobian to compute. If unspecified, will compute all actuators in configuration.
    output : str, optional
        Output file for Jacobian, including .fits extension.
    proc : int, optional
        Number of processes, defaults to 1. If set to 0, uses half the available CPU cores.
    num_threads : int, optional
        Sets os.environ['MKL_NUM_THREADS']=num_threads. Default uses os.environ['MKL_NUM_THREADS'] if already set,
        or os.environ['HOWFS_CALCJAC_NUM_THREADS'], or if none are defined, standard OS threading.
    """
    if output is not None and os.path.isfile(output):
        raise Exception("Output file exists")

    # Load cfg
    if mode == 'widefov':
        print('mode = widefov')
        cfgpath = os.path.join(os.path.dirname(
                       os.path.abspath(howfsc_file)),
                       'model', 'testdata', 'widefov', 'widefov.yaml')
        pass
    elif mode == 'narrowfov':
        print('mode = narrowfov')
        cfgpath = os.path.join(os.path.dirname(
                       os.path.abspath(howfsc_file)),
                       'model', 'testdata', 'narrowfov', 'narrowfov.yaml')
        pass
    elif mode == 'spectroscopy':
        print('mode = spectroscopy')
        cfgpath = os.path.join(os.path.dirname(
                     os.path.abspath(howfsc_file)),
                     'model', 'testdata', 'spectroscopy', 'spectroscopy.yaml')
        pass
    elif mode == 'nfov_dm':
        print('mode = nfov_dm')
        cfgpath = os.path.join(os.path.dirname(
                     os.path.abspath(howfsc_file)),
                     'model', 'testdata', 'narrowfov', 'narrowfov_dm.yaml')
        pass
    elif mode == 'nfov_flat':
        print('mode = nfov_flat')
        cfgpath = os.path.join(os.path.dirname(
                     os.path.abspath(howfsc_file)),
                     'model', 'testdata', 'narrowfov', 'narrowfov_flat.yaml')
        pass
    elif mode == 'nfov_band1_flat':
        print('mode = nfov_band1_flat')
        # TODO: fix
        cfgpath = os.path.join(os.path.dirname(
                     os.path.abspath(howfsc_file)),
                     'model', 'testdata', 'nfov_band1_flat', 'narrowfov.yaml')
        pass

    elif mode == 'wfov_band4_flat':
        print('mode = wfov_band4_flat')
        # TODO: fix
        cfgpath = os.path.join(os.path.dirname(
                     os.path.abspath(howfsc_file)),
                     'model', 'testdata', 'wfov_band4_flat', 'widefov.yaml')
        pass

    else:
        # Should never reach this if choices=[..] works right
        raise Exception('Bad input for coronagraph mode')

    logging.basicConfig(level=logging.INFO)

    cfg = CoronagraphMode(cfgpath)

    # set num_process, see doc string above for order of precedence
    num_process = proc
    if num_process is None:
        # default input argument is None
        num_process = int(os.environ.get('HOWFS_CALCJAC_NUM_PROCESS', 1))

    check.nonnegative_scalar_integer(num_process, 'num_process', TypeError)

    # special case
    if num_process == 0:
        num_process = multiprocessing.cpu_count()//2

    # set num_threads, see doc string above for order of precedence
    num_threads = num_threads
    if num_threads is None:
        # default input argument is None
        # os.environ.get('key') returns None if 'key' doesn't exist
        num_threads = os.environ.get('HOWFS_CALCJAC_NUM_THREADS')

        if num_threads is None and os.environ.get('MKL_NUM_THREADS') is None \
           and num_process > 1:

            num_threads = 1

        # if os.environ['MKL_NUM_THREADS'] is defined,
        #  leave num_threads = None,
        # if num_process == 1, leave num_threads = None,

    # check valid num_threads
    if not num_threads is None:
        # environ variables are strings, e.g. '1'
        if isinstance(num_threads, str):
            num_threads = int(num_threads)
        check.positive_scalar_integer(num_threads, 'num_threads', TypeError)

    #------------------
    # Jacobian
    #------------------

    # make ijlist using given nact
    # using None will use the initmaps starting setting built into cfg
    dm0list = None

    # use nact for actual number of actuators to run through
    if nact is not None:
        nact = int(nact)
        ijlist = range(nact)
        pass
    else:
        # Run everything
        ndmact = np.cumsum([0]+[d.registration['nact']**2
                                    for d in cfg.dmlist])[-1]
        nact = int(ndmact)
        ijlist = range(nact)
        pass

    # log start time and calc jac
    logging.info('Begin Jacobian calculation')
    t0 = time.time()

    # log number processes
    logging.info('multiprocessing using %d processes', num_process)

    # if defined, set num_threads and log
    if not num_threads is None:

        # local means we will unset after calcjacs
        bUnset_mkl_num_threads = True

        # if os.environ['MKL_NUM_THREADS'] already has a value, need to save it
        save_mkl_num_threads = os.environ['MKL_NUM_THREADS'] \
                               if 'MKL_NUM_THREADS' in os.environ else None

        # set
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        logging.info('setting os.environ[MKL_NUM_THREADS] %d threads',
                     num_threads)

    else:
        # nothing to unset
        bUnset_mkl_num_threads = False

        # if environment variable set exernally, do nothing just log
        if 'MKL_NUM_THREADS' in os.environ:
            logging.info('MKL_NUM_THREADS = %s threads',
                         os.environ['MKL_NUM_THREADS'])

    # single process or multiprocessing
    jac = calcjacs(cfg, ijlist, dm0list, jacmethod=JACMETHOD,
                   num_process=num_process)

    # return num threads to default
    if bUnset_mkl_num_threads:
        logging.info("restoring os.environ['MKL_NUM_THREADS'] to " +
                     "previous value")

        if not save_mkl_num_threads is None:
            os.environ['MKL_NUM_THREADS'] = save_mkl_num_threads

        else:
            del os.environ['MKL_NUM_THREADS']

    t1 = time.time()
    logging.info('Jacobian calculation complete: elapsed time = %f secs',
                 t1-t0)

    if output is not None:
        logging.info('Writing jac to: ' + output)
        pyfits.writeto(output, jac)
        pass


if __name__ == "__main__":

    # setup for cmd line args
    ap = argparse.ArgumentParser(prog='python jactest_mp.py', description="Compute a full or partial Jacobian using multiple processors.  Stores the output in a FITS file, if a name is provided, and prints the time spent to do the calculation.  The filename must not be used already, as this script will not overwrite an existing file.  The output file will be a 2 x (num actuators) x (num pixels) 3D array. The first axis with 2 dimensions will be the real and imaginary parts of the Jacobian, as FITS files cannot store complex data directly.")

    ap.add_argument('-n', '--nact', help="number of actuators in Jacobian to compute.  If unspecified, will compute all actuators in configuration", type=int)

    ap.add_argument('-o', '--output', help="output file for Jacobian; will be in FITS format", type=str)

    ap.add_argument('--mode', default='widefov', choices=['widefov', 'narrowfov', 'spectroscopy', 'nfov_dm', 'nfov_flat'], help="coronagraph mode from test data; must be one of 'widefov' (default), 'narrowfov', 'nfov_dm', 'nfov_flat', or 'spectroscopy'")

    ap.add_argument('-p', '--proc', default=None,
                    help="number of processes, defaults to 1", type=int)

    ap.add_argument('--num_threads', default=None,
                    help="set os.environ['MKL_NUM_THREADS']=num_threads, default uses os.environ['MKL_NUM_THREADS'] if already set, or os.environ['HOWFS_CALCJAC_NUM_THREADS'] or if none are defined standard os threading", type=int)

    # num_process: number of parallel processes to use when calculating the
    #  Jacobian. If value is 0 then then
    #  num_process = multiprocessing.cpu_count()//2, which is the number
    #  of cores on computers with Intel CPUs.
    #  If not defined (default=None), then num_process is:
    #  os.environ['HOWFS_CALCJAC_NUM_PROCESS'], if defined,
    #  else no multiprocessing
    # num_threads: the mkl library automatically uses multi-threading
    #  which causes bottle necks and results in slower performance when using
    #  parallel processes. With Intel CPUs with multiple cores, best
    #  performance is usually with parallel processing and num_threads = 1.
    #  If not defined (default=None), order of precedence is:
    #  os.environ['HOWFS_CALCJAC_NUM_THREADS'], if defined, else
    #  os.environ['MKL_NUM_THREADS'], if defined, else
    #  1, if num_process is defined and > 1, else
    #  num_threads is not set and the default OS threading is used. For most
    #  linux/Intel machines, the OS will use max number of threads.
    #  Note: if os.environ['MKL_NUM_THREADS'] is defined, this value sets
    #  number of threads for all computations, not just calcjacs(). The
    #  computation time of single process functions might be longer.

    args = ap.parse_args()

    output = args.output
    mode = args.mode
    proc = args.proc
    num_threads = args.num_threads
    nact = args.nact

    calculate_jacobian_multiprocessed(output, mode, proc, num_threads, nact)
