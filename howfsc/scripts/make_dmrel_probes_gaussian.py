# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
# pylint: disable=line-too-long
"""
Functions to build Gaussian relative DM probes for unit test configurations
"""

import os
import argparse

import numpy as np
import astropy.io.fits as pyfits

import howfsc
from howfsc.model.mode import CoronagraphMode
from howfsc.util.prop_tools import efield, open_efield, make_dmrel_probe_gaussian

if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog='python make_dmrel_probes_gaussian.py', description="Create Gaussian relative DM probe settings corresponding to a desired probe height for howfsc unit test configurations.  Run with 'python -i' to look at outputs (focal-plane intensities in npint0/npint1/npint2)")
    ap.add_argument('-t', '--target', default=1e-5, help="Target probe height, defaults to 1e-5.", type=float)
    ap.add_argument('--mode', default='nfov_dm', choices=['narrowfov', 'nfov_dm', 'nfov_flat'], help="coronagraph mode from test data; must be one of 'narrowfov', 'nfov_dm', or 'nfov_flat'.  Defaults to 'nfov_dm'.  Note: spectroscopy mode is not supported for Gaussian probes.")
    ap.add_argument('--dark_hole', default='360deg', choices=['360deg'], help="Dark hole geometry; must be one of '360deg'.  Defaults to '360deg'.")
    ap.add_argument('--write', action='store_true', help="write FITS file outputs to the howfsc/model/testdata/ directory")
    args = ap.parse_args()

    # User params
    mode = args.mode
    dark_hole = args.dark_hole
    _target = args.target
    write = True #args.write

    howfscpath = os.path.dirname(os.path.abspath(howfsc.__file__))

    # kwargs need to be tuned per mode--get area right, and most importantly
    # get center right.  If center is obscured, probe might be very large
    # artificially.  Need to look at probe images overlaid on pupil plane
    # obscurations (front + SP + Lyot).
    if mode == 'nfov_dm':
        modelpath = os.path.join(howfscpath, 'model', 'testdata', 'narrowfov')
        cfgpath = os.path.join(modelpath, 'narrowfov_dm.yaml')
        _cfg = CoronagraphMode(cfgpath)

        if '360' in dark_hole:
            sigma = 1.
            deltax_act_list = [13.5, 12.5, 13.5]
            deltay_act_list = [8.5, 8.5, 7.5]

        kwargs = {'lod_min': 6,
                  'lod_max': 9,
                  'maxiter': 5,
                  }
        pass

    elif mode == 'narrowfov':
        modelpath = os.path.join(howfscpath, 'model', 'testdata', 'narrowfov')
        cfgpath = os.path.join(modelpath, 'narrowfov.yaml')
        _cfg = CoronagraphMode(cfgpath)

        if '360' in dark_hole:
            sigma = 1.
            deltax_act_list = [13.5, 12.5, 13.5]
            deltay_act_list = [8.5, 8.5, 7.5]

        kwargs = {'lod_min': 6,
                  'lod_max': 9,
                  'maxiter': 5,
                  }
        pass

    elif mode == 'nfov_flat':
        modelpath = os.path.join(howfscpath, 'model', 'testdata', 'narrowfov')
        cfgpath = os.path.join(modelpath, 'narrowfov_flat.yaml')
        _cfg = CoronagraphMode(cfgpath)

        if '360' in dark_hole:
            sigma = 1.
            deltax_act_list = [13.5, 12.5, 13.5]
            deltay_act_list = [8.5, 8.5, 7.5]

        kwargs = {'lod_min': 6,
                  'lod_max': 9,
                  'maxiter': 5,
                  }
        pass

    else:
        # Gaussian probes not tested with WFOV or SPEC yet
        raise ValueError('Invalid mode name for Gaussian probes "' + str(mode) + '"')


    _dmlist = _cfg.initmaps
    _ind = 0
    _iopen = np.abs(open_efield(_cfg, _dmlist, _ind))**2
    _ipeak = np.max(_iopen)

    probe_name_list = ['gauss0', 'gauss1', 'gauss2']

    for index_probe in range(len(deltax_act_list)):
        xcenter = deltax_act_list[index_probe]
        ycenter = deltay_act_list[index_probe]
        probe_name = probe_name_list[index_probe]

        dpout, _probe_int, _lod_mask, _dm_surface, _pupil_mask = make_dmrel_probe_gaussian(
            cfg=_cfg, dmlist=_dmlist, xcenter=xcenter, ycenter=ycenter,
            sigma=sigma, target=_target, ind=_ind, **kwargs)

        _eplus = efield(_cfg, [_dmlist[0]+dpout, _dmlist[1]], _ind)
        _eminus = efield(_cfg, [_dmlist[0]-dpout, _dmlist[1]], _ind)
        _pampe = np.abs((_eplus - _eminus)/2j)
        locals()['npint' + str(index_probe)] = _pampe**2/_ipeak

        if write:
            fnbase = mode + '_dmrel_' + '{:.1e}'.format(_target) + '_' + probe_name + '.fits'
            fn = os.path.join(modelpath, fnbase)
            pyfits.writeto(fn, dpout, overwrite=True)
            pass

        pass

    pass