# Copyright 2025, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# Any commercial use must be negotiated with the Office of Technology Transfer
# at the California Institute of Technology.
"""
Class object for high-level coronagraph behavior
"""

import os

from howfsc.util.load import load, load_ap, load_ri
from howfsc.util.loadyaml import loadyaml

from howfsc.model.mask import (PupilMask, FocalPlaneMask, Epup, DMFace,
                               LyotStop, FieldStop)
from howfsc.model.dmobj import DM
from howfsc.model.singlelambda import SingleLambda
from howfsc.model.parse_mdf import check_mode_lists

class CoronagraphMode(object):
    """
    A class which contains all of the necessary data to operate CGI in a
    'coronagraph mode', one of the three baseline configurations for the
    instrument.

    It will contain two lists:
     sl_list (list of SingleLambda objects)
     dmlist (list of DMFace objects)
     initmaps (DM settings at which the initial wavefronts in epup apply)
    which together are enough to define a coronagraphic diffraction model.

    Arguments:
     cfgfile: a YAML file with three top-level dictionary equivalents, 'sls',
      'dms', and 'init', containing information on SingleLambda channels
      and DMFace deformable mirror objects.

    """
    def __init__(self, cfgfile):
        # Validate cfgfile against spec; no errors if this returns
        check_mode_lists(cfgfile, usefiles=True)

        # If validation succeeded, don't recheck it for errors
        y0 = loadyaml(cfgfile, custom_exception=IOError)
        localpath = os.path.dirname(os.path.abspath(
            os.path.expandvars(cfgfile)))

        # Load DMs first
        dmkeylist = ['DM1', 'DM2']
        dmlist = []
        for dmkey in dmkeylist:
            idm = y0['dms'][dmkey]

            gpath = os.path.join(localpath, idm['voltages']['gainfn'])
            fpath = os.path.join(localpath, idm['voltages']['flatfn'])
            tpath = os.path.join(localpath, idm['voltages']['tiefn'])

            # crosstalkfn is a yaml, or can be None
            xpath_tmp = idm['voltages']['crosstalkfn']
            xpath = None if xpath_tmp is None else os.path.join(localpath,
                                                                xpath_tmp)

            dmvobj = DM(vmax=idm['voltages']['vmax'],
                        vmin=idm['voltages']['vmin'],
                        vneighbor=idm['voltages']['vneighbor'],
                        vcorner=idm['voltages']['vcorner'],
                        vquant=idm['voltages']['vquant'],
                        gainmap=load(gpath),
                        flatmap=load(fpath),
                        tiemap=load(tpath),
                        crosstalkfn=xpath,
                        )

            rpath = os.path.join(localpath, idm['registration']['inffn'])
            registration = {'nact':idm['registration']['nact'],
                          'ppact_d':idm['registration']['ppact_d'],
                          'ppact_cx':idm['registration']['ppact_cx'],
                          'ppact_cy':idm['registration']['ppact_cy'],
                          'dx':idm['registration']['dx'],
                          'dy':idm['registration']['dy'],
                          'thact':idm['registration']['thact'],
                          'flipx':idm['registration']['flipx'],
                          'inf_func':load(rpath)}

            z = idm['z']
            pitch = idm['pitch']

            dmlist.append(DMFace(z, pitch, dmvobj, registration))
            pass
        self.dmlist = dmlist # assign once we loaded it successfully

        # Load DM settings used to collect channel data
        initmaps = []
        for dmkey in dmkeylist:
            ii = y0['init'][dmkey]
            ipath = os.path.join(localpath, ii['dminit'])
            initmap = load(ipath)
            initmaps.append(initmap)
            pass
        self.initmaps = initmaps

        # Load and build SingleLambdas last, one per channel
        slkeys = sorted(y0['sls'].keys()) # integers 0 to N-1
        sl_list = []
        for k in slkeys:
            isl = y0['sls'][k]
            # epup
            epup = Epup(lam=isl['lam'],
                        e=self._keyload(isl['epup'], path=localpath),
                        pixperpupil=isl['epup']['pdp'],
                        tip=isl['epup']['tip'],
                        tilt=isl['epup']['tilt'],
            )

            # sp
            pupil = PupilMask(lam=isl['lam'],
                              e=self._keyload(isl['sp'], path=localpath),
                              pixperpupil=isl['sp']['pdp']
            )

            # fpm
            fpm = FocalPlaneMask(lam=isl['lam'],
                                 e=self._keyload(isl['fpm'], path=localpath),
                                 isopen=isl['fpm']['isopen'],
                                 pixperlod=isl['fpm']['ppl']
            )

            # lyot
            lyot = LyotStop(lam=isl['lam'],
                            e=self._keyload(isl['lyot'], path=localpath),
                            pixperpupil=isl['lyot']['pdp'],
                            tip=isl['lyot']['tip'],
                            tilt=isl['lyot']['tilt'],
            )

            # fs
            fs = FieldStop(lam=isl['lam'],
                           e=self._keyload(isl['fs'], path=localpath),
                           pixperlod=isl['fs']['ppl']
            )


            # dh
            dh = FieldStop(lam=isl['lam'],
                           e=load(os.path.join(localpath, isl['dh'])),
                           pixperlod=isl['fs']['ppl'], # should match fs
            )

            # ft_dir
            ft_dir = isl['ft_dir']

            sl = SingleLambda(lam=isl['lam'],
                              epup=epup,
                              dmlist=self.dmlist,
                              pupil=pupil,
                              fpm=fpm,
                              lyot=lyot,
                              fs=fs,
                              dh=dh,
                              initmaps=self.initmaps,
                              ft_dir=ft_dir,
            )
            sl_list.append(sl)
            pass
        self.sl_list = sl_list

        pass


    def _keyload(self, d, path):
        """Handle three YAML config cases for loading a complex field

        YAML config has file names expected as FITS files.  FITS can't store
        complex-valued data, so the right way to store it is ambiguous.  Three
        cases are supported:
         - amplitude and phase
         - real and imaginary parts
         - pure real

        Arguments:
         d: a dictionary with one of {'afn', 'pfn'}, {'rfn', 'ifn'}, {'fn'} as
          keys to cover the cases of amp/phase, real/imag, and pure real. If
          none are present an error will be thrown; if more than one are
          present, an error will be thrown.
         path: path to use in load.

        Returns:
         a 2D array loaded from the file(s) specified in the key

        """

        ap = False
        ri = False
        pr = False

        try:
            if {'afn', 'pfn'}.issubset(d.keys()):
                ap = True
                pass
            if {'rfn', 'ifn'}.issubset(d.keys()):
                ri = True
                pass
            if {'fn'}.issubset(d.keys()):
                pr = True
                pass
            pass
        except AttributeError: # d is not dict-like and does not have keys
            raise TypeError('Mask specifications must be dict-like')

        # No specs
        if not any([ap, ri, pr]):
            raise ValueError('Mask field specifications must be ' +
                             'one of: amp/phase, real/imag, real only')
        # Too many specs
        if (ap and ri) or (ap and pr) or (ri and pr):
            raise ValueError('Mask field specifications must be only ' +
                             'one of: amp/phase, real/imag, real only')

        # Exactly one now, so sequential ifs are safe
        try:
            if ap:
                return load_ap(os.path.join(path, d['afn']),
                               os.path.join(path, d['pfn']))
            if ri:
                return load_ri(os.path.join(path, d['rfn']),
                               os.path.join(path, d['ifn']))
            if pr:
                return load(os.path.join(path, d['fn']))
            pass
        except IOError: # files don't exist
            raise IOError('Mask field file not found')

        pass
