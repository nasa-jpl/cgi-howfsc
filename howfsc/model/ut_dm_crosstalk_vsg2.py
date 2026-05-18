"""Verify the DM2 crosstalk model against VSG2 poke grid data."""
import os
import sys
import unittest
import warnings

from astropy.io import fits
import numpy as np

from howfsc.util.dmhtoph import dmhtoph, volts_to_dmh
from howfsc.util.loadyaml import loadyaml

# to display warnings
warnings.simplefilter('always', RuntimeWarning)

# paths to example images, config, output
HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(HERE, 'testdata', 'vsg2')

# defaults
WAVELENGTH = 633e-9  # For VSG2, not CGI


def rms(atmp):
    i_use = np.logical_not(np.logical_or(np.isnan(atmp), np.isinf(atmp)))
    return np.sqrt(np.mean(atmp[i_use]**2))


def compute_delta_phase_dm2(cmdDM_new, cmdDM_old, dmreg, crosstalkfn):
    """
    Compute the delta phase for two DM2 command maps.
    """
    # gain is m/V
    gain_cube_fn = os.path.join(INPUT_PATH, 'gain_cube_dm2.fits')
    gain_map_DM = fits.getdata(gain_cube_fn)[4, :, :]

    # apply gain to get dm actuator height
    # use volts_to_h, which incorporates crosstalk
    # Note: dorba requires all arguments are keyword
    DM_rad_new = volts_to_dmh(gainmap=gain_map_DM, volts=cmdDM_new, lam=WAVELENGTH,
                              crosstalk_fn=crosstalkfn)
    DM_rad_old = volts_to_dmh(gainmap=gain_map_DM, volts=cmdDM_old, lam=WAVELENGTH,
                              crosstalk_fn=crosstalkfn)

    # delta actuator height
    delta_DM_rad = DM_rad_new - DM_rad_old

    nr = 701
    nc = 701
    inf_func = fits.getdata(os.path.join(INPUT_PATH, dmreg['inffn']))

    # calculate delta pupil phase
    # input is actuator poke in radians, output is phase surface in radians
    delta_phase = dmhtoph(
        nrow=nr, ncol=nc, dmin=delta_DM_rad, nact=dmreg['nact'], inf_func=inf_func,
        ppact_d=dmreg['ppact_d'], ppact_cx=dmreg['ppact_cx'], ppact_cy=dmreg['ppact_cy'],
        dx=dmreg['dx'], dy=dmreg['dy'], thact=dmreg['thact'], flipx=dmreg['flipx']
    )

    return delta_phase, rms(delta_phase)


def compute_fit_residual(poke_grid_num):

    dm_fn = os.path.join(INPUT_PATH, 'howfsc_optical_model_fit_vsg2_data.yaml')

    # machinations to get the crosstalkfn and its path correct
    dmreg_dms = loadyaml(path=(dm_fn))['dms']
    crosstalkfn_dm2 = dmreg_dms['DM2']['voltages']['crosstalkfn']
    crosstalkfn_dm2 = os.path.join(
        INPUT_PATH, crosstalkfn_dm2
    ) if crosstalkfn_dm2 else None

    cmdDM2 = 40 * np.ones((48, 48))
    poke_norm = fits.getdata(os.path.join(INPUT_PATH, ('dm-mask%d.fits' % poke_grid_num)))

    if poke_grid_num == 7:
        poke_norm *= -1

    cmdDM2_new = cmdDM2 - 1*10*poke_norm
    cmdDM2_new[27, 34:36] = np.sum(cmdDM2_new[27, 34:36]/2)  # Manually enforce the tied actuators as having the mean voltage of the pair

    dm2_delta_phase_true = fits.getdata(os.path.join(INPUT_PATH, ('poke-image%d.fits' % poke_grid_num)))

    dm2_delta_surf, dm2_delta_surf_rms = compute_delta_phase_dm2(
        cmdDM2_new, cmdDM2, dmreg_dms['DM2']['registration'], crosstalkfn_dm2)

    diff = dm2_delta_phase_true-dm2_delta_surf
    diff[300:350, 20:70] = 0  # mask out the left most actuator that goes outside the 48x48 grid
    diff[300:420, 640:700] = 0  # mask out the right-most actuators that goes outside the 48x48 grid
    
    pv_diff = np.max(diff) - np.min(diff)
    pv_meas = np.max(dm2_delta_phase_true) - np.min(dm2_delta_phase_true)
    error_pv_fractional = pv_diff/pv_meas

    # import matplotlib.pyplot as plt

    # plt.figure(10)
    # plt.clf()
    # plt.title('%d. poke_norm' % poke_grid_num)
    # plt.imshow(poke_norm)
    # plt.gca().invert_yaxis()
    # plt.colorbar()

    # plt.figure(11)
    # plt.clf()
    # plt.title('%d. Measured - Modeled' % poke_grid_num)
    # plt.imshow(diff)
    # plt.gca().invert_yaxis()
    # plt.colorbar()

    # plt.figure(1)
    # plt.clf()
    # plt.title('%d. Measured' % poke_grid_num)
    # plt.imshow(dm2_delta_phase_true)
    # plt.gca().invert_yaxis()
    # plt.colorbar()

    # plt.figure(2)
    # plt.clf()
    # plt.title('%d. GSW Modeled' % poke_grid_num)
    # plt.imshow(dm2_delta_surf)
    # plt.gca().invert_yaxis()
    # plt.colorbar()

    # plt.show()

    return error_pv_fractional


class Test_dm_crosstalk_vsg2_data(unittest.TestCase):
    """Unit test to verify that the DM crosstalk is correct for surface generation."""
        
    def setUp(self):
        """Defaults for creating valid objec."""
        self.rtol = 0.12  # somewhat high because of superposition breakdown

    def testGridPattern7(self):
        """Test DM crosstalk for a specific poke grid pattern."""
        error_pv_fractional = compute_fit_residual(7)
        self.assertTrue(error_pv_fractional < self.rtol)

    def testGridPattern10(self):
        """Test DM crosstalk for a specific poke grid pattern."""
        error_pv_fractional = compute_fit_residual(10)
        self.assertTrue(error_pv_fractional < self.rtol)

    def testGridPattern14(self):
        """Test DM crosstalk for a specific poke grid pattern."""
        error_pv_fractional = compute_fit_residual(14)
        self.assertTrue(error_pv_fractional < self.rtol)


if __name__ == '__main__':
    unittest.main()
