from astropy.cosmology import Planck18
import astropy.units as u
import dataclasses
from dataclasses import dataclass
import numpy as np
import os.path as op
import pandas as pd
import paths
import seaborn as sns
from tqdm import tqdm

detection_threshold = 10.0

@dataclass
class Uncertainties(object):
    sigma_log_mc: object
    sigma_q: object
    sigma_log_dl: object

    @classmethod
    def from_snr(cls, snr):
        # These formulas come from estimations based on uncertainties in GWTC-3
        slmc = 0.05*20/snr
        sq = 0.07*20/snr
        sld = 0.2*20/snr

        return cls(slmc, sq, sld)

if __name__ == '__main__':
    rng = np.random.default_rng(181286134409181405721219170031242732711)

    inj = pd.read_hdf(op.join(paths.data, 'mock_injections.h5'), key='true_parameters')

    inj['SNR_OBS'] = inj['SNR'] + rng.normal(loc=0, scale=np.sqrt(3), size=len(inj))

    inj_det = inj[inj['SNR_OBS'] > detection_threshold].copy()
    inj_det['mc'] = inj_det['m1'] * inj_det['q']**(3/5) / (1 + inj_det['q'])**(1/5)
    inj_det['dl'] = Planck18.luminosity_distance(inj_det['z'].to_numpy()).to(u.Gpc).value
    inj_det['mc_det'] = inj_det['mc'] * (1 + inj_det['z'])

    inj_det['sigma_log_mc'] = 0.05*20/inj_det['SNR_OBS']
    inj_det['log_mc_obs'] = rng.normal(loc=np.log(inj_det['mc_det']), scale=inj_det['sigma_log_mc'])

    inj_det['sigma_q'] = 0.07*20/inj_det['SNR_OBS']
    inj_det['q_obs'] = rng.normal(loc=inj_det['q'], scale=inj_det['sigma_q'])

    inj_det['sigma_log_dl'] = 0.2*20/inj_det['SNR_OBS']
    inj_det['log_dl_obs'] = rng.normal(loc=np.log(inj_det['dl']), scale=inj_det['sigma_log_dl'])

    inj_det.to_hdf(op.join(paths.data, 'mock_observations.h5'), key='observations')