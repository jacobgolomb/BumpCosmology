from astropy.cosmology import Planck18
import astropy.units as u
import numpy as np
import os.path as op
import pandas as pd
import paths
from tqdm import tqdm
import weighting

if __name__ == '__main__':
    rng = np.random.default_rng(177043409333769410879087781513332130228)

    inj = pd.read_hdf(op.join(paths.data, 'mock_injections.h5'), key='true_parameters')
    obs = pd.read_hdf(op.join(paths.data, 'mock_observations.h5'), key='observations')

    nex = 4*np.pi*weighting.default_parameters.R*np.sum(np.exp(weighting.default_log_dNdmdqdV(obs['m1'], obs['q'], obs['z']))*Planck18.differential_comoving_volume(obs['z']).to(u.Gpc**3/u.sr).value/(1+obs['z'])/obs['pdraw_mqz'])/len(inj)

    n = rng.poisson(nex)
    wt = weighting.default_pop_wt(obs['m1'], obs['q'], obs['z']) / obs['pdraw_mqz']
    inds = rng.choice(len(obs), size=n, p=wt / np.sum(wt), replace=False)

    nsamp = 128
    df = pd.DataFrame()
    for i in tqdm(range(n)):
        evt = inds[i]
        row = obs.iloc[evt]

        size = 32*nsamp
        while True:
            m,q,z,w = weighting.draw_mock_samples(row['log_mc_obs'], row['sigma_log_mc'], 
                                                  row['q_obs'], row['sigma_q'],
                                                  row['log_dl_obs'], row['sigma_log_dl'],
                                                  size=size, rng=rng, 
                                                  output_source_frame=True)
            
            pop_wt = weighting.default_pop_wt(m,q,z)
            wt = pop_wt/w
            ne = np.square(np.sum(wt))/np.sum(np.square(wt))
            if ne < 2*nsamp:
                print(f'continuing because neff = {ne:.1f} < {2*nsamp=}')
                size = size*2
                continue
        
            samp_inds = rng.choice(np.arange(len(wt)), size=nsamp, p=wt / np.sum(wt))

            df = pd.concat((df, pd.DataFrame({'m1': m[samp_inds], 'q': q[samp_inds], 'z': z[samp_inds], 'wt': pop_wt[samp_inds], 'evt': evt})), ignore_index=True)
            break
    
    df.to_hdf(op.join(paths.data, 'mock_year_samples.h5'), key='samples')