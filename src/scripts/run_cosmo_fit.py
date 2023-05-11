import arviz as az
from astropy.cosmology import Planck18
import astropy.units as u
import intensity_models
import numpy as np
import os.path as op
import pandas as pd
import paths
import pymc as pm
import weighting

nmcmc = 1000
nchain = 4

if __name__ == '__main__':
    random_seed = 330601005597653222525092651948614145425

    pe_samples = pd.read_hdf(op.join(paths.data, 'pe-samples.h5'), 'samples')
    pe_samples['m1d'] = pe_samples['m1']*(1+pe_samples['z'])
    pe_samples['dl'] = Planck18.luminosity_distance(pe_samples['z'].to_numpy()).to(u.Gpc).value
    pe_samples['pdraw_cosmo'] = pe_samples['wt']*weighting.dm1qz_dm1dqdl(pe_samples['m1'], pe_samples['q'], pe_samples['z'])

    sel_samples = pd.read_hdf(op.join(paths.data, 'selection-samples.h5'), 'samples')
    sel_samples['m1d'] = sel_samples['m1']*(1+sel_samples['z'])
    sel_samples['dl'] = Planck18.luminosity_distance(sel_samples['z'].to_numpy()).to(u.Gpc).value
    sel_samples['pdraw_cosmo'] = sel_samples['pdraw']*weighting.dm1qz_dm1dqdl(sel_samples['m1'], sel_samples['q'], sel_samples['z'])

    evts = pe_samples.groupby('evt')
    m1s = []
    qs = []
    dls = []
    pdraws = []
    for (n, e) in evts:
        m1s.append(e['m1d'])
        qs.append(e['q'])
        dls.append(e['dl'])
        pdraws.append(e['pdraw_cosmo'])

    m1s, qs, dls, pdraws = map(np.array, [m1s, qs, dls, pdraws])

    model = intensity_models.make_pop_cosmo_model(m1s, qs, dls, pdraws, 
                                                  sel_samples['m1d'], sel_samples['q'], sel_samples['dl'], sel_samples['pdraw_cosmo'], sel_samples['ndraw'].iloc[0])
    
    with model:
        trace = pm.sample(tune=nmcmc, draws=nmcmc, chains=nchain, random_seed=random_seed)
    az.to_netcdf(trace, op.join(paths.data, 'trace_cosmo.nc'))
