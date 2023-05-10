import arviz as az
import intensity_models
import numpy as np
import os.path as op
import pandas as pd
import paths
import pymc as pm

nmcmc = 1000
nchain = 4

if __name__ == '__main__':
    random_seed = 308471063890295930777023523201258283075

    pe_samples = pd.read_hdf(op.join(paths.data, 'pe-samples.h5'), 'samples')
    sel_samples = pd.read_hdf(op.join(paths.data, 'selection-samples.h5'), 'samples')

    evts = pe_samples.groupby('evt')
    m1s = []
    qs = []
    zs = []
    pdraws = []
    for (n, e) in evts:
        m1s.append(e['m1'])
        qs.append(e['q'])
        zs.append(e['z'])
        pdraws.append(e['wt'])

    m1s, qs, zs, pdraws = map(np.array, [m1s, qs, zs, pdraws])

    model = intensity_models.make_pop_model(m1s, qs, zs, pdraws, 
                                            sel_samples['m1'], sel_samples['q'], sel_samples['z'], sel_samples['pdraw'], sel_samples['ndraw'].iloc[0])
    
    with model:
        trace = pm.sample(tune=nmcmc, draws=nmcmc, chains=nchain, random_seed=random_seed)
    az.to_netcdf(trace, op.join(paths.data, 'trace.nc'))
