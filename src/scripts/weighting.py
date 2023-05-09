import astropy.cosmology as cosmo
from astropy.cosmology import Planck18
import astropy.units as u
import h5py
import intensity_models
import numpy as np
import pytensor
import pytensor.tensor as pt

a = 1.5
b = -0.71
c = 2.3
mpisn = 32.0
mbhmax = 37.0
sigma = 2.7
fpl = 0.19
beta = -2.0
lam = 4.2
kappa = 7.8
zp = 2.1

m = pt.dvector('m1')
q = pt.dvector('q')
z = pt.dvector('z')
default_log_dNdmdqdV = pytensor.function([m,q,z], intensity_models.LogDNDMDQDV(a, b, c, mpisn, mbhmax, sigma, fpl, beta, lam, kappa, zp)(m,q,z))
default_log_dNdmdqdV.__doc__ = r"""
Default mass-redshift distribution, more-or-less a reasonable fit to O3a.
"""

def default_pop_wt(m1, q, z):
    """Weights in `(m1,q,z)` corresponding to the :func:`default_log_dNdmdqdV`."""
    log_dN = default_log_dNdmdqdV(m1, q, z)
    return np.exp(log_dN)*Planck18.differential_comoving_volume(z).to(u.Gpc**3/u.sr).value/(1+z)

def li_prior_wt(m1, q, z, cosmology_weighted=False):
    """Returns LALInference/Bilby prior over `m1`, `q`, and `z`.
    
    `cosmology_weighted` controls whether to use the default prior or one that
    uses the uniform-merger-rate-in-the-comoving-frame redshift weighting."""
    if cosmology_weighted:
        return 4*np.pi*np.square(1+z)*m1*Planck18.differential_comoving_volume(z).to(u.Gpc**3/u.sr).value/(1+z)
    else:
        return np.square(1+z)*m1*np.square(Planck18.luminosity_distance(z).to(u.Gpc).value)*(Planck18.comoving_distance(z).to(u.Gpc).value + (1+z)*Planck18.hubble_distance.to(u.Gpc).value/Planck18.efunc(z))
    
def extract_posterior_samples(file, nsamp, desired_pop_wt=None):
    """Returns posterior samples over `m1`, `q`, `z` extracted from `file`.

    The returned samples will be drawn from a density proportional to
    `desired_pop_wt` (or, if none is given, the default LALInference/Bilby
    prior).

    :param file: The file (HDF5) containing posterior samples.

    :param nsamp: The number of samples desired.  The code will raise a
        `ValueError` if too few samples exist in the file.

    :param desired_pop_wt: A function over `(m1, q, z)` giving the desired
        population weight for the returned samples.  If none given, this will be
        the default (non-cosmologically-weighted) LALInference/Bilby prior.

    :return: Arrays `(m1, q, z, pop_wt)` giving the samples and (unnormalized)
        weights according to the extracted population.    
    """
    with h5py.File(file, 'r') as f:
        if 'PublicationSamples' in f.keys():
            # O3a files
            samples = np.array(f['PublicationSamples/posterior_samples'])
        elif 'C01:Mixed' in f.keys():
            # O3b files
            samples = np.array(f['C01:Mixed/posterior_samples'])
        else:
            raise ValueError(f'could not read samples from file {file}')
        
        m1 = np.array(samples['mass_1_source'])
        q = np.array(samples['mass_ratio'])
        z = np.array(samples['redshift'])

        if desired_pop_wt is None:
            pop_wt = li_prior_wt(m1, q, z)
        else:
            pop_wt = desired_pop_wt(m1, q, z)
        wt = pop_wt / li_prior_wt(m1, q, z)
        wt = wt / np.sum(wt)

        ns = 1/np.sum(wt*wt)
        if ns < nsamp:
            raise ValueError('could not read samples from {:s} due to too few samples: {:.1f}'.format(file, ns))

        inds = np.random.choice(np.arange(len(samples)), nsamp, p=wt)
        return (m1[inds], q[inds], z[inds], pop_wt[inds])
