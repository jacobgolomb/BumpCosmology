import astropy.cosmology as cosmo
from astropy.cosmology import Planck18
import astropy.units as u
import h5py
import intensity_models
import numpy as np
import pytensor
import pytensor.tensor as pt

a = 1.0
b = -0.30
c = 1.9
mpisn = 33.0
mbhmax = 38.0
sigma = 3.4
fpl = 0.16
beta = -1.9
lam = 3.5
kappa = 7.1
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
    
def extract_posterior_samples(file, nsamp, desired_pop_wt=None, rng=None):
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

    :param rng: A random number generator used for the draws; if `None`, one
        will be initialized randomly.

    :return: Arrays `(m1, q, z, pop_wt)` giving the samples and (unnormalized)
        weights according to the extracted population.    
    """
    if rng is None:
        rng = np.random.default_rng()

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

        inds = rng.choice(np.arange(len(samples)), nsamp, p=wt)
        return (m1[inds], q[inds], z[inds], pop_wt[inds])

def extract_selection_samples(file, nsamp, desired_pop_wt=None, far_threshold=1, rng=None):
    """Return `(m1, q, z, pdraw, nsel)` to estimate selection effects.
    
    :param file: The injection file.

    :param nsamp: The number of samples to be returned.

    :param desired_pop_wt: Function giving a weight in `(m1, q, z)` from which
        the population of injections should be drawn.  If none is given, the
        reference distribution for the actual injections will be used; otherwise
        the distribution of injections will be re-weighted to achieve the
        desired poplation.

    :param far_threshold: The threshold on the FAR (per year) at which an
        injection is considered detected.

    :param rng: A random number generator for the draws; if `None`, one will be
        initialized randomly.

    :return: A tuple `(m1, q, z, pdraw, nsel)`, giving a draw of detected
        injections from the desired population.  `pdraw` is properly normalized
        for estimating detectability as in, e.g., [Farr
        (2019)](https://ui.adsabs.harvard.edu/abs/2019RNAAS...3...66F/abstract).
    """
    if rng is None:
        rng = np.random.default_rng()

    with h5py.File(file, 'r') as f:
        m1s_sel = np.array(f['injections/mass1_source'])
        qs_sel = np.array(f['injections/mass2_source'])/m1s_sel
        zs_sel = np.array(f['injections/redshift'])
        pdraw_sel = np.array(f['injections/mass1_source_mass2_source_sampling_pdf'])*np.array(f['injections/redshift_sampling_pdf'])*m1s_sel

        pycbc_far = np.array(f['injections/far_pycbc_hyperbank'])
        pycbc_bbh_far = np.array(f['injections/far_pycbc_bbh'])
        gstlal_far = np.array(f['injections/far_gstlal'])
        mbta_far = np.array(f['injections/far_mbta'])

        detected = (pycbc_far < far_threshold) | (pycbc_bbh_far < far_threshold) | (gstlal_far < far_threshold) | (mbta_far < far_threshold)

        ndraw = f.attrs['n_accepted'] + f.attrs['n_rejected']

        T = (f.attrs['end_time_s'] - f.attrs['start_time_s'])/(3600.0*24.0*365.25) 
        pdraw_sel /= T

        m1s_sel = m1s_sel[detected]
        qs_sel = qs_sel[detected]
        zs_sel = zs_sel[detected]
        pdraw_sel = pdraw_sel[detected]

        if desired_pop_wt is None:
            pop_wt = pdraw_sel
        else:
            pop_wt = desired_pop_wt(m1s_sel, qs_sel, zs_sel)

        unnorm_wt = pop_wt/pdraw_sel
        sum_norm_wt = unnorm_wt / np.sum(unnorm_wt)
        pdraw_wt = pop_wt / (np.sum(unnorm_wt) / ndraw)

        inds = rng.choice(len(m1s_sel), size=nsamp, p=sum_norm_wt)
        m1s_sel_cut = m1s_sel[inds]
        qs_sel_cut = qs_sel[inds]
        zs_sel_cut = zs_sel[inds]
        pdraw_sel_cut = pdraw_wt[inds]
        ndraw_cut = nsamp

        return m1s_sel_cut, qs_sel_cut, zs_sel_cut, pdraw_sel_cut, ndraw_cut
    
def dm1qz_dm1dqdl(m1, q, z):
    return (1+z)/(Planck18.comoving_distance(z).to(u.Gpc).value + (1+z)*Planck18.hubble_distance.to(u.Gpc).value / Planck18.efunc(z))