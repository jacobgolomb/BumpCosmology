from astropy.cosmology import Planck18
import astropy.units as u
import lal
import lalsimulation as ls
import numpy as np
import os.path as op
import sys
import pandas as pd
import paths
from tqdm import tqdm, trange
import weighting
import scipy.integrate as sint
import intensity_models

def next_pow_2(x):
    np2 = 1
    while np2 < x:
        np2 = np2 << 1
    return np2

def compute_snrs(d):
    snrhs = []
    snrls = []
    snrvs = []
    snrs = []
    for _, r in d.iterrows():
        mlow = 5*(r.z-2)/(0.75-2) + 30*(r.z-0.75)/(2-0.75)
        mhigh = 150*(r.z-2.5)/(1.5-2.5) + 70*(r.z-1.5)/(2.5-1.5)
        if r.m1 > mlow and r.m1 < mhigh:
            m2s = r.m1*r.q
            m1d = r.m1*(1+r.z)
            m2d = m2s*(1+r.z)

            a1 = np.sqrt(r.s1x*r.s1x + r.s1y*r.s1y + r.s1z*r.s1z)
            a2 = np.sqrt(r.s2x*r.s2x + r.s2y*r.s2y + r.s2z*r.s2z)
            dl = d['dL'] * 1e9*lal.PC_SI

            fmin = 9.0
            fref = fmin
            psdstart = 10.0

            T = next_pow_2(ls.SimInspiralChirpTimeBound(fmin, m1d*lal.MSUN_SI, m2d*lal.MSUN_SI, a1, a2))
            df = 1/T
            fmax = 2048.0
            psdstop = 0.95*fmax

            Nf = int(round(fmax/df)) + 1
            fs = np.linspace(0, fmax, Nf)

            hp, hc = ls.SimInspiralChooseFDWaveform(m1d*lal.MSUN_SI, m2d*lal.MSUN_SI, r.s1x, r.s1y, r.s1z, r.s2x, r.s2y, r.s2z, dl, r.iota, 0.0, 0.0, 0.0, 0.0, df, fmin, fmax, fref, None, ls.IMRPhenomXPHM)

            sn = []
            for det in ['H1', 'L1', 'V1']:
                h = lal.CreateCOMPLEX16FrequencySeries('h', hp.epoch, hp.f0, hp.deltaF, hp.sampleUnits, hp.data.data.shape[0])
                psd = lal.CreateREAL8FrequencySeries("psds", 0, 0.0, df, lal.DimensionlessUnit, fs.shape[0])
    
                dd = lal.cached_detector_by_prefix[det]
                Fp, Fc = lal.ComputeDetAMResponse(dd.response, r.ra, r.dec, r.psi, r.gmst)
    
                h.data.data = Fp*hp.data.data + Fc*hc.data.data

                if det in ['H1', 'L1']:
                    ls.SimNoisePSDaLIGODesignSensitivityP1200087(psd, psdstart)
                else:
                    ls.SimNoisePSDAdVDesignSensitivityP1200087(psd, psdstart)

                sn.append(ls.MeasureSNRFD(h, psd, psdstart, psdstop))
            sn = np.array(sn)
            snrhs.append(sn[0])
            snrls.append(sn[1])
            snrvs.append(sn[2])
            snrs.append(np.sqrt(np.sum(np.square(sn))))
        else:
            snrhs.append(0)
            snrls.append(0)
            snrvs.append(0)
            snrs.append(0)

    d['SNR_H1'] = snrhs
    d['SNR_L1'] = snrls
    d['SNR_V1'] = snrvs
    d['SNR'] = snrs

    return d

class ZPDF(object):
    def __init__(self, lam, kappa, zp, zmax, cosmo):
        self.lam = lam
        self.kappa = kappa
        self.zp = zp

        self.zmax = zmax
        self.cosmo = cosmo

        self.zinterp = np.expm1(np.linspace(np.log(1), np.log(1+self.zmax), 1024))
        self.norm = 1

        unnorm_pdf = self(self.zinterp)
        
        self.norm = 1/np.trapz(unnorm_pdf, self.zinterp)
        self.pdfinterp = unnorm_pdf * self.norm

        self.cdfinterp = sint.cumtrapz(self.pdfinterp, self.zinterp, initial=0)

    def __call__(self, z):
        if self.cosmo == 'default':
            return self.norm*(1+z)**self.lam / (1 + ((1+z)/(1+self.zp))**self.kappa) * Planck18.differential_comoving_volume(z).to(u.Gpc**3/u.sr).value / (1+z)
        else:
            return self.norm*(1+z)**self.lam / (1 + ((1+z)/(1+self.zp))**self.kappa) * self.cosmo.dVCdz(z) / (1+z)

    def icdf(self, c):
        return np.interp(c, self.cdfinterp, self.zinterp)
    
class InterpolatedPDF(object):
    def __init__(self, xs, cdfs):
        self.xs = xs
        self.cdfs = cdfs / cdfs[-1]
        self.pdfs = np.diff(cdfs) / np.diff(xs)

    def __call__(self, x):
        x = np.atleast_1d(x)
        i = np.searchsorted(self.xs, x)-1

        return self.pdfs[i]
    
    def icdf(self, c):
        return np.interp(c, self.cdfs, self.xs)

class PowerLawPDF(object):
    def __init__(self, alpha, a, b):
        assert alpha > 1

        self.alpha = alpha
        self.a = a
        self.b = b

        self.norm = (self.a - (self.a/self.b)**self.alpha*self.b)/(self.a*(self.alpha-1))

    def __call__(self, x):
        return (self.a/x)**self.alpha/self.a/self.norm
    
    def icdf(self, c):
        return ((self.a**self.alpha*self.b*c + self.a*self.b**self.alpha*(1-c))/(self.a*self.b)**self.alpha)**(1/(1-self.alpha))


def calc_nex(df_det, default_settings, **kwargs):
    for key in df_det.keys():
        df_det[key] = np.array(df_det[key])
    if default_settings:
        h, Om, w = Planck18.h, Planck18.Om0, -1
        log_dN_func = weighting.default_log_dNdmdqdV
        rate = weighting.default_parameters.R
    else:
        pop_params = {key: kwargs[key] for key in ModelParameters().keys()}
        h, Om, w = kwargs['h'], kwargs['Om'], kwargs['w']
        rate = kwargs['R']
        log_dN_func = intensity_models.LogDNDMDQDV(**pop_params)
    if "cosmo" not in kwargs.keys():
        cosmo = intensity_models.FlatwCDMCosmology(h, Om, w)
    else:
        cosmo = kwargs.get("cosmo")
    log_dN = log_dN_func(df_det['m1'], df_det['q'], df_det['z'])
    nex = np.sum(rate*np.exp(log_dN)*cosmo.dVCdz(df_det['z'])*4*np.pi/(1+df_det['z'])/df_det['pdraw_mqz'])/len(df)
    return nex



if __name__ == '__main__':

    if sys.argv[1] == 'default':
        default = True
    else:
        default = False
        custom_params = sys.argv[1]
    if sys.argv[2]:
        snr_threshold = float(sys.argv[2])
    else:
        snr_threshold = 0
    if sys.argv[3]:
        outfile = sys.argv[3]
    else:
        outfile = op.join(paths.data, 'mock_injections.h5')

    population_parameters = dict()
    if not default:
        with open("file.txt") as param_file:
            for line in param_file:
                (key, val) = line.split('=')
                population_parameters[key.strip()] = float(val.strip())

    custom_cosmo = None
    if not default:
        custom_cosmo = intensity_models.FlatwCDMCosmology(population_parameters['h'], population_parameters['Om'], population_parameters['w'])
        population_parameters['cosmo'] = custom_cosmo
    if default:
        zpdf = ZPDF(lam=2.7, kappa=5.6, zp=1.9, zmax = 20, cosmo='default')
    else:
        zpfg = ZPDF(lam=population_parameters["lam"], kappa=population_parameters["kappa"], zp=population_parameters["zp"], zmax = population_parameters.get("zmax", 20), cosmo=population_parameters["cosmo"])
    mpdf = PowerLawPDF(2.35, 5, 500)

    rng = np.random.default_rng(333165393797366967556667466879860422123)
    ndraw = 1000000
    dictlist=  []

    #df = pd.DataFrame(columns = ['m1', 'q', 'z', 'iota', 'ra', 'dec', 'psi', 'gmst', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z', 'pdraw_mqz', 'SNR_H1', 'SNR_L1', 'SNR_V1', 'SNR'])
    z = zpdf.icdf(rng.uniform(low=0, high=1, size=ndraw))
    m = mpdf.icdf(rng.uniform(low=0, high=1, size=ndraw))
    mtpdf = PowerLawPDF(2, m+5, 2*m)

    mt = mtpdf.icdf(rng.uniform(low=0, high=1, size=ndraw))

    m2 = mt - m
    q = m2/m

    pdraw = mpdf(m)*(mtpdf(mt)*m)*zpdf(z)

    m1d = m * (1 + z)
    iota = np.arccos(rng.uniform(low=-1, high=1, size=ndraw))

    ra = rng.uniform(low=0, high=2*np.pi, ndraw)
    dec = np.arcsin(rng.uniform(low=-1, high=1, size=ndraw))

    # 0 < psi < pi, uniformly distributed
    psi = rng.uniform(low=0, high=np.pi, size=ndraw)
    gmst = rng.uniform(low=0, high=2*np.pi, size=ndraw)

    s1x, s1y, s1z = rng.normal(loc=0, scale=0.2/np.sqrt(3), size=(3,ndraw))
    s2x, s2y, s2z = rng.normal(loc=0, scale=0.2/np.sqrt(3), size=(3,ndraw))

    if default:
        dL = Planck18.luminosity_distance(z).to(u.Gpc).value
        dm1sz_dm1ddl = weighting.dm1sz_dm1ddl(z, cosmo=None)
    else:
        dm1sz_dm1ddl = weighting.dm1sz_dm1ddl(z, cosmo=population_parameters['cosmo'])
        dL = population_parameters['cosmo'].dL(z)

    df = pd.DataFrame({
        'm1': m,
        'q': q,
        'z': z,
        'dL': dL,
        'm1d': m1d,
        'iota': iota,
        'ra': ra,
        'dec': dec,
        'psi': psi,
        'gmst': gmst,
        's1x': s1x,
        's1y': s1y,
        's1z': s1z,
        's2x': s2x,
        's2y': s2y,
        's2z': s2z,
        'pdraw_mqz': pdraw,
        'dm1sz_dm1ddl': dm1sz_dm1ddl
    })
    if snr_threshold > 0:
        df = compute_snrs(df)
    else:
        df['SNR'] = 10000000
    p_pop = weighting.pop_wt(np.array(df['m1']), np.array(df['q']), np.array(df['z']), default=default, **population_parameters) / df['pdraw_mqz']
    df['p_pop'] = p_pop
    df.to_hdf(outfile, key='true_parameters')

    df_det = df[df['SNR'] > snr_threshold]
    #nex = np.sum(weighting.default_parameters.R*np.exp(weighting.default_log_dNdmdqdV(df_det['m1'], df_det['q'], df_det['z']))*Planck18.differential_comoving_volume(df_det['z']).to(u.Gpc**3/u.sr).value*4*np.pi/(1+df_det['z'])/df_det['pdraw_mqz'])/len(df)
    nex = calc_nex(df_det, default_settings = default, **population_parameters)
    print('Found {:d} injections with SNR > {:d}'.format(np.sum(df['SNR'] > snr_threshold), snr_threshold))
    print('Predicting {:.0f} detections per year'.format(nex))
