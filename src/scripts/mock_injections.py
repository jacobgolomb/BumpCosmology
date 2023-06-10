from astropy.cosmology import Planck18
import astropy.units as u
import lal
import lalsimulation as ls
from multiprocessing import Pool
import numpy as np
import os.path as op
import pandas as pd
import paths
import scipy.integrate as sint
import scipy.interpolate as si
from tqdm import tqdm
import weighting

def next_pow_2(x):
    np2 = 1
    while np2 < x:
        np2 = np2 << 1
    return np2

def line_between(z0, y0, z1, y1):
    return lambda z: (z-z1)*y0/(z0-z1) + (z-z0)*y1/(z1-z0)

# mz_low = line_between(0.5, 5, 2.5, 30)
# mz_high = line_between(0, 1000.0, 2.5, 50)
# qz_low = line_between(0.6, 0.0, 2.6, 1)

z_horiz = 3.5
chirp_dist_min = 1.5

def compute_snrs(ir):
    i, r = ir
    m2s = r.m1*r.q

    mcds = r.m1*(1+r.z)*r.q**(3/5)/(1+r.q)**(1/5)
    chirp_d = mcds**(5/6) / Planck18.luminosity_distance(r.z).to(u.Gpc).value

    if r.z < z_horiz and chirp_d > chirp_dist_min:
        m1d = r.m1*(1+r.z)
        m2d = m2s*(1+r.z)

        a1 = np.sqrt(r.s1x*r.s1x + r.s1y*r.s1y + r.s1z*r.s1z)
        a2 = np.sqrt(r.s2x*r.s2x + r.s2y*r.s2y + r.s2z*r.s2z)

        dl = Planck18.luminosity_distance(r.z).to(u.Gpc).value * 1e9*lal.PC_SI

        fmin = 9.0
        fref = fmin
        psdstart = 10.0

        T = next_pow_2(ls.SimInspiralChirpTimeBound(fmin, m1d*lal.MSUN_SI, m2d*lal.MSUN_SI, a1, a2))
        df = 1/T
        fmax = 2048.0
        psdstop = 0.95*fmax

        Nf = int(round(fmax/df)) + 1
        fs = np.linspace(0, fmax, Nf)
        try:
            hp, hc = ls.SimInspiralChooseFDWaveform(m1d*lal.MSUN_SI, m2d*lal.MSUN_SI, r.s1x, r.s1y, r.s1z, r.s2x, r.s2y, r.s2z, dl, r.iota, 0.0, 0.0, 0.0, 0.0, df, fmin, fmax, fref, None, ls.IMRPhenomXPHM)
        except RuntimeError:
            return (0.0, 0.0, 0.0, 0.0)

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
        return tuple(sn) + (np.sqrt(np.sum(np.square(sn))),)
    else:
        return (0.0, 0.0, 0.0, 0.0)

class ZPDF(object):
    def __init__(self):
        self.lam = 2.7
        self.kappa = 5.6
        self.zp = 1.9

        self.zmax = z_horiz

        self.zinterp = np.expm1(np.linspace(np.log(1), np.log(1+self.zmax), 1024))
        self.norm = 1
        unnorm_pdf = self(self.zinterp)
        
        self.norm = 1/np.trapz(unnorm_pdf, self.zinterp)
        self.pdfinterp = unnorm_pdf * self.norm

        self.cdfinterp = sint.cumtrapz(self.pdfinterp, self.zinterp, initial=0)

    def __call__(self, z):
        return self.norm*(1+z)**self.lam / (1 + ((1+z)/(1+self.zp))**self.kappa) * Planck18.differential_comoving_volume(z).to(u.Gpc**3/u.sr).value/(1+z)
    
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

ndraw = 10000000

if __name__ == '__main__':
    rng = np.random.default_rng(333165393797366967556667466879860422123)

    with Pool() as pool:
        df = pd.DataFrame(columns = ['m1', 'q', 'z', 'iota', 'ra', 'dec', 'psi', 'gmst', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z', 'pdraw_mqz', 'SNR_H1', 'SNR_L1', 'SNR_V1', 'SNR'])

        # We want to draw from p(m1) ~ m1^(-2.35), p(mtotal | m1) ~ mtotal^-2, p(z) ~ (1+z)^2.7/(1 + ((1+z)/(1+zp))^(5.6))
        zpdf = ZPDF()
        z = zpdf.icdf(rng.uniform(low=0, high=1, size=ndraw))

        mpdf = PowerLawPDF(2.35, 5, 500)
        m = mpdf.icdf(rng.uniform(low=0, high=1, size=ndraw))

        mtpdf = PowerLawPDF(2, m+5, 2*m)
        mt = mtpdf.icdf(rng.uniform(low=0, high=1, size=ndraw))

        m2 = mt - m
        q = m2/m

        pdraw = mpdf(m)*(mtpdf(mt)*m)*zpdf(z)

        iota = np.arccos(rng.uniform(low=-1, high=1, size=ndraw))

        ra = rng.uniform(low=0, high=2*np.pi, size=ndraw)
        dec = np.arcsin(rng.uniform(low=-1, high=1, size=ndraw))

        # 0 < psi < pi, uniformly distributed
        psi = rng.uniform(low=0, high=np.pi, size=ndraw)
        gmst = rng.uniform(low=0, high=2*np.pi, size=ndraw)

        s1x, s1y, s1z = rng.normal(loc=0, scale=0.2/np.sqrt(3), size=(3, ndraw))
        s2x, s2y, s2z = rng.normal(loc=0, scale=0.2/np.sqrt(3), size=(3, ndraw))

        df = pd.DataFrame({
            'm1': m,
            'q': q,
            'z': z,
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
            'pdraw_mqz': pdraw
        })

        snrs = list(tqdm(pool.imap(compute_snrs, df.iterrows(), chunksize=min(1000, ndraw)), total=ndraw, smoothing=10/ndraw))
        snrs = np.array(snrs)
        df['SNR_H1'] = snrs[:,0]
        df['SNR_L1'] = snrs[:,1]
        df['SNR_V1'] = snrs[:,2]
        df['SNR'] = snrs[:,3]

    df.to_hdf(op.join(paths.data, 'mock_injections.h5'), key='true_parameters')

    df_det = df[df['SNR'] > 10]
    nex = np.sum(weighting.default_parameters.R*np.exp(weighting.default_log_dNdmdqdV(df_det['m1'], df_det['q'], df_det['z']))*Planck18.differential_comoving_volume(df_det['z']).to(u.Gpc**3/u.sr).value*4*np.pi/(1+df_det['z'])/df_det['pdraw_mqz'])/len(df)

    wt = weighting.default_pop_wt(df_det['m1'], df_det['q'], df_det['z']) / df_det['pdraw_mqz']
    
    print('Found {:d} injections with SNR > 10'.format(np.sum(df['SNR'] > 10)))
    print('Predicting {:.0f} detections per year'.format(nex))
    print('Neff from default pop model = {:.1f}'.format(np.square(np.sum(wt))/np.sum(np.square(wt))))
    print('Expected number of pop-model draws = {:.1f}'.format(np.sum(wt)/np.max(wt)))
