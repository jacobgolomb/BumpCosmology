from astropy.cosmology import Planck18
import astropy.units as u
import lal
import lalsimulation as ls
from multiprocessing import Pool
import numpy as np
import os.path as op
import pandas as pd
import paths
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

mz_low = line_between(0.5, 5, 2.5, 30)
mz_high = line_between(0, 1000.0, 2.5, 50)
qz_low = line_between(0.6, 0.0, 2.6, 1)

def compute_snrs(ir):
    i, r = ir
    m2s = r.m1*r.q
    if r.m1 > mz_low(r.z) and r.q > qz_low(r.z) and r.m1 < mz_high(r.z) and m2s > 5:
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

ndraw = 1000000

if __name__ == '__main__':
    rng = np.random.default_rng(333165393797366967556667466879860422123)

    with Pool() as pool:
        df = pd.DataFrame(columns = ['m1', 'q', 'z', 'iota', 'ra', 'dec', 'psi', 'gmst', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z', 'pdraw_mqz', 'SNR_H1', 'SNR_L1', 'SNR_V1', 'SNR'])
        m = 5/(1-rng.uniform(low=0, high=1, size=ndraw)) # p(m) = 1/5 (5/m)^2
        q = rng.uniform(low=0, high=1, size=ndraw)
        z = rng.uniform(low=0, high=2.75, size=ndraw)

        iota = np.arccos(rng.uniform(low=-1, high=1, size=ndraw))

        ra = rng.uniform(low=0, high=2*np.pi, size=ndraw)
        dec = np.arcsin(rng.uniform(low=-1, high=1, size=ndraw))

        # 0 < psi < pi, uniformly distributed
        psi = rng.uniform(low=0, high=np.pi, size=ndraw)
        gmst = rng.uniform(low=0, high=2*np.pi, size=ndraw)

        s1x, s1y, s1z = rng.normal(loc=0, scale=0.2/np.sqrt(3), size=(3, ndraw))
        s2x, s2y, s2z = rng.normal(loc=0, scale=0.2/np.sqrt(3), size=(3, ndraw))

        pdraw = 1/5*np.square(5/m)/2.75

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
