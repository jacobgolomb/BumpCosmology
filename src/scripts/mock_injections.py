from astropy.cosmology import Planck18
import astropy.units as u
import lal
import lalsimulation as ls
import numpy as np
import os.path as op
import pandas as pd
import paths
from tqdm import tqdm
import weighting

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

ndraw = 100000000
nbatch = 10000

if __name__ == '__main__':
    rng = np.random.default_rng(333165393797366967556667466879860422123)

    wt_sum = 0

    with tqdm(total=ndraw) as bar:
        df = pd.DataFrame(columns = ['m1', 'q', 'z', 'iota', 'ra', 'dec', 'psi', 'gmst', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z', 'pdraw_mqz', 'SNR_H1', 'SNR_L1', 'SNR_V1', 'SNR'])
        for _ in range(ndraw // nbatch):

            m = rng.uniform(low=5, high=150, size=nbatch)
            q = rng.uniform(low=0, high=1, size=nbatch)
            z = rng.uniform(low=0, high=2.75, size=nbatch)

            iota = np.arccos(rng.uniform(low=-1, high=1, size=nbatch))

            ra = rng.uniform(low=0, high=2*np.pi, size=nbatch)
            dec = np.arcsin(rng.uniform(low=-1, high=1, size=nbatch))

            # 0 < psi < pi, uniformly distributed
            psi = rng.uniform(low=0, high=np.pi, size=nbatch)
            gmst = rng.uniform(low=0, high=2*np.pi, size=nbatch)

            s1x, s1y, s1z = rng.normal(loc=0, scale=0.2/np.sqrt(3), size=(3, nbatch))
            s2x, s2y, s2z = rng.normal(loc=0, scale=0.2/np.sqrt(3), size=(3, nbatch))

            pdraw = 1/(150-5)*1*1/2.75

            wt = weighting.default_pop_wt(m, q, z)

            wt[m*q < 5] = 0 # Cut out events with m2 < 5
            wt_sum += np.sum(wt/pdraw)

            r = rng.uniform(low=0, high=np.max(wt), size=nbatch)
            sel = r < wt

            d = pd.DataFrame({
                'm1': m[sel],
                'q': q[sel],
                'z': z[sel],
                'iota': iota[sel],
                'ra': ra[sel],
                'dec': dec[sel],
                'psi': psi[sel],
                'gmst': gmst[sel],
                's1x': s1x[sel],
                's1y': s1y[sel],
                's1z': s1z[sel],
                's2x': s2x[sel],
                's2y': s2y[sel],
                's2z': s2z[sel],
                'pdraw_mqz': wt[sel]
            })
            d = compute_snrs(d)

            df = pd.concat((df, d))
            bar.update(nbatch)

    wt_sum /= ndraw
    df['pdraw_mqz'] = df['pdraw_mqz'] / wt_sum

    df.to_hdf(op.join(paths.data, 'mock_injections.h5'), key='true_parameters')

    df_det = df[df['SNR'] > 9]
    nex = np.sum(weighting.default_parameters.R*np.exp(weighting.default_log_dNdmdqdV(df_det['m1'], df_det['q'], df_det['z']))*Planck18.differential_comoving_volume(df_det['z']).to(u.Gpc**3/u.sr).value*4*np.pi/(1+df_det['z'])/df_det['pdraw_mqz'])/len(df)

    print('Found {:d} injections with SNR > 9'.format(np.sum(df['SNR'] > 9)))
    print('Predicting {:.0f} detections per year'.format(nex))
