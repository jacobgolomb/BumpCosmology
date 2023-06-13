from astropy.cosmology import Planck18
import astropy.units as u
import lal
import lalsimulation as ls
import numpy as np
import os.path as op
import sys
import pandas as pd
import paths
from tqdm import tqdm
import weighting
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

ndraw = 100000000
nbatch = 10000

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
    rng = np.random.default_rng(333165393797366967556667466879860422123)

    wt_sum = 0

    with tqdm(total=ndraw) as bar:
        df = pd.DataFrame()
        #df = pd.DataFrame(columns = ['m1', 'q', 'z', 'iota', 'ra', 'dec', 'psi', 'gmst', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z', 'pdraw_mqz', 'SNR_H1', 'SNR_L1', 'SNR_V1', 'SNR'])
        for _ in range(ndraw // nbatch):

            m = rng.uniform(low=5, high=150, size=nbatch)
            q = rng.uniform(low=0, high=1, size=nbatch)
            z = rng.uniform(low=0, high=20, size=nbatch)

            m1d = m * (1 + z)
            iota = np.arccos(rng.uniform(low=-1, high=1, size=nbatch))

            ra = rng.uniform(low=0, high=2*np.pi, size=nbatch)
            dec = np.arcsin(rng.uniform(low=-1, high=1, size=nbatch))

            # 0 < psi < pi, uniformly distributed
            psi = rng.uniform(low=0, high=np.pi, size=nbatch)
            gmst = rng.uniform(low=0, high=2*np.pi, size=nbatch)

            s1x, s1y, s1z = rng.normal(loc=0, scale=0.2/np.sqrt(3), size=(3, nbatch))
            s2x, s2y, s2z = rng.normal(loc=0, scale=0.2/np.sqrt(3), size=(3, nbatch))

            pdraw = 1/(150-5)*1*1/20
            if default:
                dL = Planck18.luminosity_distance(z).to(u.Gpc).value
                dm1sz_dm1ddl = weighting.dm1sz_dm1ddl(z, cosmo=None)
            else:
                dm1sz_dm1ddl = weighting.dm1sz_dm1ddl(z, cosmo=population_parameters['cosmo'])
                dL = population_parameters['cosmo'].dL(z)

            wt = np.array(weighting.pop_wt(m, q, z, default=default, **population_parameters))

            wt[m*q < 5] = 0 # Cut out events with m2 < 5
            wt_sum += np.sum(wt/pdraw)

            r = rng.uniform(low=0, high=np.max(wt), size=nbatch)
            sel = r < wt

            d = pd.DataFrame({
                'm1': m[sel],
                'q': q[sel],
                'z': z[sel],
                'dL': dL[sel],
                'm1d': m1d[sel],
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
                'pdraw_mqz': wt[sel],
                'dm1sz_dm1ddl': dm1sz_dm1ddl[sel]
            })
            if snr_threshold > 0:
                d = compute_snrs(d)
            else:
                d['SNR'] = 10000000
            df = pd.concat((df, d))
            bar.update(nbatch)

    wt_sum /= ndraw
    df['pdraw_mqz'] = df['pdraw_mqz'] / wt_sum

    df.to_hdf(outfile, key='true_parameters')

    df_det = df[df['SNR'] > snr_threshold]
    #nex = np.sum(weighting.default_parameters.R*np.exp(weighting.default_log_dNdmdqdV(df_det['m1'], df_det['q'], df_det['z']))*Planck18.differential_comoving_volume(df_det['z']).to(u.Gpc**3/u.sr).value*4*np.pi/(1+df_det['z'])/df_det['pdraw_mqz'])/len(df)
    nex = calc_nex(df_det, default_settings = default, **population_parameters)
    print('Found {:d} injections with SNR > {:d}'.format(np.sum(df['SNR'] > snr_threshold), snr_threshold))
    print('Predicting {:.0f} detections per year'.format(nex))
