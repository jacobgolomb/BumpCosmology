from astropy.cosmology import Planck18
import astropy.units as u
import dataclasses
from dataclasses import dataclass
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from utils import pt_interp, pt_cumtrapz

def mean_mbh_from_mco(mco, mpisn, mbhmax):
    """The mean black hole mass from the core-mass to remnant-mass relation.
    
    :param mco: The CO core mass.
    :param mpisn: The BH mass at which the relation starts to turn over.
    :param mbhmax: The maximum BH mass achieved by the relation.
    """
    a = 1 / (4*(mpisn - mbhmax))
    mcomax = 2*mbhmax - mpisn

    return pt.where(mco < mpisn, mco, mbhmax + a*pt.square(mco - mcomax))

def largest_mco(mpisn, mbhmax):
    """The largest CO core mass with positive BH masses."""
    mcomax = 2*mbhmax - mpisn
    return mcomax + pt.sqrt(4*mbhmax*(mbhmax - mpisn))

def log_dNdmCO(mco, a, b):
    r"""The broken power law CO core mass function.
    
    The power law breaks (smoothly) at :math:`16 \, M_\odot` (i.e. a BH mass of :math:`20 \, M_\odot`).

    :param mco: The CO core mass.
    :param a: The power law slope for small CO core masses.
    :param b: The power law slope for large CO core masses.
    """
    mtr = 20.0
    x = mco/mtr
    return pt.where(mco < mtr, -a*pt.log(x), -b*pt.log(x))

def log_smooth_turnon(m, mmin, width=0.05):
    """A function that smoothly transitions from 0 to 1.
    
    :param m: The function argument.
    :param mmin: The location around which the function transitions.
    :param width: (optional) The fractional width of the transition.
    """
    dm = mmin*width

    return np.log(2) -pt.log1p(pt.exp(-(m-mmin)/dm))

@dataclass
class LogDNDMPISN(object):
    r"""
    Function-like object representing the compact-object mass function induced
    by the PISN process.
    
    .. math::
        \frac{\mathrm{d} N}{\mathrm{d} m} = \int \mathrm{d} m_\mathrm{CO} \, \frac{\mathrm{d} N}{\mathrm{d} m_\mathrm{CO}} p\left( m \mid \mu\left( m_\mathrm{CO} \mid m_\mathrm{PISN}, m_\mathrm{BH,max} \right), \sigma \right),

    where the CO-remnant mass relationship is 

    .. math::
        \mu\left( m_\mathrm{CO} \mid m_\mathrm{PISN}, m_\mathrm{BH,max} \right) = \begin{cases}
            m_\mathrm{CO} + 4 & m_\mathrm{CO} \leq m_\mathrm{PISN} - 4 \\
            m_\mathrm{BH,max} - a \left( m_\mathrm{CO} - b \right^2 & \mathrm{otherwise}
        \end{cases},

    with :math:`a` and :math:`b` chosen so that the relation is smooth when
    :math:`\mu = m_\mathrm{PISN}`; :math:`\sigma` is the amount of scatter
    around the mean CO-remnant mass relationship, and the distribution of
    offsets represented by :math:`p` is Gaussian.  

    Parameters `a` and `b` are passed to :func:`log_dNdmCO`.

    The (optional) `n_m` parameter controls how many points are used in the CO
    mass and compact object mass grids to perform the above integral.  The
    default setting works well for PISN/max BH masses around 50 MSun and `sigma`
    parameters larger than ~1 solar mass; if sigma decreases below that value,
    more grid points are needed to resolve the narrow width of the CO
    mass-remnant mass relation, and similarly if the peak mass scale increases.
    """
    a: object
    b: object
    mpisn: object
    mbhmax: object
    sigma: object
    n_m: object = 256
    mbh_grid: object = dataclasses.field(init=False)
    log_dN_grid: object = dataclasses.field(init=False)

    def __post_init__(self):
        min_bh_mass = 3.0
        min_co_mass = 1.0
        max_bh_mass = self.mbhmax + 7*self.sigma 
        max_co_mass = largest_mco(self.mpisn, self.mbhmax)

        mbh = pt.linspace(min_bh_mass, max_bh_mass, self.n_m)
        mco = pt.linspace(min_co_mass, max_co_mass, self.n_m)

        log_wts = log_dNdmCO(mco[None,:], self.a, self.b) - 0.5*pt.square((mbh[:,None] - mean_mbh_from_mco(mco[None,:], self.mpisn, self.mbhmax))/self.sigma) - np.log(np.sqrt(2*np.pi)) - pt.log(self.sigma)
        log_trapz = np.log(0.5) + pt.logaddexp(log_wts[:,1:], log_wts[:,:-1]) + pt.log(pt.diff(mco[None,:], axis=1))
        self.log_dN_grid = pt.logsumexp(log_trapz, axis=1)
        self.mbh_grid = mbh

    def __call__(self, m):
        return pt_interp(m, self.mbh_grid, self.log_dN_grid)
    
@dataclass
class LogDNDM(object):
    """
    Function-like object representing the full compact object mass function.

    A combination of :class:`LogDNDMPISN` above and a power law that begins at
    :math:`m = m_\mathrm{BH,max}` with relative amplitude `fpl` and slope `c`.
    """
    a: object
    b: object
    c: object
    mpisn: object
    mbhmax: object
    sigma: object
    fpl: object
    mref: object = 30.0
    log_norm: object = 0.0
    log_pl_norm: object = dataclasses.field(init=False)
    log_dndm_pisn: object = dataclasses.field(init=False)

    def __post_init__(self):
        self.log_dndm_pisn = LogDNDMPISN(self.a, self.b, self.mpisn, self.mbhmax, self.sigma)
        self.log_pl_norm = pt.log(self.fpl) + self.log_dndm_pisn(self.mbhmax)

        self.log_norm = -(self(np.array(self.mref)) + pt.log(self.mref)) # normalize so that m dNdm = 1 at mref

    def __call__(self, m):
        m = pt.as_tensor(m)
        log_dNdm = self.log_dndm_pisn(m)

        log_dNdm = pt.where(m <= self.log_dndm_pisn.mbh_grid[0], np.NINF, log_dNdm)
        log_dNdm = pt.where(m >= self.log_dndm_pisn.mbh_grid[-1], np.NINF, log_dNdm)

        log_dNdm = pt.logaddexp(log_dNdm, -self.c*pt.log(m/self.mbhmax) + self.log_pl_norm + log_smooth_turnon(m, self.mbhmax))

        return log_dNdm + self.log_norm
    
@dataclass
class LogDNDV(object):
    r"""
    Madau-Dickinson-like merger rate density over cosmic time:

    .. math::
        \frac{\mathrm{d} N}{\mathrm{d} V \mathrm{d} t} \propto \frac{\left( 1 + z \right)^\lambda}{1 + \left( \frac{1 + z}{1 + z_p} \right)^\kappa}
    """
    lam: object
    kappa: object
    zp: object
    zref: object = 0.0
    log_norm: object = 0.0

    def __post_init__(self):
        self.log_norm = -self(self.zref)

    def __call__(self, z):
        z = pt.as_tensor(z)

        return self.lam*pt.log1p(z) - pt.log1p(((1+z)/(1+self.zp))**self.kappa) + self.log_norm
    
@dataclass
class LogDNDMDQDV(object):
    r"""
    TODO: Document pairing function, arguments.
    """
    a: object
    b: object
    c: object
    mpisn: object
    mbhmax: object
    sigma: object
    fpl: object
    beta: object
    lam: object
    kappa: object
    zp: object
    mref: object = 30.0
    qref: object = 1.0
    zref: object = 0.0
    log_dndm: object = dataclasses.field(init=False)
    log_dndv: object = dataclasses.field(init=False)


    def __post_init__(self):
        self.log_dndm = LogDNDM(self.a, self.b, self.c, self.mpisn, self.mbhmax, self.sigma, self.fpl, self.mref)
        self.log_dndv = LogDNDV(self.lam, self.kappa, self.zp, self.zref)

    def __call__(self, m1, q, z):
        m1 = pt.as_tensor(m1)
        q = pt.as_tensor(q)
        z = pt.as_tensor(z)

        m2 = q*m1
        mt = m1+m2

        return self.log_dndm(m1) + self.log_dndm(m2) + self.beta*pt.log(mt/(self.mref*(1 + self.qref))) + pt.log(m1) + self.log_dndv(z)

@dataclass
class FlatwCDMCosmology(object):
    """
    Function-like object representing a flat w-CDM cosmology.
    """
    h: object
    Om: object
    w: object
    zmax: object = 100.0
    ninterp: object = 1024
    zinterp: object = dataclasses.field(init=False)
    dcinterp: object = dataclasses.field(init=False)
    dlinterp: object = dataclasses.field(init=False)
    ddlinterp: object = dataclasses.field(init=False)
    vcinterp: object = dataclasses.field(init=False)
    dvcinterp: object = dataclasses.field(init=False)

    def __post_init__(self):
        self.zinterp = pt.expm1(np.linspace(np.log(1), np.log(1+self.zmax), self.ninterp))
        self.dcinterp = self.dH*pt_cumtrapz(1/self.E(self.zinterp), self.zinterp)
        self.dlinterp = self.dcinterp*(1+self.zinterp)
        self.ddlinterp = self.dcinterp + self.dH*(1+self.zinterp)/self.E(self.zinterp)
        self.vcinterp = 4/3*np.pi*self.dcinterp*self.dcinterp*self.dcinterp
        self.dvcinterp = 4*np.pi*pt.square(self.dcinterp)*self.dH/self.E(self.zinterp)

    @property
    def dH(self):
        return 2.99792 / self.h
    
    @property
    def Ol(self):
        return 1-self.Om
    
    @property
    def om(self):
        return self.Om*pt.square(self.h)
    
    @property
    def ol(self):
        return self.Ol*pt.square(self.h)
    
    def E(self, z):
        opz = 1 + z
        opz3 = opz*opz*opz
        return pt.sqrt(self.Om*opz3 + (1-self.Om)*opz**(3*(1 + self.w)))

    def dC(self, z):
        return pt_interp(z, self.zinterp, self.dcinterp)
    def dL(self, z):
        return pt_interp(z, self.zinterp, self.dlinterp)
    def VC(self, z):
        return pt_interp(z, self.zinterp, self.vcinterp)
    def dVCdz(self, z):
        return pt_interp(z, self.zinterp, self.dvcinterp)
    
    def ddL_dz(self, z):
        return pt_interp(z, self.zinterp, self.ddlinterp)

    def z_of_dC(self, dC):
        return pt_interp(dC, self.dcinterp, self.zinterp)
    def z_of_dL(self, dL):
        return pt_interp(dL, self.dlinterp, self.zinterp)

def make_pop_model(m1s, qs, zs, pdraw, m1s_sel, qs_sel, zs_sel, pdraw_sel, Ndraw):
    nobs = m1s.shape[0]
    nsamp = m1s.shape[1]

    nsel = m1s_sel.shape[0]

    log_pdraw = np.log(pdraw)
    log_pdraw_sel = np.log(pdraw_sel)

    zmax = 100
    zinterp = np.expm1(np.linspace(np.log1p(0), np.log1p(zmax), 1024))
    dVdzdt_interp = 4*np.pi*Planck18.differential_comoving_volume(zinterp).to(u.Gpc**3/u.sr).value/(1+zinterp)

    coords = {
        'obs': np.arange(nobs),
        'samp': np.arange(nsamp),
        'sel': np.arange(nsel),
        'm_grid': np.exp(np.linspace(np.log(5), np.log(150), 128)),
        'q_grid': np.linspace(0, 1, 129)[1:],
        'z_grid': np.expm1(np.linspace(np.log1p(0), np.log1p(3), 128))
    }
    
    with pm.Model(coords=coords) as model:
        m1s = pm.ConstantData('m1', m1s, dims=['obs', 'samp'])
        qs = pm.ConstantData('q', qs, dims=['obs', 'samp'])
        zs = pm.ConstantData('z', zs, dims=['obs', 'samp'])

        m1s_sel = pm.ConstantData('m1_sel', m1s_sel, dims='sel')
        qs_sel = pm.ConstantData('q_sel', qs_sel, dims='sel')
        zs_sel = pm.ConstantData('z_sel', zs_sel, dims='sel')

        a = pm.Normal('a', 2.35, 2)
        b = pm.Normal('b', 1.9, 2)
        c = pm.Normal('c', 4, 2)

        mpisn = pm.Normal('mpisn', 35.0, 5.0)
        dmbhmax = pm.Truncated('dmbhmax', pm.Normal.dist(5.0, 2.0), lower=0.0)
        mbhmax = pm.Deterministic('mbhmax', mpisn + dmbhmax)
        sigma = pm.Truncated('sigma', pm.Normal.dist(2, 2), lower=1)

        beta = pm.Normal('beta', 0, 2)

        log_fpl = pm.Uniform('log_fpl', np.log(1e-3), np.log(0.5))
        fpl = pm.Deterministic('fpl', pt.exp(log_fpl))
        
        lam = pm.Normal('lam', 2.7, 2.0)
        dkappa = pm.Truncated('dkappa', pm.Normal.dist(5.6-2.7, 2.0), lower=1)
        kappa = pm.Deterministic('kappa', lam + dkappa)
        zp = pm.Truncated('zp', pm.Normal.dist(1.9, 1), lower=0)

        log_dN = LogDNDMDQDV(a, b, c, mpisn, mbhmax, sigma, fpl, beta, lam, kappa, zp)

        log_wts = log_dN(m1s, qs, zs) + pt.log(pt_interp(zs, zinterp, dVdzdt_interp)) - log_pdraw
        log_like = pt.logsumexp(log_wts, axis=1) - pt.log(nsamp)
        _ = pm.Potential('loglike', pt.sum(log_like))

        log_sel_wts = log_dN(m1s_sel, qs_sel, zs_sel) + pt.log(pt_interp(zs_sel, zinterp, dVdzdt_interp)) - log_pdraw_sel
        log_mu_sel = pt.logsumexp(log_sel_wts) - pt.log(Ndraw)
        _ = pm.Potential('selfactor', -nobs*log_mu_sel)

        log_mu2 = pt.logsumexp(2*log_sel_wts) - 2*pt.log(Ndraw)
        log_s2 = log_mu2 + pt.log1p(-pt.exp(2*log_mu_sel - pt.log(Ndraw) - log_mu2))
        _ = pm.Deterministic('neff_sel', pt.exp(2*log_mu_sel - log_s2))

        mu_sel = pt.exp(log_mu_sel)

        R = pm.Normal('R', nobs/mu_sel, pt.sqrt(nobs)/mu_sel)

        _ = pm.Deterministic('neff', pt.exp(2*pt.logsumexp(log_wts, axis=1) - pt.logsumexp(2*log_wts, axis=1)), dims='obs')

        _ = pm.Deterministic('mdNdmdVdt_fixed_qz', coords['m_grid']*R*pt.exp(log_dN(coords['m_grid'], log_dN.qref, log_dN.zref)), dims='m_grid')
        _ = pm.Deterministic('dNdqdVdt_fixed_mz', log_dN.mref*R*pt.exp(log_dN(log_dN.mref, coords['q_grid'], log_dN.zref)), dims='q_grid')
        _ = pm.Deterministic('dNdVdt_fixed_mq', log_dN.mref*R*pt.exp(log_dN(log_dN.mref, log_dN.qref, coords['z_grid'])), dims='z_grid')

    return model

def make_pop_cosmo_model(m1s_det, qs, dls, pdraw, m1s_det_sel, qs_sel, dls_sel, pdraw_sel, Ndraw):
    nobs = m1s_det.shape[0]
    nsamp = m1s_det.shape[1]

    nsel = m1s_det_sel.shape[0]

    log_pdraw = np.log(pdraw)
    log_pdraw_sel = np.log(pdraw_sel)

    coords = {
        'obs': np.arange(nobs),
        'samp': np.arange(nsamp),
        'sel': np.arange(nsel),
        'm_grid': np.exp(np.linspace(np.log(5), np.log(150), 128)),
        'q_grid': np.linspace(0, 1, 129)[1:],
        'z_grid': np.expm1(np.linspace(np.log1p(0), np.log1p(3), 128))
    }
    
    with pm.Model(coords=coords) as model:
        m1s_det = pm.ConstantData('m1_det', m1s_det, dims=['obs', 'samp'])
        qs = pm.ConstantData('q', qs, dims=['obs', 'samp'])
        dls = pm.ConstantData('dl', dls, dims=['obs', 'samp'])

        m1s_det_sel = pm.ConstantData('m1_det_sel', m1s_det_sel, dims='sel')
        qs_sel = pm.ConstantData('q_sel', qs_sel, dims='sel')
        dls_sel = pm.ConstantData('dl_sel', dls_sel, dims='sel')

        h = pm.Truncated('h', pm.Normal.dist(0.7, 0.2), lower=0.35, upper=1.4)
        Om = pm.Truncated('Om', pm.Normal.dist(0.3, 0.15), lower=0, upper=1)
        w = pm.Truncated('w', pm.Normal.dist(-1, 0.25), lower=-1.5, upper=-0.5)

        a = pm.Normal('a', 2.35, 2)
        b = pm.Normal('b', 1.9, 2)
        c = pm.Normal('c', 4, 2)

        mpisn = pm.Normal('mpisn', 35.0, 5.0)
        dmbhmax = pm.Truncated('dmbhmax', pm.Normal.dist(5.0, 2.0), lower=0.0)
        mbhmax = pm.Deterministic('mbhmax', mpisn + dmbhmax)
        sigma = pm.Truncated('sigma', pm.Normal.dist(2, 2), lower=1)

        beta = pm.Normal('beta', 0, 2)

        log_fpl = pm.Uniform('log_fpl', np.log(1e-3), np.log(0.5))
        fpl = pm.Deterministic('fpl', pt.exp(log_fpl))
        
        lam = pm.Normal('lam', 2.7, 2.0)
        dkappa = pm.Truncated('dkappa', pm.Normal.dist(5.6-2.7, 2.0), lower=1)
        kappa = pm.Deterministic('kappa', lam + dkappa)
        zp = pm.Truncated('zp', pm.Normal.dist(1.9, 1), lower=0)

        cosmo = FlatwCDMCosmology(h, Om, w)

        log_dN = LogDNDMDQDV(a, b, c, mpisn, mbhmax, sigma, fpl, beta, lam, kappa, zp)

        zs = cosmo.z_of_dL(dls)
        m1s = m1s_det / (1 + zs)

        log_wts = log_dN(m1s, qs, zs) - 2*pt.log1p(zs) + pt.log(cosmo.dVCdz(zs)) - pt.log(cosmo.ddL_dz(zs)) - log_pdraw
        log_like = pt.logsumexp(log_wts, axis=1) - pt.log(nsamp)
        _ = pm.Potential('loglike', pt.sum(log_like))

        zs_sel = cosmo.z_of_dL(dls_sel)
        m1s_sel = m1s_det_sel / (1 + zs_sel)

        log_sel_wts = log_dN(m1s_sel, qs_sel, zs_sel) - 2*pt.log1p(zs_sel) + pt.log(cosmo.dVCdz(zs_sel)) - pt.log(cosmo.ddL_dz(zs_sel)) - log_pdraw_sel
        log_mu_sel = pt.logsumexp(log_sel_wts) - pt.log(Ndraw)
        _ = pm.Potential('selfactor', -nobs*log_mu_sel)

        log_mu2 = pt.logsumexp(2*log_sel_wts) - 2*pt.log(Ndraw)
        log_s2 = log_mu2 + pt.log1p(-pt.exp(2*log_mu_sel - pt.log(Ndraw) - log_mu2))
        _ = pm.Deterministic('neff_sel', pt.exp(2*log_mu_sel - log_s2))

        mu_sel = pt.exp(log_mu_sel)

        R = pm.Normal('R', nobs/mu_sel, pt.sqrt(nobs)/mu_sel)

        _ = pm.Deterministic('neff', pt.exp(2*pt.logsumexp(log_wts, axis=1) - pt.logsumexp(2*log_wts, axis=1)), dims='obs')

        _ = pm.Deterministic('mdNdmdVdt_fixed_qz', coords['m_grid']*R*pt.exp(log_dN(coords['m_grid'], log_dN.qref, log_dN.zref)), dims='m_grid')
        _ = pm.Deterministic('dNdqdVdt_fixed_mz', log_dN.mref*R*pt.exp(log_dN(log_dN.mref, coords['q_grid'], log_dN.zref)), dims='q_grid')
        _ = pm.Deterministic('dNdVdt_fixed_mq', log_dN.mref*R*pt.exp(log_dN(log_dN.mref, log_dN.qref, coords['z_grid'])), dims='z_grid')
        _ = pm.Deterministic('hz', cosmo.h*cosmo.E(coords['z_grid']), dims='z_grid')

    return model