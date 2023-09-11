from astropy.cosmology import Planck18
import astropy.units as u
import dataclasses
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.scipy.special as jss
import numpy as np
import numpyro 
import numpyro.distributions as dist
from utils import jnp_cumtrapz

mbh_min = 5.0

def mean_mbh_from_mco(mco, mpisn, mbhmax):
    """The mean black hole mass from the core-mass to remnant-mass relation.
    
    :param mco: The CO core mass.
    :param mpisn: The BH mass at which the relation starts to turn over.
    :param mbhmax: The maximum BH mass achieved by the relation.
    """
    a = 1 / (4*(mpisn - mbhmax))
    mcomax = 2*mbhmax - mpisn
    return jnp.where(mco < mpisn, mco, mbhmax + a*jnp.square(mco - mcomax))

def sigma_mbh_from_mco(mco, mpisn, mbhmax, sigma):
    #a = (sigma - 1) / ((mbhmax - mcomax) * (mbhmax - mpisn))
    #b = - (sigma - 1) * (mcomax + mpisn) /( (mbhmax - mcomax) * (mbhmax - mpisn))
    #c = (mbhmax**2 - mbhmax * (mcomax + mpisn) + mpisn * mcomax * sigma) / ((mbhmax - mcomax) * (mbhmax - mpisn))

    #sigma_m = jnp.where( (mco > mpisn) & (mco < mcomax), a * mco**2 + b * mco + c, 1)

    #a = (sigma - 1) / (mbhmax - 20)
    #b = (mbhmax - 20 * sigma) / (mbhmax - 20)
    #sigma_m = jnp.where(mco > 20, a * mco + b, 1)
    #sigma_m = jnp.where(mco < mbhmax, a *mco + b, 1)
    #sigma_m = jnp.where(mco > mbhmax, sigma, sigma_m)
    #return sigma_m
    return sigma #* jnp.ones(sigma_m.shape)

def largest_mco(mpisn, mbhmax):
    """The largest CO core mass with positive BH masses."""
    mcomax = 2*mbhmax - mpisn
    return mcomax + jnp.sqrt(4*mbhmax*(mbhmax - mpisn))

def log_dNdmCO(mco, a, b):
    r"""The broken power law CO core mass function.
    
    The power law breaks (smoothly) at :math:`16 \, M_\odot` (i.e. a BH mass of :math:`20 \, M_\odot`).

    :param mco: The CO core mass.
    :param a: The power law slope for small CO core masses.
    :param b: The power law slope for large CO core masses.
    """
    mtr = 20.0
    x = mco/mtr
    return jnp.where(mco < mtr, -a*jnp.log(x), -b*jnp.log(x))

def log_smooth_turnon(m, mmin, width=0.05):
    """A function that smoothly transitions from 0 to 1.
    
    :param m: The function argument.
    :param mmin: The location around which the function transitions.
    :param width: (optional) The fractional width of the transition.
    """
    dm = mmin*width

    return np.log(2) - jnp.log1p(jnp.exp(-(m-mmin)/dm))

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
    n_m: object = 500
    mbh_grid: object = dataclasses.field(init=False)
    log_dN_grid: object = dataclasses.field(init=False)

    def __post_init__(self):
        min_bh_mass = 3.0
        min_co_mass = 1.0
        max_bh_mass = self.mbhmax + 7*self.sigma 
        max_co_mass = largest_mco(self.mpisn, self.mbhmax)

        mbh = jnp.linspace(min_bh_mass, max_bh_mass, self.n_m)
        mco = jnp.linspace(min_co_mass, max_co_mass, self.n_m)
        log_wts = log_dNdmCO(mco[None,:], self.a, self.b) - 0.5*jnp.square((mbh[:,None] - mean_mbh_from_mco(mco[None,:], self.mpisn, self.mbhmax))/self.sigma) - np.log(np.sqrt(2*np.pi)) - jnp.log(self.sigma)
        log_trapz = np.log(0.5) + jnp.logaddexp(log_wts[:,1:], log_wts[:,:-1]) + jnp.log(jnp.diff(mco[None,:], axis=1))
        self.log_dN_grid = jss.logsumexp(log_trapz, axis=1)
        self.mbh_grid = mbh

    def __call__(self, m):
        return jnp.interp(m, self.mbh_grid, self.log_dN_grid)

@dataclass
class LogDNDMPISN_evolve(object):
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
    n_m: object = 800
    mbh_grid: object = dataclasses.field(init=False)
    log_dN_grid: object = dataclasses.field(init=False)

    def __post_init__(self):
        min_bh_mass = 3.0
        min_co_mass = 1.0
        max_bh_mass = jnp.max(self.mbhmax *(1 + 4 * self.sigma))
        max_co_mass = largest_mco(self.mpisn, self.mbhmax)

        mbh = jnp.linspace(min_bh_mass, max_bh_mass, self.n_m+2)
        mco = jnp.linspace(min_co_mass, max_co_mass, self.n_m)

        sigma = self.sigma#sigma_mbh_from_mco(mco[None,:], self.mpisn, self.mbhmax, self.sigma)
        mu = mean_mbh_from_mco(mco[None,:], self.mpisn, self.mbhmax)
        exponent = jnp.where(mu < 0.0, jnp.NINF, -0.5 * jnp.square((jnp.log(mbh[:,None,None]) - jnp.log(mu))/sigma) )

        log_wts = log_dNdmCO(mco[None,:], self.a, self.b) + exponent - np.log(np.sqrt(2*np.pi)) - jnp.log(sigma) - jnp.log(mbh[:,None,None]) 
        log_trapz = np.log(0.5) + jnp.logaddexp(log_wts[:,1:, :], log_wts[:,:-1, :]) + jnp.log(jnp.diff(mco[None,:], axis=1))
        self.log_dN_grid = jss.logsumexp(log_trapz, axis=1)
        self.mbh_grid = mbh

    #def __call__(self, m):
        #return jnp.interp(m, self.mbh_grid, self.log_dN_grid)


@dataclass
class LogDNDM_evolve(object):
    """
    Function-like object representing the full compact object mass function.

    A combination of :class:`LogDNDMPISN` above and a power law that begins at
    :math:`m = m_\mathrm{BH,max}` with relative amplitude `fpl` and slope `c`.
    """
    a: object
    b: object
    c: object
    mpisn: object
    mpisndot: object
    mbhmax: object
    sigma: object
    fpl: object
    mbh_min: object = mbh_min
    zmax: object = 20
    mref: object = 30.0
    zref: object = 0.01
    log_dndm_pisn: object = dataclasses.field(init=False)

    def __post_init__(self):
        self.dmbhmax = self.mbhmax - self.mpisn
        self.setup_interp()

        #mpisn_at_zref = self.mpisn + self.mpisndot * (1 - 1/(1 + self.zref))
        #mbhmax_at_zref = mpisn_at_zref + self.dmbhmax
        
       # self.log_pl_norm = jnp.log(self.fpl) + self.interp_2d_dndmpisn(mbhmax_at_zref, self.zref)

        #self.log_norm = -(self(self.mref, self.zref) + jnp.log(self.mref)) # normalize so that m dNdm = 1 at mref

    def setup_interp(self):
        self.z_array = jnp.expm1(jnp.linspace(np.log(1), jnp.log(1+self.zmax), 50))
        mpisns = self.mpisn + self.mpisndot * (1 - 1/(1+self.z_array))
        mbhmaxs = mpisns + self.dmbhmax
        self.log_dndm_pisn = LogDNDMPISN_evolve(self.a, self.b, jnp.array(mpisns), jnp.array(mbhmaxs), self.sigma)
        self.mbh_grid = self.log_dndm_pisn.mbh_grid
        self.log_dndm_pisn_grid = self.log_dndm_pisn.log_dN_grid
        self.mbhmaxs = jnp.array(mbhmaxs)
    
    def interp_2d_dndmpisn(self, m, z):
        m_indxs = jnp.searchsorted(self.mbh_grid, m)
        z_indxs = jnp.searchsorted(self.z_array, z)
        
        m_upper = self.mbh_grid[m_indxs]
        m_lower = self.mbh_grid[m_indxs - 1]
        
        z_upper = self.z_array[z_indxs]
        z_lower = self.z_array[z_indxs - 1]
        
        f1 = self.log_dndm_pisn_grid[m_indxs - 1, z_indxs - 1]
        f2 = self.log_dndm_pisn_grid[m_indxs, z_indxs - 1]
        f3 = self.log_dndm_pisn_grid[m_indxs, z_indxs]
        f4 = self.log_dndm_pisn_grid[m_indxs - 1, z_indxs]

        mdiffs = m_upper - m_lower
        zdiffs = z_upper - z_lower

        mdiffs = jnp.where(mdiffs != 0, mdiffs, np.inf)
        zdiffs = jnp.where(zdiffs != 0, zdiffs, np.inf)

        t = jnp.where(m_lower == m_upper, 0, (m - m_lower) / mdiffs)
        u = jnp.where(z_lower == z_upper, 0, (z - z_lower) / zdiffs)
        
        return (1 - t) * (1 - u) * f1 + t * (1 - u) * f2 + t * u *f3 + (1 - t) * u * f4
        
    def __call__(self, m, z):
        m = jnp.array(m)
        z = jnp.array(z) 
        log_dNdm = self.interp_2d_dndmpisn(m, z)

        #log_dNdm = jnp.where(m <= self.log_dndm_pisn.mbh_grid[0], log_dNdm[0], log_dNdm)
        log_dNdm = jnp.where(m >= self.log_dndm_pisn.mbh_grid[-1], np.NINF, log_dNdm)
        
        mbhmax_at_samples = jnp.array(self.mpisn + self.mpisndot * (1 - 1/(1+z)) + self.dmbhmax)

        log_dNdmbhmax_at_samples = self.interp_2d_dndmpisn(mbhmax_at_samples, z)
        log_dNdm = jnp.logaddexp(log_dNdm, jnp.log(self.fpl) + log_dNdmbhmax_at_samples + -self.c*jnp.log(m/mbhmax_at_samples)  + log_smooth_turnon(m, mbhmax_at_samples))
        log_dNdm = jnp.where(m < self.mbh_min, np.NINF, log_dNdm)

        return log_dNdm #+ self.log_norm

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
    mbh_min: object = mbh_min
    mref: object = 30.0
    log_norm: object = 0.0
    log_pl_norm: object = dataclasses.field(init=False)
    log_dndm_pisn: object = dataclasses.field(init=False)

    def __post_init__(self):
        self.log_dndm_pisn = LogDNDMPISN(self.a, self.b, self.mpisn, self.mbhmax, self.sigma)
        self.log_pl_norm = jnp.log(self.fpl) + self.log_dndm_pisn(self.mbhmax)

        self.log_norm = -(self(np.array(self.mref)) + jnp.log(self.mref)) # normalize so that m dNdm = 1 at mref


    def __call__(self, m):
        m = jnp.array(m)
        log_dNdm = self.log_dndm_pisn(m)

        #log_dNdm = jnp.where(m <= self.log_dndm_pisn.mbh_grid[0], np.NINF, log_dNdm)
        log_dNdm = jnp.where(m >= self.log_dndm_pisn.mbh_grid[-1], np.NINF, log_dNdm)

        log_dNdm = jnp.logaddexp(log_dNdm, -self.c*jnp.log(m/self.mbhmax) + self.log_pl_norm + log_smooth_turnon(m, self.mbhmax))
        #log_dNdm = jnp.where(m < self.mbh_min, np.NINF, log_dNdm)

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
    zref: object = 0.01
    zmax: object = 20
    log_norm: object = 0.0

    def __post_init__(self):
        self.log_norm = -self(self.zref)

    def __call__(self, z):
        z = jnp.array(z)

        return jnp.where(z < self.zmax, self.lam*jnp.log1p(z) - jnp.log1p(((1+z)/(1+self.zp))**self.kappa) + self.log_norm, -np.inf)
    
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
    zref: object = 0.01
    log_dndm: object = dataclasses.field(init=False)
    log_dndv: object = dataclasses.field(init=False)


    def __post_init__(self):
        self.log_dndm = LogDNDM(self.a, self.b, self.c, self.mpisn, self.mbhmax, self.sigma, self.fpl, mref=self.mref)
        self.log_dndv = LogDNDV(self.lam, self.kappa, self.zp, zref=self.zref)

    def __call__(self, m1, q, z):
        m1 = jnp.array(m1)
        q = jnp.array(q)
        z = jnp.array(z)

        m2 = q*m1
        mt = m1+m2

        return self.log_dndm(m1) + self.log_dndm(m2) + self.beta*jnp.log(mt/(self.mref*(1 + self.qref))) + jnp.log(m1) + self.log_dndv(z)

@dataclass
class LogDNDMDQDV_evolve(object):
    r"""
    TODO: Document pairing function, arguments.
    """
    a: object
    b: object
    c: object
    mpisn: object
    mpisndot: object
    mbhmax: object
    sigma: object
    fpl: object
    beta: object
    lam: object
    kappa: object
    zp: object
    mref: object = 30.0
    qref: object = 1.0
    zref: object = 0.01
    zmax: object = 20
    log_dndm: object = dataclasses.field(init=False)
    log_dndv: object = dataclasses.field(init=False)


    def __post_init__(self):
        self.log_dndm = LogDNDM_evolve(self.a, self.b, self.c, self.mpisn, self.mpisndot, self.mbhmax, self.sigma, self.fpl, mref=self.mref, zmax=self.zmax, zref = self.zref)
        self.log_dndv = LogDNDV(self.lam, self.kappa, self.zp, self.zref, zmax=self.zmax)
        self.log_norm = 2 * jnp.log(self.mref) + self.log_dndm(self.mref, self.zref) + self.log_dndm(self.qref * self.mref, self.zref) #+ self.log_dndv(self.zref)

    def __call__(self, m1, q, z):
        m1 = jnp.array(m1)
        q = jnp.array(q)
        z = jnp.array(z)

        m2 = q*m1
        mt = m1+m2

        return self.log_dndm(m1, z) + self.log_dndm(m2, z) + self.beta*jnp.log(mt/(self.mref*(1 + self.qref))) + jnp.log(m1) + self.log_dndv(z) - self.log_norm
        
@dataclass
class FlatwCDMCosmology(object):
    """
    Function-like object representing a flat w-CDM cosmology.
    """
    h: object
    Om: object
    w: object
    zmax: object = 20.0
    ninterp: object = 1024
    zinterp: object = dataclasses.field(init=False)
    dcinterp: object = dataclasses.field(init=False)
    dlinterp: object = dataclasses.field(init=False)
    ddlinterp: object = dataclasses.field(init=False)
    vcinterp: object = dataclasses.field(init=False)
    dvcinterp: object = dataclasses.field(init=False)

    def __post_init__(self):
        self.zinterp = jnp.expm1(np.linspace(np.log(1), np.log(1+self.zmax), self.ninterp))
        self.dcinterp = self.dH*jnp_cumtrapz(1/self.E(self.zinterp), self.zinterp)
        self.dlinterp = self.dcinterp*(1+self.zinterp)
        self.ddlinterp = self.dcinterp + self.dH*(1+self.zinterp)/self.E(self.zinterp)
        self.vcinterp = 4/3*np.pi*self.dcinterp*self.dcinterp*self.dcinterp
        self.dvcinterp = 4*np.pi*jnp.square(self.dcinterp)*self.dH/self.E(self.zinterp)

    @property
    def dH(self):
        return 2.99792 / self.h
    
    @property
    def Ol(self):
        return 1-self.Om
    
    @property
    def om(self):
        return self.Om*jnp.square(self.h)
    
    @property
    def ol(self):
        return self.Ol*jnp.square(self.h)
    
    def E(self, z):
        opz = 1 + z
        opz3 = opz*opz*opz
        return jnp.sqrt(self.Om*opz3 + (1-self.Om)*opz**(3*(1 + self.w)))

    def dC(self, z):
        return jnp.interp(z, self.zinterp, self.dcinterp)
    def dL(self, z):
        return jnp.interp(z, self.zinterp, self.dlinterp)
    def VC(self, z):
        return jnp.interp(z, self.zinterp, self.vcinterp)
    def dVCdz(self, z):
        return jnp.interp(z, self.zinterp, self.dvcinterp)
    
    def ddL_dz(self, z):
        return jnp.interp(z, self.zinterp, self.ddlinterp)

    def z_of_dC(self, dC):
        return jnp.interp(dC, self.dcinterp, self.zinterp)
    def z_of_dL(self, dL):
        return jnp.interp(dL, self.dlinterp, self.zinterp)

coords = {
    'm_grid': np.exp(np.linspace(np.log(5), np.log(150), 128)),
    'q_grid': np.linspace(0, 1, 129)[1:],
    'z_grid': np.expm1(np.linspace(np.log1p(0), np.log1p(3), 128))
}

def mass_parameters():
    a = numpyro.sample('a', dist.TruncatedNormal(2.35, 2, low=-1.65, high=6.35))
    b = numpyro.sample('b', dist.TruncatedNormal(1.9, 2, low=-2.1, high=5.9))
    c = numpyro.sample('c', dist.TruncatedNormal(4, 2, low=0, high=8))

    mpisn = numpyro.sample('mpisn', dist.TruncatedNormal(35.0, 5.0, low=20.0, high=50.0))
    dmbhmax = numpyro.sample('dmbhmax', dist.TruncatedNormal(5.0, 2.0, low=0.5, high=11.0))
    mbhmax = numpyro.deterministic('mbhmax', mpisn + dmbhmax)
    sigma = numpyro.sample('sigma', dist.TruncatedNormal(2, 4, low=1))

    beta = numpyro.sample('beta', dist.Normal(0, 2))

    log_fpl = numpyro.sample('log_fpl', dist.Uniform(np.log(1e-3), np.log(0.5)))
    fpl = numpyro.deterministic('fpl', jnp.exp(log_fpl))

    return a,b,c,mpisn,mbhmax,sigma,beta,fpl

def redshift_parameters():
    lam = numpyro.sample('lam', dist.TruncatedNormal(2.7, 2.0, low=-1.3, high=6.7))
    dkappa = numpyro.sample('dkappa', dist.TruncatedNormal(5.6-2.7, 2.0, low=1, high=9.6-2.7))
    kappa = numpyro.deterministic('kappa', lam + dkappa)
    zp = numpyro.sample('zp', dist.TruncatedNormal(1.9, 1, low=0, high=3.9))

    return lam,kappa,zp

def cosmo_parameters():
    h = numpyro.sample('h', dist.TruncatedNormal(0.63, 0.2, low=0.35, high=1.4))
    Om = numpyro.sample('Om', dist.TruncatedNormal(0.3, 0.15, low=0, high=1))
    w = numpyro.sample('w', dist.TruncatedNormal(-1, 0.25, low=-1.5, high=-0.5))

    return h,Om,w

def pop_cosmo_model(m1s_det, qs, dls, pdraw, m1s_det_sel, qs_sel, dls_sel, pdraw_sel, Ndraw, evolution = False, zmax=20, fixed_cosmo_params = None):
    m1s_det, qs, dls, pdraw, m1s_det_sel, qs_sel, dls_sel, pdraw_sel = map(jnp.array, (m1s_det, qs, dls, pdraw, m1s_det_sel, qs_sel, dls_sel, pdraw_sel))

    nobs = m1s_det.shape[0]
    nsamp = m1s_det.shape[1]

    nsel = m1s_det_sel.shape[0]

    log_pdraw = jnp.log(pdraw)
    log_pdraw_sel = jnp.log(pdraw_sel)

    if fixed_cosmo_params is None:
        h,Om,w = cosmo_parameters()
    else:
        h = fixed_cosmo_params['h']
        Om = fixed_cosmo_params['Om']
        w = fixed_cosmo_params['w']

    a = numpyro.sample('a', dist.Normal(2.35, 2))
    b = numpyro.sample('b', dist.Normal(1.9, 2))
    c = numpyro.sample('c', dist.Normal(4, 2))

    mpisn = numpyro.sample('mpisn', dist.TruncatedNormal(35.0, 5.0, low=25))
    dmbhmax = numpyro.sample('dmbhmax', dist.TruncatedNormal(5.0, 4.0, low=0.0))
    sigma = numpyro.sample('sigma', dist.Uniform(low=0.04, high=0.2))

    beta = numpyro.sample('beta', dist.Normal(0, 2))

    log_fpl = numpyro.sample('log_fpl', dist.Uniform(np.log(1e-3), np.log(0.5)))
    fpl = numpyro.deterministic('fpl', jnp.exp(log_fpl))
    
    lam,kappa,zp = redshift_parameters()

    cosmo = FlatwCDMCosmology(h, Om, w, zmax=zmax)

    if not evolution:
        mpisndot = 0
    else:
        mpisndot = numpyro.sample('mpisndot', dist.Uniform(low=0, high=8))
        
    mpisnzinf = mpisn + mpisndot
    mbhmax = numpyro.deterministic('mbhmax', mpisn + dmbhmax)
    log_dN = LogDNDMDQDV_evolve(a=a, b=b, c=c, mpisn=mpisn, mpisndot=mpisndot, mbhmax=mbhmax, sigma=sigma, fpl=fpl, beta=beta, lam=lam, kappa=kappa, zp=zp, zmax=zmax)
    zs = cosmo.z_of_dL(dls)
    m1s = m1s_det / (1 + zs)

    log_wts = log_dN(m1s, qs, zs) - 2*jnp.log1p(zs) + jnp.log(cosmo.dVCdz(zs)) - jnp.log(cosmo.ddL_dz(zs)) - log_pdraw
    log_like = jss.logsumexp(log_wts, axis=1) - jnp.log(nsamp)
    log_like = jnp.nan_to_num(jnp.nan_to_num(jnp.sum(log_like), nan=-np.inf))

    _ = numpyro.factor('loglike', log_like)

    zs_sel = cosmo.z_of_dL(dls_sel)
    m1s_sel = m1s_det_sel / (1 + zs_sel)

    log_sel_wts = log_dN(m1s_sel, qs_sel, zs_sel) - 2*jnp.log1p(zs_sel) + jnp.log(cosmo.dVCdz(zs_sel)) - jnp.log(cosmo.ddL_dz(zs_sel)) - log_pdraw_sel
    log_mu_sel = jss.logsumexp(log_sel_wts) - jnp.log(Ndraw)
    _ = numpyro.factor('selfactor', jnp.nan_to_num(jnp.nan_to_num(-nobs*log_mu_sel, nan=-np.inf)))

    log_mu2 = jss.logsumexp(2*log_sel_wts) - 2*jnp.log(Ndraw)
    log_s2 = log_mu2 + jnp.log1p(-jnp.exp(2*log_mu_sel - jnp.log(Ndraw) - log_mu2))
    _ = numpyro.deterministic('neff_sel', jnp.exp(2*log_mu_sel - log_s2))

    mu_sel = jnp.exp(log_mu_sel)

    R_unit = numpyro.sample('R_unit', dist.Normal(0, 1))
    R = numpyro.deterministic('R', nobs/mu_sel + jnp.sqrt(nobs)/mu_sel*R_unit)

    _ = numpyro.deterministic('neff', jnp.exp(2*jss.logsumexp(log_wts, axis=1) - jss.logsumexp(2*log_wts, axis=1)))

    _ = numpyro.deterministic('mdNdmdVdt_fixed_qz', coords['m_grid']*R*jnp.exp(log_dN(coords['m_grid'], log_dN.qref, log_dN.zref)))
    _ = numpyro.deterministic('dNdqdVdt_fixed_mz', log_dN.mref*R*jnp.exp(log_dN(log_dN.mref, coords['q_grid'], log_dN.zref)))
    _ = numpyro.deterministic('dNdVdt_fixed_mq', log_dN.mref*R*jnp.exp(log_dN(log_dN.mref, log_dN.qref, coords['z_grid'])))
    _ = numpyro.deterministic('hz', cosmo.h*cosmo.E(coords['z_grid']))




def pop_cosmo_model_noselection(m1s_det, qs, dls, pdraw, **kwargs):
    m1s_det, qs, dls, pdraw = map(jnp.array, (m1s_det, qs, dls, pdraw))

    nobs = m1s_det.shape[0]
    nsamp = m1s_det.shape[1]

    #log_pdraw = jnp.log(pdraw)

    h = numpyro.sample('h', dist.TruncatedNormal(0.7, 0.2, low=0.35, high=1.4))
    Om = numpyro.sample('Om', dist.TruncatedNormal(0.3, 0.15, low=0, high=1))
    w = numpyro.sample('w', dist.TruncatedNormal(-1, 0.25, low=-1.5, high=-0.5))

    a = numpyro.sample('a', dist.Normal(2.35, 2))
    b = numpyro.sample('b', dist.Normal(1.9, 2))
    c = numpyro.sample('c', dist.Normal(4, 2))

    mpisn = numpyro.sample('mpisn', dist.TruncatedNormal(35.0, 5.0, low=15))
    dmbhmax = numpyro.sample('dmbhmax', dist.TruncatedNormal(5.0, 2.0, low=0.0))
    mbhmax = numpyro.deterministic('mbhmax', mpisn + dmbhmax)
    sigma = numpyro.sample('sigma', dist.TruncatedNormal(2, 2, low=1))

    beta = numpyro.sample('beta', dist.Normal(0, 2))

    log_fpl = numpyro.sample('log_fpl', dist.Uniform(np.log(1e-3), np.log(0.5)))
    fpl = numpyro.deterministic('fpl', jnp.exp(log_fpl))
    
    lam = numpyro.sample('lam', dist.Normal(2.7, 2.0))
    dkappa = numpyro.sample('dkappa', dist.TruncatedNormal(5.6-2.7, 2.0, low=1))
    kappa = numpyro.deterministic('kappa', lam + dkappa)
    zp = numpyro.sample('zp', dist.TruncatedNormal(1.9, 1, low=0))
    """
    params_dict = dict(
            a= a,
            b =b,
            c = c,
            mpisn= mpisn,
            mbhmax = mbhmax,
            fpl = fpl,
    beta = beta,
    lam = lam,
    kappa = kappa,
    zp = zp,
    h = h,
    Om = Om,
    w = w)
    """

    cosmo = FlatwCDMCosmology(h, Om, w)
    zs = cosmo.z_of_dL(dls)
    m1s = m1s_det / (1 + zs)
    """"
    log_dN = LogDNDMDQDV(a, b, c, mpisn, mbhmax, sigma, fpl, beta, lam, kappa, zp)


    m1_line = jnp.linspace(5, 150, 300)
    q_line = jnp.linspace(0.01, 1, 200)
    mm, qq = jnp.meshgrid(m1_line, q_line)

    mass_norm_grid =  jnp.exp(log_dN(mm, qq, z=0.001))
    mass_norm = jnp.trapz(x = q_line, y = jnp.trapz(x = m1_line, y = mass_norm_grid))

    z_norm_grid = jnp.exp(log_dN(m1=30, q=1, z = cosmo.zinterp))
    z_norm = jnp.trapz(x = cosmo.zinterp, y = z_norm_grid)

    total_norm = mass_norm * z_norm

    log_wts = log_dN(m1s, qs, zs) - jnp.log(total_norm) - 2*jnp.log1p(zs) + jnp.log(cosmo.dVCdz(zs)) - jnp.log(cosmo.ddL_dz(zs)) #- log_pdraw
    """
    log_dN = LogDNDMDQDV(a, b, c, mpisn, mbhmax, sigma, fpl, beta, lam, kappa, zp)
    #log_dN = log_dN_func(m1s, qs, zs)

    log_p_unnormed = log_dN(m1s, qs, zs) + jnp.log(cosmo.dVCdz(zs)) - jnp.log1p(zs)
    m1_line = jnp.linspace(5, 150, 300)
    q_line = jnp.linspace(0.01, 1, 200)
    mm, qq = jnp.meshgrid(m1_line, q_line)

    mass_norm_grid =  jnp.exp(log_dN(mm, qq, z=0.001))
    mass_norm = jnp.trapz(x = q_line, y = jnp.trapz(x = m1_line, y = mass_norm_grid))

    z_norm_grid = jnp.exp(log_dN(m1=30, q=1, z = cosmo.zinterp)) *cosmo.dvcinterp/(1+cosmo.zinterp)
    z_norm = jnp.trapz(x = cosmo.zinterp, y = z_norm_grid)

    total_norm = mass_norm * z_norm
    log_wts  = log_p_unnormed - jnp.log(total_norm) - jnp.log1p(zs) - jnp.log(cosmo.ddL_dz(zs))
    log_like = jss.logsumexp(log_wts, axis=1) - jnp.log(nsamp)
    _ = numpyro.factor('loglike', jnp.sum(log_like))

    #VT =  jnp.trapz(x = cosmo.zinterp, y=cosmo.dVCdz(cosmo.zinterp) / (1 + cosmo.zinterp))
    R = numpyro.sample('Ntotal', dist.Normal(nobs, jnp.sqrt(nobs)))
    #R = numpyro.deterministic('R_density', Ntotal*jnp.exp(log_dN(m,q,z))/total_norm)

    _ = numpyro.deterministic('neff', jnp.exp(2*jss.logsumexp(log_wts, axis=1) - jss.logsumexp(2*log_wts, axis=1)))
    """
    _ = numpyro.deterministic('mdNdmdVdt_fixed_qz', coords['m_grid']*R*jnp.exp(log_dN(coords['m_grid'], log_dN.qref, log_dN.zref)))
    _ = numpyro.deterministic('dNdqdVdt_fixed_mz', log_dN.mref*R*jnp.exp(log_dN(log_dN.mref, coords['q_grid'], log_dN.zref)))
    _ = numpyro.deterministic('dNdVdt_fixed_mq', log_dN.mref*R*jnp.exp(log_dN(log_dN.mref, log_dN.qref, coords['z_grid'])))
    _ = numpyro.deterministic('hz', cosmo.h*cosmo.E(coords['z_grid']))
    """
"""
def pop_model(m1s, qs, zs, pdraw, m1s_sel, qs_sel, zs_sel, pdraw_sel, Ndraw):
    m1s, qs, zs, pdraw, m1s_sel, qs_sel, zs_sel, pdraw_sel = map(jnp.array, (m1s, qs, zs, pdraw, m1s_sel, qs_sel, zs_sel, pdraw_sel))

    nobs = m1s.shape[0]
    nsamp = m1s.shape[1]

    nsel = m1s_sel.shape[0]

    log_pdraw = jnp.log(pdraw)
    log_pdraw_sel = jnp.log(pdraw_sel)

    zmax = 20
    zinterp = np.expm1(np.linspace(np.log1p(0), np.log1p(zmax), 1024))
    dVdzdt_interp = 4*np.pi*Planck18.differential_comoving_volume(zinterp).to(u.Gpc**3/u.sr).value/(1+zinterp)
    
    a,b,c,mpisn,mbhmax,sigma,beta,fpl = mass_parameters()

    lam,kappa,zp = redshift_parameters()

    log_dN = LogDNDMDQDV(a, b, c, mpisn, mbhmax, sigma, fpl, beta, lam, kappa, zp)

    log_wts = log_dN(m1s, qs, zs) + jnp.log(jnp.interp(zs, zinterp, dVdzdt_interp)) - log_pdraw
    log_like = jss.logsumexp(log_wts, axis=1) - jnp.log(nsamp)
    _ = numpyro.factor('loglike', jnp.sum(log_like))

    log_sel_wts = log_dN(m1s_sel, qs_sel, zs_sel) + jnp.log(jnp.interp(zs_sel, zinterp, dVdzdt_interp)) - log_pdraw_sel
    log_mu_sel = jss.logsumexp(log_sel_wts) - jnp.log(Ndraw)
    _ = numpyro.factor('selfactor', -nobs*log_mu_sel)

    log_mu2 = jss.logsumexp(2*log_sel_wts) - 2*jnp.log(Ndraw)
    log_s2 = log_mu2 + jnp.log1p(-jnp.exp(2*log_mu_sel - jnp.log(Ndraw) - log_mu2))
    _ = numpyro.deterministic('neff_sel', jnp.exp(2*log_mu_sel - log_s2))

    mu_sel = jnp.exp(log_mu_sel)

    R_unit = numpyro.sample('R_unit', dist.Normal(0, 1))
    R = numpyro.deterministic('R', nobs/mu_sel + jnp.sqrt(nobs)/mu_sel*R_unit)

    _ = numpyro.deterministic('neff', jnp.exp(2*jss.logsumexp(log_wts, axis=1) - jss.logsumexp(2*log_wts, axis=1)))

    _ = numpyro.deterministic('mdNdmdVdt_fixed_qz', coords['m_grid']*R*jnp.exp(log_dN(coords['m_grid'], log_dN.qref, log_dN.zref)))
    _ = numpyro.deterministic('dNdqdVdt_fixed_mz', log_dN.mref*R*jnp.exp(log_dN(log_dN.mref, coords['q_grid'], log_dN.zref)))
    _ = numpyro.deterministic('dNdVdt_fixed_mq', log_dN.mref*R*jnp.exp(log_dN(log_dN.mref, log_dN.qref, coords['z_grid'])))
"""