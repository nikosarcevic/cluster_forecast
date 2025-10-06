import numpy as np
from scipy.integrate import quad, simpson
from scipy.interpolate import UnivariateSpline
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from classy import Class
import time

def setup_cosmology_parameters(params):
    """
    Setup cosmological parameters
    
    Args:
        params: array of parameters [h, omega_b, omega_cdm, ns, A_s, sigma8]
    
    Returns:
        Dictionary containing all cosmological parameters and objects
    """
    h, omega_b, omega_cdm, n_s, A_s, sigma8 = params
    omega_m = omega_b + omega_cdm
    Om0 = omega_m / h**2
    Ob0 = omega_b / h**2
    
    #Astropy cosmology
    cosmo = FlatLambdaCDM(H0=h*100, Om0=Om0)
    
    #CLASS cosmology
    cosmopars = {
        'h': h,
        'omega_b': omega_b,  
        'omega_cdm': omega_cdm,
        'n_s': n_s,
        'A_s': A_s,
        'output': 'mPk, mTk',
        'z_max_pk': 10., 
        'P_k_max_h/Mpc': 2500.,
        'N_ur': 3.04,
        'k_per_decade_for_pk': 50.0,
        'z_max_pk': 200.0,
    }
    
    cosmo_CLASS = Class()
    cosmo_CLASS.set(cosmopars)
    cosmo_CLASS.compute()
    
    #Calculate critical density
    crit_rho = cosmo.critical_density(0).to(u.Msun / u.Mpc**3).value / h**2
    rho_b_0 = Om0 * crit_rho
    
    return {
        'h': h,
        'omega_b': omega_b,
        'omega_cdm': omega_cdm,
        'omega_m': omega_m,
        'Om0': Om0,
        'Ob0': Ob0,
        'n_s': n_s,
        'A_s': A_s,
        'sigma8': sigma8,
        'cosmo': cosmo,
        'cosmo_CLASS': cosmo_CLASS,
        'crit_rho': crit_rho,
        'rho_b_0': rho_b_0,
    }

def mpk_class(cosmo_CLASS, k, z):
    """Matter Power Spectrum from CLASS"""
    z = 0.0
    return cosmo_CLASS.pk(k, z)

def growth_factor_unnormalized(Om0, z):    
    """Growth Factor for Flat ΛCDM"""
    scalar_input = np.isscalar(z)
    z_array = np.atleast_1d(z)
    D = np.zeros_like(z_array)
    
    def Ez(z):
        return np.sqrt(Om0 * (1 + z)**3 + (1.0 - Om0))
    
    def integrand(z_prime):
        return (1 + z_prime) / (Ez(z_prime))**3
    
    for i, z_val in enumerate(z_array):
        D[i] = 5.0/2.0 * Om0 * Ez(z_val) * quad(integrand, z_val, np.inf)[0]
    
    if scalar_input:
        return D[0]
    else:
        return D

def growth_factor(Om0, z):
    """Normalized Growth Factor for Flat ΛCDM"""
    scalar_input = np.isscalar(z)
    z_array = np.atleast_1d(z)
    
    if np.max(np.abs(z_array)) < 1e-10:
        D = np.ones_like(z_array)
    else:
        z_with_zero = np.unique(np.append(z_array, 0.0))
        D_unnorm = growth_factor_unnormalized(Om0, z_with_zero)
        
        D0 = D_unnorm[z_with_zero == 0.0][0]
        D_all = D_unnorm / D0
        
        indices = np.searchsorted(z_with_zero, z_array)
        D = D_all[indices]
    
    if scalar_input:
        return D[0]
    else:
        return D

def smoothing_R(M, rho_b_0):
    """Smoothing scale of the variance of linear matter power spectrum"""
    R_val = ((3 * M) / (4 * np.pi * rho_b_0)) ** (1/3)
    return R_val

def top_hat_filter(k, R):
    """Top hat filter"""
    x = R * k
    if isinstance(x, np.ndarray):
        W = np.ones_like(x)
        mask = x > 1e-8
        W[mask] = 3 * (np.sin(x[mask]) - x[mask] * np.cos(x[mask]))/(x[mask]**3)
    else:
        if x <= 1e-8:
            W = 1.0
        else:
            W = 3 * (np.sin(x) - x * np.cos(x)) / (x**3)
    return W

def sigma_unnormalized(cosmo_params, R, z):
    """Computes unnormalized sigma with minimal caching"""
    tk0 = cosmo_params['cosmo_CLASS'].get_transfer()
    k0 = tk0['k (h/Mpc)']
    k_vals = np.array([i for i in k0 if i <= 100.0])
    
    W = top_hat_filter(k_vals, R)
    pk_vals = np.array(([mpk_class(cosmo_params['cosmo_CLASS'], k, z) for k in k_vals])) * (cosmo_params['h']**3)
    
    integrand = (k_vals ** 2) * pk_vals * (W**2)
    
    integral = simpson(integrand, x=k_vals) * ((growth_factor(cosmo_params['Om0'], z)**2) / (2 * np.pi**2))
    return np.sqrt(integral)

def sigma8_normalization(cosmo_params):
    """Calculate sigma8 normalization"""
    R_8 = 8.0
    sigma8_unnormalized = sigma_unnormalized(cosmo_params, R_8, z=0)
    normalization = cosmo_params['sigma8'] / sigma8_unnormalized
    return normalization

def sigma_normalized(cosmo_params, M, z):
    """Normalized sigma"""
    R = smoothing_R(M, cosmo_params['rho_b_0'])
    sigma_normed = sigma_unnormalized(cosmo_params, R, z) * sigma8_normalization(cosmo_params)
    return sigma_normed

def compute_tabulated_sigma(cosmo_params, M_array, z):
    """Compute tabulated sigma for efficiency"""
    log_M_min = np.log10(np.min(M_array)) - 0.5
    log_M_max = np.log10(np.max(M_array)) + 0.5
    M_tab = np.logspace(log_M_min, log_M_max, 100)
    
    sigma_tab = np.array([sigma_normalized(cosmo_params, M, z) for M in M_tab])
    
    log_M_tab = np.log(M_tab)
    log_sigma_tab = np.log(sigma_tab)
    spline = UnivariateSpline(log_M_tab, log_sigma_tab, k=3, s=0)
    
    log_M = np.log(M_array)
    log_sigma = spline(log_M)
    sigma = np.exp(log_sigma)
    
    return sigma

def hmf_fit_jenkins(sigma):
    """Jenkins HMF fitting function"""
    hmf_jenkins = 0.315 * np.exp(-np.abs(np.log(1/sigma) + 0.61)**3.8)
    return hmf_jenkins

def compute_hmf_jenkins(cosmo_params, M, z):
    """
    Compute HMF using Jenkins fitting function
    
    Args:
        cosmo_params: dictionary containing cosmological parameters
        M: array of masses
        z: redshift
        
    Returns:
        dn/dlnM: halo mass function
        f_sigma: fitting function values
    """
    sigma_array = compute_tabulated_sigma(cosmo_params, M, z)
    
    f_sigma = np.array([hmf_fit_jenkins(s) for s in sigma_array])
    
    log_M = np.log(M)
    log_sigma = np.log(sigma_array)
    
    spline = UnivariateSpline(log_M, log_sigma, k=3, s=0)
    dlnsigma_dlnM = spline.derivative()(log_M)
    
    dn_dlnM = (cosmo_params['rho_b_0']/M) * f_sigma * np.abs(dlnsigma_dlnM)
    
    return dn_dlnM, f_sigma

def compute_hmf_analytical(params, mass, z):
    """ 
    Wrapper for HMF calculation using analytical approach
    
    Args:
        params: parameter values (Order: h, omega_b, omega_cdm, n_s, A_s, sigma8)
        z: redshift at which to evaluate HMF
        mass: array of masses at which to calculate HMF
    
    Returns:
        Halo Mass Function and fitting function values
    """
    cosmo_params = setup_cosmology_parameters(params)
    
    dn_dlnM, f_sigma = compute_hmf_jenkins(cosmo_params, mass, z)
    
    return dn_dlnM, f_sigma

def compute_multiple_redshifts(params, mass, z_array):
    """
    Compute HMF for multiple redshifts - reuse CLASS setup
    
    Args:
        params: parameter values
        mass: array of masses  
        z_array: array of redshifts
        
    Returns:
        Dictionary with z as keys and (dn_dlnM, f_sigma) as values
    """
    cosmo_params = setup_cosmology_parameters(params)
    
    results = {}
    for z in z_array:
        dn_dlnM, f_sigma = compute_hmf_jenkins(cosmo_params, mass, z)
        results[z] = (dn_dlnM, f_sigma)
    
    return results

