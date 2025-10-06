import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

from src.hmf_theory_nk import compute_hmf_analytical


class LSSTClusterAnalysis:
    """
    Class to handle LSST DESC-SRD cluster data analysis and comparison with theoretical HMF
    """
    
    def __init__(self):
        """Initialize LSST analysis parameters"""
        self.survey_area_deg2 = 18000 
        self.setup_binning_scheme()
        
    def setup_binning_scheme(self):
        """
        Setup LSST Y1 binning scheme based on the DESC-SRD documentation -- ****CHECK ONCE MORE****
        """
        # Y1 redshift bins (3 bins) - I used approximate centers(is that ok?)
        self.z_centers = np.array([0.3, 0.5, 0.7])
        self.z_bin_edges = np.array([0.2, 0.4, 0.6, 0.8])
        
        # Richness bins (5 bins) as specified in readme
        self.richness_bin_edges = np.array([20, 30, 45, 70, 120, 220])
        self.richness_centers = np.array([25, 37.5, 57.5, 95, 170])
        
        # Mass-richness relation parameters (from DESC-SRD Eq. 7)
        # M = M_pivot * (lambda/lambda_pivot)^A * ((1+z)/(1+z_pivot))^B
        
        # self.M_pivot = 3e14  # Solar masses/h
        # self.lambda_pivot = 30
        # self.z_pivot = 0.6
        # self.A_MOR = 1.0  # Mass-richness slope -- Check Murata et al
        # self.B_MOR = 0.0  # Redshift evolution -- Check Murata et al

        self.M_pivot = 3e14
        self.lambda_pivot = 25  # Changed from 30
        self.z_pivot = 0.6 
        self.A_MOR = 0.85  # Changed from 1.0 (average of 0.83-0.86)
        self.B_MOR = 0.0  # This is simplified - see note below

        '''NOTE: The DESC SRD and Murata et al. use a forward modeling approach where they model P(ln N | M, z) rather than M(N, z). My inverse relation is a simplified version. For LSST analysis, might want to consider implementing their more flexible parameterization with redshift evolution terms if accuracy is critical. Will be changed after first order checks are done'''
        
        self.setup_mass_bins()
        
    def setup_mass_bins(self):
        """Calculate mass bin centers from richness using mass-observable relation"""
        self.mass_centers = np.zeros((len(self.z_centers), len(self.richness_centers)))
        
        for i, z in enumerate(self.z_centers):
            for j, richness in enumerate(self.richness_centers):
                self.mass_centers[i, j] = self.mass_from_richness(richness, z)
    
    def mass_from_richness(self, richness, z):
        """
        Convert richness to mass using mass-observable relation
        Based on DESC-SRD Equation 7 - **** CHECK ****
        """
        mass = self.M_pivot * (richness / self.lambda_pivot)**self.A_MOR * \
               ((1 + z) / (1 + self.z_pivot))**self.B_MOR
        return mass
    
    def parse_lsst_data(self, data_string):
        """
        Changing LSST cluster data from string format to arrays
        Returns 2 arrays -- indices and values arrays
        """
        lines = data_string.strip().split('\n')
        indices = []
        values = []
        
        for line in lines:
            parts = line.split()
            indices.append(int(parts[0]))
            values.append(float(parts[1]))
        
        return np.array(indices), np.array(values)
    
    def extract_cluster_counts(self, values):
        """
        Extract cluster number counts from the file
        First 15 data points are cluster number counts (3 redshifts and 5richness)
        """
        cluster_counts = values[:15].reshape(3, 5)
        return cluster_counts

    def calculate_uncertainties(self, cluster_counts):
        """
        Calculate Poisson uncertainties for cluster counts
        """
        # Poisson errors
        poisson_errors = np.sqrt(cluster_counts)
        
        # Convert to fractional errors
        fractional_errors = np.divide(poisson_errors, cluster_counts, 
                                    out=np.ones_like(cluster_counts), 
                                    where=cluster_counts!=0)
        
        return poisson_errors, fractional_errors
           
    def counts_to_number_density(self, cluster_counts):
        """
        Convert cluster counts to comoving number density
        """
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)  # Fiducial cosmology
        area_sr = self.survey_area_deg2 * (np.pi/180)**2  # Converting to steradians
        
        number_density = np.zeros_like(cluster_counts, dtype=float)
        
        for i, z in enumerate(self.z_centers):
            d_c = cosmo.comoving_distance(z).value  # Mpc/h
            delta_z = 0.2  # Approximate bin width
            
            # Volume element: dV = area * d_c^2 * c/H(z) * dz / (1+z)^2
            H_z = cosmo.H(z).value  # km/s/Mpc
            c_light = 299792.458  # km/s
            
            dV_dz = area_sr * d_c**2 * c_light / H_z / (1 + z)**2
            volume = dV_dz * delta_z
            
            number_density[i, :] = cluster_counts[i, :] / volume
        
        return number_density
    
    def convert_to_hmf(self, number_density, poisson_errors):
        """
        Convert number density to dn/dlnM for comparison with our analytical HMF
        """
        dn_dlnM_data = {}
        
        for i, z in enumerate(self.z_centers):
            masses = self.mass_centers[i, :]
            n_density = number_density[i, :]
            poiss_errs = poisson_errors[i, :]
            
            # Calculate dlnM for each bin
            log_masses = np.log(masses)
            dlnM = np.diff(log_masses)
            dlnM = np.append(dlnM, dlnM[-1])  # Extending for last bin
            
            # Convert to dn/dlnM
            dn_dlnM = n_density / dlnM
            
            # Calculate volume for error propagation
            cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
            area_sr = self.survey_area_deg2 * (np.pi/180)**2
            z_val = self.z_centers[i]
            d_c = cosmo.comoving_distance(z_val).value
            delta_z = 0.2
            H_z = cosmo.H(z_val).value
            c_light = 299792.458
            dV_dz = area_sr * d_c**2 * c_light / H_z / (1 + z_val)**2
            volume = dV_dz * delta_z
            
            # Propagate Poisson errors: sigma_(dn/dlnM) = sigma_counts / (volume * dlnM)
            dn_dlnM_errors = poiss_errs / (volume * dlnM)
            
            # Calculate fractional errors for reference
            frac_errs = np.divide(dn_dlnM_errors, dn_dlnM, 
                                 out=np.ones_like(dn_dlnM), 
                                 where=dn_dlnM!=0)
            
            dn_dlnM_data[z] = {
                'mass': masses,
                'dn_dlnM': dn_dlnM,
                'dn_dlnM_errors': dn_dlnM_errors,
                'number_density': n_density,
                'poisson_errors': poiss_errs,
                'fractional_errors': frac_errs,
                'dlnM': dlnM
            }
        
        return dn_dlnM_data
    

def plot_comparison_with_theory(lsst_analysis, lsst_hmf_data, theoretical_params=None):
    """
    Create comparison plots between LSST data and theoretical HMF
    
    Args:
        lsst_analysis: 
        lsst_hmf_data: HMF data that was converted above with proper error propagation
        theoretical_params: Parameters for theoretical HMF calculation
    """
    if theoretical_params is None:
        # Default cosmological parameters (Planck-like)
        theoretical_params = [0.67, 0.022, 0.12, 0.96, 2.1e-9, 0.8]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.suptitle('Poisson Errors')
    
    colors = ['navy', 'darkred', 'darkgreen']
    mass_arr = np.logspace(13.5, 15.5, 50)
    
    for i, z in enumerate(lsst_analysis.z_centers):
        row, col = divmod(i, 2)
        ax = axes[row, col]
        
        # Plot theoretical HMF
        dn_dlnM_theory, _ = compute_hmf_analytical(theoretical_params, mass_arr, z)
        ax.loglog(mass_arr, dn_dlnM_theory, alpha=0.8,
                 label=f'Theoretical HMF (z={z:.1f})')
        
        # Plot LSST data
        data_z = lsst_hmf_data[z]
        ax.scatter(data_z['mass'], data_z['dn_dlnM'], 
                  alpha=0.9, edgecolor='black', linewidth=1,
                  label=f'LSST Y1 Data (z={z:.1f})', zorder=5)
        
        ax.errorbar(data_z['mass'], data_z['dn_dlnM'], 
                   yerr=data_z['dn_dlnM_errors'], fmt='none',
                   alpha=0.7, capsize=5, linewidth=2)
        
        ax.set_xlabel('Mass [Msun/h]', fontsize=12)
        ax.set_ylabel('dn/dlnM [h^3/Mpc^3]', fontsize=12)
        ax.set_title(f'Redshift z = {z:.1f}', fontsize=14)
        ax.legend(fontsize=10)
        
        # Set axis limits
        ax.set_xlim(1e13, 1e16)
        ax.set_ylim(1e-10, 1e-3)
        
        # Print actual error information
        # print(f"z={z:.1f}: Poisson errors range from {np.min(data_z['poisson_errors']):.1f} to {np.max(data_z['poisson_errors']):.1f} counts")
        # print(f"z={z:.1f}: Fractional errors range from {np.min(data_z['fractional_errors']):.3f} to {np.max(data_z['fractional_errors']):.3f}")
    
    # Hide unused subplot
    if len(lsst_analysis.z_centers) == 3:
        axes[1, 1].set_visible(False)
    
    plt.tight_layout()
    return fig
