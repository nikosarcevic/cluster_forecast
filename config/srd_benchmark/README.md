##### Aug 28, 2018, Tim Eifler, please send comments, critique to timeifler@gmail.com #########

This readme details the file format of covariances, Fisher matrices, data vectors, and redshift distributions that entered the WL+LSS+CL analysis and it explains how to run the corresponding Fisher forecasts. It also explains how to create the DESC-SRD individual and multi-probe plots for Y1 and Y10, including the SN and SL results (Fig. G2 in the DESC-SRD).

Note that users who wish to directly compute data vectors, covariances, Fisher matrices, and requirements can do so by going to the relevant CosmoLike repository, https://github.com/CosmoLike/DESC_SRD, and following the instructions for doing so in the README there.  The process outlined there has been significantly streamlined for ease of use. The content in this tarball/repository is the output of that process, but does not enable users to directly reproduce the process.


--------------------------------------------------------
A) Create Figure G2 from existing chains
- Download the tarball containing the DESC SRD data products from "https://zenodo.org/communities/lsst-desc" and store the chains that currently live in "chains/indiv_joint_probes" in subfolder "like/".z
- Ensure that python 2.7, matplotlib, and chainconsumer are installed 
- Type "python plot_DESC-SRD_FIG_G2.py"
--------------------------------------------------------



--------------------------------------------------------
B) Data vector filename conventions (all data vectors are stored in directory "datav/")
The data vectors are created for the following probe combinations (indicated in their filename): 
a) shear-shear (shear-shear) 
b) position-position, aka galaxy clustering (pos-pos) 
c) cluster Number Counts + Cluster Weak Lensing (clusterN_clusterWL)
d) 3x2pt (3x2pt)
e) 3x2pt+cluster Number Counts+Cluster Weak Lensing (3x2pt_clusterN_clusterWL)

For each of these probe combination cases we create several Y1 and Y10 data vectors: 
a) fiducial data vector (_fid) 
b) contaminated by shear calibration (_shear_calibration)
c) contaminated by photo-z mean bias (_mean_photo-z)
d) contaminated by photo-z sigma bias (_sigma_photo-z)

Note that for "pos-pos" there is no data vector with contaminating shear calibration uncertainties for obvious reasons.
--------------------------------------------------------



--------------------------------------------------------
C) Data vector file format (data vectors are all stored in "datav/")
The data vector is ordered as follows (for given subsets, please exclude the probes not considered from the ordering):
- shear-shear: 
	* the tomographic bin combinations are ordered as z11, z12, z13,..., z15, z22, z23,...z55)
	* each tomographic bin consists of 20 l-values ranging from 20-15000, with l-values >3000 being set to zero
	* the exact theta values can be found in "ell-values"
- galaxy-galaxy lensing: 
	* the lens-source bin combinations are only included in the data vector if the lens bin is at lower redshift than the sources (we allow for the lens bin to overlap at 10%). Given the different redshift distributions of Y1 and Y10 and the fact that Y1 and Y10 have different numbers of lens bins (10 for Y10 and 5 for Y1), the accepted lens-source pairs differ. The exact lens-source combinations that are accepted can be found in "gglensing_zbin_Y10" and "gglensing_zbin_Y1"
	* ell values correspond to the shear-shear case, but with the LSS cut excluding ell-bins for which k>0.3 h/Mpc
- galaxy clustering (pos-pos): 
	* only auto-bins are considered; the tomographic bin combinations are ordered as z11, z22, z33, z44, z55,...)
	* similar to shear-shear we consider 20 ell-bins but cut out bins where k>0.3 h/Mpc
- Cluster Number Counts:
	* binned in 3 (for Y1) and 4 (for Y10) tomographic bins
	* each tomographic bin is again binned in 5 richness bins (Richness bin 0: 2.000000e+01 - 3.000000e+01, Richness bin 1: 3.000000e+01 - 4.500000e+01, Richness bin 2: 4.500000e+01 - 7.000000e+01, Richness bin 3: 7.000000e+01 - 1.200000e+02, Richness bin 4: 1.200000e+02 - 2.200000e+02
- Cluster Weak Lensing 
	* for Y1, 6 valid combinations of clusters (foreground) and sources (background) exist; for Y10 we have 11 such tomographic power spectra.
	* each of these power spectra is again divided into 5 richness bins
	* each richness bin has 5 ell-bins, corresponding to the last 5 ell-bins in "ell-values" 

- Number of data points:
  * The Y10 full data vector (e.g., 3x2pt_clusterN_clusterWL_Y10_fid) has 1295 data points altogether, which comprises:
    # 300 data points for cosmic shear (20 ell-bins, 15 tomographic power spectra)
    # 500 data points for galaxy-galaxy lensing (20 ell bins, 25 accepted lens-source power spectra combinations)
    # 200 data points for galaxy clustering (20 ell-bins, 10 lens bins)
    # 20 data points for cluster number counts (4 redshift and 5 richness bins)
    # 275 data points for cluster weak lensing (11 allowed combinations of cluster lens bin and galaxy source bin, 5 ell-bins, 5 richness bins)

  * The Y1 full data vector (e.g., 3x2pt_clusterN_clusterWL_Y1_fid) has 705 data points altogether, which comprises:
    # 300 data points for cosmic shear (20 ell-bins, 15 tomographic power spectra)
    # 140 data points for galaxy-galaxy lensing (20 ell bins, 7 accepted lens-source power spectra combinations)
    # 100 data points for galaxy clustering (20 ell-bins, 5 lens bins)
    # 15 data points for cluster number counts (3 redshift and 5 richness bins)
    # 150 data points for cluster weak lensing (6 allowed combinations of cluster lens bin and galaxy source bin, 5 ell-bins, 5 richness bins)

Important: Within the stored data vectors (joint and individual probes) scales that are EXCLUDED from the analysis are INCLUDED in the data vector, but their values are set to zero. For example, given that cosmic shear imposes ell_max=3000, multipoles of the cosmic shear data vector index range [15-19] are zero.

--------------------------------------------------------




--------------------------------------------------------
D) Covariance file format
In the folder "cov/" the user can find the multi-probe original covariances for Y1 and Y10 "*_3x2pt_clusterN_clusterWL_cov" and the inverse covariances for the various probe combinations. The same name conventions of the data vectors hold.

Format of the covariance matrix "*_cov":
- The columns contain the following information: cov index 1 | cov index 2 | ell_1 | ell_2 | z1 | z2 | z3 | z4 | cov_Gauss | cov_Non-Gauss | 
- Note that cov_Non-Gauss does not also include cov_Gauss. In order to obtain the full cov, both must be added.
- The multi-probe covariance that can be built from this file has the same dimension as the data vectors described above, i.e. 1295 for Y10 and 705 for Y1. Please see the included "inv_cov.py" file (lines 46 to 48) for how to build the covariance correctly. Note that for computing (and storage) efficiency purposes we use several symmetry properties of the covariance and only compute the upper triangular matrix (for Y10: the file stores 886525 elements of the total 1677025 cov elements).
- The covariance "knows" about the scale cuts imposed in the analysis. For elements that are being cut out later, only a main diagonal element is computed, no off-diagonal elements. This enables a proper inversion process and later exclusion of the corresponding elements.

Inverse Covariances: 
- To create inverse covariance matrices for all the probe combinations considered from the "*cov" file, please comment/uncomment the corresponding lines (Y1 vs Y10) in the inv_cov.py file and execute "./inv_cov.py" from the command line. This will also generate a plot of the Y1 and Y10 covariance matrix, respectively.
- The resulting inverse covariance matrix files "*_inv" have the dimension squared of the corresponding data vectors.   
- The columns correspond to:  index 1 | index 2 | inverse (Gaussian+NonGaussian cov)
- Elements that should be excluded because of scale cuts are being set to zero in the inverse covariance matrix (happens in inv_cov.py after the inversion process). That is, if the data vector element "i" is zero, the corresponding row and column in the inverse also have zeros.

--------------------------------------------------------


--------------------------------------------------------
E) Redshift distributions:
- The folder "zdistris/" contains 4 redshift distributions, corresponding to the Y1 and Y10, lens and source distributions
- The columns are z_min |z_mid | z_max | dn/dz (normalized such that area under the curve is 1)
--------------------------------------------------------


--------------------------------------------------------
F) Fisher matrices and Figure of Merit calculations
- In the folder "Fishers/" we store two text files containing the Fisher matrices of WL+LSS+CL for the cosmological parameters only "Fisher_cosmo.txt" and when including the self-calibrated systematics "Fisher_all.txt". All "Fisher*.txt" files contain the Fisher matrices in a convenient np.array format, which can be immediately copied and pasted into e.g. an ipython notebook.
- The folder also contains individual files with all Fisher matrices (including the Stage 3 with and without w0wa information) as .npy files, for reading directly in when running python scripts.
- Finally, the folder contains a text file with the DETF Figures of Merit for each probe, FoM.txt. The FoM.txt file contains the DETF FoMs for the different science cases (denoted as "mode" in the file) followed by the value of "r" that is derived from the parameter biases that occur due to a systematic (please see Eq. 2 in Sect. C2 of the DESC SRD for the exact definition).  We compute FoMs and "r" values for 2 cases: including Stage 3 priors (without their constraining power on w0 and wa) and excluding those priors.
- For Fisher_cosmo, the order of the parameters is Omega_m, sigma_8, n_s, w0, wa, Omega_b, h0
- For Fisher_all, the first 7 columns also correspond to these 7 parameters, however the self-calibrated systematics parameters that are appended to the cosmological parameters differ from probe to probe:
	* shear-shear: 7 cosmological parameters and 4 intrinsic alignment (IA) parameters, specifically IA amplitude "A", IA luminosity scaling "beta", IA redshift scaling "eta", IA redshift scaling at high-z "eta_high-z"
	* position-position: 7 cosmological parameters and 5 galaxy bias parameters for Y1 and 10 galaxy bias parameters for Y10
	* cluster Number Counts + Cluster Weak Lensing: 7 cosmological parameters and 3 parameters to describe mean Mass-Observable-Relation (MOR) parameters (A, B, C as explained in Eq. 7 in the DESC-SRD) and 3 parameters to describe the mass-dependent scatter (sigma_0, q_m, qz as explained in Eq. 8 in the DESC-SRD) 
	* 3x2pt: 7 cosmological parameters followed by the 5 (Y1) or 10 (Y10) bias parameters and the 4 IA parameters
	* 3x2pt+cluster Number Counts+Cluster Weak Lensing: 7 cosmological parameters followed by the 5 (Y1) or 10 (Y10) bias parameters, the 4 IA parameters and the 6 MOR parameters as described above
--------------------------------------------------------










