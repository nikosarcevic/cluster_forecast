"""Predict cluster counts in richness and redshift bins, given cosmology, HMF, and MOR."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
import pyccl as ccl

from src.cluster_forecast.mor import ForwardMOR, InverseMOR
from src.cluster_forecast.hmf_theory import MassFunctionBase
from src.cluster_forecast.helpers import get_solid_angle_sr

__all__ = [
    "compute_forward_counts",
    "compute_inverse_counts",
]


def compute_inverse_counts(
    cosmo: ccl.Cosmology,
    mass_function_model: MassFunctionBase,
    inverse_mor: InverseMOR,
    richness_edges: ArrayLike,
    redshift_edges: ArrayLike,
    area_deg2: float,
    *,
    n_z: int = 11,
    n_m_per_bin: int = 256,
    edge_mode: str = "center",  # "center" uses z-bin center; "avg-edges" volume-weights edges
) -> np.ndarray:
    """Compute binned cluster counts using a deterministic mass–observable relation.
    This integrates the halo mass function over mass and redshift, using a
    deterministic mapping from richness (λ) to mass (M) to define the mass
    bin edges for each richness bin.

    Conventions:
      * Masses are in Msun/h.
      * HMF returns dn/dlnM in Mpc^-3.
      * Counts are integrated over the survey solid angle (area_deg2).

    Args:
        cosmo: PyCCL cosmology.
        mass_function_model: Object exposing ``mass_function(mass, z) -> dn/dlnM`` (Mpc^-3).
        inverse_mor: Inverse MOR providing ``mass_edges(richness_edges, z)
            -> mass_edges`` (Msun/h).
        richness_edges: λ (richness) bin edges, length = n_richness+
        redshift_edges: Redshift bin edges, length = n_redshift+1.
        area_deg2: Survey area in square degrees.
        n_z: Number of fine-z slices per redshift bin (integration control).
        n_m_per_bin: Number of mass grid points (in ln M) for integration per
            richness bin (integration control).
        edge_mode: How to choose mass bin edges for each richness bin:
            * "center": use inverse_mor at the center of the redshift bin.
            * "avg-edges": compute mass edges at each fine-z slice and
                average them weighted by comoving volume.

    Returns:
        np.ndarray: Array of shape (n_redshift, n_richness) with predicted
            counts per (z, λ) bin.
    Raises:
        ValueError: If inputs are invalid.
    """
    richness_edges = np.asarray(richness_edges, dtype=float)
    redshift_edges = np.asarray(redshift_edges, dtype=float)

    n_redshift = len(redshift_edges) - 1
    n_richness = len(richness_edges) - 1

    counts = np.zeros((n_redshift, n_richness), dtype=float)
    solid_angle_sr = get_solid_angle_sr(area_deg2)

    for i in range(n_redshift):
        z_lo, z_hi = float(redshift_edges[i]), float(redshift_edges[i + 1])
        z_mid, seg_vol_per_sr, w_z = _z_volume_slices(cosmo, z_lo, z_hi, n_z)
        vol_shell_per_sr = float(np.sum(seg_vol_per_sr))

        # choose lnM edges for all richness bins across this z-bin
        if edge_mode == "center":
            zc = 0.5 * (z_lo + z_hi)
            ln_mass_edges = np.log(inverse_mor.mass_edges(richness_edges, zc))
        elif edge_mode == "avg-edges":
            ln_mass_edges_z = np.stack([np.log(inverse_mor.mass_edges(richness_edges, zc)) for zc in z_mid], axis=0)
            ln_mass_edges = (w_z[:, None] * ln_mass_edges_z).sum(axis=0)
        else:
            raise ValueError("edge_mode must be 'center' or 'avg-edges'")

        # precompute HMF on a per-bin lnM grid (changes per bin because edges differ)
        for j in range(n_richness):
            l0, l1 = float(ln_mass_edges[j]), float(ln_mass_edges[j + 1])
            ls = np.linspace(l0, l1, int(n_m_per_bin))
            mass = np.exp(ls)  # Msun/h

            # evaluate HMF on mass grid at each fine-z slice
            dn_dln_mass_z = np.asarray([mass_function_model.mass_function(mass, zc) for zc in z_mid], float)
            # volume-weight dn/dlnM across z at each M, then integrate over lnM and divide by width
            dn_avg = (w_z[:, None] * dn_dln_mass_z).sum(axis=0)
            n_bin = np.trapezoid(dn_avg, ls)  # integrates per-sr number density across lnM
            counts[i, j] = n_bin * vol_shell_per_sr * solid_angle_sr
    return counts


def compute_forward_counts(
    cosmo: ccl.Cosmology,
    mass_function_model: MassFunctionBase,
    forward_mor: ForwardMOR,
    richness_edges: ArrayLike,
    redshift_edges: ArrayLike,
    area_deg2: float,
    *,
    n_z: int,
    n_mass: int,
    mass_min: float,
    mass_max: float,
) -> np.ndarray:
    """Forward-model binned cluster counts using λ|M with lognormal scatter.

    This integrates the halo mass function over mass and redshift, weighting by
    the probability that a halo of mass M at redshift z falls inside each
    richness (λ) bin according to the forward mass–observable relation.

    Conventions:
      * Masses are in Msun/h.
      * HMF returns dn/dlnM in Mpc^-3.
      * Counts are integrated over the survey solid angle (area_deg2).

    Args:
      cosmo: PyCCL cosmology.
      mass_function_model: Object exposing ``mass_function(mass, z) -> dn/dlnM`` (Mpc^-3).
      forward_mor: Forward MOR providing ``probability_in_richness_bin``.
      richness_edges: λ (richness) bin edges, length = n_richness+1.
      redshift_edges: Redshift bin edges, length = n_redshift+1.
      area_deg2: Survey area in square degrees.
      n_z: Number of fine-z slices per redshift bin (integration control).
      n_mass: Number of mass grid points (in ln M) for integration.
      mass_min: Lower mass bound (Msun/h) for the mass grid.
      mass_max: Upper mass bound (Msun/h) for the mass grid.

    Returns:
      np.ndarray: Array of shape (n_redshift, n_richness) with predicted counts per (z, λ) bin.
    """
    # Validate inputs.
    if n_z < 1 or n_mass < 2:
        raise ValueError("n_z >= 1 and n_mass >= 2 required.")
    if mass_min <= 0 or mass_max <= mass_min:
        raise ValueError(f"Require 0 < mass_min < mass_max; got {mass_min}, {mass_max}")

    # Convert edges to float arrays.
    richness_edges = np.asarray(richness_edges, dtype=float)
    redshift_edges = np.asarray(redshift_edges, dtype=float)

    # Prepare output array.
    nz = len(redshift_edges) - 1
    nr = len(richness_edges) - 1
    counts = np.empty((nz, nr), dtype=float)

    # Mass grid in ln M (fixed across redshift for speed).
    ln_mass = np.linspace(np.log(mass_min), np.log(mass_max), int(n_mass))
    mass = np.exp(ln_mass)

    # Survey solid angle in steradians.
    solid_angle_sr = get_solid_angle_sr(area_deg2)

    # Loop over redshift bins to compute counts.
    for i in range(nz):
        z_lo, z_hi = float(redshift_edges[i]), float(redshift_edges[i + 1])
        z_mid, seg_vol_per_sr, w_z = _z_volume_slices(cosmo, z_lo, z_hi, n_z)
        # Evaluate HMF on mass grid at each fine-z slice.
        dn_dln_mass_z = np.asarray([mass_function_model.mass_function(mass, zc) for zc in z_mid], float)

        for j in range(nr):
            ln_l0, ln_l1 = np.log(richness_edges[j]), np.log(richness_edges[j + 1])
            probability = np.asarray(
                [forward_mor.probability_in_richness_bin(ln_mass, zc, ln_l0, ln_l1) for zc in z_mid],
                float,
            )  # (n_z, n_mass)

            n_slice = np.trapezoid(dn_dln_mass_z * probability, ln_mass, axis=1)  # (n_z,)
            n_avg_per_sr = float(np.sum(w_z * n_slice))
            counts[i, j] = n_avg_per_sr * float(np.sum(seg_vol_per_sr)) * solid_angle_sr
    return counts


def _z_volume_slices(cosmo: ccl.Cosmology, z_lo: float, z_hi: float, n_z: int):
    """Compute fine redshift slices and their comoving volumes per steradian.

    Args:
        cosmo: PyCCL cosmology.
        z_lo: Lower redshift bound.
        z_hi: Upper redshift bound.
        n_z: Number of slices.

    Returns:
        z_mid: Midpoints of the redshift slices.
        seg_vol_per_sr: Comoving volume of each slice per steradian (Mpc^3).
        weights: Weights (seg_vol_per_sr / total_volume).
    """
    redshift_edges = np.linspace(z_lo, z_hi, int(n_z) + 1, dtype=float)
    z_mid = 0.5 * (redshift_edges[:-1] + redshift_edges[1:])
    seg_vol_per_sr = np.asarray(
        [ccl.comoving_volume(cosmo, 1.0 / (1.0 + zb), solid_angle=1.0)
            - ccl.comoving_volume(cosmo, 1.0 / (1.0 + za), solid_angle=1.0)
            for za, zb in zip(redshift_edges[:-1], redshift_edges[1:])
        ],
        dtype=float,
    )
    weights = seg_vol_per_sr / seg_vol_per_sr.sum()
    return z_mid, seg_vol_per_sr, weights
