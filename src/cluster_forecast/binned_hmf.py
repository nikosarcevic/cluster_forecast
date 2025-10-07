"""Bin-averaged halo mass function with volume weighting over redshift."""

from __future__ import annotations

import numpy as np
import pyccl as ccl

from src.cluster_forecast.hmf_theory import MassFunctionBase


__all__ = ["volume_weighted_binned_hmf"]


def volume_weighted_binned_hmf(
    cosmo: ccl.Cosmology,
    mass_fn: MassFunctionBase,
    ln_m_edges_row: np.ndarray,
    z_min: float,
    z_max: float,
    *,
    n_z: int = 11,
    n_m_per_bin: int = 256,
) -> np.ndarray:
    """
    Calculate the volume-weighted, bin-averaged halo mass function dn/dlnM
    over a redshift interval [z_min, z_max].

    Args:
        cosmo : ccl.Cosmology
        mass_fn : object
            Must implement mass_function(mass: array, z: float) -> dn/dlnM [Mpc^-3].
        ln_m_edges_row : array-like, shape (Nr+1,)
            Natural-log mass bin edges (Msun/h).
        z_min : float
            Redshift bin edges (z_max > z_min).
        z_max : float
            Redshift bin edges (z_max > z_min).
        n_z : int, optional
            Number of fine-z slices for volume weighting (default 11).
        n_m_per_bin : int, optional
            Number of points for lnM integration within each bin (default 256).

    Returns:
        np.ndarray, shape (Nr,)
            Bin-averaged dn/dlnM [Mpc^-3].
    """
    ln_m_edges_row = np.asarray(ln_m_edges_row, dtype=float)
    if ln_m_edges_row.ndim != 1 or ln_m_edges_row.size < 2:
        raise ValueError("ln_m_edges_row must be 1D with length >= 2.")
    if not np.all(np.isfinite(ln_m_edges_row)) or np.any(np.diff(ln_m_edges_row) <= 0):
        raise ValueError("ln_m_edges_row must be finite and strictly increasing.")
    if not (np.isfinite(z_min) and np.isfinite(z_max) and z_max > z_min):
        raise ValueError("Require z_max > z_min and both finite.")
    if n_z < 1 or n_m_per_bin < 2:
        raise ValueError("n_z >= 1 and n_m_per_bin >= 2 required.")

    # fine z-slices and per-steradian shell volumes (weights)
    z_fine = np.linspace(z_min, z_max, int(n_z) + 1, dtype=float)
    z_mid = 0.5 * (z_fine[:-1] + z_fine[1:])

    # per-steradian comoving volumes of each segment
    vol_segment = np.array([
        ccl.comoving_volume(cosmo, 1.0 / (1.0 + zb), solid_angle=1.0)
        - ccl.comoving_volume(cosmo, 1.0 / (1.0 + za), solid_angle=1.0)
        for za, zb in zip(z_fine[:-1], z_fine[1:])
    ], dtype=float)
    w_z = vol_segment / vol_segment.sum()

    out = []
    for l0, l1 in zip(ln_m_edges_row[:-1], ln_m_edges_row[1:]):
        ls   = np.linspace(l0, l1, int(n_m_per_bin), dtype=float)
        mass = np.exp(ls)
        dn_z = np.array([mass_fn.mass_function(mass, zc) for zc in z_mid], dtype=float)
        dn_avg = (w_z[:, None] * dn_z).sum(axis=0)
        out.append(np.trapezoid(dn_avg, ls) / (l1 - l0))
    return np.asarray(out, dtype=float)
