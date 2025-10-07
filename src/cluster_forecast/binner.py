"""Helper functions for binning and related calculations."""

from __future__ import annotations


import numpy as np
import pyccl as ccl

from src.cluster_forecast.helpers import get_solid_angle_sr, get_scale_factor_from_redshift

__all__ = [
    "comoving_volume_shell",
    "binned_mass_function",

]


def comoving_volume_shell(
    cosmo: ccl.Cosmology, zmin: float, zmax: float, area_deg2: float
) -> float:
    """Compute the comoving volume of a redshift shell over a given sky area.

    Args:
        cosmo (ccl.Cosmology): PyCCL cosmology object.
        zmin (float): Lower redshift bound.
        zmax (float): Upper redshift bound (must be > `zmin`).
        area_deg2 (float): Sky area in square degrees.

    Returns:
        float: Comoving volume of the shell in Mpc³.

    Raises:
        ValueError: If redshift bounds are invalid or area is non-positive.

    Note: for small zmax, this is not the same as the volume element
    dV = dΩ * d_c^2 * c/H(z) * dz / (1+z)^2, because the latter
    assumes a thin shell at fixed z rather than integrating over a finite z range.
    All masses are Msun/h; area_deg2 is in square degrees.
    Returns volume in (Mpc/h)^3.

    Conversions:
        1 steradian = (180/π)^2 deg^2
        1 Mpc = 1 Mpc/h * h
        1 (Mpc/h)^3 = h^3 Mpc^3
        1 Mpc^3 = (1/h^3) (Mpc/h)^3
        Thus, if volume is computed in Mpc^3, convert to (Mpc/h)^3 by multiplying by h^3.
        1 (Mpc/h)^3 = h^3 Mpc^3
        1 Mpc^3 = (1/h^3) (Mpc/h)^3

    Note: pyccl.comoving_volume returns volume in Mpc^3, so we need to convert to (Mpc/h)^3.
    """
    if not (np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin):
        raise ValueError(f"Require zmax > zmin and finite; got zmin={zmin}, zmax={zmax}")
    if area_deg2 <= 0:
        raise ValueError(f"area_deg2 must be positive; got {area_deg2}")

    scale_factor1 = get_scale_factor_from_redshift(zmin)
    scale_factor2 = get_scale_factor_from_redshift(zmax)

    solid_angle = get_solid_angle_sr(area_deg2)
    vol2 = ccl.comoving_volume(cosmo, scale_factor2, solid_angle=solid_angle)  # Mpc^3
    vol1 = ccl.comoving_volume(cosmo, scale_factor1, solid_angle=solid_angle)  # Mpc^3
    volume = vol2 - vol1  # Mpc^3

    return volume


def binned_mass_function(
    number_density: np.ndarray, mass_edges_log: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute binned mass function (dn/dlnM) from per-bin number density.

    Args:
        number_density (np.ndarray): (nz, nr) number density per bin
            (e.g., counts / volume), in Mpc^-3.
        mass_edges_log (np.ndarray): (nz, nr+1) natural-log mass edges per z-bin.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            mass_function (np.ndarray): (nz, nr) bin-averaged dn/dlnM.
            delta_log_mass (np.ndarray): (nz, nr) ln-width of each bin.

    Raises:
        ValueError: If input shapes are inconsistent or any ln-width ≤ 0.
    """
    number_density = np.asarray(number_density, dtype=float)
    mass_edges_log = np.asarray(mass_edges_log, dtype=float)

    if number_density.ndim != 2 or mass_edges_log.ndim != 2:
        raise ValueError("number_density and mass_edges_log must both be 2D arrays.")

    nz, nr = number_density.shape
    if mass_edges_log.shape != (nz, nr + 1):
        raise ValueError(
            f"shape mismatch: number_density {number_density.shape} vs "
            f"mass_edges_log {mass_edges_log.shape}; expected (nz, nr+1)."
        )

    delta_log_mass = mass_edges_log[:, 1:] - mass_edges_log[:, :-1]
    if np.any(~np.isfinite(delta_log_mass)) or np.any(delta_log_mass <= 0.0):
        raise ValueError("all ln(bin width) must be finite and positive.")

    mass_function = number_density / delta_log_mass
    return mass_function, delta_log_mass
