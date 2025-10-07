"""Helper functions for cluster_forecast."""

from __future__ import annotations

from typing import Iterable

import numpy as np

__all__ = [
    "get_solid_angle_sr",
    "mass_edges_to_msunh",
    "get_scale_factor_from_redshift",
    "get_redshift_from_scale_factor",
]


def get_solid_angle_sr(area_deg2: float) -> float:
    """Convert area in square degrees to solid angle in steradians.

    Args:
        area_deg2: Area in square degrees.
            Must be positive.
    Returns:
        Solid angle in steradians.
    Raises:
        ValueError: If area_deg2 is not positive.
    """
    if area_deg2 <= 0:
        raise ValueError(f"area_deg2 must be positive; got {area_deg2}")
    deg2_to_sr = (np.pi / 180.0) ** 2
    return float(area_deg2) * deg2_to_sr



def mass_edges_to_msunh(
    mor,
    richness_edges: Iterable[float],
    z: float,
    h: float,
    mor_units: str = "msunh",
) -> np.ndarray:
    """Return MOR mass edges converted to Msun/h (regardless of MOR’s native units).

    Args:
        mor: Object implementing ``mass_edges(richness_edges, z) -> array_like`` in its
            native mass units.
        richness_edges (Iterable[float]): Richness (λ) bin edges.
        z (float): Redshift at which to evaluate the inverse MOR.
        h (float): Little-h (H0 / 100 km/s/Mpc); used for Msun↔Msun/h conversion.
        mor_units (str, optional): Unit of the MOR output. Accepts:
            - "msunh", "msun/h", "h^-1 msun"  (already Msun/h)
            - "msun"                            (solar masses; will be scaled by h)
            Defaults to "msunh".

    Returns:
        np.ndarray: Mass edges in Msun/h.

    Raises:
        ValueError: If ``h`` is invalid or ``mor_units`` is unrecognized.
    """
    if not np.isfinite(h) or h <= 0:
        raise ValueError(f"h must be a positive finite number, got {h!r}")

    # Compute edges in MOR’s native units
    edges_native = np.asarray(mor.mass_edges(richness_edges, z), dtype=float)

    units = mor_units.strip().lower()
    if units in {"msunh", "msun/h", "h^-1 msun"}:
        return edges_native  # already Msun/h
    if units == "msun":
        return edges_native * float(h)  # convert Msun -> Msun/h
    raise ValueError(f"unrecognized mor_units {mor_units!r}; expected 'msunh' or 'msun'.")

def get_scale_factor_from_redshift(z: float) -> float:
    """Convert redshift to scale factor.

    Args:
        z: Redshift (must be >= -1).

    Returns:
        Scale factor a = 1 / (1 + z).

    Raises:
        ValueError: If z < -1.
    """
    if z < -1:
        raise ValueError(f"Redshift must be >= -1; got {z}")
    return 1.0 / (1.0 + z)


def get_redshift_from_scale_factor(a: float) -> float:
    """Convert scale factor to redshift.

    Args:
        a: Scale factor (must be > 0).

    Returns:
        Redshift z = 1/a - 1.

    Raises:
        ValueError: If a <= 0.
    """
    if a <= 0:
        raise ValueError(f"Scale factor must be > 0; got {a}")
    return (1.0 / a) - 1.0
