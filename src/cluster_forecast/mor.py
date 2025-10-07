from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Mapping, Union

import numpy as np
from scipy.special import erf

__all__ = [
    "InverseMOR",
    "ForwardMOR",
    "make_inverse_from_cfg",
    "make_forward_from_cfg",
]


@dataclass(frozen=True)
class InverseMOR:
    """Deterministic richness→mass mapping for binning/visualization.

    The mapping is:
        M = m_pivot * (λ / lambda_pivot)^a * ((1 + z) / (1 + z_pivot))^b

    All masses are in Msun/h if `m_pivot` is in Msun/h.

    Args:
        m_pivot: Pivot mass (Msun/h).
        lambda_pivot: Pivot richness (dimensionless).
        z_pivot: Pivot redshift.
        a: Power-law slope with richness.
        b: Redshift evolution slope.
    """
    m_pivot: float
    lambda_pivot: float
    z_pivot: float
    a: float
    b: float

    def mass_edges(self, richness_edges: np.ndarray, z: float) -> np.ndarray:
        """Map richness bin edges to mass edges at a given redshift.

        Args:
            richness_edges: Array of richness (λ) bin edges.
            z: Redshift.

        Returns:
            np.ndarray: Mass edges (Msun/h), same length as `richness_edges`.
        """
        lam = np.asarray(richness_edges, dtype=float)
        return (
            self.m_pivot
            * (lam / self.lambda_pivot) ** self.a
            * ((1.0 + z) / (1.0 + self.z_pivot)) ** self.b
        )


@dataclass(frozen=True)
class ForwardMOR:
    """Probabilistic richness–mass relation λ|M with lognormal scatter.

    Model:
        ln λ ~ Normal( μ, s² ), with
            μ = a + b * ln(M / m_pivot) + c * ln(1 + z)
            s = sigma_0 + q_m * ln(M / m_pivot) + q_z * ln(1 + z)

    Args:
        a: Intercept for ln λ.
        b: Slope with ln M.
        c: Redshift evolution for ln λ.
        sigma_0: Scatter at the pivots (stddev in ln λ).
        q_m: Mass dependence of scatter.
        q_z: Redshift dependence of scatter.
        m_pivot: Pivot mass (Msun/h).
    """
    a: float
    b: float
    c: float
    sigma_0: float
    q_m: float
    q_z: float
    m_pivot: float  # Msun/h

    def mean_and_scatter(self, ln_mass:  Union[np.ndarray, float], z: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (μ, s) for ln λ at given ln M and redshift.

        Args:
            ln_mass: Natural log of mass M (Msun/h); scalar or array.
            z: Redshift.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (mean, stddev) of ln λ.
        """
        ln_mass = np.asarray(ln_mass, dtype=float)
        center = ln_mass - np.log(self.m_pivot)
        mu = self.a + self.b * center + self.c * np.log1p(z)
        s = self.sigma_0 + self.q_m * center + self.q_z * np.log1p(z)
        return mu, np.maximum(s, 1e-8)

    def probability_in_richness_bin(
        self,
        ln_mass: np.ndarray,
        z: float,
        ln_lambda_lo: float,
        ln_lambda_hi: float,
    ) -> np.ndarray:
        """Probability that λ falls in [ln_lambda_lo, ln_lambda_hi] given (ln M, z).

        Args:
            ln_mass: Natural log of mass (Msun/h); vectorized.
            z: Redshift.
            ln_lambda_lo: Lower richness edge in ln λ.
            ln_lambda_hi: Upper richness edge in ln λ.

        Returns:
            np.ndarray: Probability per input ln_mass value.
        """
        mu, s = self.mean_and_scatter(ln_mass, z)
        inv = 1.0 / (s * np.sqrt(2.0))
        cdf_hi = 0.5 * (1.0 + erf((ln_lambda_hi - mu) * inv))
        cdf_lo = 0.5 * (1.0 + erf((ln_lambda_lo - mu) * inv))
        return np.clip(cdf_hi - cdf_lo, 0.0, 1.0)


def _require_number(block: Mapping[str, object], key: str) -> float:
    """Return a numeric value from a mapping, or raise with a clear message."""
    if key not in block:
        raise KeyError(f"Missing required MOR key: {key!r}")
    try:
        return float(block[key])  # type: ignore[arg-type]
    except Exception as exc:
        raise TypeError(f"MOR key {key!r} must be numeric (got {block[key]!r})") from exc


def make_inverse_from_cfg(cfg: Mapping[str, object]) -> InverseMOR:
    """Construct an `InverseMOR` from a YAML-decoded config dict.

    Expected schema:
        inverse:
          params:
            m_pivot: 3.0e14
            lambda_pivot: 25.0
            z_pivot: 0.6
            a: 0.94
            b: -0.20

    Args:
        cfg: Mapping with a nested `params` dict.

    Returns:
        InverseMOR: Instance built from the provided parameters.

    Raises:
        KeyError, TypeError: If required keys are missing or non-numeric.
    """
    if "params" not in cfg or not isinstance(cfg["params"], dict):
        raise KeyError("Inverse MOR config must contain a 'params' mapping.")
    p: Mapping[str, object] = cfg["params"]  # type: ignore[assignment]
    return InverseMOR(
        m_pivot=_require_number(p, "m_pivot"),
        lambda_pivot=_require_number(p, "lambda_pivot"),
        z_pivot=_require_number(p, "z_pivot"),
        a=_require_number(p, "a"),
        b=_require_number(p, "b"),
    )


def make_forward_from_cfg(cfg: Mapping[str, object]) -> ForwardMOR:
    """Construct a `ForwardMOR` from a YAML-decoded config dict.

    Expected schema:
        forward:
          params:
            a: 3.2
            b: 0.95
            c: -0.20
            sigma_0: 0.25
            q_m: 0.0
            q_z: 0.0
            m_pivot: 3.0e14

    Args:
        cfg: Mapping with a nested `params` dict.

    Returns:
        ForwardMOR: Instance built from the provided parameters.

    Raises:
        KeyError, TypeError: If required keys are missing or non-numeric.
    """
    if "params" not in cfg or not isinstance(cfg["params"], dict):
        raise KeyError("Forward MOR config must contain a 'params' mapping.")
    p: Mapping[str, object] = cfg["params"]  # type: ignore[assignment]
    return ForwardMOR(
        a=_require_number(p, "a"),
        b=_require_number(p, "b"),
        c=_require_number(p, "c"),
        sigma_0=_require_number(p, "sigma_0"),
        q_m=_require_number(p, "q_m"),
        q_z=_require_number(p, "q_z"),
        m_pivot=_require_number(p, "m_pivot"),
    )
