from __future__ import annotations

from typing import Mapping
import pathlib
import yaml
import pyccl as ccl

from .configs import config_dir


def load_cosmology(
    profile: str = "srd",
    yaml_path: pathlib.Path = config_dir / "cosmology_params.yaml",
) -> ccl.Cosmology:
    """Load a PyCCL cosmology from a YAML profile.

    YAML schema (example):
      srd:
        values:
          Omega_m: 0.3156
          sigma_8: 0.831
          n_s: 0.9645
          w_0: -1.0
          w_a: 0.0
          Omega_b: 0.0491685
          h: 0.6727
        sigmas:
          omega_m: 0.15
          ...

    Args:
      profile: Top-level key under which `values` and (optionally) `sigmas` live.
      yaml_path: Path to the cosmology YAML file.

    Returns:
      ccl.Cosmology: Constructed with (Omega_c, Omega_b, h, n_s, w0, wa, sigma8).

    Raises:
      FileNotFoundError: If the YAML file is missing.
      KeyError / TypeError: If required keys are missing or non-numeric.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Cosmology YAML not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        doc = yaml.safe_load(f)

    if profile not in doc:
        raise KeyError(f"Profile '{profile}' not found in {yaml_path}")

    values = doc[profile].get("values")
    if not isinstance(values, dict):
        raise KeyError(f"Profile '{profile}' must contain a 'values' mapping")

    omega_m = _num(values, "Omega_m")
    omega_b = _num(values, "Omega_b")
    omega_c = omega_m - omega_b

    cosmo = ccl.Cosmology(
        Omega_c=omega_c,
        Omega_b=omega_b,
        h=_num(values, "h"),
        n_s=_num(values, "n_s"),
        w0=_num(values, "w_0"),
        wa=_num(values, "w_a"),
        sigma8=_num(values, "sigma_8"),
    )
    return cosmo


def load_cosmo_priors(
    profile: str = "srd",
    yaml_path: pathlib.Path = config_dir / "cosmology_params.yaml",
) -> dict[str, float]:
    """Return the `sigmas` (Gaussian priors) block, if present, as floats."""
    if not yaml_path.exists():
        raise FileNotFoundError(f"Cosmology YAML not found: {yaml_path}")
    with open(yaml_path, "r") as f:
        doc = yaml.safe_load(f)
    if profile not in doc:
        raise KeyError(f"Profile '{profile}' not found in {yaml_path}")
    sigmas = doc[profile].get("sigmas", {})
    if not isinstance(sigmas, dict):
        return {}
    # coerce to float where possible
    out: dict[str, float] = {}
    for k, v in sigmas.items():
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


def _num(block: Mapping, key: str) -> float:
    """Get a required numeric key as float, with a clear error if missing/bad."""
    if key not in block:
        raise KeyError(f"Missing cosmology value: '{key}'")
    try:
        return float(block[key])
    except Exception as e:
        raise TypeError(f"Cosmology value '{key}' must be numeric (got {block[key]!r})") from e
