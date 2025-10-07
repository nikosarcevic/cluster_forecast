from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
import pyccl as ccl

from src.cluster_forecast.configs import (
    load_counts,
    load_area_deg2,
    load_richness_edges,
    load_z_edges,
    load_mor_params,
)
from src.cluster_forecast.cosmology import load_cosmology
from src.cluster_forecast.hmf_theory import get_mass_function
from src.cluster_forecast.mor import make_inverse_from_cfg, make_forward_from_cfg
from src.cluster_forecast.binner import comoving_volume_shell, binned_mass_function
from src.cluster_forecast.predict_counts import compute_forward_counts
from src.cluster_forecast.binned_hmf import volume_weighted_binned_hmf

__all__ = [
    "compute_binned_hmf_data",
]


def compute_binned_hmf_data(
    *,
    year: str,
    hmf_name: str,
    inv_profile: str = "default",
    fwd_profile: str = "default",
    forward_overrides: Optional[Dict[str, float]] = None,
    norm_all_z: bool = False,         # only used by Tinker10
    inverse_n_z: int = 11,            # fine-z for inverse volume weighting
    n_m_per_bin: int = 256,           # lnM integration points for inverse
) -> Dict[str, Any]:
    """
    Build all ingredients for the binned-HMF comparison plot.

    Returns a dict with:
      - counts: observed counts (Nz, Nr)
      - area_deg2, richness_edges (Nr+1), z_edges (Nz+1)
      - volumes: comoving shell volumes per z-bin (Nz)
      - ln_mass_edges: ln(M) bin edges per z-bin from inverse MOR (Nz, Nr+1)
      - mass_centers: M centers per bin (Nz, Nr)
      - dn_data: data dn/dln_mass (Nz, Nr)
      - dn_err: poisson-ish errors (Nz, Nr)
      - dn_fwd: forward-model dn/dln_mass (Nz, Nr)
      - dn_inv: inverse/HMF volume-weighted, bin-averaged dn/dln_mass (Nz, Nr)
      - z_centers: bin centers (Nz,)
      - median_ratio_per_z: median( data / forward ) per z row (Nz,)
      - median_ratio_overall: single median across all bins (float)
      - meta: dict with year, hmf_name, profiles, params actually used
    """
    # --- SRD configs & survey ---
    counts, nz, nr = load_counts(year)
    area_deg2 = load_area_deg2(year)
    richness_edges = load_richness_edges()
    z_edges = load_z_edges(year)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    # --- cosmology & HMF ---
    cosmo: ccl.Cosmology = load_cosmology(profile="srd")
    mf = get_mass_function(hmf_name, cosmo, norm_all_z=norm_all_z)

    # --- MOR configs (YAML) ---
    inv_cfg = load_mor_params("inverse", profile=inv_profile)
    fwd_cfg = load_mor_params("forward", profile=fwd_profile)
    if forward_overrides:
        fwd_cfg = {**fwd_cfg}
        fwd_cfg.setdefault("params", {}).update(forward_overrides)

    inverse_mor = make_inverse_from_cfg(inv_cfg)
    forward_mor = make_forward_from_cfg(fwd_cfg)

    # integrator settings for forward counts (strict)
    integ = fwd_cfg.get("integrator", {})
    for key in ("n_z", "n_m", "m_min", "m_max"):
        if key not in integ:
            raise KeyError(f"missing forward.integrator key: '{key}' in profile '{fwd_profile}'")
    n_z_fwd = int(integ["n_z"])
    n_m_fwd = int(integ["n_m"])
    m_min_fwd = float(integ["m_min"])
    m_max_fwd = float(integ["m_max"])

    # --- survey shell volumes per z-bin ---
    volumes = np.asarray(
        [comoving_volume_shell(cosmo, z_edges[i], z_edges[i + 1], area_deg2) for i in range(nz)],
        dtype=float,
    )  # (Nz,)

    # --- inverse MOR → lnM edges and mass centers per z-bin ---
    ln_mass_edges = np.empty((nz, nr + 1), dtype=float)
    mass_centers = np.empty((nz, nr), dtype=float)
    for i, zc in enumerate(z_centers):
        m_edges = inverse_mor.mass_edges(richness_edges, zc)  # Msun/h
        ln_edges = np.log(m_edges)
        ln_mass_edges[i] = ln_edges
        mass_centers[i] = np.exp(0.5 * (ln_edges[:-1] + ln_edges[1:]))

    # --- data densities and dn/dln_mass (using inverse bins) ---
    n_density = counts / volumes[:, None]  # Mpc^-3
    dn_data, dln_mass = binned_mass_function(n_density, ln_mass_edges)
    dn_err = np.sqrt(np.maximum(counts, 0.0)) / (volumes[:, None] * dln_mass)

    # --- forward counts → dn/dln_mass in the SAME mass bins ---
    counts_fwd = compute_forward_counts(
        cosmo=cosmo,
        mass_function_model=mf,
        forward_mor=forward_mor,
        richness_edges=richness_edges,
        redshift_edges=z_edges,
        area_deg2=area_deg2,
        n_z=n_z_fwd,
        n_mass=n_m_fwd,
        mass_min=m_min_fwd,
        mass_max=m_max_fwd,
    )
    n_density_fwd = counts_fwd / volumes[:, None]
    dn_fwd, _ = binned_mass_function(n_density_fwd, ln_mass_edges)

    # --- inverse/HMF overlay: volume-weighted, bin-averaged HMF (no scatter) ---
    dn_inv_rows = []
    for i in range(nz):
        dn_inv_i = volume_weighted_binned_hmf(
            cosmo=cosmo,
            mass_fn=mf,
            ln_m_edges_row=ln_mass_edges[i],
            z_min=z_edges[i],
            z_max=z_edges[i + 1],
            n_z=inverse_n_z,
            n_m_per_bin=n_m_per_bin,
        )
        dn_inv_rows.append(dn_inv_i)
    dn_inv = np.vstack(dn_inv_rows)

    # --- quick diagnostics ---
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = dn_data / np.clip(dn_fwd, 1e-300, None)
    med_per_z = np.nanmedian(ratio, axis=1)
    med_all = float(np.nanmedian(ratio))

    # --- bundle outputs ---
    return {
        "counts": counts,
        "area_deg2": float(area_deg2),
        "richness_edges": richness_edges,
        "z_edges": z_edges,
        "volumes": volumes,
        "ln_mass_edges": ln_mass_edges,
        "mass_centers": mass_centers,
        "dn_data": dn_data,
        "dn_err": dn_err,
        "dn_fwd": dn_fwd,
        "dn_inv": dn_inv,
        "dln_mass": dln_mass,  # <— add this
        "z_centers": z_centers,
        "nz": int(nz),  # <— and these two for convenience
        "nr": int(nr),
        "median_ratio_per_z": med_per_z,
        "median_ratio_overall": med_all,
        "meta": {
            "year": year,
            "hmf_name": hmf_name,
            "inv_profile": inv_profile,
            "fwd_profile": fwd_profile,
            "norm_all_z": bool(norm_all_z),
            "forward_params_used": dict(fwd_cfg.get("params", {})),
            "integrator": dict(integ),
            "inverse_n_z": int(inverse_n_z),  # <—
            "n_m_per_bin": int(n_m_per_bin),  # <—
            "cosmo_h": float(cosmo["h"]),
        },
    }
