
from __future__ import annotations

from pathlib import Path

import numpy as np
import pathlib, yaml


__all__ = [
    "config_dir",
    "load_counts",
    "load_area_deg2",
    "load_richness_edges",
    "load_z_edges",
    "load_mor_params",
]


config_dir = Path(__file__).resolve().parents[2] / "config"


def load_counts(year: str, path: pathlib.Path = config_dir/ "srd_cluster_counts_extracted.yaml"):
    """
    Returns (counts[Nz,Nr], Nz, Nr)
    counts_extracted.yaml structure is: {counts: {Y1:{shape: [Nz,Nr], values:[...]}, Y10:{...}}}
    """
    doc = yaml.safe_load(open(path, "r"))
    c = doc["counts"][year]
    arr = np.array(c["values"], float)
    shp = tuple(c["shape"])
    if arr.shape != shp:
        arr = arr.reshape(shp)
    nz, nr = arr.shape
    if nr != 5:
        raise ValueError(f"Expected 5 richness bins; got shape {arr.shape}")
    return arr, nz, nr


def load_area_deg2(year: str, path: pathlib.Path = config_dir / "survey_specs.yaml") -> float:
    """Load survey area (deg^2) for a given SRD block (Y1/Y10) from survey_specs.yaml."""
    doc = _read_yaml(path)
    area_map = doc.get("area", {})
    if year not in area_map:
        raise KeyError(f"'area' for {year} not found in {path}")
    return float(area_map[year])


def load_z_edges(year: str, path: pathlib.Path = config_dir / "survey_specs.yaml") -> np.ndarray:
    """Load redshift bin edges for a given SRD block (Y1/Y10) from survey_specs.yaml."""
    doc = _read_yaml(path)
    z_map = doc.get("z_bin_edges", {})
    if year not in z_map:
        raise KeyError(f"'z_bin_edges' for {year} not found in {path}")
    edges = np.asarray(z_map[year], dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError(f"z_bin_edges[{year}] must be strictly increasing; got {edges!r}")
    return edges



def load_richness_edges(path: pathlib.Path = config_dir/ "cluster_bins.yaml"):
    b = yaml.safe_load(open(path, "r"))
    return np.array(b["richness_bins"]["edges"], float)


def _read_yaml(path: pathlib.Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_mor_params(kind: str, profile: str, path: pathlib.Path = config_dir/ "mor_params.yaml") -> dict:
    """
    Strict: returns exactly the dict at mor.yaml[kind][profile].
    No merging, no code defaults.
    """
    doc = _read_yaml(path)
    if kind not in doc:
        raise KeyError(f"Section '{kind}' not found in {path}")
    section = doc[kind]
    if profile not in section:
        raise KeyError(f"Profile '{profile}' not found under '{kind}' in {path}")
    cfg = section[profile]
    if not isinstance(cfg, dict):
        raise TypeError(f"Profile '{profile}' under '{kind}' must be a mapping")
    return cfg
