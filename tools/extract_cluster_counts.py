#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import yaml
import sys

def _read_values_column(p: Path) -> np.ndarray:
    """Read the 2nd column ('value') from a 'index value' file, in file order."""
    vals = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) < 2:
                continue
            try:
                v = float(parts[1])
            except ValueError:
                continue
            vals.append(v)
    if not vals:
        raise RuntimeError(f"No numeric values parsed from {p}")
    return np.asarray(vals, float)

def _reshape_first_block(values: np.ndarray, nz: int, rbins: int) -> np.ndarray:
    need = nz * rbins
    if values.size < need:
        raise RuntimeError(f"Need {need} values, file has only {values.size}")
    block = values[:need].copy()
    return block.reshape(nz, rbins)

def _flat_index_map(arr: np.ndarray) -> dict:
    nz, nr = arr.shape
    out = {}
    for iz in range(nz):
        for ir in range(nr):
            flat = iz * nr + ir
            out[str(flat)] = {
                "z_bin": iz,
                "richness_bin": ir,
                "value": float(arr[iz, ir]),
            }
    return out

def write_yaml(out_path: Path,
               y1_arr: np.ndarray, y1_src: Path,
               y10_arr: np.ndarray, y10_src: Path,
               richness_edges: list[float] | None):
    doc = {
        "dataset": "DESC_SRD",
        "probe": "clusterN_clusterWL",
        "counts": {
            "Y1": {
                "shape": [int(y1_arr.shape[0]), int(y1_arr.shape[1])],
                "order": ["z_bin", "richness_bin"],
                "values": y1_arr.tolist(),
                "flat_index_map": _flat_index_map(y1_arr),
                "source_file": str(y1_src),
            },
            "Y10": {
                "shape": [int(y10_arr.shape[0]), int(y10_arr.shape[1])],
                "order": ["z_bin", "richness_bin"],
                "values": y10_arr.tolist(),
                "flat_index_map": _flat_index_map(y10_arr),
                "source_file": str(y10_src),
            },
        },
        "notes": [
            "ONLY the first nz×rbins entries are used (pure cluster counts).",
            "WL, photo-z, etc. that follow in the SRD datavectors are intentionally ignored.",
        ],
    }
    if richness_edges:
        doc["richness_bins"] = {
            "edges": [float(x) for x in richness_edges],
            "count": max(len(richness_edges) - 1, 0),
        }
    out_path.write_text(yaml.safe_dump(doc, sort_keys=False))
    print(f"[ok] wrote {out_path}")

def write_csv(prefix: Path, tag: str, arr: np.ndarray):
    nz, nr = arr.shape
    lines = ["z_bin,richness_bin,flat_index,count"]
    for iz in range(nz):
        for ir in range(nr):
            flat = iz * nr + ir
            lines.append(f"{iz},{ir},{flat},{arr[iz, ir]:.10g}")
    p = prefix.with_name(f"{prefix.name}_{tag}.csv")
    p.write_text("\n".join(lines))
    print(f"[ok] wrote {p}")

def main():
    ap = argparse.ArgumentParser(description="Extract ONLY cluster number counts (Y1/Y10) from SRD datavectors.")
    ap.add_argument("--datav-dir", default="config/srd_benchmark/datavectors", type=str,
                    help="Dir with SRD datavectors (clusterN_clusterWL_Y1_fid, _Y10_fid).")
    ap.add_argument("--out-yaml", default="config/srd_cluster_counts_extracted.yaml", type=str,
                    help="Output YAML file with Y1 & Y10 counts.")
    ap.add_argument("--csv-prefix", default=None, type=str,
                    help="Also write CSVs as <prefix>_Y1.csv and <prefix>_Y10.csv.")
    ap.add_argument("--rbins", type=int, default=5, help="# richness bins (default 5).")
    ap.add_argument("--y1-nz", type=int, default=3, help="# z bins for Y1 (default 3).")
    ap.add_argument("--y10-nz", type=int, default=4, help="# z bins for Y10 (default 4).")
    ap.add_argument("--richness-edges", default="20,30,45,70,120,220",
                    help="Comma-separated richness edges to attach in YAML ('' to omit).")
    args = ap.parse_args()

    datav_dir = Path(args.datav_dir).resolve()
    y1_file = datav_dir / "clusterN_clusterWL_Y1_fid"
    y10_file = datav_dir / "clusterN_clusterWL_Y10_fid"
    for p in (y1_file, y10_file):
        if not p.exists():
            print(f"[err] missing datavector: {p}", file=sys.stderr)
            sys.exit(1)

    y1_vals = _read_values_column(y1_file)
    y10_vals = _read_values_column(y10_file)

    y1_arr = _reshape_first_block(y1_vals, args.y1_nz, args.rbins)
    y10_arr = _reshape_first_block(y10_vals, args.y10_nz, args.rbins)

    edges = None
    if args.richness_edges.strip():
        try:
            edges = [float(x) for x in args.richness_edges.split(",") if x.strip()]
        except Exception:
            print("[warn] couldn't parse --richness-edges; omitting from YAML.", file=sys.stderr)

    out_yaml = Path(args.out_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(
        out_yaml,
        y1_arr, y1_file.relative_to(datav_dir.parent),
        y10_arr, y10_file.relative_to(datav_dir.parent),
        edges,
    )

    if args.csv_prefix:
        csv_prefix = Path(args.csv_prefix)
        csv_prefix.parent.mkdir(parents=True, exist_ok=True)
        write_csv(csv_prefix, "Y1", y1_arr)
        write_csv(csv_prefix, "Y10", y10_arr)

if __name__ == "__main__":
    main()
