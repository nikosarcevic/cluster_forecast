#!/usr/bin/env python3
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import numpy as np
import yaml

# ---------- tiny io helpers ----------

def load_datav_values(p: Path) -> np.ndarray:
    idxs, vals = [], []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                i = int(float(parts[0])); v = float(parts[1])
            except ValueError:
                continue
            idxs.append(i); vals.append(v)
    if not idxs:
        raise ValueError(f"No numeric rows parsed from {p}")
    idxs = np.array(idxs, int); vals = np.array(vals, float)
    out, expect = [], 0
    for i, v in zip(idxs, vals):
        if i == expect:
            out.append(v); expect += 1
        elif expect == 0:
            continue
        else:
            break
    if not out:
        raise ValueError(f"Couldn't find contiguous indices starting at 0 in {p}")
    return np.array(out)

def choose_shape(n: int) -> tuple[int, int]:
    cands = [(a, n//a) for a in range(1, int(np.sqrt(n))+1) if n % a == 0]
    if (3,5) in cands: return (3,5)
    if (4,5) in cands: return (4,5)
    return min(cands, key=lambda ab: abs(ab[0]-ab[1])) if cands else (1, n)

def flat_index_map(arr: np.ndarray) -> dict:
    nz, nr = arr.shape
    out = {}
    for iz in range(nz):
        for ir in range(nr):
            flat = iz*nr + ir
            out[str(flat)] = {"z_bin": iz, "richness_bin": ir, "value": float(arr[iz, ir])}
    return out

# ---------- optional scraping (only if --srd-dir is given) ----------

MAT_RE = re.compile(r"MORPRIOR\s*=\s*(\[\s*\[.*?\]\s*\])", re.DOTALL)
FID_BLOCK_RE = {
    "Y1": re.compile(r"def\s+fiducial_Y1\s*\(.*?\):(?P<body>.*?)(?=^\s*def\s|\Z)", re.S|re.M),
    "Y10": re.compile(r"def\s+fiducial_Y10\s*\(.*?\):(?P<body>.*?)(?=^\s*def\s|\Z)", re.S|re.M),
}
MLAMBDA_ASSIGN_RE = re.compile(r"c\.m_lambda\s*\[\s*:\s*\]\s*=\s*(?P<rhs>\[.*?\]|np\.repeat\([^)]*\))", re.S)

def _nearest_context_tag(txt: str, pos: int) -> str|None:
    y1m = list(re.finditer(r"if\s*\(\s*sys\.argv\[1\]\s*==\s*['\"]Y1['\"]\s*\)", txt[:pos]))
    y10m = list(re.finditer(r"if\s*\(\s*sys\.argv\[1\]\s*==\s*['\"]Y10['\"]\s*\)", txt[:pos]))
    last_y1 = y1m[-1].end() if y1m else -1
    last_y10 = y10m[-1].end() if y10m else -1
    if last_y1 > last_y10 and last_y1 != -1: return "Y1"
    if last_y10 > last_y1 and last_y10 != -1: return "Y10"
    ctx = txt[max(0, pos-800):pos+800]
    y1_here = "Y1" in ctx; y10_here = "Y10" in ctx
    if y1_here and not y10_here: return "Y1"
    if y10_here and not y1_here: return "Y10"
    return None

def scrape_morpriors(srd_dir: Path) -> dict:
    out = {"Y1": None, "Y10": None}
    for py in srd_dir.rglob("*.py"):
        try:
            txt = py.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in MAT_RE.finditer(txt):
            mat_txt = m.group(1)
            tag = _nearest_context_tag(txt, m.start())
            try:
                mat = eval(mat_txt, {"__builtins__": {}})
                if isinstance(mat, list) and all(isinstance(r, list) for r in mat):
                    if tag and out.get(tag) is None:
                        out[tag] = mat
            except Exception:
                pass
    return {k: v for k, v in out.items() if v is not None}

def _parse_rhs_list(rhs: str) -> list[float]:
    vals = eval(rhs, {"__builtins__": {}})
    if isinstance(vals, (list, tuple)): return [float(x) for x in vals]
    raise ValueError("m_lambda RHS not a list")

def _parse_np_repeat(rhs: str) -> list[float]:
    m = re.search(r"np\.repeat\(\s*([0-9.eE+\-]+)\s*,\s*([0-9]+)\s*\)", rhs)
    if not m: raise ValueError("np.repeat pattern not recognized")
    val = float(m.group(1)); n = int(m.group(2))
    return [val]*n

def scrape_mlambda_fiducials(srd_dir: Path) -> dict:
    out = {}
    for py in srd_dir.rglob("*.py"):
        try:
            txt = py.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for tag, block_rx in FID_BLOCK_RE.items():
            if tag in out: continue
            mb = block_rx.search(txt)
            if not mb: continue
            body = mb.group("body")
            ma = MLAMBDA_ASSIGN_RE.search(body)
            if not ma: continue
            rhs = ma.group("rhs")
            try:
                vals = _parse_rhs_list(rhs) if rhs.strip().startswith("[") else _parse_np_repeat(rhs)
            except Exception:
                continue
            out[tag] = vals
    return out

# ---------- scan helpers for local benchmark bundle ----------

def find_zdistris(zd_dir: Path, tag: str) -> dict|None:
    if not zd_dir or not zd_dir.exists(): return None
    src = next((p for p in zd_dir.iterdir() if p.is_file() and p.name.endswith(f"{tag}_source")), None)
    lens = next((p for p in zd_dir.iterdir() if p.is_file() and p.name.endswith(f"{tag}_lens")), None)
    if not (src and lens): return None
    return {"source": str(src.relative_to(zd_dir.parent)), "lens": str(lens.relative_to(zd_dir.parent))}

def find_invcov(cov_dir: Path, tag: str) -> str|None:
    if not cov_dir or not cov_dir.exists(): return None
    cand = cov_dir / f"{tag}_clusterN_clusterWL_inv"
    return str(cand.relative_to(cov_dir.parent)) if cand.exists() else None

def resolve_ell_path(ell_path: Path) -> str | None:
    """
    Accept either a file (e.g. config/srd_benchmark/ell-values) or a directory
    containing one or more ell-value files. Returns a path relative to the
    srd_benchmark root (the parent of the file/dir).
    """
    if not ell_path or not ell_path.exists():
        return None
    if ell_path.is_file():
        return str(ell_path.relative_to(ell_path.parent))
    if ell_path.is_dir():
        files = sorted(p for p in ell_path.iterdir() if p.is_file())
        return str(files[0].relative_to(ell_path.parent)) if files else None
    return None

# ---------- builders ----------

def build_counts_yaml(datav_dir: Path, cov_dir: Path|None, zd_dir: Path|None, ell_dir: Path|None) -> dict:
    y1_file = datav_dir / "clusterN_clusterWL_Y1_fid"
    y10_file = datav_dir / "clusterN_clusterWL_Y10_fid"
    if not y1_file.exists() or not y10_file.exists():
        raise FileNotFoundError(f"Missing datav files at {datav_dir}")

    y1_vals = load_datav_values(y1_file)
    y10_vals = load_datav_values(y10_file)
    nz1, nr1 = choose_shape(len(y1_vals))
    nz10, nr10 = choose_shape(len(y10_vals))
    y1 = y1_vals.reshape(nz1, nr1)
    y10 = y10_vals.reshape(nz10, nr10)

    doc = {
        "dataset": "DESC_SRD",
        "probe": "clusterN_clusterWL",
        "source": str(datav_dir.parent),  # config/srd_benchmark
        "Y1": {
            "counts": {"unit": "raw_counts", "shape": [int(nz1), int(nr1)], "order": ["z_bin","richness_bin"], "values": y1.tolist()},
            "flat_index_map": flat_index_map(y1),
            "datav_file": str(y1_file.relative_to(datav_dir.parent)),
        },
        "Y10": {
            "counts": {"unit": "raw_counts", "shape": [int(nz10), int(nr10)], "order": ["z_bin","richness_bin"], "values": y10.tolist()},
            "flat_index_map": flat_index_map(y10),
            "datav_file": str(y10_file.relative_to(datav_dir.parent)),
        },
        "notes": [
            "Shapes inferred automatically from leading contiguous indices.",
            "Bin edges are provided in cluster_bins.yaml.",
        ],
    }

    # optional pointers kept entirely within config/
    if cov_dir:
        y1_cov = find_invcov(cov_dir, "Y1");  y10_cov = find_invcov(cov_dir, "Y10")
        if y1_cov:  doc["Y1"]["invcov_file"]  = y1_cov
        if y10_cov: doc["Y10"]["invcov_file"] = y10_cov
    if zd_dir:
        z1 = find_zdistris(zd_dir, "Y1"); z10 = find_zdistris(zd_dir, "Y10")
        if z1:  doc["Y1"]["zdistris"]  = z1
        if z10: doc["Y10"]["zdistris"] = z10
    if ell_dir:
        ell = resolve_ell_path(ell_dir)
        if ell: doc["ell_values_file"] = ell

    return doc

def build_cluster_bins_yaml(counts_doc: dict) -> dict:
    nz1 = int(counts_doc["Y1"]["counts"]["shape"][0])
    nz10 = int(counts_doc["Y10"]["counts"]["shape"][0])
    default_edges = [20.0,30.0,45.0,70.0,120.0,220.0]
    doc = {
        "richness_bins": {"edges": default_edges, "count": len(default_edges)-1},
        "redshift_bins": {"y1_nbins": nz1, "y10_nbins": nz10},
        "source": {"counts_yaml": "desc_srd_counts.yaml", "harvested_from_config": False}
    }
    return doc

# --- MOR yaml: real build (scrape) or placeholder fallbacks ---

def build_mor_priors_yaml_from_srd(srd_dir: Path) -> dict:
    mats = scrape_morpriors(srd_dir)
    if not mats:
        raise RuntimeError("Couldn't find any MORPRIOR matrices in SRD python files.")
    out = {
        "dataset": "DESC_SRD",
        "probe": "clusterN_clusterWL",
        "MOR_prior": {},
        "source": "scraped from *.py 'MORPRIOR = [[...]]' in DESC_SRD",
    }
    for k, mat in mats.items():
        out["MOR_prior"][k] = {"matrix": mat, "shape": [len(mat), len(mat[0]) if mat else 0]}
    # Ensure both keys exist
    for tag in ("Y1","Y10"):
        out["MOR_prior"].setdefault(tag, {"matrix": None, "shape": [3,3]})
    return out

def build_mor_fiducials_yaml_from_srd(srd_dir: Path) -> dict:
    vals = scrape_mlambda_fiducials(srd_dir)
    if not vals:
        raise RuntimeError("Couldn't find m_lambda fiducial assignments in fiducial_Y* blocks.")
    alias = {
        "m_lambda_0": "sigma_0",
        "m_lambda_1": "q_m",
        "m_lambda_2": "q_z",
        "m_lambda_3": "A",
        "m_lambda_4": "B",
        "m_lambda_5": "C",
    }
    out = {
        "dataset": "DESC_SRD",
        "probe": "clusterN_clusterWL",
        "MOR_fiducial": {},
        "alias": alias,
        "source": "scraped from fiducial_Y1/Y10 in DESC_SRD",
    }
    for tag in ("Y1","Y10"):
        arr = vals.get(tag)
        if not arr or len(arr) != 6:
            out["MOR_fiducial"][tag] = {
                "values": {v: None for v in alias.values()},
                "note": "scrape missing or length != 6"
            }
        else:
            out["MOR_fiducial"][tag] = {
                "values": {alias[f"m_lambda_{i}"]: float(arr[i]) for i in range(6)},
                "as_list": [float(x) for x in arr],
            }
    return out

def build_mor_priors_yaml_placeholder() -> dict:
    return {
        "dataset": "DESC_SRD",
        "probe": "clusterN_clusterWL",
        "MOR_prior": {
            "Y1": {"matrix": None, "shape": [3,3]},
            "Y10":{"matrix": None, "shape": [3,3]},
        },
        "source": "placeholder – provide DESC_SRD to auto-scrape or fill manually."
    }

def build_mor_fiducials_yaml_placeholder() -> dict:
    alias = {
        "m_lambda_0": "sigma_0",
        "m_lambda_1": "q_m",
        "m_lambda_2": "q_z",
        "m_lambda_3": "A",
        "m_lambda_4": "B",
        "m_lambda_5": "C",
    }
    base_vals = {v: None for v in alias.values()}
    return {
        "dataset": "DESC_SRD",
        "probe": "clusterN_clusterWL",
        "alias": alias,
        "MOR_fiducial": {
            "Y1": {"values": dict(base_vals)},
            "Y10":{"values": dict(base_vals)},
        },
        "source": "placeholder – provide DESC_SRD to auto-scrape or fill manually."
    }

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datav-dir", default="config/srd_benchmark/datavectors", type=str, help="Path to SRD datavectors (within config/).")
    ap.add_argument("--cov-dir",   default="config/srd_benchmark/covmats",    type=str, help="Path to SRD covmats (within config/).")
    ap.add_argument("--zdistris-dir", default="config/srd_benchmark/zdistris", type=str, help="(optional) zdistris dir (within config/).")
    ap.add_argument("--ell-dir",      default="config/srd_benchmark/ell-values", type=str, help="(optional) ell-values file OR directory (within config/).")
    ap.add_argument("--srd-dir",   default=None, type=str, help="(optional) DESC_SRD repo; if provided we will scrape MOR YAMLs.")
    ap.add_argument("--out-dir",   default="config", type=str, help="Where to write YAMLs.")
    ap.add_argument("--fail-on-missing-mor", action="store_true", help="Instead of writing placeholders, error if MOR cannot be scraped.")
    args = ap.parse_args()

    datav_dir = Path(args.datav_dir).resolve()
    cov_dir   = Path(args.cov_dir).resolve() if args.cov_dir else None
    zd_dir    = Path(args.zdistris_dir).resolve() if args.zdistris_dir else None
    ell_dir   = Path(args.ell_dir).resolve() if args.ell_dir else None
    out_dir   = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build counts/bins from local benchmark bundle
    counts_doc = build_counts_yaml(datav_dir, cov_dir, zd_dir, ell_dir)
    (out_dir / "desc_srd_counts.yaml").write_text(yaml.safe_dump(counts_doc, sort_keys=False))
    bins_doc = build_cluster_bins_yaml(counts_doc)
    (out_dir / "cluster_bins.yaml").write_text(yaml.safe_dump(bins_doc, sort_keys=False))

    # MOR yaml behavior:
    wrote_real = False
    if args.srd_dir:
        srd_dir = Path(args.srd_dir).resolve()
        try:
            mor_doc = build_mor_priors_yaml_from_srd(srd_dir)
            (out_dir / "desc_srd_mor_priors.yaml").write_text(yaml.safe_dump(mor_doc, sort_keys=False))
            fid_doc = build_mor_fiducials_yaml_from_srd(srd_dir)
            (out_dir / "desc_srd_mor_fiducials.yaml").write_text(yaml.safe_dump(fid_doc, sort_keys=False))
            wrote_real = True
        except Exception as e:
            if args.fail_on_missing_mor:
                print(f"[error] Failed to scrape MOR from {srd_dir}: {e}", file=sys.stderr)
                sys.exit(2)
            # fall through to placeholders
    if not wrote_real:
        # ALWAYS write placeholders so files exist
        (out_dir / "desc_srd_mor_priors.yaml").write_text(
            yaml.safe_dump(build_mor_priors_yaml_placeholder(), sort_keys=False)
        )
        (out_dir / "desc_srd_mor_fiducials.yaml").write_text(
            yaml.safe_dump(build_mor_fiducials_yaml_placeholder(), sort_keys=False)
        )

    print(f"Wrote YAMLs to {out_dir}")

if __name__ == "__main__":
    main()
