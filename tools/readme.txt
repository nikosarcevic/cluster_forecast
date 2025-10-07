YAML builder for DESC SRD (clusters: NC + WL)
=============================================

Purpose
-------
Builds a self-contained set of YAML configs for the DESC SRD cluster
number counts + weak lensing benchmark, using the SRD data bundle you keep
under config/srd_benchmark/. It does NOT need CosmoLike at runtime.

Optionally, it can scrape Mass–Observable Relation (MOR) priors/fiducials
from the DESC_SRD repo. If you don’t provide that repo, it still generates
the core YAMLs; the two MOR YAMLs will be skipped unless they already exist.

What it writes (into config/)
-----------------------------
desc_srd_counts.yaml
  - Y1/Y10 data vectors as arrays
  - flat index maps
  - relative paths to:
      datavectors (Y1/Y10 fiducial),
      inverse covariances,
      photo-z distributions (lens/source),
      an ℓ-values file (if provided)

cluster_bins.yaml
  - redshift bin counts (Y1, Y10) inferred from datavector shapes
  - richness-bin edges (falls back to [20,30,45,70,120,220] if unknown)

desc_srd_mor_priors.yaml        (only when --srd-dir is provided)
  - 3×3 (A,B,C) prior matrices for Y1 & Y10 scraped from DESC_SRD

desc_srd_mor_fiducials.yaml     (only when --srd-dir is provided)
  - six MOR fiducials (sigma_0, q_m, q_z, A, B, C) scraped from DESC_SRD

Input bundle expected (already in this repo)
--------------------------------------------
config/srd_benchmark/
  datavectors/   # clusterN_clusterWL_Y1_fid, clusterN_clusterWL_Y10_fid, ...
  covmats/       # Y1_clusterN_clusterWL_inv, Y10_clusterN_clusterWL_inv
  zdistris/      # *_Y1_source, *_Y1_lens, *_Y10_source, *_Y10_lens
  ell-values     # either a single file named 'ell-values' OR a directory of ℓ files

Quick start (no DESC_SRD scraping)
----------------------------------
python tools/build_desc_srd_yaml.py

This writes:
  config/desc_srd_counts.yaml
  config/cluster_bins.yaml

If MOR YAMLs aren’t present you’ll see warnings like:
  [warn] desc_srd_mor_priors.yaml not found and --srd-dir not provided; skipping.
  [warn] desc_srd_mor_fiducials.yaml not found and --srd-dir not provided; skipping.

Generate MOR YAMLs (temporary DESC_SRD checkout)
------------------------------------------------
git clone https://github.com/CosmoLike/DESC_SRD.git
python tools/build_desc_srd_yaml.py --srd-dir ./DESC_SRD

Now config/ contains:
  desc_srd_mor_priors.yaml
  desc_srd_mor_fiducials.yaml

You can delete DESC_SRD/ after this.

Custom paths (if your bundle is elsewhere)
------------------------------------------
python tools/build_desc_srd_yaml.py \
  --datav-dir     config/srd_benchmark/datavectors \
  --cov-dir       config/srd_benchmark/covmats \
  --zdistris-dir  config/srd_benchmark/zdistris \
  --ell-dir       config/srd_benchmark/ell-values \
  --out-dir       config \
  --srd-dir       ./DESC_SRD   # optional; only if you want MOR YAMLs re-scraped

Notes:
- --ell-dir may be a file (e.g. config/srd_benchmark/ell-values) or a directory
  containing one or more ℓ files. The script records the first file it finds.
- If you don’t pass --srd-dir, MOR YAMLs are left as-is (if present) or skipped.

Requirements
------------
Python ≥ 3.8
numpy, pyyaml

Install:
python -m pip install numpy pyyaml

Troubleshooting
---------------
“Missing datav files”
  Ensure clusterN_clusterWL_Y1_fid and clusterN_clusterWL_Y10_fid exist under --datav-dir.

Unexpected shapes
  The script infers shapes from leading contiguous indices (starting at 0)
  inside the datavector files. Fix the files if indices don’t start at 0
  or have gaps.

MOR YAMLs not created
  Provide --srd-dir ./DESC_SRD to scrape them, or commit existing MOR YAMLs
  to config/ so the script doesn’t need DESC_SRD.

After generation
----------------
Once YAMLs are in config/, you can remove the DESC_SRD/ directory. Your code
should rely only on the YAMLs and the config/srd_benchmark/ data bundle.
