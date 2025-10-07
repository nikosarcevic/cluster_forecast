from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import numpy as np
import cmasher as cmr

__all__ = ["hmf_label", "plot_binned_hmf"]


def hmf_label(name_or_obj):
    """Map HMF internal names to pretty labels."""
    name = name_or_obj if isinstance(name_or_obj, str) else name_or_obj.__class__.__name__
    key = str(name).strip().lower()
    return {
        "tinker10": "Tinker10",
        "tinker08": "Tinker08",
        "jenkins":  "Jenkins",
    }.get(key, str(name))


def plot_binned_hmf(
    year: str,
    hmf_name: str,
    z_centers: np.ndarray,
    z_edges: np.ndarray,
    nz: int,
    mass_centers: np.ndarray,
    dn_fwd: np.ndarray,
    dn_inv: np.ndarray,
    dn_data: np.ndarray,
    area_deg2: float,
    plt_dir: Path = Path("plots"),
    file_ext: str = "pdf",
    fwd_profile: str | None = None,  # optional but handy in filename
):
    plt_dir = Path(plt_dir)
    plt_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = cmr.take_cmap_colors('viridis', nz, cmap_range=(0., 0.5), return_fmt='hex')
    fs = 15; lw = 3; markersize = 100

    for i, zc in enumerate(z_centers):
        color = colors[i]
        ax.plot(mass_centers[i], dn_fwd[i], lw=lw, color=color)
        ax.plot(mass_centers[i], dn_inv[i], lw=lw, ls="--", color=color)
        ax.scatter(mass_centers[i], dn_data[i], s=markersize, edgecolor=color, facecolor="yellow", zorder=5)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$M\ [M_\odot/h]$", fontsize=fs)
    ax.set_ylabel(r"$\frac{\mathrm{d}n}{\mathrm{d}\ln M}\;[\mathrm{Mpc}^{-3}]$", fontsize=fs)
    ax.set_title(f"LSST {year}  (area={area_deg2:.0f} deg$^2$)", fontsize=fs+2)

    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="major", length=6, width=1.0)
    ax.tick_params(axis="both", which="minor", length=3, width=0.9)

    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2,10)*0.1))
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2,10)*0.1))
    ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext())
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    # z-slice legend
    z_labels = [f"{z_edges[i]:.1f}–{z_edges[i+1]:.1f}" for i in range(nz)]
    z_handles = [mlines.Line2D([0],[0], color=colors[i], lw=3, label=z_labels[i]) for i in range(nz)]
    leg_z = ax.legend(handles=z_handles, title="$z$ slice", frameon=False, fontsize=fs-1, title_fontsize=fs-1, loc="upper right")

    # style legend
    hmf_str = f"HMF: {hmf_label(hmf_name)}"
    style_handles = [
        mlines.Line2D([0],[0], color="white", lw=lw, label=hmf_str),
        mlines.Line2D([0],[0], color='k', lw=lw, label='forward (with scatter)'),
        mlines.Line2D([0],[0], color='k', lw=lw, ls='--', label='inverse (no scatter)'),
        mlines.Line2D([0],[0], marker='o', color='k', lw=0, markersize=lw+4, label='SRD data'),
    ]
    ax.legend(handles=style_handles, frameon=False, fontsize=fs-1, loc="lower left")
    ax.add_artist(leg_z)

    fig.tight_layout()

    # build filename safely
    suffix = f"_{fwd_profile}" if fwd_profile else ""
    out = plt_dir / f"binned_hmf_{year}_{hmf_name}{suffix}.{file_ext}"

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {out.resolve()}")
