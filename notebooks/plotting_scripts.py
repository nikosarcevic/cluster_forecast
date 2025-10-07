import sys, pathlib
sys.path.append(str(pathlib.Path().resolve()))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import cmasher as cmr

__all = [
    "hmf_label",
    "plot_binned_hmf",
]


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
    dn_err: np.ndarray,
    area_deg2: float,
    plt_dir: pathlib.Path = pathlib.Path("plots/"),
    file_ext: str = "pdf"
):
    """Plot binned HMF (dn/dlnM) from forward and inverse modeling, compared to data.


    Args:
        year: SRD block (e.g., "Y1", "Y10").
        hmf_name: Name of the halo mass function (e.g., "tinker10").
        z_centers: Array of shape (nz,) with redshift bin centers.
        z_edges: Array of shape (nz+1,) with redshift bin edges.
        nz: Number of redshift bins.
        mass_centers: Array of shape (nz, nr) with mass bin centers (Msun/h).
        dn_fwd: Array of shape (nz, nr) with forward-model HMF (Mpc^-3).
        dn_inv: Array of shape (nz, nr) with inverse-model HMF (Mpc^-3).
        dn_data: Array of shape (nz, nr) with SRD data HMF (Mpc^-3).
        dn_err: Array of shape (nz, nr) with uncertainties on dn_data (Mpc^-3).
        area_deg2: Survey area in square degrees.
        plt_dir: Directory in which to save the plot PDF.
        file_ext: File extension for the saved plot (default: "pdf").

    Returns:
        None. Saves the plot as a PDF file.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = cmr.take_cmap_colors('viridis', nz, cmap_range=(0., 0.5), return_fmt='hex')
    fs = 15
    lw = 3
    markersize = 100

    for i, zc in enumerate(z_centers):
        color = colors[i]
        # forward (filled)
        ax.plot(mass_centers[i], dn_fwd[i], lw=lw, color=color)
        # inverse overlay (dashed)
        ax.plot(mass_centers[i], dn_inv[i], lw=lw, ls="--", color=color)
        # data (red)
        ax.scatter(mass_centers[i], dn_data[i], s=markersize, edgecolor=color, facecolor="yellow", zorder=5)
        #ax.errorbar(M_centers[i], dn_data[i], yerr=dn_err[i], fmt="none", ecolor=color, alpha=0.6, capsize=3, lw=2)

    ax.set_xscale("log")
    ax.set_yscale("log")

    xlabel = r"$M\ [M_\odot/h]$"
    ylabel = r"$\frac{\mathrm{d}n}{\mathrm{d}\ln M}\;[\mathrm{Mpc}^{-3}]$"
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs)
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

    # Legend for z slices
    if 'z_edges' in locals():
        z_labels = [f"{z_edges[i]:.1f}–{z_edges[i+1]:.1f}" for i in range(nz)]
    else:
        z_labels = [f"{zc:.2f}" for zc in z_centers]  # fallback to centers
    z_handles = [mlines.Line2D([0],[0], color=colors[i], lw=3, label=z_labels[i])
                 for i in range(nz)]
    leg_z = ax.legend(handles=z_handles, title="$z$ slice", frameon=False,
                      fontsize=fs-1, title_fontsize=fs-1, loc="upper right")

    # Legend for styles
    hmf_str = f"HMF: {hmf_label(hmf_name)}"
    style_handles = [
        mlines.Line2D([0], [0], color="white", lw=lw, label=hmf_str),
        mlines.Line2D([0],[0], color='k', lw=lw, label='forward (with scatter)'),
        mlines.Line2D([0],[0], color='k', lw=lw, ls='--', label='inverse (no scatter)'),
        mlines.Line2D([0],[0], marker='o', color='k', lw=0, markersize=lw+4, label='SRD data'),
    ]
    leg_style = ax.legend(handles=style_handles, frameon=False,
                          fontsize=fs-1, loc="lower left")
    ax.add_artist(leg_z)

    fig.tight_layout()
    plt.savefig(f"{plt_dir}binned_hmf_{year}_{hmf_name}.{file_ext}", dpi=150)
    plt.show()
    print(f"Saved to {plt_dir}")
