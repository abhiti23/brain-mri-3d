"""
Plot original vs reconstructed brain (and their difference) across three
anatomical views. Pure visualization — no fitting, no rescaling.

Inputs: pre-computed .npy arrays from the pipeline:
  - original:      group_average_gm_T1w.npy
  - reconstructed: produced by reconstruct_from_saved_model.py --save-recon

Outputs: three PNGs (sagittal, coronal, axial), each a 3x3 grid:
    row 1: original
    row 2: reconstruction (same intensity scale)
    row 3: difference (diverging scale)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_view(original, recon, axis, view_name, filename,
              n_slices=3, slice_fracs=(0.35, 0.5, 0.65)):
    dim_size = original.shape[axis]
    slices_to_plot = [int(dim_size * f) for f in slice_fracs[:n_slices]]

    # Shared intensity scale for original and recon (honest comparison)
    vmax = max(original.max(), recon.max())
    # Diff scale — symmetric, driven by larger of abs errors
    diff = recon - original
    dmax = np.abs(diff).max() * 0.6  # 0.6 is a cosmetic saturation nudge

    fig, axes = plt.subplots(3, n_slices, figsize=(4 * n_slices, 11))
    fig.suptitle(f"Original vs Reconstruction — {view_name}", fontsize=14)

    for i, idx in enumerate(slices_to_plot):
        if axis == 0:
            s_orig, s_recon, s_diff = original[idx], recon[idx], diff[idx]
        elif axis == 1:
            s_orig, s_recon, s_diff = original[:, idx], recon[:, idx], diff[:, idx]
        else:
            s_orig, s_recon, s_diff = original[:, :, idx], recon[:, :, idx], diff[:, :, idx]

        s_orig, s_recon, s_diff = s_orig.T, s_recon.T, s_diff.T

        im0 = axes[0, i].imshow(s_orig, cmap='magma', origin='lower', vmin=0, vmax=vmax)
        axes[0, i].set_title(f'Original (slice {idx})')
        axes[0, i].axis('off')

        axes[1, i].imshow(s_recon, cmap='magma', origin='lower', vmin=0, vmax=vmax)
        axes[1, i].set_title(f'Reconstruction (slice {idx})')
        axes[1, i].axis('off')

        im2 = axes[2, i].imshow(s_diff, cmap='RdBu_r', origin='lower',
                                vmin=-dmax, vmax=dmax)
        axes[2, i].set_title(f'Recon − Original (slice {idx})')
        axes[2, i].axis('off')

    # Two colorbars: intensity (rows 1-2) and diff (row 3)
    cbar1 = fig.add_axes([0.25, 0.055, 0.2, 0.012])
    fig.colorbar(im0, cax=cbar1, orientation='horizontal', label='Intensity')
    cbar2 = fig.add_axes([0.55, 0.055, 0.2, 0.012])
    fig.colorbar(im2, cax=cbar2, orientation='horizontal', label='Residual')

    plt.tight_layout(rect=[0, 0.09, 1, 0.96])
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"   -> Saved {filename}")


def plot_scatter_and_histograms(original, recon, threshold, filename):
    """Pixel-wise scatter + residual histogram — diagnostic of systematic error."""
    mask = original > threshold
    a = original[mask]
    b = recon[mask]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Scatter
    # Subsample to keep figure reasonable
    n = len(a)
    if n > 50_000:
        idx = np.random.RandomState(0).choice(n, 50_000, replace=False)
        a_plot, b_plot = a[idx], b[idx]
    else:
        a_plot, b_plot = a, b
    axes[0].scatter(a_plot, b_plot, s=1, alpha=0.1)
    lims = [0, max(a.max(), b.max())]
    axes[0].plot(lims, lims, 'r--', lw=1, label='y = x')
    axes[0].set_xlabel('Original intensity')
    axes[0].set_ylabel('Reconstructed intensity')
    axes[0].set_title(f'Voxel-wise agreement\nPearson r = {np.corrcoef(a, b)[0,1]:.4f}')
    axes[0].legend()

    # Residual histogram
    resid = b - a
    axes[1].hist(resid, bins=80, color='gray', edgecolor='black')
    axes[1].axvline(0, color='red', lw=1)
    axes[1].set_xlabel('Reconstruction − Original')
    axes[1].set_ylabel('Voxel count')
    axes[1].set_title(f'Residuals\nmean={resid.mean():.4f}  std={resid.std():.4f}')

    # Intensity histograms overlaid
    axes[2].hist(a, bins=80, alpha=0.5, label='original', color='steelblue')
    axes[2].hist(b, bins=80, alpha=0.5, label='reconstruction', color='darkorange')
    axes[2].set_xlabel('Intensity')
    axes[2].set_ylabel('Voxel count')
    axes[2].set_title('Intensity distributions')
    axes[2].legend()
    axes[2].set_yscale('log')

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"   -> Saved {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('original', help="Original brain .npy")
    parser.add_argument('reconstruction', help="Reconstructed brain .npy "
                                               "(from reconstruct_from_saved_model.py)")
    parser.add_argument('--threshold', type=float, default=0.005)
    parser.add_argument('--prefix', default='recon',
                        help="Output filename prefix (default: recon)")
    args = parser.parse_args()

    print(f"[*] Loading {args.original}")
    original = np.squeeze(np.load(args.original))
    print(f"[*] Loading {args.reconstruction}")
    recon = np.load(args.reconstruction)

    assert original.shape == recon.shape, \
        f"Shape mismatch: {original.shape} vs {recon.shape}"

    # Quick numerical summary
    mask = original > args.threshold
    print(f"\n    Original:        sum={original.sum():.2f}  max={original.max():.4f}")
    print(f"    Reconstruction:  sum={recon.sum():.2f}  max={recon.max():.4f}")
    print(f"    Mass ratio:      {recon.sum() / original.sum():.4f} (should be ~1.0)")
    print(f"    Pearson (mask):  {np.corrcoef(original[mask], recon[mask])[0,1]:.4f}")

    print("\n[*] Generating plots...")
    plot_view(original, recon, axis=0, view_name='Sagittal',
              filename=f'{args.prefix}_sagittal.png')
    plot_view(original, recon, axis=1, view_name='Coronal',
              filename=f'{args.prefix}_coronal.png')
    plot_view(original, recon, axis=2, view_name='Axial',
              filename=f'{args.prefix}_axial.png')
    plot_scatter_and_histograms(original, recon, args.threshold,
                                filename=f'{args.prefix}_diagnostics.png')


if __name__ == "__main__":
    main()
