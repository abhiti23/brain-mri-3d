"""
Stage 1: fit an intensity-weighted GMM on the group-average brain with
FREE means and covariances. The resulting means become the fixed centers
used for all subsequent per-subject fits.

This is deliberately a full weighted-EM fit (not weighted k-means) so that
elongated anatomical structures can be covered by correspondingly elongated
components — means and covariances co-adapt.

Output: centers.npy (shape 500 x 3)

Where this fits in the pipeline:
  [THIS SCRIPT]            — learn 500 landmarks from average brain
  fit_weighted_gmm_single  — fit each subject with means locked to those landmarks
  reconstruct_and_compare  — validate reconstructions on a few subjects
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky


class WeightedGMM(GaussianMixture):
    """
    Full GMM where each sample carries a weight (for us: the voxel intensity).
    Unlike the fixed-means version, means here are updated each M-step.
    """

    def __init__(self, **kwargs):
        super().__init__(covariance_type='full', **kwargs)
        self._sample_weights = None

    def fit_weighted(self, X, sample_weights):
        self._sample_weights = np.asarray(sample_weights, dtype=np.float64)
        return self.fit(X)

    def _e_step(self, X):
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        w = self._sample_weights
        weighted_lb = np.sum(w * log_prob_norm) / w.sum()
        return weighted_lb, log_resp

    def _m_step(self, X, log_resp):
        n_features = X.shape[1]
        resp = np.exp(log_resp)
        w = self._sample_weights

        wresp = resp * w[:, np.newaxis]                    # rho_ik = w_i * gamma_ik
        nk = wresp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps

        # 1. Mixing weights
        self.weights_ = nk / w.sum()

        # 2. MEANS — updated here (the key difference from fixed-means version)
        self.means_ = (wresp.T @ X) / nk[:, np.newaxis]

        # 3. Covariances around the updated means
        covariances = np.empty((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = X - self.means_[k]
            covariances[k] = (wresp[:, k] * diff.T) @ diff / nk[k]
            covariances[k].flat[::n_features + 1] += self.reg_covar
        self.covariances_ = covariances
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, 'full'
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('brain', help="Path to group-average 3D .npy")
    parser.add_argument('--out', default='centers.npy',
                        help="Where to save fitted means (default: centers.npy)")
    parser.add_argument('--n-components', type=int, default=500)
    parser.add_argument('--threshold', type=float, default=0.005)
    parser.add_argument('--max-iter', type=int, default=100)
    parser.add_argument('--tol', type=float, default=1e-3)
    parser.add_argument('--save-full', default=None,
                        help="Optional: also save full model state "
                             "(means, covs, weights) to this path")
    parser.add_argument('--init', choices=['kmeans', 'random'], default='kmeans',
                        help="Initialization strategy (default kmeans)")
    args = parser.parse_args()

    print(f"[*] Loading {args.brain}")
    brain = np.squeeze(np.load(args.brain))
    mask = brain > args.threshold
    coords = np.argwhere(mask).astype(np.float64)
    intensities = brain[mask].astype(np.float64)

    print(f"    Brain shape         : {brain.shape}")
    print(f"    Suprathreshold vox  : {mask.sum()}")
    print(f"    Total intensity T   : {intensities.sum():.2f}")
    print(f"    Max voxel value     : {brain.max():.3f}"
          f"{'  (> 1 → likely modulated VBM)' if brain.max() > 1 else ''}")

    print(f"\n[*] Fitting weighted GMM with K={args.n_components}, free means...")
    start = time.time()
    model = WeightedGMM(
        n_components=args.n_components,
        random_state=42,
        max_iter=args.max_iter,
        tol=args.tol,
        init_params=args.init,
        verbose=1,
        verbose_interval=5,
    )
    model.fit_weighted(coords, intensities)
    elapsed = (time.time() - start) / 60
    print(f"    -> converged={model.converged_}  iterations={model.n_iter_}  "
          f"time={elapsed:.1f} min")

    np.save(args.out, model.means_)
    print(f"[*] Saved centers to {args.out}  (shape {model.means_.shape})")

    if args.save_full:
        np.savez(args.save_full,
                 means=model.means_,
                 covariances=model.covariances_,
                 weights=model.weights_,
                 total_intensity=intensities.sum())
        print(f"[*] Saved full model state to {args.save_full}")


if __name__ == "__main__":
    main()
