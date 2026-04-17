"""
Fit an intensity-weighted, fixed-means GMM on a single 3D MRI volume.

Output: per-component [pi_k * T, c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]
where T = sum(intensities) over the suprathreshold mask.

Multiplying pi_k by T recovers absolute regional GM mass, so the simplex
constraint no longer throws away global volume.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky


class IntensityWeightedFixedMeansGMM(GaussianMixture):
    def __init__(self, fixed_means, **kwargs):
        super().__init__(covariance_type='full', **kwargs)
        self.fixed_means = np.asarray(fixed_means, dtype=np.float64)
        self._sample_weights = None

    def fit_weighted(self, X, sample_weights):
        self._sample_weights = np.asarray(sample_weights, dtype=np.float64)
        return self.fit(X)

    def _initialize_parameters(self, X, random_state):
        super()._initialize_parameters(X, random_state)
        self.means_ = self.fixed_means.copy()

    def _e_step(self, X):
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        w = self._sample_weights
        weighted_lb = np.sum(w * log_prob_norm) / w.sum()
        return weighted_lb, log_resp

    def _m_step(self, X, log_resp):
        n_features = X.shape[1]
        resp = np.exp(log_resp)
        w = self._sample_weights
        wresp = resp * w[:, np.newaxis]
        nk = wresp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps

        self.weights_ = nk / w.sum()          # pi_k (sums to 1)
        self.means_ = self.fixed_means.copy()

        covariances = np.empty((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = X - self.means_[k]
            covariances[k] = (wresp[:, k] * diff.T) @ diff / nk[k]
            covariances[k].flat[::n_features + 1] += self.reg_covar
        self.covariances_ = covariances
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, 'full'
        )


def fit_brain(brain_path, centers_path, n_components=500, threshold=0.005,
              output_path=None):
    brain = np.squeeze(np.load(brain_path))
    mask = brain > threshold
    coords = np.argwhere(mask).astype(np.float64)
    intensities = brain[mask].astype(np.float64)
    T = intensities.sum()

    fixed_means = np.load(centers_path)
    assert fixed_means.shape[0] == n_components, \
        f"Centers file has {fixed_means.shape[0]} components, expected {n_components}"

    model = IntensityWeightedFixedMeansGMM(
        fixed_means=fixed_means,
        n_components=n_components,
        random_state=42,
    )
    model.fit_weighted(coords, intensities)

    # Pack: [pi_k * T, cxx, cxy, cxz, cyy, cyz, czz] per component
    params = []
    for k in range(n_components):
        scaled_weight = model.weights_[k] * T
        cov = model.covariances_[k]
        params.extend([
            scaled_weight,
            cov[0, 0], cov[0, 1], cov[0, 2],
            cov[1, 1], cov[1, 2],
            cov[2, 2],
        ])
    params = np.array(params)

    if output_path:
        np.save(output_path, params)
        print(f"[*] Saved {len(params)} features to {output_path}")

    print(f"    Total intensity T = {T:.2f}")
    print(f"    Mask voxels       = {mask.sum()}")
    print(f"    Brain shape       = {brain.shape}")

    return {
        "weights_pi": model.weights_,
        "weights_scaled": model.weights_ * T,
        "means": model.means_,
        "covariances": model.covariances_,
        "total_intensity": T,
        "brain_shape": brain.shape,
        "threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('brain', help="Path to 3D .npy brain volume")
    parser.add_argument('centers', help="Path to fixed means .npy file")
    parser.add_argument('--out', default='fit_params.npy')
    parser.add_argument('--n-components', type=int, default=500)
    parser.add_argument('--threshold', type=float, default=0.005)
    args = parser.parse_args()

    fit_brain(args.brain, args.centers, args.n_components,
              args.threshold, args.out)


if __name__ == "__main__":
    main()
