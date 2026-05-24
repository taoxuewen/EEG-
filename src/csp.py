"""
Common Spatial Patterns (CSP) algorithm for motor imagery EEG.
"""

import numpy as np
from scipy.linalg import eigh


class CSP:
    """
    Common Spatial Patterns for two-class motor imagery classification.

    Finds spatial filters that maximize the variance difference between
    two classes by simultaneous diagonalization of covariance matrices.
    """

    def __init__(self, n_components=4):
        """
        Parameters
        ----------
        n_components : int
            Number of CSP components (must be even). Half from each
            end of the eigenvalue spectrum.
        """
        if n_components % 2 != 0:
            raise ValueError("n_components must be even")
        self.n_components = n_components
        self.filters_ = None
        self.patterns_ = None
        self.eigvals_ = None
        self.ch_names = None

    def fit(self, X, y):
        """
        Fit CSP spatial filters.

        Parameters
        ----------
        X : ndarray (n_trials, n_channels, n_times)
        y : ndarray (n_trials,) binary labels
        """
        n_trials, n_channels, _ = X.shape
        labels = np.unique(y)
        if len(labels) != 2:
            raise ValueError(f"CSP requires exactly 2 classes, got {len(labels)}")

        idx_a = np.where(y == labels[0])[0]
        idx_b = np.where(y == labels[1])[0]

        # Average covariance matrices per class
        cov_a = np.mean([np.cov(X[i]) for i in idx_a], axis=0)
        cov_b = np.mean([np.cov(X[i]) for i in idx_b], axis=0)

        # Composite covariance
        cov_comp = cov_a + cov_b

        # Eigenvalue decomposition of composite covariance
        evals, evecs = eigh(cov_comp)

        # Sort descending
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

        # Whitening transform: W @ cov_comp @ W.T = I
        D_inv_sqrt = np.diag(1.0 / np.sqrt(evals))
        W = D_inv_sqrt @ evecs.T

        # Whiten class-specific covariances
        S_a = W @ cov_a @ W.T
        S_b = W @ cov_b @ W.T

        # Eigendecomposition of S_a (S_b shares eigenvectors with complementary eigenvalues)
        evals_s, evecs_s = eigh(S_a)

        # Sort descending by eigenvalue (largest -> class a dominates, smallest -> class b dominates)
        order_s = np.argsort(evals_s)[::-1]
        evals_s = evals_s[order_s]
        evecs_s = evecs_s[:, order_s]

        # Full set of spatial filters: each row is a filter
        A = evecs_s.T @ W  # (n_channels, n_channels)

        # Spatial patterns = inverse of filter matrix
        self.patterns_ = np.linalg.inv(A).T  # (n_channels, n_channels)

        # Select most discriminative filters: m largest + m smallest eigenvalues
        m = self.n_components // 2
        selected = list(range(m)) + list(range(n_channels - m, n_channels))
        self.filters_ = A[selected]  # (n_components, n_channels)
        self.eigvals_ = evals_s

        return self

    def transform(self, X):
        """
        Project data through CSP filters and compute log-variance features.

        Parameters
        ----------
        X : ndarray (n_trials, n_channels, n_times)

        Returns
        -------
        features : ndarray (n_trials, n_components)
        """
        if self.filters_ is None:
            raise RuntimeError("CSP must be fitted before transform.")

        n_trials = X.shape[0]
        Z = np.array([self.filters_ @ X[i] for i in range(n_trials)])
        features = np.log(np.var(Z, axis=2) + 1e-10)
        return features

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_spatial_patterns(self):
        """Return spatial patterns (columns = components)."""
        if self.patterns_ is None:
            raise RuntimeError("CSP must be fitted first.")
        m = self.n_components // 2
        n_ch = self.patterns_.shape[0]
        selected = list(range(m)) + list(range(n_ch - m, n_ch))
        return self.patterns_[:, selected]
