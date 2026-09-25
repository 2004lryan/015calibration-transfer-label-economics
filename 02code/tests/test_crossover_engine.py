"""Unit tests for 62_crossover_engine: preprocessing, alignment, CORAL, drift features."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import load_script

ce = load_script("62_crossover_engine.py", "crossover_engine")


# ---------------------------------------------------------------- SNV
def test_snv_standardises_each_spectrum_independently() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 2.0, size=(6, 40))
    Z = ce.snv(X)
    assert np.allclose(Z.mean(axis=1), 0.0, atol=1e-8)
    assert np.allclose(Z.std(axis=1), 1.0, atol=1e-4)


def test_snv_removes_per_spectrum_offset_and_scale() -> None:
    """SNV must be invariant to a multiplicative/additive change of one spectrum."""
    rng = np.random.default_rng(1)
    X = rng.normal(0.0, 1.0, size=(3, 30))
    Y = X.copy()
    Y[0] = 3.0 * X[0] + 7.0  # scatter-like distortion of a single sample
    assert np.allclose(ce.snv(X)[0], ce.snv(Y)[0], atol=1e-6)


def test_snv_does_not_divide_by_zero_on_a_flat_spectrum() -> None:
    out = ce.snv(np.ones((1, 10)))
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------- RMSE
def test_rmse_is_zero_for_identical_vectors() -> None:
    a = np.array([1.0, 2.0, 3.0])
    assert ce.rmse(a, a) == pytest.approx(0.0)


def test_rmse_matches_the_closed_form() -> None:
    a, b = np.array([1.0, 2.0]), np.array([2.0, 4.0])
    assert ce.rmse(a, b) == pytest.approx(np.sqrt((1.0 + 4.0) / 2))


# ---------------------------------------------------------------- wavelength alignment
def test_align_common_is_a_noop_when_grids_already_match() -> None:
    Xs, Xt = np.zeros((2, 5)), np.ones((3, 5))
    wl = np.arange(5.0)
    out_s, out_t = ce.align_common(Xs, wl, Xt, wl)
    assert out_s.shape == (2, 5)
    assert out_t.shape == (3, 5)


def test_align_common_maps_onto_the_shorter_grid() -> None:
    """The apple benchmark mixes a 1213-channel and a 229-channel instrument."""
    wl_short = np.array([600.0, 700.0, 800.0])
    wl_long = np.array([600.0, 650.0, 700.0, 750.0, 800.0])
    Xs = np.zeros((4, 3))                      # already on the short grid
    Xt = np.arange(5.0).reshape(1, 5)          # on the long grid
    out_s, out_t = ce.align_common(Xs, wl_short, Xt, wl_long)
    assert out_s.shape[1] == out_t.shape[1] == 3
    # nearest-wavelength selection picks long-grid columns 0, 2, 4
    assert out_t[0].tolist() == [0.0, 2.0, 4.0]


def test_align_common_is_symmetric_in_which_side_is_longer() -> None:
    wl_short = np.array([600.0, 700.0, 800.0])
    wl_long = np.array([600.0, 650.0, 700.0, 750.0, 800.0])
    Xlong = np.arange(5.0).reshape(1, 5)
    Xshort = np.zeros((2, 3))
    a, b = ce.align_common(Xlong, wl_long, Xshort, wl_short)
    assert a.shape[1] == b.shape[1] == 3
    assert a[0].tolist() == [0.0, 2.0, 4.0]


# ---------------------------------------------------------------- CORAL
def test_coral_moves_target_moments_towards_the_source() -> None:
    rng = np.random.default_rng(2)
    Xs = rng.normal(0.0, 1.0, size=(200, 4))
    Xt = rng.normal(3.0, 2.0, size=(200, 4))          # shifted and rescaled
    Xt_aligned = ce.coral(Xs, Xt)
    before = np.linalg.norm(Xt.mean(0) - Xs.mean(0))
    after = np.linalg.norm(Xt_aligned.mean(0) - Xs.mean(0))
    assert after < before
    # second-order moments must also come closer
    cov_before = np.linalg.norm(np.cov(Xt, rowvar=False) - np.cov(Xs, rowvar=False))
    cov_after = np.linalg.norm(np.cov(Xt_aligned, rowvar=False) - np.cov(Xs, rowvar=False))
    assert cov_after < cov_before


def test_coral_preserves_shape_and_stays_finite() -> None:
    rng = np.random.default_rng(3)
    Xs, Xt = rng.normal(size=(50, 6)), rng.normal(size=(30, 6))
    out = ce.coral(Xs, Xt)
    assert out.shape == (30, 6)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------- drift features
def test_mmd_is_near_zero_for_two_samples_of_one_distribution() -> None:
    rng = np.random.default_rng(4)
    Xs, Xt = rng.normal(size=(80, 3)), rng.normal(size=(80, 3))
    assert abs(ce.mmd2_rbf(Xs, Xt, rng)) < 0.05


def test_mmd_grows_with_separation() -> None:
    rng = np.random.default_rng(5)
    Xs = rng.normal(0.0, 1.0, size=(80, 3))
    near = ce.mmd2_rbf(Xs, rng.normal(0.5, 1.0, size=(80, 3)), rng)
    far = ce.mmd2_rbf(Xs, rng.normal(6.0, 1.0, size=(80, 3)), rng)
    assert far > near


def test_sliced_w1_is_near_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(6)
    X = rng.normal(size=(200, 4))
    assert ce.sliced_w1(X, X.copy(), rng) == pytest.approx(0.0, abs=1e-9)


def test_sliced_w1_recovers_a_known_mean_shift() -> None:
    """For a pure translation the sliced W1 distance approximates the shift size."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(400, 2))
    shifted = X + 2.0
    # projections are unit-norm, so a shift of 2 along both axes projects to <= 2*sqrt(2)
    d = ce.sliced_w1(X, shifted, rng)
    assert 1.0 < d < 2.0 * np.sqrt(2) + 0.2


def test_pca_dim_is_one_for_a_rank_one_matrix() -> None:
    """participation ratio = 1 when all variance sits on a single component."""
    base = np.linspace(0.0, 1.0, 12)
    X = np.outer(np.arange(1.0, 21.0), base)
    assert ce.pca_dim(X) == pytest.approx(1.0, abs=1e-6)


def test_pca_dim_approaches_the_rank_for_isotropic_data() -> None:
    rng = np.random.default_rng(8)
    X = rng.normal(size=(2000, 5))
    assert 4.0 < ce.pca_dim(X) <= 5.0


def test_task_features_exposes_the_documented_keys() -> None:
    rng = np.random.default_rng(9)
    Xs, Xt = rng.normal(size=(40, 5)), rng.normal(size=(35, 5))
    ys, yt = rng.normal(size=40), rng.normal(size=35)
    feats = ce.task_features(Xs, Xt, ys, yt, rng)
    for key in ("mmd", "sliced_w1", "pca_dim_src", "pca_dim_tgt", "y_std_src", "n_src", "n_tgt"):
        assert key in feats
    assert feats["n_src"] == 40
    assert feats["n_tgt"] == 35
