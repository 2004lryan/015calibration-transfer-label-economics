"""Unit tests for 65_paper_analysis: effect size, clustered bootstrap CI, task aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import load_script

pa = load_script("65_paper_analysis.py", "paper_analysis")


# ---------------------------------------------------------------- Cliff's delta
def test_cliffs_delta_is_plus_one_when_a_dominates_b() -> None:
    assert pa.cliffs_delta([5.0, 6.0, 7.0], [1.0, 2.0]) == pytest.approx(1.0, abs=1e-6)


def test_cliffs_delta_is_minus_one_when_b_dominates_a() -> None:
    assert pa.cliffs_delta([1.0, 2.0], [5.0, 6.0, 7.0]) == pytest.approx(-1.0, abs=1e-6)


def test_cliffs_delta_is_zero_for_identical_samples() -> None:
    x = [1.0, 2.0, 3.0]
    assert pa.cliffs_delta(x, list(x)) == pytest.approx(0.0, abs=1e-6)


def test_cliffs_delta_is_antisymmetric() -> None:
    a, b = [1.0, 4.0, 6.0], [2.0, 3.0, 9.0]
    assert pa.cliffs_delta(a, b) == pytest.approx(-pa.cliffs_delta(b, a), abs=1e-9)


def test_cliffs_delta_stays_within_the_unit_interval() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=25), rng.normal(1.0, size=25)
    assert -1.0 <= pa.cliffs_delta(a, b) <= 1.0


# ---------------------------------------------------------------- cluster bootstrap CI
def test_cluster_boot_ci_brackets_the_median() -> None:
    rng = np.random.default_rng(1)
    vals = rng.normal(5.0, 1.0, size=200)
    lo, hi = pa.cluster_boot_ci(vals, n=400)
    assert lo < np.median(vals) < hi


def test_cluster_boot_ci_is_degenerate_for_a_constant_sample() -> None:
    lo, hi = pa.cluster_boot_ci([2.0] * 30, n=200)
    assert lo == pytest.approx(2.0)
    assert hi == pytest.approx(2.0)


def test_cluster_boot_ci_returns_nan_below_three_values() -> None:
    lo, hi = pa.cluster_boot_ci([1.0, 2.0], n=100)
    assert np.isnan(lo)
    assert np.isnan(hi)


def test_cluster_boot_ci_ignores_non_finite_entries() -> None:
    lo, hi = pa.cluster_boot_ci([1.0, np.nan, 1.0, np.inf, 1.0, 1.0], n=200)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_cluster_boot_ci_narrows_as_the_sample_grows() -> None:
    rng = np.random.default_rng(2)
    small = pa.cluster_boot_ci(rng.normal(0.0, 1.0, size=15), n=800)
    large = pa.cluster_boot_ci(rng.normal(0.0, 1.0, size=600), n=800)
    assert (large[1] - large[0]) < (small[1] - small[0])


# ---------------------------------------------------------------- task aggregation
def test_task_median_collapses_seeds_to_a_median_per_cell() -> None:
    df = pd.DataFrame(
        {
            "benchmark": ["corn"] * 3,
            "shift_type": ["instrument"] * 3,
            "task_id": ["t1"] * 3,
            "method": ["coral"] * 3,
            "n_cal": [0.0] * 3,
            "seed": [1, 2, 3],
            "rmsep": [1.0, 2.0, 9.0],
        }
    )
    out = pa.task_median(df)
    assert len(out) == 1
    # median, not mean: the outlying seed must not drag the summary
    assert out["rmsep"].iloc[0] == pytest.approx(2.0)


def test_task_median_keeps_distinct_cells_separate() -> None:
    df = pd.DataFrame(
        {
            "benchmark": ["corn", "corn", "mango"],
            "shift_type": ["instrument", "instrument", "season"],
            "task_id": ["t1", "t1", "t2"],
            "method": ["coral", "sbc", "coral"],
            "n_cal": [0.0, 5.0, 0.0],
            "rmsep": [1.0, 2.0, 3.0],
        }
    )
    out = pa.task_median(df)
    assert len(out) == 3
    assert set(out["method"]) == {"coral", "sbc"}
