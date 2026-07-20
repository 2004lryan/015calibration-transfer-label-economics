"""Unit tests for 63_crossover_analysis: budget logic, n* definitions, log-log fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import load_script

ca = load_script("63_crossover_analysis.py", "crossover_analysis")

GRID = [0.0, 5.0, 10.0, 20.0, 40.0]


# ---------------------------------------------------------------- best_at_budget
def test_best_at_budget_only_spends_labels_it_has() -> None:
    mat = {"sbc": {0.0: np.nan, 5.0: 2.0, 20.0: 1.0}}
    # at a budget of 5 the 20-label result is not reachable
    assert ca.best_at_budget(mat, ["sbc"], 5.0, GRID) == pytest.approx(2.0)
    assert ca.best_at_budget(mat, ["sbc"], 20.0, GRID) == pytest.approx(1.0)


def test_zero_label_methods_are_available_at_any_budget() -> None:
    """coral / zero_shot need no target labels, so they count even at budget 0."""
    mat = {"coral": {0.0: 1.5}}
    assert ca.best_at_budget(mat, ["coral"], 0.0, GRID) == pytest.approx(1.5)


def test_best_at_budget_ignores_missing_methods_and_nans() -> None:
    mat = {"sbc": {5.0: np.nan}}
    assert np.isnan(ca.best_at_budget(mat, ["sbc", "absent_method"], 40.0, GRID))


def test_best_at_budget_takes_the_minimum_error_across_methods() -> None:
    mat = {"sbc": {5.0: 2.0}, "model_update": {5.0: 1.2}}
    assert ca.best_at_budget(mat, ["sbc", "model_update"], 5.0, GRID) == pytest.approx(1.2)


# ---------------------------------------------------------------- compute_nstar
def _task_frame(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["method", "n_cal", "rmsep"])


def test_nstar_usable_is_the_first_budget_reaching_the_rpd_threshold() -> None:
    """USABLE_RPD = 2.0, so with y_std = 2.0 a usable model needs RMSEP <= 1.0."""
    y_std = 2.0
    df = _task_frame(
        [
            ("zero_shot", 0.0, 4.0),   # RPD 0.5
            ("sbc", 5.0, 1.5),         # RPD 1.33
            ("sbc", 10.0, 0.8),        # RPD 2.5 -> first usable
            ("sbc", 20.0, 0.7),
        ]
    )
    out = ca.compute_nstar(df, y_std, GRID)
    assert out["n_usable"] == pytest.approx(10.0)


def test_nstar_usable_is_nan_when_no_budget_ever_suffices() -> None:
    df = _task_frame([("zero_shot", 0.0, 9.0), ("sbc", 40.0, 8.0)])
    out = ca.compute_nstar(df, 2.0, GRID)
    assert np.isnan(out["n_usable"])


def test_simple_suffices_triggers_once_simple_catches_the_expensive_arm() -> None:
    """The crossover budget n*: cheap correction matches the expensive arm."""
    y_std = 1.0  # tolerance = TOL_FRAC * y_std = 0.03
    df = _task_frame(
        [
            ("sbc", 5.0, 3.0), ("model_update", 5.0, 1.0),    # simple far behind
            ("sbc", 10.0, 1.0), ("model_update", 10.0, 1.0),  # caught up here
        ]
    )
    out = ca.compute_nstar(df, y_std, GRID)
    assert out["n_simple_suffices"] == pytest.approx(10.0)


def test_zero_label_coral_can_make_simple_suffice_immediately() -> None:
    """The headline instrument-drift case: CORAL wins at zero labels."""
    df = _task_frame([("coral", 0.0, 1.0), ("model_update", 0.0, 2.0)])
    out = ca.compute_nstar(df, 1.0, GRID)
    assert out["n_simple_suffices"] == pytest.approx(0.0)


def test_nstar_reports_zero_shot_and_coral_rpd() -> None:
    df = _task_frame([("zero_shot", 0.0, 4.0), ("coral", 0.0, 1.0)])
    out = ca.compute_nstar(df, 2.0, GRID)
    assert out["rpd_zeroshot"] == pytest.approx(0.5)
    assert out["rpd_coral0"] == pytest.approx(2.0)


# ---------------------------------------------------------------- loglog_fit
def test_loglog_fit_recovers_a_known_power_law() -> None:
    """y = 3 * x^2 must give exponent 2 and R^2 = 1."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = 3.0 * x**2
    coef, r2, n = ca.loglog_fit(x.reshape(-1, 1), y)
    assert coef is not None
    assert coef[0] == pytest.approx(2.0, abs=1e-8)      # exponent
    assert np.exp(coef[-1]) == pytest.approx(3.0, abs=1e-8)  # prefactor
    assert r2 == pytest.approx(1.0, abs=1e-9)
    assert n == 6


def test_loglog_fit_reports_low_r2_for_unrelated_data() -> None:
    """The paper's negative result rests on this: a low R^2 must actually show up."""
    rng = np.random.default_rng(0)
    x = rng.uniform(1.0, 100.0, size=60)
    y = rng.uniform(1.0, 100.0, size=60)   # independent of x
    _, r2, _ = ca.loglog_fit(x.reshape(-1, 1), y)
    assert r2 < 0.25


def test_loglog_fit_declines_when_too_few_valid_points() -> None:
    x = np.array([1.0, 2.0])
    coef, r2, n = ca.loglog_fit(x.reshape(-1, 1), np.array([1.0, 2.0]))
    assert coef is None
    assert np.isnan(r2)
    assert n == 2


def test_loglog_fit_drops_nonpositive_and_nonfinite_rows() -> None:
    """log-log is undefined at x <= 0 or y <= 0; those rows must be excluded."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, 7.0])
    y = np.array([3.0, 12.0, 27.0, 48.0, 75.0, 108.0, 5.0, np.nan])
    coef, r2, n = ca.loglog_fit(x.reshape(-1, 1), y)
    assert n == 6                       # the -1 and the NaN rows are gone
    assert coef is not None
    assert coef[0] == pytest.approx(2.0, abs=1e-8)
    assert r2 == pytest.approx(1.0, abs=1e-9)
