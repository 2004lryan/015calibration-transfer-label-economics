"""Unit tests for export_utils: pure helpers and LLH-GBID mathematical identities."""

from __future__ import annotations

import hashlib
import os
import tempfile
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from conftest import load_script

eu = load_script("01_export_utils.py", "export_utils")


# ---------------------------------------------------------------- utility mapping
def test_utility_benefit_is_linear_between_baseline_and_ideal() -> None:
    out = eu.compute_utility(np.array([0.5]), I=1.0, L=0.0, direction="benefit")
    assert out["u"][0] == pytest.approx(0.5)


def test_utility_clips_outside_the_unit_interval() -> None:
    out = eu.compute_utility(np.array([2.0, -1.0]), I=1.0, L=0.0, direction="benefit")
    assert out["u"].tolist() == [1.0, 0.0]
    # the pre-clip values are retained for diagnostics
    assert out["u_tilde"].tolist() == [2.0, -1.0]


def test_utility_cost_direction_inverts_the_scale() -> None:
    # cost metric: ideal = 0, baseline = 0.05 (e.g. calibration error)
    out = eu.compute_utility(np.array([0.0, 0.05]), I=0.0, L=0.05, direction="cost")
    assert out["u"].tolist() == pytest.approx([1.0, 0.0])


def test_utility_rejects_a_degenerate_metric() -> None:
    with pytest.raises(ValueError, match="退化"):
        eu.compute_utility(np.array([0.5]), I=1.0, L=1.0)


def test_utility_rejects_an_unknown_direction() -> None:
    with pytest.raises(ValueError, match="direction"):
        eu.compute_utility(np.array([0.5]), I=1.0, L=0.0, direction="sideways")


# ---------------------------------------------------------------- GBID
def _equal_weights(
    dim_groups: list[list[int]],
) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]]:
    # cast: numpy 2.2 types `np.full(int, ...)` as the 1-D `ndarray[tuple[int], ...]`,
    # which is invariant against the general `npt.NDArray` shape parameter.
    W = cast("npt.NDArray[np.float64]", np.full(len(dim_groups), 1.0 / len(dim_groups), dtype=np.float64))
    v = [cast("npt.NDArray[np.float64]", np.full(len(g), 1.0 / len(g), dtype=np.float64)) for g in dim_groups]
    return W, v


def test_gbid_of_a_perfect_model_is_zero_and_score_is_100() -> None:
    groups = [[0], [1, 2]]
    W, v = _equal_weights(groups)
    res = eu.compute_gbid(np.ones((1, 3)), W, v, [np.array(g) for g in groups])
    assert float(res["gbid"][0]) == pytest.approx(0.0, abs=1e-12)
    assert float(res["gbids"][0]) == pytest.approx(100.0)


def test_gbid_decomposition_identity_at_q_equals_two() -> None:
    """§15.2 mandates GBID² = (1-ū)² + C_B² + C_W² when q_B = q_W = 2."""
    groups = [[0, 1], [2, 3]]
    W, v = _equal_weights(groups)
    u = np.array([[0.9, 0.6, 0.3, 0.8]])
    res = eu.compute_gbid(u, W, v, [np.array(g) for g in groups], q_B=2.0, q_W=2.0)
    lhs = float(res["gbid"][0]) ** 2
    rhs = (
        (1.0 - float(res["u_bar"][0])) ** 2
        + float(res["C_B"][0]) ** 2
        + float(res["C_W"][0]) ** 2
    )
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_gbid_is_monotone_in_utility() -> None:
    """Uniformly better utilities must not produce a worse composite score."""
    groups = [[0, 1]]
    W, v = _equal_weights(groups)
    u = np.array([[0.4, 0.5], [0.6, 0.7]])
    res = eu.compute_gbid(u, W, v, [np.array(g) for g in groups])
    assert float(res["gbid"][1]) < float(res["gbid"][0])
    assert float(res["gbids"][1]) > float(res["gbids"][0])


def test_gbid_rejects_weights_that_do_not_sum_to_one() -> None:
    groups = [np.array([0]), np.array([1, 2])]
    v = [np.array([1.0]), np.array([0.5, 0.5])]
    with pytest.raises(ValueError, match="之和必须为 1"):
        eu.compute_gbid(np.ones((1, 3)), np.array([1.0, 1.0]), v, groups)


def test_gbid_rejects_q_below_one() -> None:
    groups = [np.array([0, 1])]
    with pytest.raises(ValueError, match="≥ 1"):
        eu.compute_gbid(np.ones((1, 2)), np.array([1.0]), [np.array([0.5, 0.5])], groups, q_B=0.5)


def test_shortfall_sensitivity_penalises_imbalance() -> None:
    """Raising q_W must not reward a model that is uneven within a dimension."""
    groups = [np.array([0, 1])]
    W, v = np.array([1.0]), [np.array([0.5, 0.5])]
    uneven = np.array([[1.0, 0.2]])
    even = np.array([[0.6, 0.6]])  # identical mean utility
    g_uneven_2 = float(eu.compute_gbid(uneven, W, v, groups, q_W=2.0)["gbid"][0])
    g_even_2 = float(eu.compute_gbid(even, W, v, groups, q_W=2.0)["gbid"][0])
    assert g_uneven_2 > g_even_2


# ---------------------------------------------------------------- misc helpers
def test_multiple_comparison_correction_stays_in_range_and_is_monotone() -> None:
    out = eu.apply_multi_comp_correction([0.01, 0.02, 0.03], method="holm")
    adj = out["p_corrected"]
    assert all(0.0 <= p <= 1.0 for p in adj)
    # a correction may only ever increase a p-value
    assert all(a >= r for a, r in zip(adj, out["p_raw"], strict=True))
    assert len(out["rejected"]) == 3


def test_bh_is_less_conservative_than_holm() -> None:
    p = [0.01, 0.2, 0.5]
    holm = eu.apply_multi_comp_correction(p, method="holm")["p_corrected"]
    bh = eu.apply_multi_comp_correction(p, method="bh")["p_corrected"]
    assert all(b <= h + 1e-12 for b, h in zip(bh, holm, strict=True))


def test_multiple_comparison_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="未知 method"):
        eu.apply_multi_comp_correction([0.01], method="bonferroni")


def test_sheet_name_strips_illegal_characters_and_truncates() -> None:
    name = eu.table_sheet_name(0, "a/b[c]d:e*f?g" + "x" * 40)
    assert len(name) <= 31
    assert not set(name) & set('[]:*?/\\')


def test_sha256_matches_hashlib() -> None:
    with tempfile.NamedTemporaryFile("wb", delete=False) as fh:
        fh.write(b"calibration-transfer")
        path = fh.name
    try:
        assert eu.compute_sha256(path) == hashlib.sha256(b"calibration-transfer").hexdigest()
    finally:
        os.unlink(path)
