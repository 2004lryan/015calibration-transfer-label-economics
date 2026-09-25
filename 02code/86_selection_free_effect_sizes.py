#!/usr/bin/env python3
"""86_selection_free_effect_sizes.py — 免选择比较（表 5）的效应量，两种口径并排算。

起因：审稿面板指出表 4 与表 5 都把效应量列叫 "Cliff's $\\delta$"，但两者算的不是同一个量——
表 4（脚本 65）用的是**非配对**优势度（跨族全部任务对上
$P(\\text{deep}>\\text{simple})-P(\\text{deep}<\\text{simple})$），
表 5（脚本 67/68）用的是**配对符号优势度**（$2p-1$，$p$ 为简单族更优的任务比例），后者恒等于
"Simple-win %" 的线性变换。两者在同一篇里同名不同义，且正文曾拿表 4 的 $-0.09$ 与表 5 的
$+0.28$ 直接对比，那是两种统计量之间的对比，会把结论说得比证据强。

本脚本从 62/64 号原始曲线重算表 5 的每一行，同时给出两种效应量，供正文统一到**非配对**口径
（与表 4 一致），配对信息则由 "Simple-win %" 列承载。

用法::

    python3 02code/86_selection_free_effect_sizes.py            # 打印全网格
    python3 02code/86_selection_free_effect_sizes.py --tex      # 只打印进表 5 的行（LaTeX 片段）
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import wilcoxon

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "04outputs")
SIMPLE = {"coral", "sbc"}
DEEP = {"cnn_zeroshot", "cnn_finetune", "dann", "deepcoral", "physbl_zeroshot", "physbl_ft"}
SHIFT_LABEL = {"instrument": "Instrument", "season": "Season",
               "origin_year_instrument": "Origin/year", "lab": "Lab (MIR)"}
F64 = npt.NDArray[np.float64]


def _load() -> pd.DataFrame:
    cl = pd.read_excel(os.path.join(_OUT, "62_crossover_engine.xlsx"), sheet_name="curves")
    dp = pd.read_excel(os.path.join(_OUT, "64_deep_transfer_server.xlsx"), sheet_name="curves")
    allr = pd.concat([cl, dp], ignore_index=True)
    med = allr.groupby(["task_id", "shift_type", "method", "n_cal"], as_index=False)["rmsep"].median()
    ns = pd.read_excel(os.path.join(_OUT, "63_crossover_analysis.xlsx"), sheet_name="nstar_per_task")
    med = med.merge(ns[["task_id", "y_std_tgt"]].drop_duplicates("task_id"), on="task_id", how="left")
    med["nrmsep"] = med["rmsep"] / med["y_std_tgt"]
    return med


def cliff_unpaired(simple: F64, deep: F64) -> float:
    """表 4 口径：跨族全部任务对上的非配对优势度，正值表示简单族误差更小。"""
    diff = deep[:, None] - simple[None, :]
    gt = float(np.count_nonzero(diff > 0))
    lt = float(np.count_nonzero(diff < 0))
    return (gt - lt) / (len(deep) * len(simple))


def dominance_paired(simple: F64, deep: F64) -> float:
    """配对符号优势度：简单族更优的任务比例减去更差的比例（恒等于 2p-1）。"""
    diff = deep - simple
    wins = int(np.count_nonzero(diff > 0))
    losses = int(np.count_nonzero(diff < 0))
    return (wins - losses) / len(diff)


def rows(med: pd.DataFrame) -> pd.DataFrame:
    out = []
    for shift in SHIFT_LABEL:
        for n in (0, 5, 10, 20, 40):
            at = med[(med["shift_type"] == shift) & (med["n_cal"] == n)]
            if at.empty:
                continue
            for kind in ("representative pair", "family mean"):
                if kind == "representative pair":
                    if n != 0:
                        continue  # 代表性配对（CORAL vs DeepCORAL）只在零标签下二者都免标签
                    s = at[at["method"] == "coral"].set_index("task_id")[["rmsep", "nrmsep"]]
                    d = at[at["method"] == "deepcoral"].set_index("task_id")[["rmsep", "nrmsep"]]
                else:
                    s = at[at["method"].isin(SIMPLE)].groupby("task_id")[["rmsep", "nrmsep"]].mean()
                    d = at[at["method"].isin(DEEP)].groupby("task_id")[["rmsep", "nrmsep"]].mean()
                idx = s.index.intersection(d.index)
                if len(idx) < 3:
                    continue
                s, d = s.loc[idx], d.loc[idx]
                sv, dv = s["rmsep"].to_numpy(), d["rmsep"].to_numpy()
                p = float(wilcoxon(sv, dv).pvalue)
                out.append({
                    "comparison": kind, "shift": SHIFT_LABEL[shift], "n": n, "tasks": len(idx),
                    "rmsep_s": round(float(np.median(sv)), 3), "rmsep_d": round(float(np.median(dv)), 3),
                    "nrmsep_s": round(float(np.median(s["nrmsep"])), 3),
                    "nrmsep_d": round(float(np.median(d["nrmsep"])), 3),
                    "simple_win_pct": round(float((sv < dv).mean()) * 100),
                    "delta_unpaired": round(cliff_unpaired(sv, dv), 2),
                    "delta_unpaired_nrmsep": round(
                        cliff_unpaired(s["nrmsep"].to_numpy(), d["nrmsep"].to_numpy()), 2),
                    "delta_paired": round(dominance_paired(sv, dv), 2),
                    "wilcoxon_p": p,
                })
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", action="store_true", help="只打印稿件表 5 收录的行")
    args = ap.parse_args()
    df = rows(_load())
    keep = [("representative pair", n) for n in (0,)] + [("family mean", n) for n in (0, 20)]
    view = df[[(c, n) in keep for c, n in zip(df["comparison"], df["n"], strict=True)]] if args.tex else df
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(view.to_string(index=False))
    print("\n注：delta_unpaired 为表 4 口径（非配对，跨族任务对）；delta_paired 为 2p-1，"
          "与 simple_win_pct 线性等价。正文与表 5 统一取 delta_unpaired。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
