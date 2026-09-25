#!/usr/bin/env python3
"""87_prescription_paired_and_deep_sanity.py — 两项被审稿面板点名缺失的检验。

**(1) 处方之间的配对比较。** 表 6 只给各处方各自的中位遗憾与区间，读者无法判断"最简单可用
校正"相对亚军（始终模型更新）的优势是否真的被数据分开——两者区间重叠。既然所有处方都在
**同一批任务–预算单元**上评分，配对比较是免费的：本脚本按任务聚类 bootstrap 给出配对差
（处方 A 减处方 B 的逐单元遗憾差的中位数）及其 95\\% 区间。只在两条处方都有定义的预算上比
（模型更新等需要标签，n=0 无定义），并如实报告区间是否跨零。

**(2) 深度基线是否"根本没学会"。** 表 4/5 的 NRMSEP$_\\mathrm{D}$ 几乎都在 1.0 附近，
NRMSEP $=1$ 恰是"预测目标域均值"的误差水平，因此有理由怀疑深度模型在目标域根本没有有效预测。
本脚本给出深度族逐任务 NRMSEP 的分布：低于 0.8 / 0.9 的任务占比，以及各基准的最小值——
若存在相当一批任务显著低于 1，则"从未学会"不成立，深度模型是**在部分任务上有效、在多数任务
上不敌简单校正**，这两种说法的科学含义完全不同。

用法::

    python3 02code/87_prescription_paired_and_deep_sanity.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "04outputs")
DEEP = ["cnn_zeroshot", "cnn_finetune", "dann", "deepcoral", "physbl_zeroshot", "physbl_ft"]
SIMPLE = ["coral", "sbc"]
NBOOT = 2000
RNG = np.random.default_rng(20060515)


def prescription_pairs() -> None:
    pt = pd.read_excel(os.path.join(_OUT, "73_heuristic_map_lobo.xlsx"), sheet_name="per_task")
    # 与表 6、图 3b 取同一个方法宇宙 fig4（各预算只含该预算下有曲线的方法）。
    # 原来这里取 extended（免标签方法的零标签结果顺延到各预算），而表 6 取 fig4，
    # 于是同一段正文里前一句的中位遗憾与后一句的配对差跑在两个候选集上。
    sub = pt[pt["universe"] == "fig4"]
    piv = sub.pivot_table(index=["task_id", "n_cal"], columns="rule", values="regret")
    cols = ["always_simple", "always_model_update", "default_deep", "B_shift_unknown", "always_target_only"]
    piv = piv.dropna(subset=[c for c in cols if c in piv.columns]).reset_index()
    tasks = piv["task_id"].unique()
    print(f"配对比较的单元数 {len(piv)}（{piv['task_id'].nunique()} 任务 × "
          f"预算 {sorted(piv['n_cal'].unique())}；n=0 无定义的处方被排除）\n")
    groups = {t: piv[piv["task_id"] == t] for t in tasks}
    for other in ("always_model_update", "B_shift_unknown", "default_deep", "always_target_only"):
        if other not in piv.columns:
            continue
        obs = float(np.median(piv["always_simple"] - piv[other]))
        boot = []
        for _ in range(NBOOT):
            pick = RNG.choice(tasks, size=len(tasks), replace=True)
            rows = pd.concat([groups[t] for t in pick])
            boot.append(float(np.median(rows["always_simple"] - rows[other])))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        frac = float((piv["always_simple"] < piv[other]).mean())
        sep = "分开" if hi < 0 or lo > 0 else "**未分开（区间跨零）**"
        print(f"最简单可用校正 − {other:<20} 配对中位差 {obs * 100:+5.1f} pp "
              f"[{lo * 100:+5.1f}, {hi * 100:+5.1f}]　更优单元 {frac * 100:3.0f}%　{sep}")


def deep_sanity() -> None:
    cl = pd.read_excel(os.path.join(_OUT, "62_crossover_engine.xlsx"), sheet_name="curves")
    dp = pd.read_excel(os.path.join(_OUT, "64_deep_transfer_server.xlsx"), sheet_name="curves")
    allr = pd.concat([cl, dp], ignore_index=True)
    med = allr.groupby(["task_id", "benchmark", "method", "n_cal"], as_index=False)["rmsep"].median()
    ns = pd.read_excel(os.path.join(_OUT, "63_crossover_analysis.xlsx"), sheet_name="nstar_per_task")
    med = med.merge(ns[["task_id", "y_std_tgt"]].drop_duplicates("task_id"), on="task_id", how="left")
    med["nrmsep"] = med["rmsep"] / med["y_std_tgt"]
    keys = ["task_id", "benchmark", "n_cal"]
    best_deep = med[med["method"].isin(DEEP)].groupby(keys, as_index=False)["nrmsep"].min()
    best_simple = med[med["method"].isin(SIMPLE)].groupby(keys, as_index=False)["nrmsep"].min()
    print("\n深度族逐任务最优 NRMSEP 的分布（NRMSEP=1 即预测目标均值的水平）")
    for label, df in (("深度族", best_deep), ("简单族", best_simple)):
        print(f"  {label}：中位 {df['nrmsep'].median():.3f}　<0.9 的任务-预算单元 "
              f"{(df['nrmsep'] < 0.9).mean() * 100:.0f}%　<0.8 {(df['nrmsep'] < 0.8).mean() * 100:.0f}%　"
              f"<0.5 {(df['nrmsep'] < 0.5).mean() * 100:.0f}%　最小 {df['nrmsep'].min():.3f}")
    print("  按基准的深度族最小 NRMSEP：" + "　".join(
        f"{b}={g['nrmsep'].min():.2f}" for b, g in best_deep.groupby("benchmark")))


def main() -> int:
    prescription_pairs()
    deep_sanity()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
