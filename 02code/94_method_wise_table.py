#!/usr/bin/env python3
"""94_method_wise_table.py — 十二个候选方法的逐方法误差表（补充材料用）。

为什么需要：正文的表 4/表 5 只报「简单族 vs 深度族」两栏，于是全文没有任何一处给出 PDS 的
精度数字，经典模型更新与目标自建也只出现在图 2 的曲线里。审稿人按表核数时，这几处是空的。
本脚本不重训任何模型，只用 62（经典）与 64（深度）已落地的曲线，按与 75 号完全相同的口径
（先在跨种子×重复上取中位得逐任务值，NRMSEP = RMSEP/σ_{y,tgt} = 1/RPD）汇总到
（基准 × 预算 × 方法），并给出该格里方法可运行的任务数——不可运行记为缺格而非最差。

用法::

    python3 02code/94_method_wise_table.py

输出文件:
    04outputs/94_method_wise_table.xlsx — by_benchmark / by_shift / latex_si
"""
from __future__ import annotations

import importlib.util
import os
from typing import Any

import pandas as pd

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, ".."))
_OUT = os.path.join(_BASE, "04outputs")
_es = importlib.util.spec_from_file_location("export_utils", os.path.join(_CODE, "01_export_utils.py"))
assert _es is not None
assert _es.loader is not None
_eu = importlib.util.module_from_spec(_es)
_es.loader.exec_module(_eu)

NCAL = [0, 5, 10, 20, 40]
ORDER = ["zero_shot", "coral", "sbc", "pds", "model_update", "target_only",
         "cnn_zeroshot", "cnn_finetune", "physbl_zeroshot", "physbl_ft", "dann", "deepcoral"]


def _task_medians() -> pd.DataFrame:
    cl = pd.read_excel(os.path.join(_OUT, "62_crossover_engine.xlsx"), sheet_name="curves")
    dp = pd.read_excel(os.path.join(_OUT, "64_deep_transfer_server.xlsx"), sheet_name="curves")
    allr = pd.concat([cl, dp], ignore_index=True)
    tm = allr.groupby(["benchmark", "shift_type", "task_id", "method", "n_cal"],
                      as_index=False)["rmsep"].median()
    ns = pd.read_excel(os.path.join(_OUT, "63_crossover_analysis.xlsx"), sheet_name="nstar_per_task")
    tm = tm.merge(ns[["task_id", "y_std_tgt"]].drop_duplicates("task_id"), on="task_id", how="left")
    tm["nrmsep"] = tm["rmsep"] / tm["y_std_tgt"]
    return tm[tm["n_cal"].isin(NCAL)]


def _aggregate(tm: pd.DataFrame, key: str) -> pd.DataFrame:
    g = tm.groupby([key, "n_cal", "method"]).agg(
        n_tasks=("task_id", "nunique"),
        rmsep_median=("rmsep", "median"),
        nrmsep_median=("nrmsep", "median"),
    ).reset_index()
    g["rmsep_median"] = g["rmsep_median"].round(3)
    g["nrmsep_median"] = g["nrmsep_median"].round(3)
    g["_ord"] = g["method"].map({m: i for i, m in enumerate(ORDER)}).fillna(99)
    return g.sort_values([key, "n_cal", "_ord"]).drop(columns="_ord").reset_index(drop=True)


def main() -> int:
    logger = _eu.get_logger(os.path.basename(__file__).replace(".py", ""))
    tm = _task_medians()
    missing = sorted(set(tm["method"]) - set(ORDER))
    if missing:
        logger.log(f"⚠ 曲线里出现未登记的方法名: {missing}")
    by_bench = _aggregate(tm, "benchmark")
    by_shift = _aggregate(tm, "shift_type")

    # 补充材料用的宽表：行 = 方法，列 = 预算，值 = NRMSEP 中位（括号内为可运行任务数）
    wide = by_shift.pivot_table(index=["shift_type", "method"], columns="n_cal",
                                values="nrmsep_median", aggfunc="first")
    cnt = by_shift.pivot_table(index=["shift_type", "method"], columns="n_cal",
                               values="n_tasks", aggfunc="first")
    cells: dict[str, Any] = {}
    for c in wide.columns:
        cells[f"n={c}"] = [
            "--" if pd.isna(v) else f"{v:.3f} ({int(n)})"
            for v, n in zip(wide[c], cnt[c], strict=True)
        ]
    latex_si = pd.DataFrame(cells, index=wide.index).reset_index()

    path = _eu.write_script_workbook(__file__, {
        0: ("by_benchmark", by_bench),
        "by_shift": by_shift,
        "latex_si": latex_si,
    })
    logger.log(f"落盘: {path}")
    print(by_shift.to_string(index=False))
    print("\n注：不可运行的（方法，预算）组合记为缺格，不按最差计分；括号内为该格可运行的任务数。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
