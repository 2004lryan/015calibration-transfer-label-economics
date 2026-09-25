#!/usr/bin/env python3
"""101_pair_level_inference.py — 把主比较的推断单位从「任务」换成「源–目标域对」的敏感性。

为什么要有这一支：正文的主推断按任务聚类（126 个任务），但 126 个任务只落在 98 个域对、
21 个目标域上——同一个目标域上的多个属性、多条源域路径并不独立。审稿人有理由问：把单位
换成更保守的域对，结论还成立吗？

做法：先把每个（任务，方法，预算）在 5 个种子上取中位（与表 4/表 5 同一口径），再取族均值，
然后把同一个域对下的多个任务折成一个值（取中位），在域对层面重算 Cliff's δ 与 Wilcoxon P。
季节与产地/年基准每个任务本来就是一个域对，折叠后不变；仪器基准 30 个任务折成 8 个域对，
土壤 12 个任务折成 6 个。

用法::

    python3 02code/101_pair_level_inference.py          # 写 04outputs/101_pair_level_inference.xlsx
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASE = Path(__file__).resolve().parent.parent
SHIFTS = ("instrument", "season", "origin_year_instrument", "lab")
BUDGETS = (0, 5, 10, 20, 40)


def _load_83() -> types.ModuleType:
    """复用 83 号的任务级中位与族定义，避免第二套口径。"""
    argv, sys.argv = sys.argv, ["83"]
    spec = importlib.util.spec_from_file_location("m83", BASE / "02code" / "83_revision_number_receipt.py")
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


def _cliff(a: Any, b: Any) -> float:
    """非配对支配统计量，与 83 号 _cliff_unpaired 同定义（正值＝简单族更小）。"""
    return float(np.sign(np.asarray(b)[None, :] - np.asarray(a)[:, None]).mean())


def main() -> int:
    m = _load_83()
    med = m._task_medians()
    med["pair"] = med["task_id"].str.split("|").str[:2].str.join("|")
    rows = []
    for shift in SHIFTS:
        for n in BUDGETS:
            at = med[(med["shift_type"] == shift) & (med["n_cal"] == n)]
            s = at[at["method"].isin(m.SIMPLE_FAM)].groupby(["task_id", "pair"])["rmsep"].mean()
            d = at[at["method"].isin(m.DEEP_FAM)].groupby(["task_id", "pair"])["rmsep"].mean()
            idx = s.index.intersection(d.index)
            if len(idx) == 0:
                continue
            s, d = s.loc[idx], d.loc[idx]
            sp = s.reset_index().groupby("pair")["rmsep"].median()
            dp = d.reset_index().groupby("pair")["rmsep"].median()
            rows.append({
                "shift_type": shift, "n_cal": n,
                "n_task": len(idx), "n_pair": len(sp),
                "delta_task": round(_cliff(s.to_numpy(), d.to_numpy()), 4),
                "p_task": float(wilcoxon(s.to_numpy(), d.to_numpy()).pvalue),
                "delta_pair": round(_cliff(sp.to_numpy(), dp.to_numpy()), 4),
                "p_pair": float(wilcoxon(sp.to_numpy(), dp.to_numpy()).pvalue),
            })
    out = pd.DataFrame(rows)
    dst = BASE / "04outputs" / "101_pair_level_inference.xlsx"
    with pd.ExcelWriter(dst) as xw:
        out.to_excel(xw, sheet_name="域对级对任务级", index=False)
    print(out.to_string(index=False))
    print(f"\n→ {dst.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
