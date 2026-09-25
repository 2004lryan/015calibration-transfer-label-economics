#!/usr/bin/env python3
"""102_budget_sensitivity.py — 把优化预算统一放大后重跑源域阳性对照，看 S14 的读数是否成立。

为什么要有这一支：S14 报的是深度基线在自己源域留出样本上也只停在常数预测器水平。
这句话有两种读法——「迁移失败」与「训练失败」——而 S14 本身分不开。把 88 号的更新步数
上限从 2000 抬到 8000、五个基准一视同仁地重跑（服务器脚本 100），就能把两者分开：

  * 源域拟合不动 ⇒ 该预算本来就不是约束，模型是收敛之后仍停在常数预测器水平；
  * 源域拟合改善而目标域误差不跟着改善 ⇒ 迁移失败，不是训练失败；
  * 目标域误差也随之改善 ⇒ 原来的读法站不住，主比较必须按新预算重跑。

三条判据在脚本里逐基准算出来，不靠肉眼看表。

口径与 S14 完全相同：同一组 (基准, 源域, 属性, 种子, 方法)，同一套划分（88 号按种子
确定性切分），所以两次运行逐格可配对。键集合对不上就报错退出，不做静默降级。

两点必须在结论里说明，否则会把机制讲错：

  * 四个训练器都带早停（源域留出集上每 25 步评一次，连续 20 次无改善即停）。
    ``_batches`` 由同一个种子生成、只按步数预算截断，所以在 2000 步之前就早停的运行，
    把上限抬到 8000 会给出逐位相同的结果——这不是「没跑」，恰恰是「预算不是约束」的证据。
  * DANN 的梯度反转系数是 ``0.5 * (2 / (1 + exp(-10 * step / steps)) - 1)``，
    显式依赖步数上限。它在任何基准上都会变，变的是对抗强度的时间表，不只是训练更久。

输出::

    04outputs/102_budget_sensitivity.xlsx  逐基准逐方法长表 + 判据表

用法::

    python3 02code/102_budget_sensitivity.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, ".."))
_OUT = os.path.join(_BASE, "04outputs")

BENCH = ["corn", "tablet", "mango", "ossl_mir", "apple"]
KEY = ["benchmark", "src", "prop", "seed", "method"]
# DANN 的对抗强度时间表按步数上限参数化，放大预算不只是「训练更久」，单列出来
SCHEDULE_DEPENDENT = {"dann"}


def _read(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"缺输入：{path}")
    d = pd.read_excel(path, sheet_name="indomain")
    d["prop"] = d["prop"].astype(str)
    return d


def _paired_p(a: pd.Series, b: pd.Series) -> float:
    """逐格配对 Wilcoxon；两列完全相同（早停未被上限截断）时没有可检验的差异。"""
    if np.allclose(a.to_numpy(), b.to_numpy(), rtol=0, atol=0):
        return float("nan")
    return float(wilcoxon(a.to_numpy(), b.to_numpy()).pvalue)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000, help="放大后的步数上限")
    ap.add_argument("--out", default=os.path.join(_OUT, "102_budget_sensitivity.xlsx"))
    args = ap.parse_args()

    rows = []
    for b in BENCH:
        base = _read(os.path.join(_OUT, f"88_deep_indomain_sanity_{b}.xlsx"))
        big = _read(os.path.join(_OUT, f"88_deep_indomain_sanity_steps{args.steps}_{b}.xlsx"))
        ka = set(map(tuple, base[KEY].astype(str).to_numpy()))
        kb = set(map(tuple, big[KEY].astype(str).to_numpy()))
        if ka != kb:
            raise SystemExit(f"{b}：两次运行的 (基准,源域,属性,种子,方法) 键集合不同，无法配对"
                             f"（只在基线 {len(ka - kb)} 个，只在放大后 {len(kb - ka)} 个）")
        m = base.merge(big, on=KEY, suffixes=("_base", "_big")).sort_values(KEY)
        for meth, g in m.groupby("method"):
            same = int(np.isclose(g["nrmsep_source_base"], g["nrmsep_source_big"],
                                  rtol=0, atol=0).sum())
            rows.append({
                "基准": b, "方法": meth, "运行数": len(g),
                "逐位相同的运行数": same,
                "源域中位_基线": g["nrmsep_source_base"].median(),
                f"源域中位_{args.steps}步": g["nrmsep_source_big"].median(),
                "源域配对P": _paired_p(g["nrmsep_source_big"], g["nrmsep_source_base"]),
                "目标域中位_基线": g["nrmsep_target_base"].median(),
                f"目标域中位_{args.steps}步": g["nrmsep_target_big"].median(),
                "目标域配对P": _paired_p(g["nrmsep_target_big"], g["nrmsep_target_base"]),
                "预算是约束": same < len(g),
                "时间表依赖步数": meth in SCHEDULE_DEPENDENT,
            })
    long = pd.DataFrame(rows)

    # 逐基准判据：只看不受时间表影响的三个方法，DANN 单列
    crit = []
    for b, g in long[~long["时间表依赖步数"]].groupby("基准"):
        src_gain = (g["源域中位_基线"] - g[f"源域中位_{args.steps}步"]).median()
        tgt_gain = (g["目标域中位_基线"] - g[f"目标域中位_{args.steps}步"]).median()
        binding = bool(g["预算是约束"].any())
        crit.append({
            "基准": b,
            "预算对任一方法是约束": binding,
            "源域中位改善": src_gain,
            "目标域中位改善": tgt_gain,
            "读法": ("预算本来就不是约束：早停先于上限触发，收敛后仍停在此水平"
                     if not binding else
                     "源域拟合改善而目标域未跟进：迁移失败而非训练失败"
                     if src_gain > 0 >= tgt_gain else
                     "源域与目标域同向改善：原读法不成立，主比较须按新预算重跑"
                     if src_gain > 0 and tgt_gain > 0 else
                     "源域拟合未改善"),
        })
    crit_df = pd.DataFrame(crit)

    # 汇总表：把三个「时间表不依赖步数」的方法在一个基准内的全部运行汇到一起取中位。
    # DANN 不进这张表——它的对抗强度按步数上限参数化，换上限换的是时间表，不只是训练更久。
    pooled = []
    for b in BENCH:
        base = _read(os.path.join(_OUT, f"88_deep_indomain_sanity_{b}.xlsx"))
        big = _read(os.path.join(_OUT, f"88_deep_indomain_sanity_steps{args.steps}_{b}.xlsx"))
        m = base.merge(big, on=KEY, suffixes=("_base", "_big"))
        m = m[~m["method"].isin(SCHEDULE_DEPENDENT)]
        same = int(np.isclose(m["nrmsep_source_base"], m["nrmsep_source_big"],
                              rtol=0, atol=0).sum())
        pooled.append({
            "基准": b, "运行数": len(m), "逐位相同的运行数": same,
            "源域中位_基线": m["nrmsep_source_base"].median(),
            f"源域中位_{args.steps}步": m["nrmsep_source_big"].median(),
            "源域配对P": _paired_p(m["nrmsep_source_big"], m["nrmsep_source_base"]),
            "目标域中位_基线": m["nrmsep_target_base"].median(),
            f"目标域中位_{args.steps}步": m["nrmsep_target_big"].median(),
            "目标域配对P": _paired_p(m["nrmsep_target_big"], m["nrmsep_target_base"]),
        })
    pooled_df = pd.DataFrame(pooled)

    with pd.ExcelWriter(args.out) as w:
        long.to_excel(w, sheet_name="逐基准逐方法", index=False)
        pooled_df.to_excel(w, sheet_name="汇总", index=False)
        crit_df.to_excel(w, sheet_name="判据", index=False)
    print(pooled_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    pd.set_option("display.width", 200)
    print(long.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(crit_df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
