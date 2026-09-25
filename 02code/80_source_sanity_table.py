#!/usr/bin/env python3
"""80_source_sanity_table.py — 把 88/89 的源域阳性对照合成补充材料 S14 的那张表。

为什么要单独一支：这张表是审稿面板判定"深度对照未被证明是能工作的源域预测器"之后新增的
直接答复，它的每个数字都必须能被脚本重算出来，而不是从两份 xlsx 里手抄再排版。

输入::

    04outputs/88_deep_indomain_sanity_<基准>.xlsx  sheet indomain  四个深度源模型
    04outputs/89_source_reference_baselines.xlsx  sheet reference PLSR 与常数预测器（同一划分）

88 号在服务器上按基准拆成 5 个进程并行跑（它单进程单线程，不拆就吃不到多核），
所以深度那一侧默认是 5 份分基准 xlsx；``--deep`` 可以改成任意几份，合并时按
(基准, 源域, 属性, 种子, 方法) 查重，两份文件覆盖同一格就报错退出。

两份输入必须来自同一组 (基准, 源域, 属性, 种子)——88 与 89 用同一个 rng 序列切划分，
本脚本据此逐条核对键集合，对不上就报错退出，不做"能对上多少算多少"的静默降级。

输出::

    04outputs/80_source_sanity_table.xlsx  长表 + 中位透视表
    04outputs/80_source_sanity_table.tex   S14 的 tabular（中英双稿共用同一份数字）
    04outputs/80_source_sanity_table.md    同样内容的可读版，供核对

用法::

    python3 02code/80_source_sanity_table.py
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import pandas as pd

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, ".."))
_OUT = os.path.join(_BASE, "04outputs")

KEY = ["benchmark", "src", "prop", "seed"]
# 列的先后就是表里从左到右的先后：四个被质疑的深度模型，然后是同划分上的两个参照。
ORDER = ["cnn", "physbl", "dann", "deepcoral", "plsr", "source_mean"]
LABEL_EN = {
    "cnn": "CNN", "physbl": "Phys+BL", "dann": "DANN", "deepcoral": "Deep CORAL",
    "plsr": "PLSR", "source_mean": "Constant",
}
BENCH_EN = {
    "corn": "Corn", "tablet": "Tablets", "mango": "Mango",
    "ossl_mir": "Soil (MIR)", "apple": "Apple",
}
BENCH_ORDER = ["corn", "tablet", "mango", "ossl_mir", "apple"]
_DEEP_DEFAULT = [os.path.join(_OUT, f"88_deep_indomain_sanity_{b}.xlsx") for b in BENCH_ORDER]


def _read(path: str, sheet: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"缺输入：{path}")
    df = pd.read_excel(path, sheet_name=sheet)
    df["prop"] = df["prop"].astype(str)
    df["src"] = df["src"].astype(str)
    return df


def _read_many(paths: list[str], sheet: str) -> pd.DataFrame:
    df = pd.concat([_read(p, sheet) for p in paths], ignore_index=True)
    dup = df.duplicated(subset=[*KEY, "method"], keep=False)
    if bool(dup.any()):
        raise SystemExit(
            "深度那一侧有重复的 (基准, 源域, 属性, 种子, 方法)——大概是同一个基准的旧单文件"
            "和新分基准文件一起传进来了，两份数字来自不同的取样口径，不能合。\n"
            f"  重复行数 {int(dup.sum())}，例如 {df.loc[dup, [*KEY, 'method']].head(3).to_dict('records')}")
    return df


def _pivot(df: pd.DataFrame, value: str, dec: int | None = 3) -> pd.DataFrame:
    piv = (df.groupby(["benchmark", "method"])[value].median().unstack("method"))
    cols = [c for c in ORDER if c in piv.columns]
    piv = piv[cols]
    rows = [b for b in BENCH_ORDER if b in piv.index]
    out = piv.loc[rows]
    return out if dec is None else out.round(dec)


def _latex(src: pd.DataFrame, tgt: pd.DataFrame) -> str:
    cols = list(src.columns)
    head = " & ".join(LABEL_EN.get(c, c) for c in cols)
    lines = [
        r"\begin{tabular}{l" + "r" * (len(cols) + 1) + "}",
        r"\hline",
        r"Benchmark & " + head + r" & Deep median (target) \\",
        r"\hline",
    ]
    for b in src.index:
        cells = " & ".join(f"{src.loc[b, c]:.3f}" for c in cols)
        # 这一列是「四个方法各自的中位」再取中位，必须用未舍入的中位去聚合——
        # 先舍到 3 位再取中位，玉米与芒果都会在第三位上偏 0.001。
        deep_cols = [c for c in ("cnn", "physbl", "dann", "deepcoral") if c in tgt.columns]
        deep_tgt = float(tgt.loc[b, deep_cols].median())
        lines.append(f"{BENCH_EN.get(str(b), str(b))} & {cells} & {deep_tgt:.3f} " + r"\\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deep", nargs="+", default=_DEEP_DEFAULT,
                    help="88 号的输出，可给多份（按基准拆分并行跑时每个基准一份）")
    ap.add_argument("--ref", default=os.path.join(_OUT, "89_source_reference_baselines.xlsx"))
    ap.add_argument("--out-stem", default=os.path.join(_OUT, "80_source_sanity_table"))
    args = ap.parse_args()

    deep = _read_many(list(args.deep), "indomain")
    ref = _read(args.ref, "reference")

    k_deep = set(map(tuple, deep[KEY].drop_duplicates().to_numpy().tolist()))
    k_ref = set(map(tuple, ref[KEY].drop_duplicates().to_numpy().tolist()))
    if k_deep != k_ref:
        only_d, only_r = sorted(k_deep - k_ref)[:5], sorted(k_ref - k_deep)[:5]
        raise SystemExit(
            "88 与 89 的 (基准, 源域, 属性, 种子) 键集合不一致，两者的划分就不是同一个，"
            f"表不能合。\n  只在 88: {only_d}\n  只在 89: {only_r}")

    long = pd.concat([deep, ref], ignore_index=True)
    src = _pivot(long, "nrmsep_source")
    tgt = _pivot(long, "nrmsep_target")
    tgt_raw = _pivot(long, "nrmsep_target", dec=None)

    rows: list[dict[str, Any]] = []
    for b in src.index:
        for c in src.columns:
            rows.append({"benchmark": b, "method": c,
                         "nrmsep_source_median": src.loc[b, c],
                         "nrmsep_target_median": tgt.loc[b, c]})
    tidy = pd.DataFrame(rows)

    with pd.ExcelWriter(args.out_stem + ".xlsx") as w:
        long.to_excel(w, sheet_name="long", index=False)
        tidy.to_excel(w, sheet_name="median", index=False)
        src.to_excel(w, sheet_name="source_pivot")
        tgt.to_excel(w, sheet_name="target_pivot")

    tex = _latex(src, tgt_raw)
    with open(args.out_stem + ".tex", "w", encoding="utf-8") as f:
        f.write(tex + "\n")
    with open(args.out_stem + ".md", "w", encoding="utf-8") as f:
        f.write("# 源域留出 NRMSEP（中位）\n\n" + src.to_markdown() + "\n\n")
        f.write("# 目标域测试 NRMSEP（中位）\n\n" + tgt.to_markdown() + "\n")

    print("源域留出 NRMSEP（中位）")
    print(src.to_string())
    print("\n目标域测试 NRMSEP（中位）")
    print(tgt.to_string())
    n_bad = int((src[[c for c in ("cnn", "physbl", "dann", "deepcoral") if c in src.columns]]
                 >= 1.0).sum().sum())
    print(f"\n源域 NRMSEP ≥ 1.0（即不如预测源域均值）的格子：{n_bad} 个")
    print(f"→ {args.out_stem}.xlsx / .tex / .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
