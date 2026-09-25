#!/usr/bin/env python3
"""83_revision_number_receipt.py — 返修二轮新增/改动数字的确定性复算收据。

背景：`refine-logs/forensics_receipts/2026-09-08_revision_numbers.txt` 覆盖的是 73/74/75 号
分析产生的数值（可用 `evidence_check.py` 逐个在 xlsx 里找到）。本轮（表 3 适用性矩阵、
§2.3 协议段、词数、文献表）新增的是**聚合口径**——候选集大小、每格运行次数、留出比例、
胜率区间、文献时效分布——它们不是某个单元格里的一个数，机械查值查不到，必须**从原始
结果表与源码常量重新算一遍**。本脚本就是那一遍，输出逐条 PASS/FAIL 收据。

判据来源（稿件里的原话，见 `06doc/01manuscript/Elsevier_en.tex`）：
  * 表 3 表注：``the denominator ... is 2 classical candidates at $n=0$ and, at $n\\ge5$, 4 on
    the instrument benchmarks and 3 elsewhere; the oracle ... chooses among 6 label-free
    methods at $n=0$ and, at $n\\ge5$, among 12 on the instrument benchmarks and 11 elsewhere``
  * §2.3：``The classical methods repeat this draw five times per fixed seed (25 runs per
    cell), the deep methods once per seed (5 runs per cell)``、``a source-domain hold-out of
    about one sixth``
  * §3.2：``CORAL ... wins on 74--100\\% of the tasks of every shift type``

用法::

    python3 02code/83_revision_number_receipt.py            # 打印收据
    python3 02code/83_revision_number_receipt.py --write    # 另写入 refine-logs/forensics_receipts/
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import glob
import hashlib
import json
import math
import importlib.util
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, wilcoxon

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "04outputs"
MS = BASE / "06doc" / "01manuscript"
XL62 = OUT / "62_crossover_engine.xlsx"
XL63 = OUT / "63_crossover_analysis.xlsx"
XL64 = OUT / "64_deep_transfer_server.xlsx"
XL50 = OUT / "50_formal_multiseed_benchmark.xlsx"
XL55 = OUT / "55_bl_risk_diagnosis_validation.xlsx"
XL58 = OUT / "58_cluster_robust_and_strata.xlsx"
XL60S = OUT / "60sensitivity_domain_clustering.xlsx"
XLP0 = OUT / "c6_p0.xlsx"
XLP15 = OUT / "c6_p15.xlsx"
CASE_LOGS = BASE / "05logs" / "server"
# Phys 与 Phys+BL 在 13 号修正回归项配对（038bfed）后由 103 号在本机补跑，50 号以
# --override_dir 04outputs/50_shards_physfix 合并；它们「整格无结果」的痕迹只在补跑日志里
PHYSFIX_LOGS = BASE / "05logs"
PHYSFIX_METHODS = ("Phys", "Phys+BL")
XL03 = OUT / "03_exp_BL_validation.xlsx"
XL28 = OUT / "28_exp_true_BL_decomp.xlsx"
XL35 = OUT / "35_BL_source_heldout_validation.xlsx"
XL71 = OUT / "71_bl_vs_generic_distance.xlsx"
XL72 = OUT / "72_convergence_survivorship.xlsx"
XL59 = OUT / "59_reviewer_experiments.xlsx"   # 「原始」表 = 原分片 + 04outputs/59_shards_physfix 覆盖
INSTRUMENT_BENCHMARKS = {"corn", "tablet"}
LABEL_FREE = {"zero_shot", "coral", "cnn_zeroshot", "dann", "deepcoral", "physbl_zeroshot"}

Row = tuple[str, str, str, str]  # (verdict, claim, recomputed, source)
_rows: list[Row] = []


def check(claim: str, expected: object, got: object, source: str) -> None:
    ok = str(expected) == str(got)
    _rows.append(("PASS" if ok else "FAIL", claim, f"稿件={expected} 复算={got}", source))


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _curves() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.read_excel(XL62, sheet_name="curves"),
            pd.read_excel(XL64, sheet_name="curves"))


def check_candidate_sets() -> None:
    c62, c64 = _curves()
    src = "04outputs/62_crossover_engine.xlsx[curves] + 64_deep_transfer_server.xlsx[curves]"
    zero = set(c62[c62.n_cal == 0].method) | set(c64[c64.n_cal == 0].method)
    check("表 3 表注：n=0 预言机候选 6 个免标签方法", 6, len(zero), src)
    check("表 3 表注：n=0 候选集与 LABEL_FREE 定义一致", "True", str(zero == LABEL_FREE), src)

    zero_classical = set(c62[c62.n_cal == 0].method)
    check("表 3 表注：图 3(a) n=0 分母 2 个经典候选", 2, len(zero_classical), src)
    check("  该 2 个是直接迁移与 CORAL", "True",
          str(zero_classical == {"zero_shot", "coral"}), src)

    for bench in sorted(set(c62.benchmark)):
        labelled62 = set(c62[(c62.benchmark == bench) & (c62.n_cal >= 5)].method) - LABEL_FREE
        labelled64 = set(c64[(c64.benchmark == bench) & (c64.n_cal >= 5)].method) - LABEL_FREE
        want_classical = 4 if bench in INSTRUMENT_BENCHMARKS else 3
        want_oracle = 12 if bench in INSTRUMENT_BENCHMARKS else 11
        check(f"图 3(a) n>=5 分母（{bench}）", want_classical, len(labelled62), src)
        check(f"表 6 预言机 n>=5 候选（{bench}）", want_oracle,
              len(labelled62 | labelled64 | LABEL_FREE), src)

    check("表 3 表注：PDS 仅玉米/药片可行", "['corn', 'tablet']",
          str(sorted(set(c62[c62.method == "pds"].benchmark))), src)


def _task_medians() -> pd.DataFrame:
    c62, c64 = _curves()
    cur = pd.concat([c62, c64], ignore_index=True)
    med = cur.groupby(["task_id", "shift_type", "method", "n_cal"], as_index=False)["rmsep"].median()
    ns = pd.read_excel(XL63, sheet_name="nstar_per_task")[["task_id", "y_std_tgt"]].drop_duplicates("task_id")
    med = med.merge(ns, on="task_id", how="left")
    med["nrmsep"] = med["rmsep"] / med["y_std_tgt"]
    return med


def _cliff_unpaired(simple: pd.Series, deep: pd.Series) -> float:
    diff = deep.to_numpy()[:, None] - simple.to_numpy()[None, :]
    return float((np.count_nonzero(diff > 0) - np.count_nonzero(diff < 0)) / diff.size)


# ── 判据直接取自稿件本身 ────────────────────────────────────────────────
# 原来表 4/表 5 的「稿件=」是手抄进本脚本的常量。2026-09-10 全量重跑后改了稿件的
# 30 行表格数据，收据却纹丝不动地报着旧值——手抄的判据在改稿后会静默过期，而收据
# 恰恰是用来发现过期的。现在逐行从 .tex 里读，稿件改一个数收据就跟着改一个数。
SHIFT_TEX = {
    "Instrument": "instrument", "Season": "season",
    "Origin/year": "origin_year_instrument", "Lab (MIR)": "lab",
    "仪器": "instrument", "季节": "season",
    "产地/年": "origin_year_instrument", "实验室(MIR)": "lab",
}
KIND_TEX = {"CORAL vs.\\ DeepCORAL": "pair", "CORAL 对 DeepCORAL": "pair",
            "Family mean": "mean", "族均值": "mean"}
SIMPLE_FAM = ("coral", "sbc")
DEEP_FAM = ("cnn_zeroshot", "cnn_finetune", "dann", "deepcoral", "physbl_zeroshot", "physbl_ft")


def _tex(name: str) -> str:
    return (MS / name).read_text(encoding="utf-8")


def _flat(name: str) -> str:
    """正文压成单空格：句子级判据不能被 LaTeX 的换行位置左右。"""
    return re.sub(r"\s+", " ", _tex(name))


def _table_rows(text: str, label: str) -> list[list[str]]:
    """取 \\label{label} 那张表 \\midrule 与 \\bottomrule 之间的数据行（分节 \\midrule 丢弃）。"""
    i = text.index("\\label{" + label + "}")
    seg = text[i:text.index("\\bottomrule", i)]
    seg = seg[seg.index("\\midrule") + len("\\midrule"):]
    rows = []
    for line in seg.split("\\\\"):
        line = line.replace("\\midrule", "").strip()
        if line:
            rows.append([c.strip() for c in line.split("&")])
    return rows


def _cell(c: str) -> tuple[float, int]:
    """把稿件单元格解析成（数值，稿件写的小数位数）——复算按稿件自己的精度舍入再比。"""
    s = c.replace("$", "").replace("+", "").replace("\\,", "").strip()
    if not re.fullmatch(r"-?\d+(\.\d+)?", s):
        raise SystemExit(f"表格单元格无法解析为数字：{c!r}")
    return float(s), (len(s.split(".")[1]) if "." in s else 0)


def _numchk(tag: str, cell: str, got: float, src: str) -> None:
    """按稿件自己写的小数位数舍入复算值再比。

    舍入用 Python 的 round()，与产出这些数字的分析脚本同一套；判据要回答的是
    「稿件有没有如实抄下脚本算出来的值」，所以两边必须用同一种舍入，
    换成四舍五入会把三处本来正确的表格单元格判成错的。
    """
    want, dec = _cell(cell)
    v = round(float(got), dec)
    check(tag, want, 0.0 if v == 0 else v, src)


def _pchk(tag: str, cell: str, a: pd.Series, b: pd.Series, src: str) -> None:
    """Wilcoxon 配对 P：稿件写 $<10^{-k}$ 就验不等式，写具体值就按其精度比。"""
    p = float(wilcoxon(a.to_numpy(), b.to_numpy()).pvalue)
    s = cell.replace(" ", "")
    m = re.fullmatch(r"\$<10\^\{(-\d+)\}\$", s)
    if m:
        check(f"{tag} {cell}", "True", str(p < 10.0 ** int(m.group(1))),
              f"{src}（实际 P={p:.3g}）")
        return
    want, dec = _cell(cell)
    check(tag, want, round(p, dec), src)


def _scichk(tag: str, cell: str, got: float, src: str) -> None:
    """稿件写成 $a\\times 10^{-b}$ 的 P 值：按稿件自己的有效位比尾数，再比指数。"""
    m = re.fullmatch(r"\$([\d.]+)\\times\s*10\^\{(-?\d+)\}\$", cell.replace(" ", "").replace("\\,", ""))
    if not m:
        check(f"{tag}（单元格可解析）", "True", "False", src)
        return
    want, dec = _cell(m.group(1))
    exp = int(m.group(2))
    got_exp = math.floor(math.log10(got))
    check(f"{tag} 指数", str(exp), str(got_exp), src)
    check(f"{tag} 尾数", str(want), str(round(got / 10.0 ** got_exp, dec)), src)


S14_BENCH = ["corn", "tablet", "mango", "ossl_mir", "apple"]
S14_BENCH_EN = {"corn": "Corn", "tablet": "Tablets", "mango": "Mango",
                "ossl_mir": "Soil (MIR)", "apple": "Apple"}
S14_METH = ["cnn", "physbl", "dann", "deepcoral", "plsr", "source_mean"]
S14_DEEP = ["cnn", "physbl", "dann", "deepcoral"]
_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
          "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
          "seventeen", "eighteen", "nineteen", "twenty"]


def check_s14() -> None:
    """S14 源域阳性对照：整张表逐格 + 三段正文里的每一个数，全部从 80 号的长表重算。

    这张表是本轮唯一新做的实验，也是稿件里最容易失聪的一处：88 号在服务器上按基准拆成
    五个进程跑，产物是五份 xlsx，数字落进 .tex 之后没有任何东西再去逐格比对它们。
    """
    long = pd.read_excel(OUT / "80_source_sanity_table.xlsx", sheet_name="long")
    src = "04outputs/80_source_sanity_table.xlsx[long]（88 号五份分基准输出 + 89 号合并）"
    deep = long[long["method"].isin(S14_DEEP)]
    med_s = long.groupby(["benchmark", "method"])["nrmsep_source"].median()
    med_t = long.groupby(["benchmark", "method"])["nrmsep_target"].median()

    rows = _table_rows(_tex("Supplementary_en.tex"), "tab:si-source-sanity")
    if len(rows) != len(S14_BENCH):
        raise SystemExit(f"S14 表有 {len(rows)} 行，应为 {len(S14_BENCH)} 行")
    for r, b in zip(rows, S14_BENCH, strict=True):
        check(f"S14 表 {b} 行标签", S14_BENCH_EN[b], r[0], src)
        for cell, meth in zip(r[1:7], S14_METH, strict=True):
            _numchk(f"S14 表 {b} {meth} 源域", cell, float(med_s.loc[(b, meth)]), src)
        # 末列与 80 号 _latex 同一算法：先按方法取中位，再对这四个中位取中位。
        _numchk(f"S14 表 {b} 深度族目标域中位", r[7],
                float(med_t.loc[b].reindex(S14_DEEP).median()), src)

    t = _flat("Supplementary_en.tex")
    combos = {b: int(deep[deep["benchmark"] == b][["src", "prop"]].drop_duplicates().shape[0])
              for b in S14_BENCH}
    m = re.search(r"\(corn (\d+), tablets (\d+), mango (\d+), soil (\d+), apple (\d+), "
                  r"(\w+(?:-\w+)?) in total\)\. Each combination is run with all five fixed "
                  r"seeds and all four models, giving (\d+) trainings", t)
    if not m:
        raise SystemExit("S14 的取样句没解析出来——句式改了就得同步改这里")
    for i, b in enumerate(S14_BENCH):
        check(f"S14 取样 {b} 组合数", str(combos[b]), m.group(i + 1), src)
    check("S14 取样 合计组合数（英文数词）", "thirty-four" if sum(combos.values()) == 34
          else str(sum(combos.values())), m.group(6), src)
    check("S14 训练次数", str(len(deep)), m.group(7), src)

    m = re.search(r"Pooled over the (\d+) trainings, the deep family's median NRMSEP is "
                  r"([\d.]+) on its own source domain against ([\d.]+) on the target domain "
                  r"of the same splits, while the constant predictor is at ([\d.]+) on the "
                  r"source hold-out and ([\d.]+) on the target domain \(it returns the "
                  r"source-domain training mean, which is only approximately the target mean\), "
                  r"and the PLS regression reaches ([\d.]+)--([\d.]+) across benchmarks", t)
    if not m:
        raise SystemExit("S14 的汇总句没解析出来——句式改了就得同步改这里")
    check("S14 汇总 训练次数", str(len(deep)), m.group(1), src)
    _numchk("S14 汇总 深度族源域中位", m.group(2), deep["nrmsep_source"].median(), src)
    _numchk("S14 汇总 深度族目标域中位", m.group(3), deep["nrmsep_target"].median(), src)
    const = long[long["method"] == "source_mean"]
    _numchk("S14 汇总 恒定预测器源域", m.group(4), const["nrmsep_source"].median(), src)
    _numchk("S14 汇总 恒定预测器目标域", m.group(5), const["nrmsep_target"].median(), src)
    pls = long[long["method"] == "plsr"].groupby("benchmark")["nrmsep_source"].median()
    _numchk("S14 汇总 PLSR 下界", m.group(6), pls.min(), src)
    _numchk("S14 汇总 PLSR 上界", m.group(7), pls.max(), src)

    corn = deep[(deep["benchmark"] == "corn") & (deep["method"] == "physbl")]["nrmsep_source"]
    m = re.search(r"Phys\+BL on corn is at the constant-predictor level in all (\d+) of its "
                  r"runs \(source-domain NRMSEP ([\d.]+)--([\d.]+), median ([\d.]+)\)", t)
    if not m:
        raise SystemExit("S14 玉米 Phys+BL 那句没解析出来——句式改了就得同步改这里")
    check("S14 例外 玉米 Phys+BL 运行数", str(len(corn)), m.group(1), src)
    _numchk("S14 例外 玉米 Phys+BL 最小", m.group(2), corn.min(), src)
    _numchk("S14 例外 玉米 Phys+BL 最大", m.group(3), corn.max(), src)
    _numchk("S14 例外 玉米 Phys+BL 中位", m.group(4), corn.median(), src)

    ap = deep[deep["benchmark"] == "apple"]
    ap_med = ap.groupby("method")["nrmsep_source"].median()
    m = re.search(r"the four per-method medians are ([\d.]+)--([\d.]+) and (\d+) of the (\d+) "
                  r"apple runs \((\d+)\\%\) are at or above 1", t)
    if not m:
        raise SystemExit("S14 苹果那句没解析出来——句式改了就得同步改这里")
    _numchk("S14 例外 苹果 各方法中位 下界", m.group(1), ap_med.min(), src)
    _numchk("S14 例外 苹果 各方法中位 上界", m.group(2), ap_med.max(), src)
    check("S14 例外 苹果 ≥1 的运行数", str(int((ap["nrmsep_source"] >= 1).sum())), m.group(3), src)
    check("S14 例外 苹果 运行总数", str(len(ap)), m.group(4), src)
    _numchk("S14 例外 苹果 ≥1 占比%", m.group(5), 100 * (ap["nrmsep_source"] >= 1).mean(), src)

    cells = med_s.loc[[(b, mm) for b in S14_BENCH for mm in S14_DEEP]]
    bad = [k for k, v in cells.items() if v >= 1]
    m = re.search(r"Of the (\w+) benchmark--method cells, (\w+) have medians below 1; "
                  r"the exception is Phys\+BL on corn\.", t)
    if not m:
        raise SystemExit("S14 的 20 格计数句没解析出来——句式改了就得同步改这里")
    check("S14 格子总数（英文数词）", _WORDS[len(cells)], m.group(1), src)
    check("S14 中位 <1 的格子数（英文数词）", _WORDS[len(cells) - len(bad)], m.group(2), src)
    check("S14 中位 ≥1 的格子", "[('corn', 'physbl')]", str(bad), src)

    soil = med_s.loc[[("ossl_mir", mm) for mm in S14_DEEP]]
    rest = med_s.loc[[(b, mm) for b in S14_BENCH if b != "ossl_mir" for mm in S14_DEEP]]
    m = re.search(r"cross-laboratory soil \(([\d.]+)--([\d.]+) against ([\d.]+)--([\d.]+) "
                  r"elsewhere\)", t)
    if not m:
        raise SystemExit("S14 土壤那句没解析出来——句式改了就得同步改这里")
    _numchk("S14 土壤 深度族源域 下界", m.group(1), soil.min(), src)
    _numchk("S14 土壤 深度族源域 上界", m.group(2), soil.max(), src)
    _numchk("S14 其余基准 深度族源域 下界", m.group(3), rest.min(), src)
    _numchk("S14 其余基准 深度族源域 上界", m.group(4), rest.max(), src)

    m = re.search(r"median source-domain NRMSEP ([\d.]+) against ([\d.]+) for a constant "
                  r"predictor and ([\d.]+)--([\d.]+) for PLS\)", _flat("Elsevier_en.tex"))
    if not m:
        raise SystemExit("§4.4 第八条引 S14 的那句没解析出来——句式改了就得同步改这里")
    _numchk("§4.4 深度族源域中位", m.group(1), deep["nrmsep_source"].median(), src)
    _numchk("§4.4 恒定预测器", m.group(2), const["nrmsep_source"].median(), src)
    _numchk("§4.4 PLSR 下界", m.group(3), pls.min(), src)
    _numchk("§4.4 PLSR 上界", m.group(4), pls.max(), src)


CASE_BARE = ["PLSR", "SVR", "CNN", "CNN+MMD", "Phys+BL"]
CASE_ALL = ["PLSR", "SVR", "CNN", "CNN+MMD", "CNN+BL", "Phys", "Phys+BL"]


def _case_a2() -> pd.DataFrame:
    """50 号共同场景子集汇总表（案例研究所有聚合量的唯一来源）。"""
    return pd.read_excel(XL50, sheet_name="表a2：共同场景子集").set_index("方法")


def _case_pivot() -> pd.DataFrame:
    """每场景跨种子均值透视到 场景 × 方法_校正，只留全方法收敛的共同场景。"""
    g = pd.read_excel(XL50, sheet_name="表g：每场景跨种子均值")
    p = g.pivot(index="场景", columns="方法_校正", values="RMSE")
    return p.loc[p[CASE_ALL].dropna().index]


def _case_divergence() -> dict[str, tuple[int, int, int]]:
    """(RMSE>10 的运行数, 训练出非有限值而整格无结果的次数, 分母) —— 逐方法。

    只数 RMSE>10 会低估：物理架构的不稳定多半表现为整格不落行，分片 CSV 里
    既无行也无 NaN，唯一的痕迹在分片日志里。故两类都数，分母取应有运行数。
    """
    raw = pd.read_excel(XL50, sheet_name="表h：全部原始结果")
    raw = raw[raw["校正"] == "裸"]
    common = set(_case_pivot().index)
    seeds = sorted(raw["种子"].unique())
    den = len(common) * len(seeds)
    pat = re.compile(r"\[(PLSR|SVR|CNN\+MMD|CNN\+BL|CNN|Phys\+BL|Phys)\]\s+([^:]+):\s*(.+)")
    nan_hit: dict[str, set[tuple[str, int]]] = {m: set() for m in CASE_ALL}
    logs = sorted(CASE_LOGS.glob("50_shard_*.out"))
    fix_logs = sorted(PHYSFIX_LOGS.glob("103_case_*.out"))
    if not logs:
        raise SystemExit("缺 05logs/server/50_shard_*.out —— 发散统计无法复算")
    if not fix_logs:
        raise SystemExit("缺 05logs/103_case_*.out —— Phys/Phys+BL 的发散统计无法复算")
    keep: Callable[[str], bool]
    for group, rx, keep in ((logs, r"50_shard_(\d+)_", lambda m: m not in PHYSFIX_METHODS),
                            (fix_logs, r"103_case_(\d+)_", lambda m: m in PHYSFIX_METHODS)):
        for f in group:
            seed_m = re.search(rx, f.name)
            seed = int(seed_m.group(1)) if seed_m else -1
            for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                m = pat.search(line)
                if m and keep(m.group(1)) and "nan" in m.group(3).lower():
                    nan_hit[m.group(1)].add((m.group(2).strip(), seed))
    out = {}
    for meth in CASE_ALL:
        r = raw[(raw["方法"] == meth) & (raw["场景"].isin(common))]
        big = int((r["RMSE"] > 10).sum())
        nan_n = len({k for k in nan_hit[meth] if k[0] in common})
        out[meth] = (big, nan_n, den)
    return out


_IV = r"\[([^\],]+),\s*([^\]]+)\]"
_SCI = r"([\d.]+\\times ?10\^\{-?\d+\})"


def _sgn(tok: str) -> str:
    """稿件里的 $-$0.104 / $+$0.44 / 0.44 → 带符号的纯数字串，交给 _numchk 按稿件精度比。"""
    s = tok.replace("\\,", "").replace(" ", "")
    return s.replace("$-$", "-").replace("$+$", "").replace("$", "")


def _rx(tag: str, pattern: str, text: str, src: str) -> re.Match[str] | None:
    """句式解析不到本身就是一条 FAIL：稿件改了句子而判据没跟上，比数字错更该露出来。"""
    m = re.search(pattern, text)
    if not m:
        check(f"{tag}（句式可解析）", "True", "False", src)
    return m


def _case_runs() -> tuple[dict[str, pd.Series], dict[str, pd.Series], list[str], list[int]]:
    """共同场景 × 种子 的逐次运行：整格无结果（表 h 缺行）与 RMSE>10 各一张指示表。"""
    raw = pd.read_excel(XL50, sheet_name="表h：全部原始结果")
    bare = raw[raw["校正"] == "裸"]
    common = list(_case_pivot().index)
    seeds = sorted(int(s) for s in bare["种子"].unique())
    idx = pd.MultiIndex.from_product([common, seeds], names=["场景", "种子"])
    miss, big = {}, {}
    for meth in CASE_ALL:
        r = bare[(bare["方法"] == meth) & (bare["场景"].isin(common))]
        r = r.set_index(["场景", "种子"])["RMSE"].reindex(idx)
        miss[meth], big[meth] = r.isna(), r > 10
    return miss, big, common, seeds


def check_case_study() -> None:
    """苹果案例研究（补充材料 S3/S8~S12、正文 §3.5/§4.5 与两封回复信）：表逐格 + 正文逐值全部现算。

    判据一律从 04outputs 重算，稿件只作被检对象。13 号回归项配对修正（038bfed）后，Phys 与
    Phys+BL 有 1.7%~4.6% 的运行发散，个别种子能把场景均值推到上千，所以稿件对 RMSE 同报中位数
    与均值：中位数回答「典型场景谁更好」，均值回答「连同失败在内期望误差多大」。两套都判，
    连同它们的方向（未校正：中位 Phys+BL 领先、均值 CNN 领先；校正后两者同向）。
    """
    a2 = _case_a2()
    piv = _case_pivot()
    n_common = len(piv)
    SI = "Supplementary_en.tex"
    MAIN = "Elsevier_en.tex"
    src50 = "04outputs/50_formal_multiseed_benchmark.xlsx[表a2/表g]"
    t_en = _flat(SI)
    t_main = _flat(MAIN)
    miss, big, common, seeds = _case_runs()
    dv = {k: miss[k] | big[k] for k in CASE_ALL}
    n_runs = len(common) * len(seeds)
    src_ab = src50 + " + 表h 缺行（与 05logs 的 nan 痕迹互核）"
    tc = pd.read_excel(XL50, sheet_name="表c：预注册配对检验")

    def hyp(a: str, b: str) -> pd.Series:
        return tc[(tc["A"] == a) & (tc["B"] == b)].iloc[0]

    def rate(meth: str) -> float:
        return 100 * float(dv[meth].mean())

    # ── S3：覆盖、共同子集，以及「无结果」两条通路互核 ───────────────────
    check("S3 节：案例研究共同场景数", str(n_common), str(int(a2["n_scenarios"].iloc[0])), src50)
    m = _rx("S3 共同场景数", r"common subset of (\d+) scenarios in which all seven methods", t_en, SI)
    if m:
        check("S3 节正文写的共同场景数", str(n_common), m.group(1), SI)
    raw = pd.read_excel(XL50, sheet_name="表h：全部原始结果")
    cov = raw[raw["校正"] == "裸"].groupby("方法")["场景"].nunique()
    m = _rx("S3 收敛场景数", r"PLSR and support vector regression \(SVR\) cover all (\d+) scenarios.{0,200}?"
            r"scenarios with a result for the deep models are (\d+) for the convolutional neural network "
            r"\(CNN\) and (\d+) for CNN\+maximum mean discrepancy \(MMD\), CNN\+BL, Phys and Phys\+BL",
            t_en, SI)
    if m:
        for meth in ("PLSR", "SVR"):
            check(f"S3 {meth} 覆盖场景数", m.group(1), str(int(cov[meth])), src50)
        check("S3 CNN 收敛场景数", m.group(2), str(int(cov["CNN"])), src50)
        for meth in ("CNN+MMD", "CNN+BL", "Phys", "Phys+BL"):
            check(f"S3 {meth} 收敛场景数", m.group(3), str(int(cov[meth])), src50)
    logs = _case_divergence()
    for meth in CASE_ALL:
        check(f"S10 {meth} 无结果运行：表 h 缺行数与分片日志 nan 痕迹一致",
              str(int(miss[meth].sum())), str(logs[meth][1]),
              src50 + " vs 05logs（server/50_shard_*.out；Phys、Phys+BL 取 103_case_*.out）")

    # ── 表 S1 逐格（列：中位、均值、均值的 CI、R² 中位、RPD、Bias）───────────
    bare5 = piv[CASE_BARE]
    rows = _table_rows(_tex(SI), "tab:apple5")
    check("表 S1 行数", "5", str(len(rows)), SI)
    for row in rows:
        meth = row[0].strip()
        r = a2.loc[meth]
        lo, hi = [x.strip(" []") for x in row[3].split(",")]
        for tag, cell, got in (("RMSE 中位", row[1], r["RMSE_median"]),
                               ("RMSE 均值", row[2], r["RMSE_mean"]),
                               ("CI 下", lo, r["RMSE_CI95_low"]),
                               ("CI 上", hi, r["RMSE_CI95_high"]),
                               ("R² 中位", row[4], r["R2_median"]),
                               ("RPD", row[5], r["RPD_mean"]),
                               ("Bias", row[6], r["Bias_mean"])):
            _numchk(f"表 S1 {meth} {tag}", cell, got, src50)
        _numchk(f"表 S1 {meth} 中位与逐场景透视同源", row[1], float(piv[meth].median()), src50 + "[表g]")

    # ── 表 S3 逐格（两列 RMSE 为中位；增益为逐场景降幅的中位）───────────────
    for row in _table_rows(_tex(SI), "tab:sbc"):
        meth = row[0].strip()
        u, c = piv[meth], piv[meth + " + SB"]
        cs = a2.loc[meth + " + SB"]
        for tag, cell, got in (("未校正 RMSE 中位", row[1], u.median()),
                               ("校正后 RMSE 中位", row[2], c.median()),
                               ("校正增益中位%", row[3], (100 * (u - c) / u).median()),
                               ("校正后 RPD", row[4], cs["RPD_mean"]),
                               ("校正后 R² 中位", row[5], cs["R2_median"]),
                               ("校正后误判率%", row[6], 100 * cs["品质误判率_mean"])):
            _numchk(f"表 S3 {meth} {tag}", cell, got, src50)
    phsb = piv["Phys+BL + SB"]
    m = _rx("表 S3 表注 Phys+BL+SB 发散", r"Phys\+BL\+SB still exceeds 10\\,\\Brix\{\} in (\d+) scenarios, "
            r"which is why its mean RMSE \(([\d.]+)\\,\\Brix\{\}\)", t_en, SI)
    if m:
        check("表 S3 表注 Phys+BL+SB >10 的场景数", m.group(1), str(int((phsb > 10).sum())), src50)
        _numchk("表 S3 表注 Phys+BL+SB 均值 RMSE", m.group(2), phsb.mean(), src50)

    # ── 表 S2（消融）逐格 ───────────────────────────────────────────
    for row in _table_rows(_tex(SI), "tab:ablation"):
        label = row[0].replace("$^{\\dagger}$", "").replace("$^{\\ddagger}$", "")
        to_m, from_m = [x.strip() for x in label.split(" vs ")]
        a, b = piv[to_m], piv[from_m]
        _numchk(f"表 S2 {row[0]} 起始中位", row[1], b.median(), src50)
        _numchk(f"表 S2 {row[0]} 终止中位", row[2], a.median(), src50)
        _numchk(f"表 S2 {row[0]} 改进中位%", row[3], (100 * (b - a) / b).median(), src50)
        _scichk(f"表 S2 {row[0]} Wilcoxon P", row[4],
                float(wilcoxon(a.to_numpy(), b.to_numpy()).pvalue), src50)
        for meth, cell in zip((from_m, to_m), row[5].split("/"), strict=True):
            _numchk(f"表 S2 {row[0]} {meth} 发散率%", cell.replace("\\%", ""), rate(meth), src_ab)
        check(f"表 S2 {row[0]} n", str(n_common), row[6].strip(), src50)
    h1 = hyp("Phys+BL", "CNN")
    m = _rx("表 S2 表注 Holm", r"the Holm-adjusted values are \$" + _SCI + r"\$ for Phys\+BL against the CNN \(the "
            r"value quoted in Section~S9, against the \$" + _SCI + r"\$ printed here\) and ([\d.]+) for the two loss "
            r"ablations", t_en, SI)
    if m:
        _scichk("表 S2 表注 Phys+BL 对 CNN 的 Holm P", f"${m.group(1)}$", float(h1["Wilcoxon_p_holm"]),
                src50 + "[表c H1]")
        _scichk("表 S2 表注 Phys+BL 对 CNN 的未校正 P", f"${m.group(2)}$", float(h1["Wilcoxon_p"]), src50 + "[表c H1]")
        for a_, b_ in (("CNN+BL", "CNN"), ("Phys+BL", "Phys")):
            _numchk(f"表 S2 表注 {a_} 对 {b_} 的 Holm P", m.group(3), float(hyp(a_, b_)["Wilcoxon_p_holm"]),
                    src50 + "[表c]")
    check("表 S2 表注写明表中与 S10 正文的 P 为未校正值", "True",
          str("here and in the text of Section~S10 are the uncorrected two-sided Wilcoxon values" in t_en), SI)
    ph_, pb_ = dv["Phys"], dv["Phys+BL"]
    n10, n01 = int((ph_ & ~pb_).sum()), int((~ph_ & pb_).sum())
    p_mc = float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue)
    m = _rx("表 S2 表注 McNemar", r"for Phys vs Phys\+BL the exact McNemar test over the discordant runs gives "
            r"\$P=" + _SCI + r"\$", t_en, SI)
    if m:
        _scichk("表 S2 表注 McNemar 精确 P", f"${m.group(1)}$", p_mc, src_ab + f"（不一致对 {n10}/{n01}）")

    # ── S9 段1：中位与均值两个答案 ─────────────────────────────────────
    ph, cnn = piv["Phys+BL"], piv["CNN"]
    imp = 100 * (cnn - ph) / cnn
    m = _rx("S9 段1 中位句", r"its median RMSE, ([\d.]+)\\,\\Brix\{\}, is the lowest of the five \(CNN "
            r"([\d.]+)\\,\\Brix\{\}\), it has the lower error in (\d+) of the (\d+) common scenarios, and the "
            r"median paired improvement over the CNN is ([\d.]+)\\% \(two-sided Wilcoxon, \$P=" + _SCI
            + r"\$ after Holm correction\)", t_en, SI)
    if m:
        _numchk("S9 段1 Phys+BL 中位 RMSE", m.group(1), ph.median(), src50)
        check("S9 段1 Phys+BL 中位为五法最低", "True", str(bare5.median().idxmin() == "Phys+BL"), src50)
        _numchk("S9 段1 CNN 中位 RMSE", m.group(2), cnn.median(), src50)
        check("S9 段1 Phys+BL 更优的场景数", m.group(3), str(int((ph < cnn).sum())), src50)
        check("S9 段1 共同场景数", m.group(4), str(n_common), src50)
        _numchk("S9 段1 配对改进中位%", m.group(5), imp.median(), src50)
        _scichk("S9 段1 Holm 校正 P", f"${m.group(6)}$", float(h1["Wilcoxon_p_holm"]), src50 + "[表c H1]")
    p_h1 = float(wilcoxon(ph.to_numpy(), cnn.to_numpy()).pvalue)
    check("表 c H1 的未校正 P 与现算一致（相对差 <1e-6）", "True",
          str(abs(p_h1 - float(h1["Wilcoxon_p"])) <= 1e-6 * p_h1), src50 + "[表c H1] vs 现算")
    m = _rx("S9 段1 均值句", r"Its mean RMSE, ([\d.]+)\\,\\Brix\{\}, is nonetheless the highest of the five: (\d+) of "
            r"its (\d+) scenario--seed runs diverge \((\d+) with a test-set RMSE above 10\\,\\Brix\{\} and (\d+) that "
            r"produced non-finite values and returned no result; Section~S10\) while no CNN run does", t_en, SI)
    if m:
        _numchk("S9 段1 Phys+BL 均值 RMSE", m.group(1), ph.mean(), src50)
        check("S9 段1 Phys+BL 均值为五法最高", "True", str(bare5.mean().idxmax() == "Phys+BL"), src50)
        check("S9 段1 Phys+BL 发散运行数", m.group(2), str(int(dv["Phys+BL"].sum())), src_ab)
        check("S9 段1 运行总数", m.group(3), str(n_runs), src_ab)
        check("S9 段1 其中 RMSE>10", m.group(4), str(int(big["Phys+BL"].sum())), src_ab)
        check("S9 段1 其中无结果", m.group(5), str(int(miss["Phys+BL"].sum())), src_ab)
        check("S9 段1 CNN 无一发散", "0", str(int(dv["CNN"].sum())), src_ab)

    c60a = pd.read_excel(XL60S, sheet_name="表a：三档聚类稳健性")
    c60b = pd.read_excel(XL60S, sheet_name="表b：中位配对差三档聚类")
    for d in (c60a, c60b):
        d["聚类单位"] = d["聚类单位"].astype(str).str.strip()
    src60 = "04outputs/60sensitivity_domain_clustering.xlsx[表a 均值 / 表b 中位]"
    UNC, PLS = "未校正 Phys+BL − CNN", "校正后 PLSR+SB − Phys+BL+SB"

    def r60(tab: pd.DataFrame, key: str, unit: str) -> pd.Series:
        return tab[(tab["比较"] == key) & tab["聚类单位"].str.startswith(unit)].iloc[0]

    m = _rx("S9 段1 场景聚类句", r"clustering unit gives a median paired RMSE difference of (\S+)\\,\\Brix\{\} "
            r"\(95\\% confidence interval " + _IV + r"\\,\\Brix\{\}\) and a mean paired difference of "
            r"(\S+)\\,\\Brix\{\} \(" + _IV + r"\\,\\Brix\{\}\); Cliff's \$\\delta\$ is (\S+), a negligible effect",
            t_en, SI)
    if m:
        rb, ra = r60(c60b, UNC, "场景"), r60(c60a, UNC, "场景")
        _numchk("S9 段1 中位配对差", _sgn(m.group(1)), float(rb["配对差中位"]), src60)
        _numchk("S9 段1 中位配对差与逐场景透视同源", _sgn(m.group(1)), float((ph - cnn).median()), src50)
        _numchk("S9 段1 中位配对差 CI 下", _sgn(m.group(2)), float(rb["CI下界"]), src60)
        _numchk("S9 段1 中位配对差 CI 上", _sgn(m.group(3)), float(rb["CI上界"]), src60)
        _numchk("S9 段1 均值配对差", _sgn(m.group(4)), float(ra["配对差均值"]), src60)
        _numchk("S9 段1 均值配对差 CI 下", _sgn(m.group(5)), float(ra["CI下界"]), src60)
        _numchk("S9 段1 均值配对差 CI 上", _sgn(m.group(6)), float(ra["CI上界"]), src60)
        _numchk("S9 段1 Cliff δ", _sgn(m.group(7)), float(h1["Cliffs_delta"]), src50 + "[表c H1]")
        check("S9 段1 Cliff δ 属可忽略量级（|δ|<0.147）", "True",
              str(abs(float(h1["Cliffs_delta"])) < 0.147), src50 + "[表c H1]")

    # ── S9 段2：三档聚类，中位与均值各一套 ───────────────────────────────
    m = _rx("S9 段2 共同场景的聚类数", r"draw on (\d+) of the target sets and on all (\d+) source sets", t_en, SI)
    if m:
        check("S9 段2 目标集聚类数", m.group(1), str(int(r60(c60b, UNC, "目标集")["聚类数"])), src60)
        check("S9 段2 源集聚类数", m.group(2), str(int(r60(c60b, UNC, "源集")["聚类数"])), src60)
    m = _rx("S9 段2 未校正三档区间", r"confidence intervals of the median paired difference under scenario, "
            r"target-set and source-set clustering are " + _IV + ", " + _IV + " and " + _IV + r"\\,\\Brix\{\}, "
            r"and those of the mean paired difference " + _IV + ", " + _IV + " and " + _IV + r"\\,\\Brix\{\}",
            t_en, SI)
    if m:
        for j, (tab, stat) in enumerate(((c60b, "中位"), (c60a, "均值"))):
            for i, unit in enumerate(("场景", "目标集", "源集")):
                r = r60(tab, UNC, unit)
                off = 6 * j + 2 * i
                _numchk(f"S9 段2 未校正{stat}配对差 {unit}聚类 CI 下", _sgn(m.group(off + 1)), float(r["CI下界"]),
                        src60)
                _numchk(f"S9 段2 未校正{stat}配对差 {unit}聚类 CI 上", _sgn(m.group(off + 2)), float(r["CI上界"]),
                        src60)
    ub, ua = c60b[c60b["比较"] == UNC], c60a[c60a["比较"] == UNC]
    check("S9 段2 未校正：中位配对差三档 CI 全在 0 以下（典型场景领先）", "True",
          str(len(ub) == 3 and bool((ub["CI上界"] < 0).all())), src60)
    check("S9 段2 未校正：均值配对差三档 CI 全在 0 以上（平均意义落后）", "True",
          str(len(ua) == 3 and bool((ua["CI下界"] > 0).all())), src60)
    m = _rx("S9 段2 校正后三档区间", r"PLSR\+SB relative to Phys\+BL\+SB gives a median difference of "
            r"(\S+)\\,\\Brix\{\} with intervals " + _IV + ", " + _IV + " and " + _IV + r"\\,\\Brix\{\}, and a mean "
            r"difference of (\S+)\\,\\Brix\{\} with intervals " + _IV + ", " + _IV + " and " + _IV, t_en, SI)
    if m:
        for j, (tab, stat, col) in enumerate(((c60b, "中位", "配对差中位"), (c60a, "均值", "配对差均值"))):
            base = 7 * j
            _numchk(f"S9 段2 校正后 PLSR+SB−Phys+BL+SB {stat}配对差", _sgn(m.group(base + 1)),
                    float(r60(tab, PLS, "场景")[col]), src60)
            for i, unit in enumerate(("场景", "目标集", "源集")):
                r = r60(tab, PLS, unit)
                _numchk(f"S9 段2 校正后{stat}配对差 {unit}聚类 CI 下", _sgn(m.group(base + 2 + 2 * i)),
                        float(r["CI下界"]), src60)
                _numchk(f"S9 段2 校正后{stat}配对差 {unit}聚类 CI 上", _sgn(m.group(base + 3 + 2 * i)),
                        float(r["CI上界"]), src60)
    PAIRS2 = ("校正后 PLSR+SB − Phys+BL+SB", "校正后 CNN+SB − Phys+BL+SB")
    SVRK = "校正后 SVR+SB − Phys+BL+SB"
    for tab, stat in ((c60a, "均值"), (c60b, "中位")):
        sub = tab[tab["比较"].isin(PAIRS2)]
        check(f"S9 段2 校正后两条对比的六档{stat} CI 全在 0 以下", "True",
              str(len(sub) == 6 and bool((sub["CI上界"] < 0).all())), src60)
    svr_a, svr_b = c60a[c60a["比较"] == SVRK], c60b[c60b["比较"] == SVRK]
    # S11 段2 的「方向在中位与均值上都成立」只说校正后两处差距：句中须点明「校正后」，并与未校正时均值反向对照
    t_zh_si = _flat("Supplementary_zh.tex")
    for lang, text, pat in (("英", t_en, r"These corrected margins are small, but, unlike the uncorrected comparison, "
                                        r"their direction holds on the median and the mean alike"),
                            ("中", t_zh_si, r"校正后的这两处差距不大，但与未校正时不同，其方向在中位数与均值上")):
        check(f"S11 段2 {lang}「中位与均值同向」句指明校正后两处差距", "True", str(bool(re.search(pat, text))),
              "Supplementary_{en,zh}.tex si:apple-sbc")
    svr_mean_all = len(svr_a) == 3 and bool((svr_a["CI上界"] < 0).all())
    svr_med_scene_only = (len(svr_b) == 3 and float(r60(c60b, SVRK, "场景")["CI上界"]) < 0
                          and all(float(r60(c60b, SVRK, u)["CI下界"]) < 0 < float(r60(c60b, SVRK, u)["CI上界"])
                                  for u in ("目标集", "源集")))
    m = _rx("S9 段2 SVR+SB 三档", r"SVR\+SB is ahead of Phys\+BL\+SB under all three clustering units on the mean, "
            r"but on the median only under scenario clustering \(target-set interval " + _IV + r", source-set "
            r"interval " + _IV + r"\\,\\Brix\{\}\)", t_en, SI)
    if m:
        check("S9 段2 SVR+SB 均值三档 CI 全在 0 以下", "True", str(svr_mean_all), src60)
        check("S9 段2 SVR+SB 中位只在场景聚类下不跨 0", "True", str(svr_med_scene_only), src60)
        for i, unit in enumerate(("目标集", "源集")):
            r = r60(c60b, SVRK, unit)
            _numchk(f"S9 段2 SVR+SB 中位配对差 {unit}聚类 CI 下", _sgn(m.group(2 * i + 1)), float(r["CI下界"]), src60)
            _numchk(f"S9 段2 SVR+SB 中位配对差 {unit}聚类 CI 上", _sgn(m.group(2 * i + 2)), float(r["CI上界"]), src60)
    _rx("S9 段2 立论句点名 PLSR 与通用 CNN", r"lets PLSR and the generic CNN overtake the physics architecture, "
        r"therefore holds under the most conservative clustering and on either statistic", t_en, SI)

    # ── S9 段2：同年/跨年分层（58 号）与热图 ─────────────────────────────
    st = pd.read_excel(XL58, sheet_name="分层")
    src58 = "04outputs/58_cluster_robust_and_strata.xlsx[分层]"
    SY, CY = "同年跨产地", "跨年(含跨仪器)"

    def s58(lab: str, meth: str, col: str) -> float:
        return float(st[(st["分层"] == lab) & (st["方法"] == meth)][col].iloc[0])

    n_sy = int(st[st["分层"] == SY]["场景数"].iloc[0])
    n_cy = int(st[st["分层"] == CY]["场景数"].iloc[0])
    check("S9 分层 两层场景数之和等于共同场景数", str(n_common), str(n_sy + n_cy), src58)
    m = _rx("S9 分层场景数", r"same-year cross-origin \((\d+) scenarios\) and cross-year including "
            r"cross-instrument \((\d+) scenarios\)", t_en, SI)
    if m:
        for i, lab in enumerate((SY, CY)):
            check(f"S9 分层场景数（{lab}）", m.group(i + 1), str(int(st[st["分层"] == lab]["场景数"].iloc[0])), src58)
    m = _rx("S9 分层校正后", r"PLSR\+SB against Phys\+BL\+SB, median ([\d.]+) against ([\d.]+)\\,\\Brix\{\} "
            r"same-year and ([\d.]+) against ([\d.]+)\\,\\Brix\{\} cross-year\)", t_en, SI)
    if m:
        for i, lab in enumerate((SY, CY)):
            _numchk(f"S9 分层 PLSR+SB 中位（{lab}）", m.group(2 * i + 1), s58(lab, "PLSR + SB", "中位RMSE"), src58)
            _numchk(f"S9 分层 Phys+BL+SB 中位（{lab}）", m.group(2 * i + 2), s58(lab, "Phys+BL + SB", "中位RMSE"),
                    src58)
    for lab in (SY, CY):
        for col in ("中位RMSE", "均值RMSE"):
            check(f"S9 分层 校正后 PLSR+SB 领先（{lab}，{col}）", "True",
                  str(s58(lab, "PLSR + SB", col) < s58(lab, "Phys+BL + SB", col)), src58)
    m = _rx("S9 分层 未校正跨年", r"carried by the cross-year stratum \(median ([\d.]+)\\,\\Brix\{\} for "
            r"Phys\+BL against ([\d.]+)\\,\\Brix\{\} for the CNN\)", t_en, SI)
    if m:
        _numchk("S9 分层 跨年 Phys+BL 中位", m.group(1), s58(CY, "Phys+BL", "中位RMSE"), src58)
        _numchk("S9 分层 跨年 CNN 中位", m.group(2), s58(CY, "CNN", "中位RMSE"), src58)
    m = _rx("S9 分层 同年深度模型", r"within ([\d.]+)\\,\\Brix\{\} of each other on either statistic \(median "
            r"([\d.]+) against ([\d.]+), mean ([\d.]+) against ([\d.]+)\\,\\Brix\{\}\)", t_en, SI)
    if m:
        _numchk("S9 分层 同年 Phys+BL 中位", m.group(2), s58(SY, "Phys+BL", "中位RMSE"), src58)
        _numchk("S9 分层 同年 CNN 中位", m.group(3), s58(SY, "CNN", "中位RMSE"), src58)
        _numchk("S9 分层 同年 Phys+BL 均值", m.group(4), s58(SY, "Phys+BL", "均值RMSE"), src58)
        _numchk("S9 分层 同年 CNN 均值", m.group(5), s58(SY, "CNN", "均值RMSE"), src58)
        gap = max(abs(s58(SY, "Phys+BL", c) - s58(SY, "CNN", c)) for c in ("中位RMSE", "均值RMSE"))
        check(f"S9 分层 同年两深度模型相差在 {m.group(1)} 以内", "True", str(gap < float(m.group(1))),
              src58 + f"（实测 {gap:.3f}）")
    m = _rx("S9 分层 PLSR 均值", r"mean PLSR RMSE rising from ([\d.]+) to ([\d.]+)\\,\\Brix", t_en, SI)
    if m:
        for i, lab in enumerate((SY, CY)):
            _numchk(f"S9 分层 PLSR 均值 RMSE（{lab}）", m.group(i + 1), s58(lab, "PLSR", "均值RMSE"), src58)
    hraw = raw[(raw["方法"] == "Phys+BL") & (raw["校正"] == "裸")].copy()
    sp = hraw["场景"].astype(str).str.split("→", expand=True)
    hraw["src"], hraw["tgt"] = sp[0], sp[1]
    hraw = hraw[~hraw["src"].str.contains(r"\+") & ~hraw["tgt"].str.contains(r"\+")]
    cell = hraw.groupby(["src", "tgt"])["RMSE"].median()
    by_t = cell.groupby(level="tgt").median().sort_values()
    by_s = cell.groupby(level="src").median()
    src_h = "04outputs/50_formal_multiseed_benchmark.xlsx[表h]（一源→一目标、Phys+BL 跨种子中位，按目标域取中位）"
    if _rx("S9 热图句", r"transfer to 2018 Xinjiang and 2025 Shandong being hardest and to 2018 Shandong easiest",
           t_en, SI):
        check("S9 热图 最难的两个目标域", "['2018_新疆', '2025_山东']", str(sorted(by_t.index[-2:])), src_h)
        check("S9 热图 最易的目标域", "2018_山东", str(by_t.index[0]), src_h)
        check("S9 热图 难度主要由目标域决定（目标域中位的极差大于源域）", "True",
              str(float(by_t.max() - by_t.min()) > float(by_s.max() - by_s.min())), src_h)

    # ── S9 段3：0%/10%/15% 三档阈值（各档在各自的七法共同场景上汇总）──────────────
    lv = {}
    for tag, xl in (("0", XLP0), ("10", XL50), ("15", XLP15)):
        g_ = (pd.read_excel(xl, sheet_name="表g：每场景跨种子均值")
              .pivot(index="场景", columns="方法_校正", values="RMSE"))
        q_ = g_.loc[g_[CASE_ALL].dropna().index]
        raw_ = pd.read_excel(xl, sheet_name="表h：全部原始结果")
        bare_ = raw_[raw_["校正"] == "裸"]
        sd_ = sorted(int(x) for x in bare_["种子"].unique())
        ix_ = pd.MultiIndex.from_product([list(q_.index), sd_], names=["场景", "种子"])
        dv_ = {}
        for mth in ("Phys", "Phys+BL"):   # 发散 = 共同场景 × 种子里无结果或 RMSE>10 的运行
            r_ = (bare_[(bare_["方法"] == mth) & bare_["场景"].isin(q_.index)]
                  .set_index(["场景", "种子"])["RMSE"].reindex(ix_))
            dv_[mth] = 100 * float(r_.isna().sum() + (r_ > 10).sum()) / len(ix_)
        lv[tag] = {"q": q_, "med": q_.median(), "mean": q_.mean(), "div": dv_}
    srcp = "04outputs/c6_p0.xlsx + 50_formal_multiseed_benchmark.xlsx + c6_p15.xlsx[表g/表h]（各档七法共同场景）"
    tags = ("0", "10", "15")
    PBSB = "Phys+BL + SB"
    m = _rx("S9 阈值 各档共同场景数",
            r"its own common scenarios \((\d+), (\d+) and (\d+) at 0\\%, 10\\% and 15\\% removal\)", t_en, SI)
    if m:
        for i, tag in enumerate(tags):
            check(f"S9 阈值 {tag}% 档共同场景数", m.group(i + 1), str(len(lv[tag]["q"])), srcp)
    m = _rx("S9 阈值校正后", r"PLSR\+SB and CNN\+SB stay ahead of Phys\+BL\+SB at every threshold and on both "
            r"statistics \(median ([\d.]+), ([\d.]+) and ([\d.]+)\\,\\Brix\{\} for PLSR\+SB and "
            r"([\d.]+), ([\d.]+) and ([\d.]+)\\,\\Brix\{\} "
            r"for CNN\+SB against ([\d.]+), ([\d.]+) and ([\d.]+)\\,\\Brix\{\} at 0\\%, 10\\% and 15\\% removal\); "
            r"SVR\+SB does "
            r"so at 0\\% and 10\\%, but at 15\\% it is level with Phys\+BL\+SB on the median "
            r"\(([\d.]+) against ([\d.]+)\\,\\Brix\{\}, "
            r"paired Wilcoxon \$P=([\d.]+)\$\) while remaining lower on the mean", t_en, SI)
    if m:
        for i, tag in enumerate(tags):
            _numchk(f"S9 阈值 {tag}% 档 PLSR+SB 中位", m.group(1 + i), lv[tag]["med"]["PLSR + SB"], srcp)
            _numchk(f"S9 阈值 {tag}% 档 CNN+SB 中位", m.group(4 + i), lv[tag]["med"]["CNN + SB"], srcp)
            _numchk(f"S9 阈值 {tag}% 档 Phys+BL+SB 中位", m.group(7 + i), lv[tag]["med"][PBSB], srcp)
        q15 = lv["15"]["q"]
        _numchk("S9 阈值 15% 档 SVR+SB 中位", m.group(10), lv["15"]["med"]["SVR + SB"], srcp)
        _numchk("S9 阈值 15% 档 Phys+BL+SB 中位（对 SVR 一句）", m.group(11), lv["15"]["med"][PBSB], srcp)
        _numchk("S9 阈值 15% 档 SVR+SB 对 Phys+BL+SB 配对 P", m.group(12),
                float(wilcoxon(q15["SVR + SB"].to_numpy(), q15[PBSB].to_numpy()).pvalue), srcp)
    for tag in tags:
        md_, mn_ = lv[tag]["med"], lv[tag]["mean"]
        for c in ("PLSR + SB", "CNN + SB"):
            check(f"S9 阈值 {tag}% 档 {c} 中位与均值都低于 Phys+BL+SB", "True",
                  str(bool(md_[c] < md_[PBSB] and mn_[c] < mn_[PBSB])), srcp)
        check(f"S9 阈值 {tag}% 档 SVR+SB 均值低于 Phys+BL+SB", "True", str(bool(mn_["SVR + SB"] < mn_[PBSB])), srcp)
    for tag in ("0", "10"):
        check(f"S9 阈值 {tag}% 档 SVR+SB 中位低于 Phys+BL+SB", "True",
              str(bool(lv[tag]["med"]["SVR + SB"] < lv[tag]["med"][PBSB])), srcp)
    m = _rx("S9 阈值未校正", r"Phys\+BL keeps the lowest median at 10\\% and 15\\% removal "
            r"\(([\d.]+) and ([\d.]+)\\,\\Brix\{\}\) "
            r"but not at 0\\%, where CNN\+MMD is lower \(([\d.]+) against ([\d.]+)\\,\\Brix\{\}\) "
            r"and the edge of Phys\+BL over "
            r"the CNN \(([\d.]+) against ([\d.]+)\\,\\Brix\{\}, lower in (\d+)\\% of the scenarios\) is not resolvable "
            r"\(Wilcoxon \$P=([\d.]+)\$\)", t_en, SI)
    if m:
        q0 = lv["0"]["q"]
        _numchk("S9 阈值 10% 档 Phys+BL 中位", m.group(1), lv["10"]["med"]["Phys+BL"], srcp)
        _numchk("S9 阈值 15% 档 Phys+BL 中位", m.group(2), lv["15"]["med"]["Phys+BL"], srcp)
        _numchk("S9 阈值 0% 档 CNN+MMD 中位", m.group(3), lv["0"]["med"]["CNN+MMD"], srcp)
        _numchk("S9 阈值 0% 档 Phys+BL 中位", m.group(4), lv["0"]["med"]["Phys+BL"], srcp)
        _numchk("S9 阈值 0% 档 Phys+BL 中位（对 CNN 一句）", m.group(5), lv["0"]["med"]["Phys+BL"], srcp)
        _numchk("S9 阈值 0% 档 CNN 中位", m.group(6), lv["0"]["med"]["CNN"], srcp)
        _numchk("S9 阈值 0% 档 Phys+BL 低于 CNN 的场景占比%", m.group(7),
                100 * float((q0["Phys+BL"] < q0["CNN"]).mean()), srcp)
        _numchk("S9 阈值 0% 档 Phys+BL 对 CNN 的 Wilcoxon P", m.group(8),
                float(wilcoxon(q0["Phys+BL"].to_numpy(), q0["CNN"].to_numpy()).pvalue), srcp)
    for tag, want in (("0", "CNN+MMD"), ("10", "Phys+BL"), ("15", "Phys+BL")):
        check(f"S9 阈值 {tag}% 档 未校正中位最低者", want, str(lv[tag]["med"][CASE_BARE].idxmin()), srcp)
    m = _rx("S9 阈值发散", r"falls as more outliers are removed: ([\d.]+)\\% of Phys and ([\d.]+)\\% of Phys\+BL "
            r"scenario--seed "
            r"runs diverge at 0\\%, ([\d.]+)\\% and ([\d.]+)\\% at 10\\%, and ([\d.]+)\\% and ([\d.]+)\\% at 15\\%",
            t_en, SI)
    if m:
        for i, tag in enumerate(tags):
            _numchk(f"S9 阈值 {tag}% 档 Phys 发散率%", m.group(2 * i + 1), lv[tag]["div"]["Phys"], srcp)
            _numchk(f"S9 阈值 {tag}% 档 Phys+BL 发散率%", m.group(2 * i + 2), lv[tag]["div"]["Phys+BL"], srcp)
    for mth in ("Phys", "Phys+BL"):
        check(f"S9 阈值 {mth} 发散率随剔除比例升高而下降", "True",
              str(lv["0"]["div"][mth] > lv["10"]["div"][mth] > lv["15"]["div"][mth]), srcp)
    m = _rx("S9 阈值均值", r"the uncorrected mean RMSE of Phys\+BL is ([\d.]+)\\,\\Brix\{\} at 0\\% but at 15\\% "
            r"the lowest of the "
            r"five methods \(([\d.]+) against ([\d.]+)\\,\\Brix\{\} for the CNN\)", t_en, SI)
    if m:
        _numchk("S9 阈值 0% 档 Phys+BL 未校正均值", m.group(1), lv["0"]["mean"]["Phys+BL"], srcp)
        _numchk("S9 阈值 15% 档 Phys+BL 未校正均值", m.group(2), lv["15"]["mean"]["Phys+BL"], srcp)
        _numchk("S9 阈值 15% 档 CNN 未校正均值", m.group(3), lv["15"]["mean"]["CNN"], srcp)
        check("S9 阈值 15% 档 未校正均值最低者", "Phys+BL", str(lv["15"]["mean"][CASE_BARE].idxmin()), srcp)

    # ── S9：幸存者敏感性（72 号：三种补齐口径 × 两统计量 + 秩口径）──────────────
    t72a = pd.read_excel(XL72, sheet_name="表a：缺失结构诊断")
    t72c = pd.read_excel(XL72, sheet_name="表c：核心排序是否翻转")
    t72d = pd.read_excel(XL72, sheet_name="表d：S4秩口径")
    src72 = "04outputs/72_convergence_survivorship.xlsx"
    m = _rx("S9 幸存者段舍去场景数", r"at the price of dropping (\d+) scenarios", t_en, SI)
    if m:
        check("S9 幸存者段舍去的场景数", m.group(1), str(int(t72a["覆盖场景"].max()) - n_common), src72)
    m = _rx("S9 缺失结构诊断", r"the PLSR error is higher than on the rest \(median ([\d.]+) against "
            r"([\d.]+)\\,\\Brix\{\}\), although the difference is not resolved "
            r"\(Mann--Whitney \$P=([\d.]+)\$\)", t_en, SI)
    if m:
        row = t72a[t72a["方法"] == "Phys+BL"].iloc[0]
        _numchk("S9 缺失场景的 PLSR 中位 RMSE", m.group(1), float(row["缺失场景_PLSR中位RMSE"]), src72)
        _numchk("S9 其余场景的 PLSR 中位 RMSE", m.group(2), float(row["其余场景_PLSR中位RMSE"]), src72)
        _numchk("S9 缺失与其余的 Mann-Whitney P", m.group(3), float(row["MannWhitney_P"]), src72)
    # 稿件的三种补齐口径 = 表 c 的 S0（各自跑出的场景）、S2（最坏观测）、S3（发散阈值）；S1 即主分析
    three = t72c[t72c["口径"].astype(str).str.match(r"S[023]\b")]
    check("S9 幸存者 三种补齐口径齐全", "3", str(len(three)), src72 + "[表c]")
    for col, want in (("排序①成立_median", "True"), ("排序②成立_median", "True"),
                      ("排序②成立_mean", "True"), ("排序①成立_mean", "False")):
        check(f"S9 幸存者 三口径 {col} 全为 {want}", "True",
              str(bool(three[col].astype(str).eq(want).all())), src72 + "[表c]")
    check("S9 幸存者 均值下未校正最优者是 CNN+MMD 或 CNN", "True",
          str(bool(three["未校正最优_mean"].isin(["CNN+MMD", "CNN"]).all())), src72 + "[表c]")
    m = _rx("S9 秩口径", r"Phys\+BL holds the best average rank when uncorrected \(([\d.]+)\) while "
            r"PLSR\+SB holds it after correction \(([\d.]+)\)", t_en, SI)
    if m:
        bare_r = t72d[t72d["校正"] == "未校正"].set_index("方法")["平均名次"]
        sb_r = t72d[t72d["校正"] == "+SB"].set_index("方法")["平均名次"]
        _numchk("S9 秩口径 未校正 Phys+BL 平均名次", m.group(1), float(bare_r["Phys+BL"]), src72)
        _numchk("S9 秩口径 校正后 PLSR 平均名次", m.group(2), float(sb_r["PLSR"]), src72)
        check("S9 秩口径 未校正名次最优者是 Phys+BL", "True", str(bare_r.idxmin() == "Phys+BL"), src72)
        check("S9 秩口径 校正后名次最优者是 PLSR", "True", str(sb_r.idxmin() == "PLSR"), src72)

    # ── S9：严格零标签对照与经典预处理基线（59 号）────────────────────────
    if not XL59.exists():
        check("S9 零标签合并表存在", "True", "False", "04outputs/59_reviewer_experiments.xlsx")
    else:
        z = pd.read_excel(XL59, sheet_name="原始")
        src59 = "04outputs/59_reviewer_experiments.xlsx[原始]（59 号严格零标签对照）"
        zp = z.pivot_table(index="场景", columns="方法", values="RMSE", aggfunc="mean")
        deep3 = ["CNN", "CNN+MMD", "Phys+BL"]
        cls4 = ["PLSR+SNV", "PLSR+MSC", "PLSR+SG1", "PLSR+SG2"]
        zc = zp[deep3 + cls4].dropna().index
        zmed = zp.loc[zc].median()
        m = _rx("S9 严格零标签", r"Over the (\d+) scenarios in which all seven methods returned a result, and even "
                r"stripped of target-label early stopping, Phys\+BL still attains the lowest median RMSE, "
                r"([\d.]+)\\,\\Brix\{\}, ahead of the generic convolutional network \(([\d.]+)\\,\\Brix\{\}\), "
                r"CNN\+MMD \(([\d.]+)\\,\\Brix\{\}\) and the best classical baseline PLSR\+SNV \(([\d.]+)", t_en, SI)
        if m:
            check("S9 零标签共同场景数", m.group(1), str(len(zc)), src59)
            for i, meth in enumerate(("Phys+BL", "CNN", "CNN+MMD", "PLSR+SNV")):
                _numchk(f"S9 零标签 {meth} 中位 RMSE", m.group(i + 2), float(zmed[meth]), src59)
            check("S9 零标签下中位最低的是 Phys+BL", "True", str(zmed[deep3 + cls4].idxmin() == "Phys+BL"), src59)
        cov59 = z.groupby("方法")["场景"].nunique()
        m = _rx("S9 零标签收敛数", r"the numbers of scenarios with a result are (\d+) for the CNN and (\d+) for "
                r"CNN\+MMD and Phys\+BL", t_en, SI)
        if m:
            check("S9 零标签 CNN 收敛场景数", m.group(1), str(int(cov59["CNN"])), src59)
            check("S9 零标签 CNN+MMD 收敛场景数", m.group(2), str(int(cov59["CNN+MMD"])), src59)
            check("S9 零标签 Phys+BL 收敛场景数", m.group(2), str(int(cov59["Phys+BL"])), src59)
        m = _rx("S9 零标签失败尾", r"Phys\+BL exceeds 10\\,\\Brix\{\} in (\d+) of its (\d+) scenarios, the CNN in "
                r"none of its (\d+)", t_en, SI)
        if m:
            pz, cz = zp["Phys+BL"].dropna(), zp["CNN"].dropna()
            check("S9 零标签 Phys+BL >10 的场景数", m.group(1), str(int((pz > 10).sum())), src59)
            check("S9 零标签 Phys+BL 场景数", m.group(2), str(len(pz)), src59)
            check("S9 零标签 CNN 场景数", m.group(3), str(len(cz)), src59)
            check("S9 零标签 CNN 无一 >10", "0", str(int((cz > 10).sum())), src59)
        both = sorted(set(zc) & set(piv.index))
        m = _rx("S9 经典基线", r"these, PLSR\+SNV, gives a median RMSE of ([\d.]+)\\,\\Brix\{\}, behind both the "
                r"generic convolutional network \(([\d.]+)\\,\\Brix\{\}\) and the physics factorisation "
                r"architecture when uncorrected \(([\d.]+)\\,\\Brix\{\}\).{0,200}?over the (\d+) scenarios the two "
                r"experiments have in common", t_en, SI)
        if m:
            check("S9 两实验共同场景数", m.group(4), str(len(both)), src59 + " + " + src50)
            _numchk("S9 经典基线 PLSR+SNV 中位", m.group(1), float(zp.loc[both, "PLSR+SNV"].median()), src59)
            _numchk("S9 20% 早停 CNN 中位", m.group(2), float(piv.loc[both, "CNN"].median()), src50)
            _numchk("S9 20% 早停 Phys+BL 中位", m.group(3), float(piv.loc[both, "Phys+BL"].median()), src50)
        m = _rx("S9 经典基线不稳定", r"exceeds 10\\,\\Brix\{\} in (\d+) and (\d+) of the (\d+) scenarios "
                r"respectively, against (\d+) for SNV", t_en, SI)
        if m:
            for i, meth in enumerate(("PLSR+MSC", "PLSR+SG2")):
                col = zp[meth].dropna()
                check(f"S9 {meth} 中位 RMSE>10 的场景数", m.group(i + 1), str(int((col > 10).sum())), src59)
            check("S9 经典基线的场景总数", m.group(3), str(int(zp["PLSR+SNV"].notna().sum())), src59)
            check("S9 PLSR+SNV 中位 RMSE>10 的场景数", m.group(4),
                  str(int((zp["PLSR+SNV"].dropna() > 10).sum())), src59)
        m = _rx("S9 零标签对 20% 早停", r"its median RMSE being ([\d.]+)\\,\\Brix\{\} under the strict zero-label "
                r"protocol against ([\d.]+)\\,\\Brix\{\} under 20\\% label early stopping over the (\d+) scenarios "
                r"the two experiments share", t_en, SI)
        if m:
            _numchk("S9 Phys+BL 严格零标签中位（两实验交集）", m.group(1), float(zp.loc[both, "Phys+BL"].median()),
                    src59)
            _numchk("S9 Phys+BL 20% 早停中位（两实验交集）", m.group(2), float(piv.loc[both, "Phys+BL"].median()),
                    src50)
            check("S9 该对比的场景数", m.group(3), str(len(both)), src59 + " + " + src50)
        sub59 = re.sub(r"\s+", " ",
                       (BASE / "06doc/02sub/revision/_submitted_Elsevier_en.tex").read_text(encoding="utf-8"))
        m = _rx("R1 零标签", r"still has the lowest median of the seven methods compared \(([\d.]+)\\,\\Brix\{\}\), "
                r"but "
                r"target-label early stopping now helps it \(([\d.]+)\\,\\Brix\{\} on the scenarios the two "
                r"experiments "
                r"share\)", _letter("Response_R1.tex"), "06doc/02sub/revision/Response_R1.tex")
        if m:
            _numchk("R1 零标签 Phys+BL 中位", m.group(1), float(zmed["Phys+BL"]), src59)
            _numchk("R1 20% 早停 Phys+BL 中位（两实验交集）", m.group(2), float(piv.loc[both, "Phys+BL"].median()),
                    src50)
            check("R1 所引投出稿「两种协议几乎相同」出自原稿", "True",
                  str("is almost identical to its level under 20" in sub59), "_submitted_Elsevier_en.tex")
    sb_all = piv[[c for c in piv.columns if c.endswith(" + SB")]].median()
    m = _rx("S9 校正后最优", r"After slope/bias correction, PLSR\+SB \(([\d.]+)\\,\\Brix\{\}\) remains the best "
            r"overall", t_en, SI)
    if m:
        _numchk("S9 PLSR+SB 中位", m.group(1), float(sb_all["PLSR + SB"]), src50)
        check("S9 校正后中位最低者是 PLSR+SB（七个 +SB 变体中）", "PLSR + SB", str(sb_all.idxmin()), src50)

    # ── S9：RPD、偏差、R²>0、CNN+MMD、迁移类型 ───────────────────────────
    m = _rx("S9 RPD 与偏差", r"The mean RPD of Phys\+BL is the highest of the five, ([\d.]+) against ([\d.]+) for "
            r"the CNN, and the mean bias is negative for every method: (\S+)\\,\\Brix\{\} for PLSR, (\S+) for SVR, "
            r"(\S+) for the CNN, (\S+) for Phys\+BL and (\S+)\\,\\Brix\{\} for CNN\+MMD", t_en, SI)
    if m:
        _numchk("S9 Phys+BL 均值 RPD", m.group(1), float(a2.loc["Phys+BL", "RPD_mean"]), src50)
        _numchk("S9 CNN 均值 RPD", m.group(2), float(a2.loc["CNN", "RPD_mean"]), src50)
        check("S9 均值 RPD 最高者是 Phys+BL", "Phys+BL", str(a2.loc[CASE_BARE, "RPD_mean"].idxmax()), src50)
        for i, meth in enumerate(("PLSR", "SVR", "CNN", "Phys+BL", "CNN+MMD")):
            _numchk(f"S9 {meth} 均值偏差", _sgn(m.group(i + 3)), float(a2.loc[meth, "Bias_mean"]), src50)
        check("S9 五法均值偏差全为负", "True", str(bool((a2.loc[CASE_BARE, "Bias_mean"] < 0).all())), src50)
    m = _rx("S9 R²>0", r"greater than zero are only: "
            + ", ".join(re.escape(x) + r" (\d+)/(\d+)" for x in CASE_BARE[:-1])
            + r" and Phys\+BL (\d+)/(\d+)", t_en, SI)
    if m:
        got = list(m.groups())
        for i, meth in enumerate(CASE_BARE):
            check(f"S9 R²>0 场景数 {meth}", got[2 * i], str(int(a2.loc[meth, "R2大于0场景数"])), src50)
            check(f"S9 R²>0 分母 {meth}", got[2 * i + 1], str(n_common), src50)
    m = _rx("S9 R²>0 占比", r"genuinely learns something useful in only ([\d.]+)\\% of scenarios", t_en, SI)
    if m:
        _numchk("S9 Phys+BL R²>0 占比%", m.group(1), 100 * float(a2.loc["Phys+BL", "R2大于0场景数"]) / n_common, src50)
        check("S9 R²>0 最多者是 Phys+BL", "Phys+BL", str(a2.loc[CASE_BARE, "R2大于0场景数"].idxmax()), src50)
    m = _rx("S9 CNN+MMD 与 CNN", r"CNN\+MMD is on a par with the CNN \(mean ([\d.]+) against ([\d.]+)\\,\\Brix\{\}, "
            r"median ([\d.]+) against ([\d.]+)\\,\\Brix\{\}\)", t_en, SI)
    if m:
        for i, (meth, stat) in enumerate((("CNN+MMD", "mean"), ("CNN", "mean"),
                                          ("CNN+MMD", "median"), ("CNN", "median"))):
            _numchk(f"S9 {meth} RMSE {stat}", m.group(i + 1), float(getattr(piv[meth], stat)()), src50)
    g = pd.read_excel(XL50, sheet_name="表g：每场景跨种子均值")
    typ = g.drop_duplicates("场景").set_index("场景")["迁移类型"].reindex(piv.index)
    by_type = (ph - cnn).groupby(typ).median()
    m = _rx("S9 迁移类型", r"largest in the one-source\$\\to\$one-target scenarios \(median paired difference "
            r"(\S+)\\,\\Brix\{\}\) and between (\S+) and (\S+)\\,\\Brix\{\} in the other three types", t_en, SI)
    if m:
        src_t = src50 + "[表g 迁移类型]"
        _numchk("S9 迁移类型 1→1 中位配对差", _sgn(m.group(1)), float(by_type["1源→1目标"]), src_t)
        rest = by_type.drop(index="1源→1目标")
        _numchk("S9 迁移类型 其余三类下界", _sgn(m.group(2)), float(rest.min()), src_t)
        _numchk("S9 迁移类型 其余三类上界", _sgn(m.group(3)), float(rest.max()), src_t)
        check("S9 迁移类型 1→1 的优势最大", "1源→1目标", str(by_type.idxmin()), src_t)

    # ── S10：三组消融、发散与 McNemar、逐种子 ─────────────────────────────
    phys, cbl = piv["Phys"], piv["CNN+BL"]
    p_arch = float(wilcoxon(phys.to_numpy(), cnn.to_numpy()).pvalue)
    check("S10 消融组数写作三组", "True", str("three ablations were run over the" in t_en), SI)
    m = _rx("S10 消融一", r"CNN\+BL has the lower median RMSE \(([\d.]+) versus ([\d.]+)\\,\\Brix\{\}\), yet scenario "
            r"by scenario the paired improvement has a median of (\S+)\\%, and the two-sided Wilcoxon test "
            r"resolves no difference \(\$P=([\d.]+)\$, n=(\d+)\); neither variant ever diverges", t_en, SI)
    if m:
        _numchk("S10 消融一 CNN+BL 中位", m.group(1), cbl.median(), src50)
        _numchk("S10 消融一 CNN 中位", m.group(2), cnn.median(), src50)
        _numchk("S10 消融一 配对改进中位%", _sgn(m.group(3)), (100 * (cnn - cbl) / cnn).median(), src50)
        _numchk("S10 消融一 P", m.group(4), float(wilcoxon(cbl.to_numpy(), cnn.to_numpy()).pvalue), src50)
        check("S10 消融一 n", str(n_common), m.group(5), src50)
        check("S10 消融一 两者都无发散", "0", str(int(dv["CNN+BL"].sum() + dv["CNN"].sum())), src_ab)
    any_p = dv["Phys"].groupby(level="场景").any()
    any_pp = (dv["Phys"] | dv["Phys+BL"]).groupby(level="场景").any()
    ok2 = any_pp[~any_pp].index
    m = _rx("S10 消融二", r"\(Phys, median RMSE ([\d.]+)\\,\\Brix\{\}\) and Phys\+BL \(median ([\d.]+)\\,\\Brix\{\}\) "
            r"differ by a median paired improvement of (\S+)\\% \(\$P=([\d.]+)\$\), and over the (\d+) scenarios in "
            r"which neither physics model diverged in any seed the difference is (\S+)\\% \(\$P=([\d.]+)\$\)", t_en, SI)
    if m:
        _numchk("S10 消融二 Phys 中位", m.group(1), phys.median(), src50)
        _numchk("S10 消融二 Phys+BL 中位", m.group(2), ph.median(), src50)
        _numchk("S10 消融二 配对改进中位%", _sgn(m.group(3)), (100 * (phys - ph) / phys).median(), src50)
        _numchk("S10 消融二 P", m.group(4), float(wilcoxon(ph.to_numpy(), phys.to_numpy()).pvalue), src50)
        check("S10 消融二 两物理模型均未发散的场景数", m.group(5), str(len(ok2)), src_ab)
        a, b = ph[ok2], phys[ok2]
        _numchk("S10 消融二 未发散场景上的改进中位%", _sgn(m.group(6)), (100 * (b - a) / b).median(), src_ab)
        _numchk("S10 消融二 未发散场景上的 P", m.group(7), float(wilcoxon(a.to_numpy(), b.to_numpy()).pvalue), src_ab)
    m = _rx("S10 发散率", r"the plain physics model diverges in ([\d.]+)\\% of scenario--seed runs \((\d+) of (\d+), "
            r"of which (\d+) returned no result\) whereas Phys\+BL diverges in ([\d.]+)\\% \((\d+) of (\d+), (\d+) of "
            r"them returning no result\), an exact McNemar test over the discordant runs giving \$P=" + _SCI + r"\$",
            t_en, SI)
    if m:
        for meth, g0 in (("Phys", 1), ("Phys+BL", 5)):
            _numchk(f"S10 {meth} 发散率%", m.group(g0), rate(meth), src_ab)
            check(f"S10 {meth} 发散次数", m.group(g0 + 1), str(int(dv[meth].sum())), src_ab)
            check(f"S10 {meth} 运行数", m.group(g0 + 2), str(n_runs), src_ab)
            check(f"S10 {meth} 其中无结果产出", m.group(g0 + 3), str(int(miss[meth].sum())), src_ab)
        _scichk("S10 McNemar 精确 P", f"${m.group(9)}$", p_mc, src_ab)
    fold = rate("Phys") / rate("Phys+BL")
    m = _rx("S10 降低倍数", r"reducing divergence roughly ([\d.]+)-fold without removing it", t_en, SI)
    if m:
        _numchk("S10 BL 损失降低发散的倍数", m.group(1), fold, src_ab)
    m = _rx("S10 消融三", r"Over all (\d+) scenarios the plain physics model reaches a median RMSE of "
            r"([\d.]+)\\,\\Brix\{\} against the CNN's ([\d.]+)\\,\\Brix\{\}, a median paired improvement of "
            r"\$\+\$([\d.]+)\\% \(\$P=([\d.]+)\$, n=(\d+)\), and is the better model in (\d+) of the (\d+) scenarios",
            t_en, SI)
    if m:
        check("S10 消融三 场景数", str(n_common), m.group(1), src50)
        _numchk("S10 消融三 Phys 中位 RMSE", m.group(2), phys.median(), src50)
        _numchk("S10 消融三 CNN 中位 RMSE", m.group(3), cnn.median(), src50)
        _numchk("S10 消融三 配对改进中位%", m.group(4), (100 * (cnn - phys) / cnn).median(), src50)
        _numchk("S10 消融三 P", m.group(5), p_arch, src50)
        check("S10 消融三 n", str(n_common), m.group(6), src50)
        check("S10 消融三 Phys 更优的场景数", str(int((phys < cnn).sum())), m.group(7), src50)
        check("S10 消融三 分母", str(n_common), m.group(8), src50)
    check("S10 消融三 「0.05 水平下未分辨」与 P 一致", "True",
          str(("it is not resolved at the 0.05 level even before any correction for multiplicity" in t_en)
              and p_arch > 0.05), src50 + f"（实际 P={p_arch:.3g}）")
    ok3 = any_p[~any_p].index
    m = _rx("S10 消融三 发散与条件化", r"the plain physics model's mean RMSE over these scenarios is "
            r"([\d.]+)\\,\\Brix\{\} "
            r"against the CNN's ([\d.]+)\\,\\Brix\{\}, and (\d+) of the (\d+) scenarios contain at least one divergent "
            r"Phys run\. Over the (\d+) scenarios in which the plain physics model never diverged, the contrast is "
            r"\$\+\$([\d.]+)\\% \(\$P=" + _SCI + r"\$\)", t_en, SI)
    if m:
        _numchk("S10 消融三 Phys 均值 RMSE", m.group(1), phys.mean(), src50)
        _numchk("S10 消融三 CNN 均值 RMSE", m.group(2), cnn.mean(), src50)
        check("S10 消融三 含 Phys 发散运行的场景数", m.group(3), str(int(any_p.sum())), src_ab)
        check("S10 消融三 场景总数", m.group(4), str(n_common), src50)
        check("S10 消融三 Phys 从未发散的场景数", m.group(5), str(len(ok3)), src_ab)
        a, b = phys[ok3], cnn[ok3]
        _numchk("S10 消融三 未发散场景上的改进中位%", m.group(6), (100 * (b - a) / b).median(), src_ab)
        _scichk("S10 消融三 未发散场景上的 P", f"${m.group(7)}$",
                float(wilcoxon(a.to_numpy(), b.to_numpy()).pvalue), src_ab)
    m = _rx("S10 两者组合对 CNN", r"that is resolved against the CNN over all (\d+) scenarios \(\$\+\$([\d.]+)\\%, "
            r"\$P=" + _SCI + r"\$\)", t_en, SI)
    if m:
        check("S10 组合对 CNN 场景数", str(n_common), m.group(1), src50)
        _numchk("S10 组合对 CNN 改进中位%", m.group(2), imp.median(), src50)
        _scichk("S10 组合对 CNN 未校正 P", f"${m.group(3)}$", float(h1["Wilcoxon_p"]), src50 + "[表c H1]")
    m = _rx("S10 逐种子", r"same 572 scenarios \(seeds ([\d, and]+?) in that order\), the plain physics model diverges "
            r"([\d, and]+?) times \(([\d.]+)\\% to ([\d.]+)\\% of that seed's runs\) while Phys\+BL diverges "
            r"([\d, and]+?) times \(([\d.]+)\\% to ([\d.]+)\\%\)", t_en, SI)
    if m:
        order = [int(v) for v in re.findall(r"\d+", m.group(1))]
        check("S10 逐种子 所列种子即五粒正式种子", str(sorted(seeds)), str(sorted(order)), src_ab)
        for meth, gl, glo, ghi in (("Phys", 2, 3, 4), ("Phys+BL", 5, 6, 7)):
            per_s = dv[meth].groupby(level="种子").sum()
            got = [int(per_s.get(sd, -1)) for sd in order]
            check(f"S10 逐种子 {meth} 发散次数（按种子对齐）", str([int(v) for v in re.findall(r"\d+", m.group(gl))]),
                  str(got), src_ab)
            _numchk(f"S10 逐种子 {meth} 最低占比%", m.group(glo), 100 * min(got) / n_common, src_ab)
            _numchk(f"S10 逐种子 {meth} 最高占比%", m.group(ghi), 100 * max(got) / n_common, src_ab)
        per_p, per_b = dv["Phys"].groupby(level="种子").sum(), dv["Phys+BL"].groupby(level="种子").sum()
        check("S10 逐种子 每粒种子上朴素模型发散都多于 Phys+BL", "True", str(bool((per_p > per_b).all())), src_ab)
        per = sorted(int(v) for v in dv["Phys"].groupby(level="种子").sum())
        m2 = _rx("S10 逐种子倍差", r"the divergence rate it yields varies (\w+)fold from seed to seed", t_en, SI)
        if m2 and per[0] > 0:
            check("S10 逐种子 Phys 发散率的倍差", str(_WORDS.index(m2.group(1))), str(round(per[-1] / per[0])), src_ab)
    m = _rx("S10 收束段", r"\(\$\+\$([\d.]+)\\% over the (\d+) scenarios without a divergent Phys run\); the explicit "
            r"BL reconstruction loss adds nothing resolvable to accuracy, (\S+)\\% on a generic CNN \(\$P=([\d.]+)\$\) "
            r"and (\S+)\\% on the physics architecture \(\$P=([\d.]+)\$\)", t_en, SI)
    if m:
        a, b = phys[ok3], cnn[ok3]
        _numchk("S10 收束段 未发散场景上的架构改进%", m.group(1), (100 * (b - a) / b).median(), src_ab)
        check("S10 收束段 未发散场景数", str(len(ok3)), m.group(2), src_ab)
        _numchk("S10 收束段 BL 加在 CNN 上%", _sgn(m.group(3)), (100 * (cnn - cbl) / cnn).median(), src50)
        _numchk("S10 收束段 BL 加在 CNN 上的 P", m.group(4), float(wilcoxon(cbl.to_numpy(), cnn.to_numpy()).pvalue),
                src50)
        _numchk("S10 收束段 BL 加在物理架构上%", _sgn(m.group(5)), (100 * (phys - ph) / phys).median(), src50)
        _numchk("S10 收束段 BL 加在物理架构上的 P", m.group(6), float(wilcoxon(ph.to_numpy(), phys.to_numpy()).pvalue),
                src50)
    m = _rx("S10 收束段 倍数与失败率", r"reduces the physics model's divergence rate roughly ([\d.]+)-fold\..{0,400}?"
            r"the physics model fails in ([\d.]+)\\% of runs where the generic network fails in none", t_en, SI)
    if m:
        _numchk("S10 收束段 降低倍数", m.group(1), fold, src_ab)
        _numchk("S10 收束段 Phys+BL 发散率%", m.group(2), rate("Phys+BL"), src_ab)

    # ── S11：校正前后 ─────────────────────────────────────────────────
    sb4 = ["PLSR + SB", "SVR + SB", "CNN + SB", "Phys+BL + SB"]
    cls3 = ["PLSR + SB", "SVR + SB", "CNN + SB"]
    m = _rx("S11 开篇", r"When uncorrected, Phys\+BL has the lower median RMSE, "
            r"([\d.]+)\\,\\Brix\{\} against the CNN's "
            r"([\d.]+)\\,\\Brix\{\} \(median paired difference (\S+)\\,\\Brix\{\}, whose cluster-robust 95\\% "
            r"confidence interval excludes zero under scenario, target-set and source-set clustering alike, whereas "
            r"the mean paired difference, (\S+)\\,\\Brix\{\}, favours the CNN", t_en, SI)
    if m:
        _numchk("S11 开篇 Phys+BL 中位", m.group(1), ph.median(), src50)
        _numchk("S11 开篇 CNN 中位", m.group(2), cnn.median(), src50)
        _numchk("S11 开篇 中位配对差", _sgn(m.group(3)), float(r60(c60b, UNC, "场景")["配对差中位"]), src60)
        _numchk("S11 开篇 均值配对差", _sgn(m.group(4)), float(r60(c60a, UNC, "场景")["配对差均值"]), src60)
    h3, h2 = hyp("Phys+BL + SB", "CNN + SB"), hyp("Phys+BL + SB", "PLSR + SB")
    m = _rx("S11 校正后", r"converge to almost the same level \(median RMSE ([\d.]+)--([\d.]+)\\,\\Brix\{\}, "
            r"Fig\.~\\ref\{fig:sbc\}a\), and Phys\+BL is left the highest of the four \(([\d.]+)\\,\\Brix\{\}\): it is "
            r"behind CNN\+correction in (\d+) of the (\d+) scenarios \(median paired difference ([\d.]+)\\%, \$P="
            + _SCI + r"\$ after Holm correction\) and behind the simplest PLSR\+correction in (\d+) \(([\d.]+)\\%, "
            r"\$P=" + _SCI + r"\$\)", t_en, SI)
    if m:
        _numchk("S11 三种经典/通用方法校正后中位下限", m.group(1), float(piv[cls3].median().min()), src50)
        _numchk("S11 三种经典/通用方法校正后中位上限", m.group(2), float(piv[cls3].median().max()), src50)
        _numchk("S11 Phys+BL+SB 中位", m.group(3), phsb.median(), src50)
        check("S11 校正后四法中 Phys+BL+SB 中位最高", "Phys+BL + SB", str(piv[sb4].median().idxmax()), src50)
        for tag, other, gc, gd, gp, gr in (("CNN+SB", "CNN + SB", 4, 5, 6, 7),
                                           ("PLSR+SB", "PLSR + SB", 8, None, 9, 10)):
            o = piv[other]
            check(f"S11 Phys+BL+SB 落后于 {tag} 的场景数", m.group(gc), str(int((phsb > o).sum())), src50)
            if gd:
                check(f"S11 对 {tag} 的场景总数", m.group(gd), str(n_common), src50)
            _numchk(f"S11 对 {tag} 的配对差中位%", m.group(gp), (100 * (phsb - o) / phsb).median(), src50)
            hh = h3 if tag == "CNN+SB" else h2
            _scichk(f"S11 对 {tag} 的 Holm P", f"${m.group(gr)}$", float(hh["Wilcoxon_p_holm"]), src50 + "[表c]")
    gains = {k: float((100 * (piv[k] - piv[k + " + SB"]) / piv[k]).median()) for k in ("PLSR", "SVR", "CNN", "Phys+BL")}
    m = _rx("S11 增益", r"median of the per-scenario RMSE reductions\) is ([\d.]+)\\% for PLSR, ([\d.]+)\\% for SVR, "
            r"([\d.]+)\\% for the CNN and ([\d.]+)\\% for Phys\+BL", t_en, SI)
    if m:
        for i, k in enumerate(("PLSR", "SVR", "CNN", "Phys+BL")):
            _numchk(f"S11 {k} 校正增益中位%", m.group(i + 1), gains[k], src50)
        check("S11 增益最小者是 Phys+BL", "Phys+BL", min(gains, key=gains.__getitem__), src50)
    m = _rx("S11 校正后收拢", r"after correction the four methods end within ([\d.]+)\\,\\Brix\{\} of one another, "
            r"the physics model highest among them", t_en, SI)
    if m:
        spread = float(piv[sb4].median().max() - piv[sb4].median().min())
        check(f"S11 校正后四法中位相差在 {m.group(1)} 以内", "True", str(spread < float(m.group(1))),
              src50 + f"（实测 {spread:.3f}）")
    m = _rx("S12 残余 R² 中位", r"a median coefficient of determination that does not exceed ([\d.]+) remain", t_en, SI)
    if m:
        top = float(a2.loc[sb4, "R2_median"].max())
        check(f"S12 校正后 R² 中位均不超过 {m.group(1)}", "True", str(top <= float(m.group(1))),
              src50 + f"（实测最大 {top:.3f}）")
    m = _rx("S11 残余 R²", r"left with an error that none of them explains \(median \$R\^\{2\}\\le([\d.]+)\$\)",
            t_en, SI)
    if m:
        top = float(a2.loc[sb4, "R2_median"].max())
        check(f"S11 校正后 R² 中位均不超过 {m.group(1)}", "True", str(top <= float(m.group(1))),
              src50 + f"（实测最大 {top:.3f}）")
    m = _rx("S11 校正后 RPD 区间", r"mean ratio of performance to deviation of the four methods spans only "
            r"([\d.]+)--([\d.]+);", t_en, SI)
    if m:
        _numchk("S11 校正后 RPD 区间下", m.group(1), a2.loc[sb4, "RPD_mean"].min(), src50)
        _numchk("S11 校正后 RPD 区间上", m.group(2), a2.loc[sb4, "RPD_mean"].max(), src50)
    m = _rx("S11 RPD 分组", r"the target-domain mean \(([\d.]+)--([\d.]+)\) while Phys\+BL stays below it "
            r"\(([\d.]+)\)", t_en, SI)
    if m:
        _numchk("S11 三法 RPD 下", m.group(1), a2.loc[cls3, "RPD_mean"].min(), src50)
        _numchk("S11 三法 RPD 上", m.group(2), a2.loc[cls3, "RPD_mean"].max(), src50)
        _numchk("S11 Phys+BL+SB RPD", m.group(3), float(a2.loc["Phys+BL + SB", "RPD_mean"]), src50)
    m = _rx("S11 误判率", r"misclassification rate remains as high as ([\d.]+)\\%--([\d.]+)\\%", t_en, SI)
    if m:
        _numchk("S11 误判率下限%", m.group(1), 100 * a2.loc[sb4, "品质误判率_mean"].min(), src50)
        _numchk("S11 误判率上限%", m.group(2), 100 * a2.loc[sb4, "品质误判率_mean"].max(), src50)

    # ── S12：三处回指 ────────────────────────────────────────────────
    m = _rx("S12 发散回指", r"which still diverges in ([\d.]+)\\% of runs \(Section~S10\)", t_en, SI)
    if m:
        _numchk("S12 Phys+BL 发散率%", m.group(1), rate("Phys+BL"), src_ab)
    m = _rx("S12 增益回指", r"\(median per scenario: PLSR ([\d.]+)\\%, CNN ([\d.]+)\\%, Phys\+BL ([\d.]+)\\%\)",
            t_en, SI)
    if m:
        for i, k in enumerate(("PLSR", "CNN", "Phys+BL")):
            _numchk(f"S12 {k} 校正增益中位%", m.group(i + 1), gains[k], src50)
    m = _rx("S12 稳定性回指", r"divergence rate roughly ([\d.]+)-fold \(([\d.]+)\\% to ([\d.]+)\\% of scenario--seed "
            r"runs, McNemar \$P=" + _SCI + r"\$\)", t_en, SI)
    if m:
        _numchk("S12 降低倍数", m.group(1), fold, src_ab)
        _numchk("S12 Phys 发散率%", m.group(2), rate("Phys"), src_ab)
        _numchk("S12 Phys+BL 发散率%", m.group(3), rate("Phys+BL"), src_ab)
        _scichk("S12 McNemar P", f"${m.group(4)}$", p_mc, src_ab)

    # ── S8：BL 残差放大（03 号原始残差 + 55 号的独立复算）───────────────────
    bl_a = pd.read_excel(XL03, sheet_name="表a：BL违反统计检验")
    bl_c = pd.read_excel(XL03, sheet_name="表c：全量样本BL残差")
    src03 = "04outputs/03_exp_BL_validation.xlsx[表a/表c]"
    check("S8：同年一源一目标场景数", "12", str(len(bl_a)), src03)
    m = _rx("S8 放大倍数范围", r"with ratios from ([\d.]+) \(Shaanxi\$\\to\$Shandong, geographically closest\) "
            r"to ([\d.]+) \(Xinjiang\$\\to\$Shaanxi\)", t_en, SI)
    if m:
        _numchk("S8 放大倍数最小", m.group(1), bl_a["倍数变化（跨域/域内）"].min(), src03)
        _numchk("S8 放大倍数最大", m.group(2), bl_a["倍数变化（跨域/域内）"].max(), src03)
        lo_row = bl_a.loc[bl_a["倍数变化（跨域/域内）"].idxmin(), "场景"]
        hi_row = bl_a.loc[bl_a["倍数变化（跨域/域内）"].idxmax(), "场景"]
        check("S8 最小放大的场景是陕西→山东", "True", str("陕西→山东" in lo_row), src03)
        check("S8 最大放大的场景是新疆→陕西", "True", str("新疆→陕西" in hi_row), src03)
    # 逐年：该年全部样本的残差汇总取中位（与稿件同口径，不是场景中位的中位）
    bl_c = bl_c.assign(year=bl_c["scenario"].str.slice(1, 5).astype(int))
    m = _rx("S8 逐年放大", r"\(2018, ([\d.]+)\$\\to\$([\d.]+), a ([\d.]+)-fold amplification; "
            r"2025, ([\d.]+)\$\\to\$([\d.]+), a ([\d.]+)-fold amplification\)", t_en, SI)
    if m:
        for i, y in enumerate((2018, 2025)):
            yb = bl_c[bl_c["year"] == y]
            sm = yb[yb["domain_type"].str.startswith("源域")]["bl_residual"].median()
            tm = yb[yb["domain_type"].str.startswith("目标域")]["bl_residual"].median()
            _numchk(f"S8 {y} 源域残差中位", m.group(3 * i + 1), sm, src03)
            _numchk(f"S8 {y} 目标域残差中位", m.group(3 * i + 2), tm, src03)
            _numchk(f"S8 {y} 放大倍数", m.group(3 * i + 3), tm / sm, src03)
    # 55 号（免标签预测子）与 03 号是两条独立通路，同年一源一目标的放大倍数必须逐场景吻合
    p55 = pd.read_excel(XL55, sheet_name="表a：免标签预测子").set_index("场景")
    worst, missing = 0.0, 0
    for _, r in bl_a.iterrows():
        yr = r["场景"][1:5]                       # "Y2018-S1: 山东→新疆" → "2018"
        a_reg, b_reg = r["场景"].split(": ", 1)[1].split("→")
        key = f"{yr}_{a_reg}→{yr}_{b_reg}"        # → "2018_山东→2018_新疆"
        if key not in p55.index:
            missing += 1
            continue
        worst = max(worst, abs(float(p55.loc[key, "BL放大倍数"]) - float(r["倍数变化（跨域/域内）"])))
    check("S8：55 号与 03 号的 12 个放大倍数逐场景可配对", "0", str(missing),
          src03 + " vs 04outputs/55_bl_risk_diagnosis_validation.xlsx[表a]")
    check("S8：两条通路的放大倍数最大差 <0.01", "True", str(worst < 0.01),
          src03 + f" vs 55 号（实测最大差 {worst:.4f}）")

    # ── S8：源域留出复核与三个低秩代理的一致性 ─────────────────────────────
    hd = pd.read_excel(XL35, sheet_name="all_results")
    src35 = "04outputs/35_BL_source_heldout_validation.xlsx[all_results]"
    m = _rx("S8 留出复核", r"held-out source samples: (\d+) of (\d+) comparisons remain significant", t_en, SI)
    if m:
        check("S8 留出复核显著数", m.group(1), str(int((hd["p_held_vs_tgt"] < 0.05).sum())), src35)
        check("S8 留出复核比较总数", m.group(2), str(len(hd)), src35)
    cr = pd.read_excel(XL28, sheet_name="表b：surrogate相关性")
    src28 = "04outputs/28_exp_true_BL_decomp.xlsx[表b：surrogate相关性]"
    three_s = {"PCA-10", "NMF-3", "SVD-3"}
    ind = cr[cr["a"].isin(three_s) & cr["b"].isin(three_s)]
    m = _rx("S8 代理相关", r"surrogates \(PCA-10, NMF-3, SVD-3\) are all at least ([\d.]+) \(\$P<([\d.]+)\$\)",
            t_en, SI)
    if m:
        check("S8 三个独立代理的两两比较数", "3", str(len(ind)), src28)
        _numchk("S8 代理相关最小 r", m.group(1), ind["Pearson r"].min(), src28)
        check(f"S8 代理相关最大 p < {m.group(2)}", "True", str(float(ind["p_value"].max()) < float(m.group(2))),
              src28 + f"（实测最大 p={ind['p_value'].max():.4f}）")
    nn = cr[((cr["a"] == "NMF-3") & (cr["b"] == "NNLS-3"))]
    m = re.search(r"bit-wise identical results \(Pearson r=([\d.]+)\)", t_en)
    if m and len(nn):
        _numchk("S8 NMF-3 与 NNLS-3 的 r", m.group(1), float(nn["Pearson r"].iloc[0]), src28)
    else:
        check("S8 NNLS-3 句/行可解析", "True", "False", src28)

    # ── S8：BL 放大比与七个通用免标签距离的对照（71 号）─────────────────────
    d71 = pd.read_excel(XL71, sheet_name="表d：配对Δrho")
    e71 = pd.read_excel(XL71, sheet_name="表e：偏相关")
    src71 = "04outputs/71_bl_vs_generic_distance.xlsx[表d/表e]"
    m = _rx("S8 对照段配对计数", r"Over the (\d+) method--baseline pairs \((\w+) prediction methods \$\\times\$ (\w+) "
            r"baselines\), the BL amplification ratio predicts cross-domain RMSE better in (\d+) pairs and "
            r"indistinguishably in the other (\d+), and worse in none", t_en, SI)
    if m:
        vc = d71["判定"].value_counts()
        n_better = int(vc.get("✅BL更强", 0))
        n_tie = int(vc.get("⚪无法区分", 0))
        check("S8 对照总配对数", m.group(1), str(len(d71)), src71)
        check("S8 对照 预测方法数", str(_WORDS.index(m.group(2))), str(d71["方法"].nunique()), src71)
        check("S8 对照 基线数", str(_WORDS.index(m.group(3))), str(d71["基线"].nunique()), src71)
        check("S8 对照 BL 更强组数", m.group(4), str(n_better), src71)
        check("S8 对照 无法区分组数", m.group(5), str(n_tie), src71)
        check("S8 对照 BL 更弱组数为 0", "0", str(int(len(d71) - n_better - n_tie)), src71)
    m = _rx("S8 对照段偏相关", r"significant for three of the five methods \(PLSR ([\d.]+), CNN ([\d.]+) and "
            r"Phys\+BL ([\d.]+), all \$P<([\d.]+)\$\)", t_en, SI)
    if m:
        e = e71.set_index("方法")
        for i, meth in enumerate(("PLSR", "CNN", "Phys+BL")):
            _numchk(f"S8 对照 {meth} 偏相关", m.group(i + 1), float(e.loc[meth, "偏相关_rho"]), src71)
        sig = e71[e71["P"] < float(m.group(4))]
        check("S8 对照 偏相关显著的方法数", "3", str(len(sig)), src71)
        check("S8 对照 显著的三个方法就是稿件点名的那三个", "True", str(set(sig["方法"]) == {"PLSR", "CNN", "Phys+BL"}),
              src71)

    # ── S8 与正文第一条：BL 残差相关的 ρ 中位 ─────────────────────────────
    b55 = pd.read_excel(XL55, sheet_name="表b：相关分析")
    rmse_rows = b55[b55["结局"].str.contains("跨域 RMSE")]
    rho_med = float(rmse_rows["Spearman_rho"].median())
    src55 = "04outputs/55_bl_risk_diagnosis_validation.xlsx[表b：相关分析]"
    m = _rx("S8 ρ", r"median \$\\rho\$ across methods ([\d.]+), range ([\d.]+)--([\d.]+)", t_en, SI)
    if m:
        _numchk("S8 跨方法 ρ 中位", m.group(1), rho_med, src55)
        _numchk("S8 ρ 范围下限", m.group(2), rmse_rows["Spearman_rho"].min(), src55)
        _numchk("S8 ρ 范围上限", m.group(3), rmse_rows["Spearman_rho"].max(), src55)
    check("S8 相关分析逐方法的场景数都等于共同场景数", str([n_common] * len(rmse_rows)),
          str([int(v) for v in rmse_rows["n"]]), src55)
    rr = rmse_rows.set_index("结局")["Spearman_rho"]
    m = _rx("S8 最强两法", r"being strongest for partial least squares regression \(\$\\rho=([\d.]+)\$\) and the "
            r"convolutional neural network \(\$\\rho=([\d.]+)\$\)", t_en, SI)
    if m:
        _numchk("S8 PLSR ρ", m.group(1), float(rr["PLSR 跨域 RMSE"]), src55)
        _numchk("S8 CNN ρ", m.group(2), float(rr["CNN 跨域 RMSE"]), src55)
        check("S8 ρ 最大的两法是 PLSR 与 CNN", "['CNN 跨域 RMSE', 'PLSR 跨域 RMSE']",
              str(sorted(rr.sort_values().index[-2:])), src55)
    d55 = pd.read_excel(XL55, sheet_name="表d：留一产地汇总")
    m = _rx("S8 留一产地", r"remains positive for origins never used in fitting \(([\d.]+)--([\d.]+)\)", t_en, SI)
    if m:
        _numchk("S8 留一产地 中位 ρ 下限", m.group(1), float(d55["中位_rho"].min()), src55 + "[表d]")
        _numchk("S8 留一产地 中位 ρ 上限", m.group(2), float(d55["中位_rho"].max()), src55 + "[表d]")
        check("S8 留一产地 各方法中位 ρ 全为正", "True", str(bool((d55["中位_rho"] > 0).all())), src55 + "[表d]")
    m = _rx("S8 对照段场景集", r"seven label-free alternatives over the (\d+) common scenarios", t_en, SI)
    if m:
        check("S8 对照段场景数等于共同场景数", m.group(1), str(n_common), src71)
    m = _rx("正文第一条 ρ", r"median\s+\$\\rho=([\d.]+)\$; Supplementary Section~S8", t_main, MAIN)
    if m:
        _numchk("正文第一条 ρ 中位与 S8 同源", m.group(1), rho_med, src55)

    # ── 正文 §3.5 第二、三条（跨年承载、稳定性、误判率与 §4.5 发散率在 S9–S11，由上面的 S 判据覆盖）──
    m = _rx("正文第二条 典型场景句", r"Phys\+BL is the least bad only in a typical scenario \(median RMSE "
            r"([\d.]+)\\,\\Brix\{\} against ([\d.]+)\\,\\Brix\{\} for the CNN over the (\d+) scenarios in which "
            r"every method returned a result\) and has the highest mean \(([\d.]+) against ([\d.]+)\\,\\Brix\{\}\) because "
            r"([\d.]+)\\% of its runs diverge", t_main, MAIN)
    if m:
        _numchk("正文第二条 Phys+BL 中位", m.group(1), ph.median(), src50)
        _numchk("正文第二条 CNN 中位", m.group(2), cnn.median(), src50)
        check("正文第二条 共同场景数", str(n_common), m.group(3), src50)
        _numchk("正文第二条 Phys+BL 均值", m.group(4), ph.mean(), src50)
        _numchk("正文第二条 CNN 均值", m.group(5), cnn.mean(), src50)
        _numchk("正文第二条 Phys+BL 发散率%", m.group(6), rate("Phys+BL"), src_ab)
        check("正文第二条 均值最高者是 Phys+BL（五法）", "Phys+BL", str(a2.loc[CASE_BARE, "RMSE_mean"].idxmax()), src50)
    m = _rx("正文第三条 校正后中位", r"reach median\s+RMSEs of ([\d.]+)--([\d.]+)\\,\\Brix\{\} \(PLSR "
            r"([\d.]+)\$\\to\$([\d.]+), SVR ([\d.]+)\$\\to\$([\d.]+)\\,\\Brix\{\}\) and overtake the corrected "
            r"physics model \(([\d.]+)\$\\to\$([\d.]+)\\,\\Brix\{\}\)", t_main, MAIN)
    if m:
        cl2 = piv[["PLSR + SB", "SVR + SB"]].median()
        _numchk("正文第三条 经典基线校正后中位下限", m.group(1), float(cl2.min()), src50)
        _numchk("正文第三条 经典基线校正后中位上限", m.group(2), float(cl2.max()), src50)
        for i, k in enumerate(("PLSR", "SVR", "Phys+BL")):
            _numchk(f"正文第三条 {k} 未校正中位", m.group(3 + 2 * i), piv[k].median(), src50)
            _numchk(f"正文第三条 {k} 校正后中位", m.group(4 + 2 * i), piv[k + " + SB"].median(), src50)
    m = _rx("正文第三条 RPD", r"with RPD at most ([\d.]+), the most economical option", t_main, MAIN)
    if m:
        _numchk("正文第三条 校正后 RPD 上限", m.group(1), a2.loc[sb4, "RPD_mean"].max(), src50)

    # ── 回复信与 cover letter：写给审稿人/编辑的每个数都要与数据、与投稿原稿同源 ──────
    sub = re.sub(r"\s+", " ", (BASE / "06doc/02sub/revision/_submitted_Elsevier_en.tex").read_text(encoding="utf-8"))
    r1, r2, cl = _letter("Response_R1.tex"), _letter("Response_R2.tex"), _letter("CoverLetter_R1.tex")
    L1, L2 = "06doc/02sub/revision/Response_R1.tex", "06doc/02sub/revision/Response_R2.tex"
    m = _rx("R1 典型场景", r"over the (\d+) common scenarios its median RMSE is ([\d.]+) against "
            r"([\d.]+)\\,\\Brix\{\} and "
            r"it is lower in (\d+)\\% of them", r1, L1)
    if m:
        check("R1 共同场景数", str(n_common), m.group(1), src50)
        _numchk("R1 Phys+BL 中位", m.group(2), ph.median(), src50)
        _numchk("R1 CNN 中位", m.group(3), cnn.median(), src50)
        _numchk("R1 更优场景占比%", m.group(4), 100 * float((ph < cnn).mean()), src50)
    m = _rx("R1 均值", r"its mean RMSE is ([\d.]+) against ([\d.]+)\\,\\Brix\{\}, because ([\d.]+)\\% of its runs "
            r"diverge and none of the generic network's do", r1, L1)
    if m:
        _numchk("R1 Phys+BL 均值", m.group(1), ph.mean(), src50)
        _numchk("R1 CNN 均值", m.group(2), cnn.mean(), src50)
        _numchk("R1 Phys+BL 发散率%", m.group(3), rate("Phys+BL"), src_ab)
    ms = re.search(r"Phys\+BL attains the lowest mean RMSE, ([\d.]+)\\,\\Brix\{\}, [\d.]+\\% lower in aggregate than "
                   r"the CNN \(([\d.]+)\\,\\Brix\{\}\)", sub)
    m = _rx("R1 投稿原稿均值", r"the submitted manuscript reported a lower mean \(([\d.]+) against "
            r"([\d.]+)\\,\\Brix\{\}\)", r1, L1)
    if m:
        check("R1 所引投稿原稿句可在原稿找到", "True", str(bool(ms)), "_submitted_Elsevier_en.tex")
        if ms:
            _numchk("R1 投稿原稿 Phys+BL 均值", m.group(1), float(ms.group(1)), "_submitted_Elsevier_en.tex")
            _numchk("R1 投稿原稿 CNN 均值", m.group(2), float(ms.group(2)), "_submitted_Elsevier_en.tex")
    ms2 = re.search(r"the median ([\d.]+)\\%, with an extremely small nominal", sub)
    ms3 = re.search(r"Cliff's \$\\delta = (-?[\d.]+)\$, a large effect", sub)
    m = _rx("R1 效应量", r"the median paired improvement is ([\d.]+)\\% and Cliff's \$\\delta\$ is \$(-?[\d.]+)\$ "
            r"\(negligible\), against ([\d.]+)\\% and \$(-?[\d.]+)\$ \(large\) in the submitted manuscript", r1, L1)
    if m:
        _numchk("R1 配对改进中位%", m.group(1), imp.median(), src50)
        _numchk("R1 Cliff δ", m.group(2), float(h1["Cliffs_delta"]), src50 + "[表c H1]")
        check("R1 所引投出稿的配对改进中位与 δ 可在原稿找到", "True", str(bool(ms2 and ms3)),
              "_submitted_Elsevier_en.tex")
        if ms2 and ms3:
            _numchk("R1 投出稿配对改进中位%", m.group(3), float(ms2.group(1)), "_submitted_Elsevier_en.tex")
            check("R1 投出稿 Cliff δ", ms3.group(1), m.group(4), "_submitted_Elsevier_en.tex")
    m = _rx("R1 同年层", r"in the (\d+) same-year scenarios the two networks are within ([\d.]+)\\,\\Brix\{\} "
            r"of each other, "
            r"whereas the submitted manuscript found the physics architecture best in both strata", r1, L1)
    if m:
        check("R1 同年场景数", m.group(1), str(n_sy), src58)
        gap = max(abs(s58(SY, "Phys+BL", c) - s58(SY, "CNN", c)) for c in ("中位RMSE", "均值RMSE"))
        check(f"R1 同年两网络相差在 {m.group(2)} 以内", "True", str(gap < float(m.group(2))),
              src58 + f"（实测 {gap:.3f}）")
        check("R1 所引投出稿「两层都最优」出自原稿", "True", str("Phys+BL is best when uncorrected" in sub),
              "_submitted_Elsevier_en.tex")
    m = _rx("R1 损失消融", r"on the physics architecture \(\$P=([\d.]+)\$\) or on a generic CNN \(\$P=([\d.]+)\$\)",
            r1, L1)
    if m:
        _numchk("R1 Phys+BL 对 Phys 的 P", m.group(1), float(wilcoxon(ph.to_numpy(), phys.to_numpy()).pvalue), src50)
        _numchk("R1 CNN+BL 对 CNN 的 P", m.group(2), float(wilcoxon(cbl.to_numpy(), cnn.to_numpy()).pvalue), src50)
        check("R1 所引投出稿两处「可分辨」出自原稿", "True",
              str("statistically resolvable degradation" in sub and "resolvable across five seeds" in sub),
              "_submitted_Elsevier_en.tex")
    m = _rx("R1 稳定性（现定义）", r"it cuts the divergence rate from ([\d.]+)\\% to ([\d.]+)\\% of scenario--seed "
            r"runs "
            r"\(McNemar \$P=" + _SCI + r"\$\)", r1, L1)
    if m:
        _numchk("R1 Phys 发散率%", m.group(1), rate("Phys"), src_ab)
        _numchk("R1 Phys+BL 发散率%", m.group(2), rate("Phys+BL"), src_ab)
        _scichk("R1 McNemar P", f"${m.group(3)}$", p_mc, src_ab)
    # 投出稿的发散定义：只数有结果的运行里 RMSE>10，分母为有结果的运行；McNemar 在两者都有结果的运行上配对
    big_rate = {k: 100 * float(big[k].sum()) / float((~miss[k]).sum()) for k in ("Phys", "Phys+BL")}
    both_ok = ~miss["Phys"] & ~miss["Phys+BL"]
    a_, b_ = big["Phys"][both_ok], big["Phys+BL"][both_ok]
    s10, s01 = int((a_ & ~b_).sum()), int((~a_ & b_).sum())
    p_sub = float(binomtest(min(s10, s01), s10 + s01, 0.5).pvalue)
    m = _rx("R1 稳定性（投出稿定义）", r"the reduction is from ([\d.]+)\\% to ([\d.]+)\\% \(\$P=([\d.]+)\$\), where "
            r"that manuscript reported roughly ninefold", r1, L1)
    if m:
        _numchk("R1 投出稿定义 Phys 发散率%", m.group(1), big_rate["Phys"], src_ab)
        _numchk("R1 投出稿定义 Phys+BL 发散率%", m.group(2), big_rate["Phys+BL"], src_ab)
        _numchk("R1 投出稿定义 McNemar P", m.group(3), p_sub, src_ab + f"（两者都有结果的运行，不一致对 {s10}/{s01}）")
        check("R1 所引「roughly ninefold」出自投稿原稿", "True", str("roughly ninefold" in sub),
              "_submitted_Elsevier_en.tex")
    m = _rx("R1 架构消融", r"not over all scenarios \(\$\+([\d.]+)\\%\$, \$P=([\d.]+)\$\)", r1, L1)
    if m:
        _numchk("R1 架构消融 改进中位%", m.group(1), (100 * (cnn - phys) / cnn).median(), src50)
        _numchk("R1 架构消融 P", m.group(2), p_arch, src50)
    m = _rx("R1 阈值", r"\(g\) The outlier-threshold check of Supplementary Section~S9 is narrower: the submitted "
            r"manuscript reported that the physics architecture kept the lowest uncorrected median at 0\\%, 10\\% and "
            r"15\\% removal; it now does so at 10\\% and 15\\% but not without removal, where CNN\+MMD is lower "
            r"\(([\d.]+) against ([\d.]+)\\,\\Brix\{\}\) and its edge over the generic network is not resolvable "
            r"\(\$P=([\d.]+)\$\)\. The corrected comparison reported there, PLSR\+SB below Phys\+BL\+SB, holds at all "
            r"three thresholds", r1, L1)
    if m:
        c6 = {}
        for tag, xl in (("0", XLP0), ("10", XL50), ("15", XLP15)):
            g0 = (pd.read_excel(xl, sheet_name="表g：每场景跨种子均值")
                  .pivot(index="场景", columns="方法_校正", values="RMSE"))
            c6[tag] = g0.loc[g0[CASE_ALL].dropna().index]
        src_c6 = "04outputs/c6_p0.xlsx、50_formal_multiseed_benchmark.xlsx、c6_p15.xlsx[表g]"
        _numchk("R1 0% 档 CNN+MMD 中位", m.group(1), c6["0"]["CNN+MMD"].median(), src_c6)
        _numchk("R1 0% 档 Phys+BL 中位", m.group(2), c6["0"]["Phys+BL"].median(), src_c6)
        _numchk("R1 0% 档 Phys+BL 对 CNN 的 P", m.group(3),
                float(wilcoxon(c6["0"]["Phys+BL"].to_numpy(), c6["0"]["CNN"].to_numpy()).pvalue), src_c6)
        check("R1 三档 PLSR+SB 中位都低于 Phys+BL+SB", "True",
              str(all(c6[t]["PLSR + SB"].median() < c6[t]["Phys+BL + SB"].median() for t in c6)), src_c6)
        check("R1 三档 未校正中位最低者为 CNN+MMD / Phys+BL / Phys+BL", "True",
              str([c6[t][CASE_BARE].median().idxmin() for t in ("0", "10", "15")] == ["CNN+MMD", "Phys+BL", "Phys+BL"]),
              src_c6)
        check("R1 所引投出稿「三档都保持最低中位」与「PLSR+SB 低于 Phys+BL+SB」出自原稿", "True",
              str("uncorrected, Phys+BL retains the lowest median RMSE (1.74, 1.71 and 1.71" in sub
                  and "PLSR+SB falls below Phys+BL+SB" in sub), "_submitted_Elsevier_en.tex")
    m = _rx("R1 差距", r"in a typical scenario CNN\+SB and PLSR\+SB lead by ([\d.]+)\\% and ([\d.]+)\\% "
            r"\(median paired "
            r"differences\), whereas on the mean they lead by (\d+)\\% and (\d+)\\%, a margin inflated by the (\w+) "
            r"scenarios in which the corrected physics model still exceeds 10\\,\\Brix\{\}; the submitted manuscript "
            r"reported ([\d.]+)\\% and ([\d.]+)\\% on the mean", r1, L1)
    if m:
        med_d = [float((100 * (phsb - piv[o]) / phsb).median()) for o in ("CNN + SB", "PLSR + SB")]
        mean_d = [100 * float(phsb.mean() - piv[o].mean()) / float(piv[o].mean()) for o in ("CNN + SB", "PLSR + SB")]
        _numchk("R1 典型场景差距 CNN+SB%", m.group(1), med_d[0], src50)
        _numchk("R1 典型场景差距 PLSR+SB%", m.group(2), med_d[1], src50)
        _numchk("R1 均值差距 CNN+SB%", m.group(3), mean_d[0], src50)
        _numchk("R1 均值差距 PLSR+SB%", m.group(4), mean_d[1], src50)
        check("R1 Phys+BL+SB >10 的场景数", str(_WORDS.index(m.group(5))), str(int((phsb > 10).sum())), src50)
        check("R1 所引投出稿 15.5%/16.7% 出自原稿", "True",
              str(f"{m.group(6)}\\% worse than CNN+correction" in sub
                  and f"{m.group(7)}\\% worse than the simplest" in sub), "_submitted_Elsevier_en.tex")
    m = _rx("R1 校正增益",
            r"median per-scenario gain ([\d.]+)\\%, against ([\d.]+)--([\d.]+)\\% for the other methods\)", r1, L1)
    if m:
        others = [gains[k] for k in ("PLSR", "SVR", "CNN")]
        _numchk("R1 Phys+BL 校正增益%", m.group(1), gains["Phys+BL"], src50)
        _numchk("R1 其余方法增益下限%", m.group(2), min(others), src50)
        _numchk("R1 其余方法增益上限%", m.group(3), max(others), src50)
    m = _rx("R1 投稿原稿增益", r"which rested on a gain of ([\d.]+)\\%", r1, L1)
    if m:
        check("R1 所引 1.2% 增益出自投稿原稿", "True",
              str(f"but only {m.group(1)}\\% for Phys+BL" in sub), "_submitted_Elsevier_en.tex")
    m = _rx("R1 中心结论", r"the classical baselines \(median ([\d.]+)--([\d.]+)\\,\\Brix\{\}\) overtake the corrected "
            r"physics "
            r"model \(([\d.]+)\\,\\Brix\{\}\)", r1, L1)
    if m:
        cl2 = piv[["PLSR + SB", "SVR + SB"]].median()
        _numchk("R1 经典基线校正后中位下限", m.group(1), float(cl2.min()), src50)
        _numchk("R1 经典基线校正后中位上限", m.group(2), float(cl2.max()), src50)
        _numchk("R1 Phys+BL+SB 中位", m.group(3), phsb.median(), src50)
    src60 = "04outputs/60sensitivity_domain_clustering.xlsx[表a 均值 / 表b 中位]"
    if _rx("R1 聚类稳健性的适用范围", r"For the two comparisons the submitted manuscript reported, "
           r"CNN\+SB and PLSR\+SB "
           r"against Phys\+BL\+SB, it now holds for the median and the mean alike and under all three clusterings; "
           r"SVR\+SB is ahead on both statistics as well, but its lead in a typical scenario is resolvable only under "
           r"scenario clustering", r1, L1):
        for tab, stat in ((c60a, "均值"), (c60b, "中位")):
            sub2 = tab[tab["比较"].isin(PAIRS2)]
            check(f"R1 CNN+SB、PLSR+SB 的六档{stat} CI 全在 0 以下", "True",
                  str(len(sub2) == 6 and bool((sub2["CI上界"] < 0).all())), src60)
        check("R1 SVR+SB 中位与均值都低于 Phys+BL+SB", "True",
              str(bool(piv["SVR + SB"].median() < phsb.median() and piv["SVR + SB"].mean() < phsb.mean())), src50)
        check("R1 SVR+SB 均值三档 CI 全在 0 以下", "True", str(svr_mean_all), src60)
        check("R1 SVR+SB 中位只在场景聚类下不跨 0", "True", str(svr_med_scene_only), src60)
    if _rx("R2 聚类稳健性的适用范围", r"After correction, PLSR and the generic network lead the physics "
           r"architecture by small margins, in the same direction on the median and the mean and under scenario, "
           r"target-set and source-set clustering; SVR leads it on the mean under all three, but on the median its "
           r"interval excludes zero only under scenario clustering, and there only narrowly", r2, L2):
        rs = c60b[(c60b["比较"].str.contains("SVR")) & (c60b["聚类单位"].str.startswith("场景"))].iloc[0]
        check("R2 SVR+SB 场景聚类中位 CI 上界离 0 不足 0.001（only narrowly）", "True",
              str(bool(-0.001 < float(rs["CI上界"]) < 0)), src60)
        for tab, stat in ((c60a, "均值"), (c60b, "中位")):
            sub2 = tab[tab["比较"].isin(PAIRS2)]
            check(f"R2 CNN+SB、PLSR+SB 的六档{stat} CI 全在 0 以下", "True",
                  str(len(sub2) == 6 and bool((sub2["CI上界"] < 0).all())), src60)
        check("R2 SVR 均值三档领先、中位只在场景聚类下可分辨", "True", str(svr_mean_all and svr_med_scene_only), src60)
    # R1 (h)：投出稿里其余已变的描述性说法，新旧两侧各自核对
    m = _rx("R1 (h) 偏差", r"the mean uncorrected bias of Phys\+BL is \$(-[\d.]+)\$ rather than "
            r"\$(-[\d.]+)\$\\,\\Brix\{\}, "
            r"larger in magnitude than that of PLSR, SVR and the CNN", r1, L1)
    if m:
        _numchk("R1 (h) Phys+BL 均值偏差", m.group(1), float(a2.loc["Phys+BL", "Bias_mean"]), src50)
        check("R1 (h) Phys+BL 偏差绝对值大于 PLSR、SVR、CNN", "True",
              str(all(abs(a2.loc["Phys+BL", "Bias_mean"]) > abs(a2.loc[k, "Bias_mean"])
                      for k in ("PLSR", "SVR", "CNN"))),
              src50)
        check("R1 (h) 所引投出稿偏差出自原稿", "True",
              str(f"its bias is $-${m.group(2)[1:]}\\,\\Brix{{}}" in sub
                  and "is the smallest among all methods, so its uncorrected predictions are already the closest to "
                      "unbiased" in sub), "_submitted_Elsevier_en.tex")
    m = _rx("R1 (h) 校正后 RPD", r"PLSR, SVR and the CNN reach a mean RPD of ([\d.]+)--([\d.]+) and the physics model "
            r"([\d.]+), whereas the submitted manuscript reported ([\d.]+)--([\d.]+) for the four and concluded "
            r"that the "
            r"RPD stayed below 1", r1, L1)
    if m:
        cls3 = ["PLSR + SB", "SVR + SB", "CNN + SB"]
        _numchk("R1 (h) 三法校正后 RPD 下", m.group(1), float(a2.loc[cls3, "RPD_mean"].min()), src50)
        _numchk("R1 (h) 三法校正后 RPD 上", m.group(2), float(a2.loc[cls3, "RPD_mean"].max()), src50)
        _numchk("R1 (h) Phys+BL+SB RPD", m.group(3), float(a2.loc["Phys+BL + SB", "RPD_mean"]), src50)
        check("R1 (h) 所引投出稿 RPD 区间与「仍低于 1」出自原稿", "True",
              str(f"the four methods is only {m.group(4)}--{m.group(5)}" in sub
                  and "even after correction the RPD remains below 1" in sub), "_submitted_Elsevier_en.tex")
    if _rx("R1 (h) 热图", r"the hardest target domains are 2018 Xinjiang and 2025 Shandong rather than 2025 Gansu",
           r1, L1):
        check("R1 (h) 热图 最难的两个目标域", "['2018_新疆', '2025_山东']", str(sorted(by_t.index[-2:])), src_h)
        check("R1 (h) 所引投出稿「2025 甘肃最难」出自原稿", "True", str("2025 Gansu being hardest" in sub),
              "_submitted_Elsevier_en.tex")
    if _rx("R1 (h) 迁移类型", r"the advantage of Phys\+BL over the CNN stands out only in the "
           r"one-source\$\\to\$one-target scenarios rather than in both single-source types", r1, L1):
        rest = by_type.drop(index="1源→1目标")
        check("R1 (h) 1→1 的优势最大，且大于其余三类最大者的两倍", "True",
              str(by_type.idxmin() == "1源→1目标" and float(by_type["1源→1目标"]) < 2 * float(rest.min())),
              src50 + "[表g 迁移类型]")
        check("R1 (h) 所引投出稿「单源两类最明显」出自原稿", "True",
              str("most pronounced in single-source scenarios (1$\\to$1 and 1$\\to$multiple)" in sub),
              "_submitted_Elsevier_en.tex")
    m = _rx("R1 (h) PLSR+SNV", r"PLSR\+SNV trails the CNN \(median ([\d.]+) against ([\d.]+)\\,\\Brix\{\}\) "
            r"rather than "
            r"matching it", r1, L1)
    if m and XL59.exists():
        _numchk("R1 (h) PLSR+SNV 中位", m.group(1), float(zp.loc[both, "PLSR+SNV"].median()), src59)
        _numchk("R1 (h) CNN 中位（两实验共同场景）", m.group(2), float(piv.loc[both, "CNN"].median()), src50)
        check("R1 (h) 所引投出稿「与 CNN 相当」出自原稿", "True",
              str("PLSR+SNV, gives a median RMSE of 2.01\\,\\Brix{}, comparable to the generic convolutional network"
                  in sub), "_submitted_Elsevier_en.tex")
    if _rx("R1 (h) CNN+MMD", r"CNN\+MMD is on a par with the CNN rather than marginally worse", r1, L1):
        check("R1 (h) CNN+MMD 的中位与均值都不高于 CNN", "True",
              str(bool(piv["CNN+MMD"].median() <= cnn.median() and piv["CNN+MMD"].mean() <= cnn.mean())), src50)
        check("R1 (h) 所引投出稿「略差于 CNN」出自原稿", "True", str("is marginally worse than the CNN" in sub),
              "_submitted_Elsevier_en.tex")
    m = _rx("R2 典型场景", r"holds in a typical scenario \(median RMSE ([\d.]+) against ([\d.]+)\\,\\Brix\{\}\) "
            r"but not on "
            r"average, because ([\d.]+)\\% of its runs diverge", r2, L2)
    if m:
        _numchk("R2 Phys+BL 中位", m.group(1), ph.median(), src50)
        _numchk("R2 CNN 中位", m.group(2), cnn.median(), src50)
        _numchk("R2 Phys+BL 发散率%", m.group(3), rate("Phys+BL"), src_ab)
    m = _rx("R2 效应量", r"Cliff's \$\\delta\$ \$(-?[\d.]+)\$ rather than \$(-?[\d.]+)\$", r2, L2)
    if m:
        _numchk("R2 Cliff δ", m.group(1), float(h1["Cliffs_delta"]), src50 + "[表c H1]")
        check("R2 所引投出稿 δ 出自原稿", "True", str(f"Cliff's $\\delta = {m.group(2)}$" in sub),
              "_submitted_Elsevier_en.tex")
    m = _rx("R2 校正增益", r"because that gain is now ([\d.]+)\\%", r2, L2)
    if m:
        _numchk("R2 Phys+BL 校正增益%", m.group(1), gains["Phys+BL"], src50)
    # 修正前的说法：回归项修正后已不成立的数与措辞（R1 引投稿原稿的「roughly ninefold」是合法对照，不在此列）
    old_num = r"(?<![\d.])1\.892(?!\d)"
    for name, text, pats in (
            ("Response_R1.tex", r1, (r"no result changed", r"three statements of the case study change",
                                     r"the analyses, tables and figures are the same ones", r"same five seeds",
                                     r"median beside the mean throughout", r"classical baselines lead by",
                                     r"under all three clusterings: once")),
            ("Response_R2.tex", r2, (r"reaches zero under", r"keep their direction", r"One statement is narrower",
                                     r"is unchanged and holds for the median")),
            (SI, t_en, (r"ninefold", old_num, r"not resolved under that unit", r"reaches zero under",
                        r"leaves the method ranking unchanged", r"insensitive to the outlier-removal threshold",
                        r"lets the classical baselines overtake",
                        r"so the accuracy advantage is carried by the architecture and")),
            (MAIN, t_main, (r"ninefold", old_num, r"(?<![\d.])7\.7\\%", r"leave the method ranking unchanged")),
            ("Supplementary_zh.tex", _flat("Supplementary_zh.tex"), (r"9 倍", old_num, r"并未被分辨出来",
                                                                  r"方法排序不变",
                                                                  r"让经典基线反超物理架构", r"故精度优势由架构承载")),
            ("Elsevier_zh.tex", _flat("Elsevier_zh.tex"), (r"9 倍", old_num, r"(?<![\d.])7\.7\\%", r"不改变方法排序"))):
        left = [p for p in pats if re.search(p, text)]
        check(f"{name} 不再含修正前的说法", "[]", str(left), name)
    check("cover letter 披露第五处问题、影响范围及其后果", "True",
          str("A fifth fault" in cl and "is not affected" in cl
              and "typical scenario but not on average and as depending on outlier removal" in cl),
          "CoverLetter_R1.tex")


def check_table4() -> None:
    """补充表 S9（族内最优包络，原正文表 4）每一行从原始曲线重算：族内取最优、该预算下可运行的成员、非配对 Cliff δ。"""
    med = _task_medians()
    src = "62/64 号 curves 重算（族内每任务取最优，成员限该预算可运行者）"
    rows = _table_rows(_tex("Supplementary_en.tex"), "tab:si-envelope")
    check("补充表 S9 共 9 行", 9, len(rows), "06doc/01manuscript/Supplementary_en.tex")
    for r in rows:
        shift, n = SHIFT_TEX[r[0]], int(r[1])
        at = med[(med["shift_type"] == shift) & (med["n_cal"] == n)]
        s = at[at["method"].isin(SIMPLE_FAM)].groupby("task_id")[["rmsep", "nrmsep"]].min()
        d = at[at["method"].isin(DEEP_FAM)].groupby("task_id")[["rmsep", "nrmsep"]].min()
        idx = s.index.intersection(d.index)
        s, d = s.loc[idx], d.loc[idx]
        tag = f"补充表 S9 {shift} n={n}"
        _numchk(f"{tag} 任务数", r[2], len(idx), src)
        _numchk(f"{tag} RMSEP_S", r[3], s["rmsep"].median(), src)
        _numchk(f"{tag} RMSEP_D", r[4], d["rmsep"].median(), src)
        _numchk(f"{tag} NRMSEP_S", r[5], s["nrmsep"].median(), src)
        _numchk(f"{tag} NRMSEP_D", r[6], d["nrmsep"].median(), src)
        _numchk(f"{tag} 简单胜率%", r[7], (s["rmsep"] < d["rmsep"]).mean() * 100, src)
        _numchk(f"{tag} Cliff δ", r[8], _cliff_unpaired(s["rmsep"], d["rmsep"]), src)
        _pchk(f"{tag} Wilcoxon P", r[9], s["rmsep"], d["rmsep"], src)


def check_table5() -> None:
    """表 3（免选择比较）每一行重算：代表性配对 CORAL/DeepCORAL 与族均值，效应量取非配对 δ。"""
    med = _task_medians()
    src = "62/64 号 curves 重算（族均值取该预算可运行成员的逐任务均值）"
    rows = _table_rows(_tex("Elsevier_en.tex"), "tab:fair")
    check("表 3 共 10 行", 10, len(rows), "06doc/01manuscript/Elsevier_en.tex")
    for r in rows:
        kind, shift, n = KIND_TEX[r[0]], SHIFT_TEX[r[1]], int(r[2])
        at = med[(med["shift_type"] == shift) & (med["n_cal"] == n)]
        if kind == "pair":
            s = at[at["method"] == "coral"].set_index("task_id")[["rmsep", "nrmsep"]]
            d = at[at["method"] == "deepcoral"].set_index("task_id")[["rmsep", "nrmsep"]]
        else:
            s = at[at["method"].isin(SIMPLE_FAM)].groupby("task_id")[["rmsep", "nrmsep"]].mean()
            d = at[at["method"].isin(DEEP_FAM)].groupby("task_id")[["rmsep", "nrmsep"]].mean()
        idx = s.index.intersection(d.index)
        s, d = s.loc[idx], d.loc[idx]
        tag = f"表 3 {kind} {shift} n={n}"
        _numchk(f"{tag} RMSEP_S", r[3], s["rmsep"].median(), src)
        _numchk(f"{tag} RMSEP_D", r[4], d["rmsep"].median(), src)
        _numchk(f"{tag} NRMSEP_S", r[5], s["nrmsep"].median(), src)
        _numchk(f"{tag} NRMSEP_D", r[6], d["nrmsep"].median(), src)
        _numchk(f"{tag} 简单胜率%", r[7], (s["rmsep"] < d["rmsep"]).mean() * 100, src)
        _numchk(f"{tag} Cliff δ（非配对）", r[8], _cliff_unpaired(s["rmsep"], d["rmsep"]), src)
        _pchk(f"{tag} Wilcoxon P", r[9], s["rmsep"], d["rmsep"], src)


TAB6_RULE = {
    "Simplest applicable correction (CORAL at $n=0$, SBC otherwise)": "always_simple",
    "Always model update": "always_model_update",
    "Always target-only calibration": "always_target_only",
    "Frequency map as a rule, shift type unknown": "B_shift_unknown",
    "Frequency map as a rule, shift type known (instrument tasks only)": "A_shift_known",
    "Default deep model (zero-shot CNN at $n=0$, fine-tuned CNN otherwise)": "default_deep",
}


def _regret(universe: str) -> pd.DataFrame:
    s = pd.read_excel(OUT / "73_heuristic_map_lobo.xlsx", sheet_name="summary")
    s = s[(s["universe"] == universe) & (s["benchmark"].astype(str) == "ALL")].copy()
    s["n"] = s["n_cal"].astype(str)
    return s.set_index(["rule", "n"])


def check_table6() -> None:
    """表 4（遗憾）逐格从 73 号 summary（fig4 宇宙、五基准汇总）重算，判据取自稿件表格本身。"""
    reg = _regret("fig4")
    src = "04outputs/73_heuristic_map_lobo.xlsx[summary]（universe=fig4, benchmark=ALL）"
    rows = _table_rows(_tex("Elsevier_en.tex"), "tab:regret")
    check("表 4 共 6 行", 6, len(rows), "06doc/01manuscript/Elsevier_en.tex")
    for r in rows:
        rule = TAB6_RULE[r[0].strip()]
        tag = f"表 4 {rule}"
        for cell, n in zip(r[1:6], ("0", "5", "10", "20", "40"), strict=True):
            if not cell.strip():
                check(f"{tag} n={n} 稿件留空", "True", str((rule, n) not in reg.index), src)
                continue
            _numchk(f"{tag} n={n}", cell, 100 * float(reg.loc[(rule, n), "median_regret"]), src)
        m = re.fullmatch(r"([\d.]+) \[([\d.]+), ([\d.]+)\]", r[6].strip())
        if not m:
            raise SystemExit(f"表 4 汇总格解析不了：{r[6]!r}")
        row = reg.loc[(rule, "pooled")]
        _numchk(f"{tag} 汇总", m.group(1), 100 * float(row["median_regret"]), src)
        _numchk(f"{tag} 汇总 CI 下", m.group(2), 100 * float(row["ci95_lo"]), src)
        _numchk(f"{tag} 汇总 CI 上", m.group(3), 100 * float(row["ci95_hi"]), src)
        _numchk(f"{tag} 10% 以内占比", r[7].replace("\\%", "").strip(),
                100 * float(row["within10_frac"]), src)

    # 表注里的备选加权口径
    t = _flat("Elsevier_en.tex")
    m2 = re.search(r"Weighting each\s*training benchmark equally when re-deriving the map gives "
                   r"([\d.]+)\\% \[([\d.]+), ([\d.]+)\]", t)
    if not m2:
        raise SystemExit("表 4 表注里的等权变体没解析出来")
    bal = reg.loc[("B_shift_unknown_bal", "pooled")]
    _numchk("表 4 注 各训练基准等权 汇总", m2.group(1), 100 * float(bal["median_regret"]), src)
    _numchk("表 4 注 各训练基准等权 CI 下", m2.group(2), 100 * float(bal["ci95_lo"]), src)
    _numchk("表 4 注 各训练基准等权 CI 上", m2.group(3), 100 * float(bal["ci95_hi"]), src)


def check_universe_sensitivity() -> None:
    """候选集口径敏感性（正文 3.2 节一句带过，数字在补充材料 S16 节）：五个汇总值、「漂移类型已知」规则、
    结转口径下频率图的两个端点，逐个从 extended 宇宙重算；规则与表 4 同一口径（非等权）。
    两句定性说法（覆盖全部任务的规则遗憾都上升、最简单可用校正仍最便宜、频率图与模型更新换位）按 fig4 与 extended 两个宇宙对照核。"""
    t = _flat("Supplementary_en.tex")
    reg, base = _regret("extended"), _regret("fig4")
    src = "04outputs/73_heuristic_map_lobo.xlsx[summary]（universe=extended, benchmark=ALL）"
    m = re.search(
        r"raises the regret of every rule that covers all tasks: simplest\s*"
        r"applicable correction ([\d.]+)\\%, always model update ([\d.]+)\\%, default deep model\s*"
        r"([\d.]+)\\%, always target-only ([\d.]+)\\%, frequency map with the shift type unknown\s*"
        r"([\d.]+)\\%\.[\s\S]*?costs nothing at all \(([\d.]+)\\%\)", t)
    if not m:
        raise SystemExit("S16 的候选集口径敏感性句没解析出来")
    rules = ("always_simple", "always_model_update", "default_deep", "always_target_only",
             "B_shift_unknown", "A_shift_known")
    for claim, rule in zip(m.groups(), rules, strict=True):
        _numchk(f"S16 敏感性 {rule} 汇总", claim,
                100 * float(reg.loc[(rule, "pooled"), "median_regret"]), src)
    m2 = re.search(r"under the\s*carried-forward convention \(([\d.]+)\\% against ([\d.]+)\\%; "
                   r"([\d.]+)\\% at five labels and ([\d.]+)\\% at forty\)", t)
    if not m2:
        raise SystemExit("S16 敏感性里频率图对固定处方与两端点没解析出来")
    for claim, (rule, n) in zip(m2.groups(), (("B_shift_unknown", "pooled"), ("always_simple", "pooled"),
                                                ("B_shift_unknown", "5"), ("B_shift_unknown", "40")), strict=True):
        _numchk(f"S16 敏感性 结转口径 {rule} n={n}", claim,
                100 * float(reg.loc[(rule, n), "median_regret"]), src)
    pooled = {r: float(reg.loc[(r, "pooled"), "median_regret"]) for r in rules}
    pooled0 = {r: float(base.loc[(r, "pooled"), "median_regret"]) for r in rules}
    ratio = pooled["B_shift_unknown"] / pooled["always_simple"]
    check("S16 敏感性「about twice」（频率图/最简单可用校正 ∈[1.75, 2.25]）", True,
          bool(1.75 <= ratio <= 2.25), src + f"；比值 {ratio:.2f}")
    allt = rules[:5]
    check("§3.2/S16 覆盖全部任务的规则遗憾都上升（extended > fig4）", True,
          all(pooled[r] > pooled0[r] for r in allt), src + " vs universe=fig4")
    check("§3.2/S16「漂移类型已知」规则的失效消失（fig4 >50%，extended =0）", True,
          bool(pooled0["A_shift_known"] > 0.5 and pooled["A_shift_known"] == 0), src + " vs universe=fig4")
    check("§3.2/S16 最简单可用校正仍是覆盖全部任务的规则中最便宜的", "always_simple",
          min(allt, key=pooled.get), src)
    check("S16 频率图与模型更新换位（fig4 模型更新更便宜，extended 频率图更便宜）", True,
          bool(pooled0["always_model_update"] < pooled0["B_shift_unknown"]
               and pooled["B_shift_unknown"] < pooled["always_model_update"]), src + " vs universe=fig4")


def check_tables_bilingual() -> None:
    """中英稿同一张表的数字必须逐格相同——两稿分开改，一稿改漏是最常见的失配。"""
    for stem, label, skip in (("Supplementary", "tab:si-envelope", 1),
                              ("Elsevier", "tab:fair", 2),
                              ("Elsevier", "tab:regret", 1),
                              ("Elsevier", "tab:struct", 2),
                              ("Supplementary", "tab:si-source-sanity", 1),
                              ("Supplementary", "tab:apple5", 1),
                              ("Supplementary", "tab:sbc", 1),
                              ("Supplementary", "tab:ablation", 1),
                              ("Supplementary", "tab:si-budget", 1)):
        en, zh = _tex(f"{stem}_en.tex"), _tex(f"{stem}_zh.tex")
        src = f"06doc/01manuscript/{stem}_en.tex vs {stem}_zh.tex"
        a = [[c.replace(" ", "") for c in r[skip:]] for r in _table_rows(en, label)]
        b = [[c.replace(" ", "") for c in r[skip:]] for r in _table_rows(zh, label)]
        if len(a) != len(b):
            bad = "行数不同"
        else:
            bad = next((str(i) for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y), "无")
        check(f"中英 {label} 数字逐格一致（不一致的首行序号）", "无", bad, src)


def _prescription_claims() -> list[tuple[float, float, float]]:
    """§4.2 四条处方的配对差与区间，按稿件里出现的先后取（模型更新、频率图、深度默认、只用目标域）。"""
    t = _flat("Elsevier_en.tex")
    i = t.index("its regret differs from always updating the model by")
    seg = t[i:t.index("script 87", i)]
    tri = re.findall(r"\$(-?[\d.]+)\$[^$]*?\$\[(-?[\d.]+),\s*\+?(-?[\d.]+)\]\$", seg)
    if len(tri) != 4:
        raise SystemExit(f"§4.2 配对差句子解析出 {len(tri)} 组区间，应为 4 组——句式改了就得同步改这里")
    return [(float(a), float(b), float(c)) for a, b, c in tri]


def _deep_level_claims() -> list[float]:
    """补充材料 S16「两族的绝对精度」的六个数：深度族 <0.9 / <0.8 / 中位，简单族 <0.9 / <0.8 / 中位。"""
    m = re.search(
        r"below 0\.9 on (\d+)\\%\s*of task--budget cells and below 0\.8 on (\d+)\\% "
        r"\(median ([\d.]+)\), against (\d+)\\% and\s*(\d+)\\% \(median ([\d.]+)\)",
        _flat("Supplementary_en.tex"))
    if not m:
        raise SystemExit("S16 的深度族水平句子没解析出来——句式改了就得同步改这里")
    return [float(x) for x in m.groups()]


def check_prescription_pairs_and_deep_level() -> None:
    """§4.2 的处方配对差与 §4.4 第八条的深度族 NRMSEP 分布（与脚本 87 同一算法）。"""
    pt = pd.read_excel(OUT / "73_heuristic_map_lobo.xlsx", sheet_name="per_task")
    piv = pt[pt["universe"] == "fig4"].pivot_table(
        index=["task_id", "n_cal"], columns="rule", values="regret").dropna(
        subset=["always_simple", "always_model_update", "default_deep",
                "B_shift_unknown", "always_target_only"]).reset_index()
    src = "04outputs/73_heuristic_map_lobo.xlsx[per_task]（fig4 宇宙，n>=5，脚本 87 同一算法）"
    rng = np.random.default_rng(20060515)
    tasks = piv["task_id"].unique()
    groups = {task: piv[piv["task_id"] == task] for task in tasks}
    rules = ["always_model_update", "B_shift_unknown", "default_deep", "always_target_only"]
    for rule, (med_pp, lo_pp, hi_pp) in zip(rules, _prescription_claims(), strict=True):
        obs = float(np.median(piv["always_simple"] - piv[rule])) * 100
        boot = [float(np.median(pd.concat([groups[x] for x in rng.choice(tasks, len(tasks), True)]).eval(
            "always_simple - " + rule))) * 100 for _ in range(2000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        check(f"§4.2 配对差 vs {rule} 中位", med_pp, round(obs, 1), src)
        check(f"§4.2 配对差 vs {rule} CI 下", lo_pp, round(float(lo), 1), src)
        check(f"§4.2 配对差 vs {rule} CI 上", hi_pp, round(float(hi), 1), src)

    med = _task_medians()
    keys = ["task_id", "n_cal"]
    bd = med[med["method"].isin(DEEP_FAM)].groupby(keys, as_index=False)["nrmsep"].min()
    bs = med[med["method"].isin(SIMPLE_FAM)].groupby(keys, as_index=False)["nrmsep"].min()
    src2 = "62/64 号 curves 重算（逐任务族内最优 NRMSEP）"
    d09, d08, dmed, s09, s08, smed = _deep_level_claims()
    check("S16 绝对精度 深度族 NRMSEP<0.9 占比", d09, float(round(float((bd["nrmsep"] < 0.9).mean()) * 100)), src2)
    check("S16 绝对精度 深度族 <0.8 占比", d08, float(round(float((bd["nrmsep"] < 0.8).mean()) * 100)), src2)
    check("S16 绝对精度 深度族中位", dmed, round(float(bd["nrmsep"].median()), 3), src2)
    check("S16 绝对精度 简单族 <0.9 占比", s09, float(round(float((bs["nrmsep"] < 0.9).mean()) * 100)), src2)
    check("S16 绝对精度 简单族 <0.8 占比", s08, float(round(float((bs["nrmsep"] < 0.8).mean()) * 100)), src2)
    check("S16 绝对精度 简单族中位", smed, round(float(bs["nrmsep"].median()), 3), src2)


def check_runs_per_cell() -> None:
    c62, c64 = _curves()
    src = "同上（按 task_id×method×n_cal 分组计数）"
    n62 = sorted(int(v) for v in c62.groupby(["task_id", "method", "n_cal"]).size().unique())
    n64 = sorted(int(v) for v in c64.groupby(["task_id", "method", "n_cal"]).size().unique())
    check("§2.3：经典方法每格 25 次运行", "[25]", str(n62), src)
    check("§2.3：深度方法每格 5 次运行", "[5]", str(n64), src)
    check("§2.3：5 个固定种子（经典）", 5, c62.seed.nunique(), src)
    check("§2.3：5 个固定种子（深度）", 5, c64.seed.nunique(), src)
    check("§2.3：经典每种子 5 次划分", 5, c62.rep.nunique(), src)
    check("§2.3：深度每种子 1 次", 1, c64.rep.nunique(), src)
    check("统一基准任务数 126（经典）", 126, c62.task_id.nunique(), src)
    check("统一基准任务数 126（深度）", 126, c64.task_id.nunique(), src)


def check_protocol_constants() -> None:
    code = (BASE / "02code" / "64_deep_transfer_server.py").read_text(encoding="utf-8")
    src = "02code/64_deep_transfer_server.py"
    holdout = set(re.findall(r"nv = max\(4, n // (\d+)\)", code))
    check("§2.3：源域留出约 1/6 用于早停", "['6']", str(sorted(holdout)), src)
    ft = re.search(r"def finetune\([^)]*epochs: int = (\d+)", code, re.S)
    check("§2.3：微调固定轮数（无验证集）", "80", ft.group(1) if ft else "未找到", src)
    check("§2.3：微调不切验证集（finetune 内无 nv=）", "True",
          str("nv = " not in code[code.index("def finetune"):code.index("def finetune") + 700]), src)
    dann = "Xt_un" in code and "train_dann" in code
    check("§2.3：DANN/Deep CORAL 只用标定池无标签谱（形参 Xt_un）", "True", str(dann), src)

    code62 = (BASE / "02code" / "62_crossover_engine.py").read_text(encoding="utf-8")
    src62 = "02code/62_crossover_engine.py"
    grid = re.search(r"NCAL_GRID = \[([^\]]*)\]", code62)
    seeds = re.search(r"SEEDS = \[([^\]]*)\]", code62)
    check("§2.3：预算网格含 {0,5,10,20,40}", "True",
          str(grid is not None and all(f"{n}" in grid.group(1) for n in (0, 5, 10, 20, 40))), src62)
    check("§2.3：5 个固定种子写死在源码", 5,
          len(seeds.group(1).split(",")) if seeds else 0, src62)


def check_win_rates() -> None:
    """§3.2 的两句胜率：两经典候选之间的 CORAL 胜率，以及扩到六个免标签候选后的最频胜者份额。

    这两句问的不是同一件事：前一句只在直接迁移与 CORAL 之间比（图 3a 的 n=0 行就是这两个
    候选），后一句把四个零样本深度模型也放进来。原来这里拿 phase_diagram 的 win_coral
    （全体候选口径）去核前一句，口径本身就对不上。
    """
    t = _flat("Elsevier_en.tex")
    med = _task_medians()
    z = med[med["n_cal"] == 0]
    src = "62/64 号 curves 重算（n=0 逐任务最小 RMSEP 的胜者份额）"

    m = re.search(r"wins on (\d+)--(\d+)\\% of the tasks of every shift type", t)
    if not m:
        raise SystemExit("§3.2 两经典候选胜率句没解析出来")
    lo_claim, hi_claim = int(m.group(1)), int(m.group(2))
    two = z[z["method"].isin(["zero_shot", "coral"])]
    w2 = two.loc[two.groupby(["shift_type", "task_id"])["rmsep"].idxmin()]
    frac = w2.groupby("shift_type")["method"].apply(lambda x: float((x == "coral").mean()) * 100)
    check("§3.2：两经典候选下 CORAL 零标签胜率下界", lo_claim, round(float(frac.min())), src)
    check("§3.2：两经典候选下 CORAL 零标签胜率上界", hi_claim, round(float(frac.max())), src)
    check("§3.2：四类漂移都不低于下界", 4, int((frac >= lo_claim).sum()), src)

    six = z[z["method"].isin(LABEL_FREE)]
    w6 = six.loc[six.groupby(["shift_type", "task_id"])["rmsep"].idxmin()]
    share = (w6.groupby("shift_type")["method"].value_counts(normalize=True) * 100)
    m2 = re.search(r"instrument \((\d+)\\%\),\s*seasonal \((\d+)\\%\)\s*and cross-origin \((\d+)\\%\)", t)
    m3 = re.search(r"DeepCORAL wins (\d+)\\% of the tasks\s*against CORAL's (\d+)\\%", t)
    if not (m2 and m3):
        raise SystemExit("§3.2 六个免标签候选的份额句没解析出来")
    src6 = "62/64 号 curves 重算（n=0 六个免标签候选之间的最频胜者份额）"
    for shift, claim in zip(("instrument", "season", "origin_year_instrument"), m2.groups(), strict=True):
        check(f"§3.2：六候选下 CORAL 最频胜者份额（{shift}）", int(claim),
              round(float(share.get((shift, "coral"), 0.0))), src6)
    check("§3.2：土壤上 DeepCORAL 份额", int(m3.group(1)),
          round(float(share.get(("lab", "deepcoral"), 0.0))), src6)
    check("§3.2：土壤上 CORAL 份额", int(m3.group(2)),
          round(float(share.get(("lab", "coral"), 0.0))), src6)
    check("§3.2：土壤上 CORAL 不是最频胜者", "True",
          str(float(share.get(("lab", "deepcoral"), 0.0)) > float(share.get(("lab", "coral"), 0.0))), src6)


def _decimals(path: Path) -> collections.Counter[str]:
    """一份稿里出现的全部小数（去掉注释、文献表、DOI、\\ref 之类的花括号参数）。"""
    t = path.read_text(encoding="utf-8")
    t = re.sub(r"(?m)^\s*%.*$", " ", t)          # 整行注释
    t = re.sub(r"(?<!\\)%.*", " ", t)            # 行尾注释
    i = t.find("\\begin{thebibliography}")
    if i > 0:
        t = t[:i]
    t = re.sub(r"\\(cite|ref|label|includegraphics|input|bibitem)\{[^}]*\}", " ", t)
    t = re.sub(r"doi:\s*\S+", " ", t)
    # 先把控制字命令换成空白：英文写 $\rho\approx0.65$、中文写 $\rho$$\approx$0.65，
    # 数字前一个字符一个是字母一个是 $，不归一化就会把排版差异误报成内容差异。
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    return collections.Counter(re.findall(r"(?<![\w.])\d+\.\d+(?![\w])", t))


def check_bilingual_numbers() -> None:
    """中英两稿的小数多重集必须完全相同——两稿分开改，漏改一处就会在这里露出来。

    只比数值本身、不比措辞：一个数在英文稿出现 3 次而中文稿 2 次，说明有一处没跟着改。
    """
    for zh_name, en_name in (("Elsevier_zh.tex", "Elsevier_en.tex"),
                             ("Supplementary_zh.tex", "Supplementary_en.tex")):
        en, zh = _decimals(MS / en_name), _decimals(MS / zh_name)
        diff = sorted({k for k in set(en) | set(zh) if en.get(k, 0) != zh.get(k, 0)})
        detail = "；".join(f"{k} 英{en.get(k, 0)}中{zh.get(k, 0)}" for k in diff[:6])
        check(f"中英 {en_name[:-4]} 小数多重集一致", "[]", str(diff),
              f"{en_name} vs {zh_name}" + (f"（{detail}）" if detail else ""))


def check_shift_structure() -> None:
    """§3.4 的漂移结构：五个基准的低阶份额、CORAL 回收幅度，以及那段基准层面的具体读法。

    2026-09-10 在这里查到过一处事实错误：正文四处写「低阶份额最低（残余最大）的基准正是
    深度族获胜的那一个」，而 74 号的按基准表给出的顺序是芒果 0.733 < 苹果 0.744 <
    土壤 0.775——残余最大的是芒果，深度族赢的是土壤。所以这一组数字必须逐个对表。
    """
    b = pd.read_excel(OUT / "74_shift_structure_diagnosis.xlsx", sheet_name="表a：by_benchmark")
    b = b.set_index("benchmark")
    src = "04outputs/74_shift_structure_diagnosis.xlsx[表a：by_benchmark]"
    t = _flat("Elsevier_en.tex")

    inst = [float(b.loc[x, "f_coral_sw1"]) for x in ("corn", "tablet")]
    pop = {x: float(b.loc[x, "f_coral_sw1"]) for x in ("mango", "apple", "ossl_mir")}
    m = re.search(r"On the two instrument benchmarks (\d+)\\% of the sliced-Wasserstein "
                  r"distance is removed by the first two moments \((\d+)--(\d+)\\% by the mean "
                  r"alone\) and the residual is (\d+)\\%; on the laboratory, seasonal and "
                  r"origin/year benchmarks the low-order share falls to (\d+)--(\d+)\\% and the "
                  r"residual rises to (\d+)--(\d+)\\%", t)
    if not m:
        raise SystemExit("§3.4 低阶份额那句没解析出来")
    check("§3.4 仪器基准低阶份额", int(m.group(1)), round(min(inst) * 100), src)
    check("§3.4 仪器基准残余", int(m.group(4)), round((1 - min(inst)) * 100), src)
    check("§3.4 群体类基准份额下界", int(m.group(5)), round(min(pop.values()) * 100), src)
    check("§3.4 群体类基准份额上界", int(m.group(6)), round(max(pop.values()) * 100), src)
    check("§3.4 群体类基准残余下界", int(m.group(7)), round((1 - max(pop.values())) * 100), src)
    check("§3.4 群体类基准残余上界", int(m.group(8)), round((1 - min(pop.values())) * 100), src)

    m2 = re.search(r"since mango \((\d+)\\%\) and apple \((\d+)\\%\) leave a "
                   r"larger residual than soil \((\d+)\\%\)", t)
    if not m2:
        raise SystemExit("§3.4 基准层面读法那句没解析出来")
    for name, claim in zip(("mango", "apple", "ossl_mir"), m2.groups(), strict=True):
        check(f"§3.4 {name} 低阶份额", int(claim), round(pop[name] * 100), src)
    check("§3.4 土壤的残余并非最大（芒果、苹果更大）", "True",
          str(pop["ossl_mir"] > pop["mango"] and pop["ossl_mir"] > pop["apple"]), src)

    m3 = re.search(r"label-free alignment recovers only (\d+)\\% of its zero-label error "
                   r"\((\d+)--(\d+)\\% on mango and apple, (\d+)--(\d+)\\% on the "
                   r"instrument benchmarks\) and leaves it at NRMSEP ([\d.]+)", t)
    if not m3:
        raise SystemExit("§3.4 误差空间那句没解析出来")
    g = {x: float(b.loc[x, "coral_gain"]) for x in b.index}
    check("§3.4 土壤 CORAL 回收幅度", int(m3.group(1)), round(g["ossl_mir"] * 100), src)
    check("§3.4 芒果/苹果回收下界", int(m3.group(2)),
          round(min(g["mango"], g["apple"]) * 100), src)
    check("§3.4 芒果/苹果回收上界", int(m3.group(3)),
          round(max(g["mango"], g["apple"]) * 100), src)
    check("§3.4 仪器基准回收下界", int(m3.group(4)),
          round(min(g["corn"], g["tablet"]) * 100), src)
    check("§3.4 仪器基准回收上界", int(m3.group(5)),
          round(max(g["corn"], g["tablet"]) * 100), src)
    check("§3.4 土壤对齐后 NRMSEP", float(m3.group(6)),
          round(float(b.loc["ossl_mir", "nrmsep_coral0"]), 2), src)
    check("§3.4 土壤确是回收最少的那个基准", "True",
          str(min(g, key=lambda k: g[k]) == "ossl_mir"), src)
    check("§3.4 土壤确是深度族唯一获胜的基准", "True",
          str([x for x in b.index if float(b.loc[x, "margin20_deep_minus_simple"]) < 0]
              == ["ossl_mir"]), src)


SRC74_TASK = "04outputs/74_shift_structure_diagnosis.xlsx[per_task] 全精度重算 Spearman"


def _rho_full(pred: str, target: str) -> float:
    """点估计直接在 126 个任务上重算：74 号 correlations 表只存三位小数，
    稿件再舍入到两位会出现连环舍入（f_coral_sw1 对 CORAL 增益 0.4951 → 0.495 → 0.49）。"""
    t = pd.read_excel(OUT / "74_shift_structure_diagnosis.xlsx", sheet_name="per_task")
    return float(spearmanr(t[pred], t[target]).correlation)


_CI74: dict[tuple[str, str], tuple[float, float, float]] = {}


def _ci74_full() -> dict[tuple[str, str], tuple[float, float, float]]:
    """重放 74 号第 4 步的域对聚类 bootstrap（同一种子 20060515、NBOOT=2000、同一 preds×targets 顺序），
    取全精度的 ρ 与 95% 区间。存档只留三位小数，再舍到两位会连环舍入（A 对零标签边际的上界
    0.5646 → 0.565 → 0.57，实为 0.56），所以两位小数一律按全精度判；重放本身先逐格对存档三位小数。"""
    if _CI74:
        return _CI74
    pt = pd.read_excel(OUT / "74_shift_structure_diagnosis.xlsx", sheet_name="per_task")
    ref = pd.read_excel(OUT / "74_shift_structure_diagnosis.xlsx", sheet_name="correlations")
    ref = ref.set_index(["predictor", "target"])
    rng = np.random.default_rng(20060515)
    cl_all = pt["pair"].to_numpy().astype(str)
    preds = ["f_coral_mmd2", "f_coral_sw1", "residual_mmd2", "A_lowrank", "A_lowrank_oos", "mmd2_raw"]
    targets = ["coral_gain", "nrmsep_zero_shot0", "nrmsep_coral0", "margin0_deep_minus_simple",
               "margin20_deep_minus_simple"]
    bad = []
    for pr in preds:
        for tg in targets:
            x, y = pt[pr].to_numpy(float), pt[tg].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            x, y, cl = x[ok], y[ok], cl_all[ok]
            rho = float(spearmanr(x, y).correlation)
            ucl = np.unique(cl)
            idx = {c: np.where(cl == c)[0] for c in ucl}
            boots = []
            for _ in range(2000):
                pick = rng.choice(ucl, len(ucl), replace=True)
                ii = np.concatenate([idx[c] for c in pick])
                if len(np.unique(x[ii])) < 3 or len(np.unique(y[ii])) < 3:
                    continue
                boots.append(spearmanr(x[ii], y[ii])[0])
            full = (rho, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
            r = ref.loc[(pr, tg)]
            if tuple(round(v, 3) for v in full) != (r["spearman_rho"], r["ci95_lo"], r["ci95_hi"]):
                bad.append((pr, tg))
            _CI74[(pr, tg)] = full
    check("74 号 bootstrap 重放与存档三位小数逐格一致（30 组）", "[]", str(bad),
          "04outputs/74_shift_structure_diagnosis.xlsx[correlations]")
    return _CI74


def check_structure_headlines_and_soil() -> None:
    """摘要、§4.1、结论里的份额与相关同用切片 Wasserstein 口径；土壤的深度族与简单族对比同用
    逐任务族内最优（75 号 reckoning_curves_nrmsep）；§4.3 的 R² 分定义报；以及已改掉的旧说法。"""
    g_sw = round(_rho_full("f_coral_sw1", "coral_gain"), 2)
    m_sw = round(_rho_full("f_coral_sw1", "margin0_deep_minus_simple"), 2)
    t_en, t_zh = _flat("Elsevier_en.tex"), _flat("Elsevier_zh.tex")
    pats_en = (
        ("摘要", r"correlates with the CORAL gain \(\$\\rho=(\d+\.\d+)\$\) and the simple family's\s+"
                 r"zero-label margin \(\$\\rho=(\d+\.\d+)\$\)"),
        ("§3.4", r"the sliced-Wasserstein version gives \$\\rho=(\d+\.\d+)\$, \$-[\d.]+\$, \$(\d+\.\d+)\$"),
    )
    pats_zh = (
        ("摘要", r"CORAL 增益（\$\\rho=(\d+\.\d+)\$）和零标签处的简单族优势（\$\\rho=(\d+\.\d+)\$）"),
        ("§3.4", r"切片 Wasserstein 版本给出 \$\\rho=(\d+\.\d+)\$、\$-[\d.]+\$、\$(\d+\.\d+)\$"),
    )
    for lang, text, pats in (("英", t_en, pats_en), ("中", t_zh, pats_zh)):
        for tag, pat in pats:
            m = re.search(pat, text)
            if not m:
                check(f"{lang}文{tag} 低阶份额相关句可解析", "True", "False",
                      f"Elsevier_{'en' if lang == '英' else 'zh'}.tex")
                continue
            check(f"{lang}文{tag} 份额与 CORAL 增益 ρ（切片 Wasserstein，与 92%/73–78% 同口径）",
                  float(m.group(1)), g_sw, SRC74_TASK)
            check(f"{lang}文{tag} 份额与零标签边际 ρ（切片 Wasserstein）", float(m.group(2)), m_sw, SRC74_TASK)
    # 结论只报份额与简单族优势的那一个 ρ
    for lang, text, pat in (("英", t_en, r"tracked the simple family's margin\s+\(\$\\rho=(\d+\.\d+)\$\)"),
                            ("中", t_zh, r"追踪简单族的优势（\$\\rho=(\d+\.\d+)\$）")):
        m = re.search(pat, text)
        check(f"{lang}文结论 份额与零标签边际 ρ（切片 Wasserstein）", m_sw,
              float(m.group(1)) if m else "句子未解析", SRC74_TASK)

    # 土壤：深度族与简单族都取逐任务族内最优
    rc = pd.read_excel(OUT / "75_nrmsep_table_columns.xlsx", sheet_name="reckoning_curves_nrmsep")
    lab = rc[rc["shift_type"] == "lab"].set_index(["n_cal", "group"])["median_nrmsep"]
    src75 = "04outputs/75_nrmsep_table_columns.xlsx[reckoning_curves_nrmsep] lab"
    d0, d20, d40 = (float(lab[(n, "best deep/physics")]) for n in (0, 20, 40))
    s40 = float(lab[(40, "best simple")])
    smax = max(float(lab[(n, "best simple")]) for n in (0, 5, 10, 20, 40))
    rx_en = (
        ("§3.1", r"its NRMSEP falling from (\d+\.\d+) at zero labels to (\d+\.\d+) at 40 while the simple family\s+"
                 r"stays between (\d+\.\d+) and (\d+\.\d+)", (d0, d40, smax, s40)),
        ("§4.2", r"a fine-tuned network \(NRMSEP (\d+\.\d+) at 20 labels\)", (d20,)),
        ("结论", r"reaching NRMSEP (\d+\.\d+) where the simple family stopped at\s+(\d+\.\d+)", (d40, s40)),
    )
    rx_zh = (
        ("§3.1", r"其 NRMSEP 由零标签处的 (\d+\.\d+) 降到 40 标签处的 (\d+\.\d+)，"
                 r"而简单族始终在 (\d+\.\d+)\$\\sim\$(\d+\.\d+) 之间",
         (d0, d40, smax, s40)),
        ("§4.2", r"（20 标签处 NRMSEP (\d+\.\d+)）", (d20,)),
        ("结论", r"NRMSEP 达到 (\d+\.\d+)，而简单族止步于 (\d+\.\d+)", (d40, s40)),
    )
    for lang, text, rxs in (("英", t_en, rx_en), ("中", t_zh, rx_zh)):
        for tag, pat, want in rxs:
            m = re.search(pat, text)
            if not m:
                check(f"{lang}文{tag} 土壤 NRMSEP 句可解析", "True", "False",
                      f"Elsevier_{'en' if lang == '英' else 'zh'}.tex")
                continue
            for i, w in enumerate(want, 1):
                _numchk(f"{lang}文{tag} 土壤 NRMSEP 第 {i} 个数", m.group(i), w, src75)
    r1 = (REV / "Response_R1.tex").read_text(encoding="utf-8")
    m = re.search(r"reaching NRMSEP (\d+\.\d+) where the simple family\s+stops at (\d+\.\d+)", r1)
    if m:
        _numchk("R1 土壤 深度族 40 标签", m.group(1), d40, src75)
        _numchk("R1 土壤 简单族 40 标签", m.group(2), s40, src75)
    else:
        check("R1 土壤 NRMSEP 句可解析", "True", "False", "Response_R1.tex")

    # §3.3 的 R²：主文定义与 n_usable 定义分开报
    fit = pd.read_excel(OUT / "63_crossover_analysis.xlsx", sheet_name="scaling_fit")
    src63 = "04outputs/63_crossover_analysis.xlsx[scaling_fit]"
    main_def = fit[fit["target"] == "n_simple_vs_deep"]["r2"]
    usable = fit[fit["target"] == "n_usable"]
    m = re.search(r"reach \$R\^\{2\}\$ of only (\d+\.\d+)--(\d+\.\d+) \(at most (\d+\.\d+) under the "
                  r"\$n_\{\\text\{usable\}\}\$ definition, defined on (\d+) tasks\)", t_en)
    if not m:
        check("§3.3 R² 句可解析", "True", "False", "Elsevier_en.tex")
    else:
        _numchk("§3.3 主文定义 R² 下限", m.group(1), float(main_def.min()), src63)
        _numchk("§3.3 主文定义 R² 上限", m.group(2), float(main_def.max()), src63)
        _numchk("§3.3 n_usable 定义 R² 上限", m.group(3), float(usable["r2"].max()), src63)
        check("§3.3 n_usable 任务数", int(m.group(4)), int(usable["n"].iloc[0]), src63)
    check("§2 与 §3.3 的预测量只列 63 号实际拟合过的三组（无信噪比）", "['mmd', 'mmd+dim', 'w1+dim']",
          str(sorted(fit["features"].unique())), src63)

    # 已改掉的说法不得残留
    si_en, si_zh = _flat("Supplementary_en.tex"), _flat("Supplementary_zh.tex")
    for name, text, stale_pats in (
            ("Elsevier_en.tex", t_en, (r"no first- and second-moment correction can", r"neither can touch",
                                       r"spent instead", r"drops the law", r"signal-to-noise",
                                       r"no low-order correction can reach", r"treated exactly like the four public",
                                       r"advantage widens to", r"NRMSEP falls to 0\.50", r"reaching NRMSEP 0\.50")),
            ("Elsevier_zh.tex", t_zh, (r"信噪比", r"任何一阶、二阶矩校正都无法消除", r"两者都触及不到", r"改用于",
                                       r"干脆不要该定律", r"完全同等处理", r"NRMSEP 降到 0\.50", r"NRMSEP 达到 0\.50")),
            ("Supplementary_en.tex", si_en, (r"simple-minus-deep", r"Spectra from the two seasons were aligned")),
            ("Supplementary_zh.tex", si_zh, (r"简单减深度", r"两季光谱统一对齐")),
            ("Response_R1.tex", r1, (r"that neither can remove", r"no first- and second-moment correction can remove",
                                     r"reaching NRMSEP 0\.50"))):
        left = [p for p in stale_pats if re.search(p, text)]
        check(f"{name} 不再含取证第三轮已改掉的说法", "[]", str(left), name)

    # 书目：与出版方记录一致的作者
    for name in ("Elsevier_en.tex", "Elsevier_zh.tex"):
        txt = (MS / name).read_text(encoding="utf-8")
        check(f"{name} LiL2021WS 第二作者按 Crossref 记为 JANG X", "True",
              str("\\bibitem{LiL2021WS} LI L, JANG X, LI B" in txt), name)
        check(f"{name} OSSL2025 第三作者按 Crossref 记为 PARENTE L L", "True",
              str("\\bibitem{OSSL2025} SAFANELLI J L, HENGL T, PARENTE L L" in txt), name)
    bib = (MS / "refs.bib").read_text(encoding="utf-8")
    check("refs.bib 机构作者不被 and 拆开", "0", str(bib.count("Eigenvector Research and Inc")), "refs.bib")


XL106 = OUT / "106_season_scaler_sensitivity.xlsx"


def _f(x: float, nd: int) -> str:
    """按稿件的位数格式化；负号统一成 ASCII，-0.000 记作 0.000。"""
    v = round(float(x), nd)
    return f"{0.0 if v == 0 else v:.{nd}f}"


def check_season_scaler_s2() -> None:
    """补充材料 S2 的年份级标准化一段：两层标准化的说明与 106 号三种拟合口径的结果，中英各核一遍。"""
    src = "04outputs/106_season_scaler_sensitivity.xlsx"
    b = pd.read_excel(XL106, sheet_name="表b：§3.5四个比较")
    b = b[b["场景集"] == "三口径共同"].set_index(["口径", "比较"])
    c = pd.read_excel(XL106, sheet_name="表c：口径相对all的位移")
    c = c[(c["场景集"] == "三口径共同") & (c["口径"] == "no_target")].set_index("方法_校正")
    d = pd.read_excel(XL106, sheet_name="表d：all口径对正典（跨平台浮点差）").set_index(["方法", "校正"])
    e = pd.read_excel(XL106, sheet_name="表e：同年舍入噪声与跨年位移")
    cnt = pd.read_excel(XL106, sheet_name="表f：场景计数").iloc[0]
    rng = pd.read_excel(XL106, sheet_name="表g：各季未标准化光谱范围").set_index("年份")
    V = ("all", "no_test", "no_target")
    phys_cnn = [b.loc[(v, "未校正 Phys+BL − CNN")] for v in V]
    plsr = [b.loc[(v, "校正后 PLSR+SB − Phys+BL+SB")] for v in V]
    others = [b.loc[(v, f"校正后 {m}+SB − Phys+BL+SB")] for v in V for m in ("SVR", "CNN")]
    svr_nt = b.loc[("no_target", "校正后 SVR+SB − Phys+BL+SB")]
    noise = e[(e["场景类型"] == "同年")].set_index(["口径", "方法_校正"])["|差|中位"]
    repro = max(float(d.loc[(m, k), "|差|P90"]) for m in ("PLSR", "SVR", "Phys+BL") for k in ("裸", "+SB"))

    # 与稿件无关的前提：六个主比较的区间都不含零；全部光谱口径下 SVR+SB、CNN+SB 的区间都含零
    check("S2 六个主比较的 95% 区间都不含零", "True",
          str(all(r["CI上界"] < 0 for r in phys_cnn + plsr)), src + "[表b] 三口径共同")
    check("S2 全部光谱口径下 SVR+SB、CNN+SB 与 Phys+BL+SB 不可分辨", "True",
          str(all(b.loc[("all", f"校正后 {m}+SB − Phys+BL+SB"), "CI下界"] <= 0
                  <= b.loc[("all", f"校正后 {m}+SB − Phys+BL+SB"), "CI上界"] for m in ("SVR", "CNN"))),
          src + "[表b]")
    check("S2 校正后 SVR、CNN 与物理模型的中位配对差都不超过 0.006", "True",
          str(max(abs(float(r["配对差中位"])) for r in others) <= 0.006), src + "[表b]")
    check("S2 每种校正后方法扣除目标域的位移都小于 0.01", "True",
          str(max(abs(float(c.loc[m, "相对all配对差中位"]))
                  for m in ("PLSR + SB", "SVR + SB", "CNN + SB", "Phys+BL + SB")) < 0.01), src + "[表c]")
    check("S2 全部光谱口径复现主实验：PLSR/SVR/Phys+BL 的 90% 分位差 ≤0.005", "True", str(repro <= 0.005),
          src + "[表d]")

    want = {
        "n_def": str(int(cnt["no_target可定义场景数"])), "n_all": str(int(cnt["场景总数"])),
        "n_rest": str(int(cnt["场景总数"] - cnt["no_target可定义场景数"])), "seed": str(int(cnt["种子"])),
        "n_com": str(int(cnt["三口径共同场景"])), "n_same": str(int(cnt["同年场景数"])),
        "cnn_repro": _f(d.loc[("CNN", "裸"), "|差|中位"], 2),
        "pc": [_f(r["配对差中位"], 2) for r in phys_cnn], "pl": [_f(r["配对差中位"], 3) for r in plsr],
        "svr": _f(svr_nt["配对差中位"], 3), "svr_lo": _f(svr_nt["CI下界"], 3), "svr_hi": _f(svr_nt["CI上界"], 3),
        "plsr_shift": _f(c.loc["PLSR", "相对all配对差中位"], 2), "cnn_shift": _f(c.loc["CNN", "相对all配对差中位"], 2),
        "noise_cnn": sorted({_f(noise[(v, "CNN")], 2) for v in ("no_test", "no_target")}),
        "noise_phys": (_f(min(noise[(v, "Phys+BL")] for v in ("no_test", "no_target")), 2),
                       _f(max(noise[(v, "Phys+BL")] for v in ("no_test", "no_target")), 2)),
        "r18": (f"{rng.loc[2018, '最小']:.0f}", f"{rng.loc[2018, '最大'] / 1e4:.1f}"),
        "r19": (f"{rng.loc[2019, '最小'] / 1e3:.1f}", f"{rng.loc[2019, '最大'] / 1e4:.1f}"),
        "r25": (f"{rng.loc[2025, '最小']:.2f}", f"{rng.loc[2025, '最大']:.0f}"),
    }
    num = r"(-?\d+(?:\.\d+)?)"
    pats: dict[str, list[tuple[str, str, Callable[[dict[str, Any]], list[Any]]]]] = {
        "en": [
            ("季内范围", r"about " + num + r" to \$" + num + r"\\times10\^\{4\}\$ in 2018 and \$" + num
             + r"\\times10\^\{3\}\$ to \$" + num + r"\\times10\^\{4\}\$ in 2019, reflectance of " + num + r" to "
             + num + r" in 2025", lambda w: [*w["r18"], *w["r19"], *w["r25"]]),
            ("可扣除场景", r"possible in (\d+) of the (\d+) scenarios; in the other (\d+) the target",
             lambda w: [w["n_def"], w["n_all"], w["n_rest"]]),
            ("种子", r"with one seed \((\d+)\)", lambda w: [w["seed"]]),
            ("CNN 复现差", r"library version, differs by a median " + num, lambda w: [w["cnn_repro"]]),
            ("共同场景", r"Over the (\d+) scenarios in which every method", lambda w: [w["n_com"]]),
            ("两组中位配对差", r"median paired differences are \$" + num + r"\$, \$" + num + r"\$ and \$" + num
             + r"\$\\,\\Brix\{\} for the first and \$" + num + r"\$, \$" + num + r"\$ and \$" + num + r"\$",
             lambda w: [*w["pc"], *w["pl"]]),
            ("SVR+SB 落后", r"ahead of SVR\+SB by " + num + r"\\,\\Brix\{\} \[" + num + r", " + num + r"\]",
             lambda w: [w["svr"], w["svr_lo"], w["svr_hi"]]),
            ("未校正位移", r"PLSR by a median \$\+" + num + r"\$ and of the CNN by a median \$\+" + num + r"\$",
             lambda w: [w["plsr_shift"], w["cnn_shift"]]),
            ("同年噪声", r"in the (\d+) same-year scenarios, .{0,120}?by a median of " + num + r" and " + num
             + r"--" + num + r"\\,\\Brix\{\} in absolute value",
             lambda w: [w["n_same"], *w["noise_cnn"], *w["noise_phys"]]),
        ],
        "zh": [
            ("季内范围", r"约 " + num + r"～\$" + num + r"\\times10\^\{4\}\$；2019 年约 \$" + num
             + r"\\times10\^\{3\}\$～\$" + num + r"\\times10\^\{4\}\$；2025 年为反射率，" + num + r"～" + num,
             lambda w: [*w["r18"], *w["r19"], *w["r25"]]),
            ("可扣除场景", r"(\d+) 个场景中 (\d+) 个可以扣除；其余 (\d+) 个",
             lambda w: [w["n_all"], w["n_def"], w["n_rest"]]),
            ("种子", r"一粒种子（(\d+)）", lambda w: [w["seed"]]),
            ("CNN 复现差", r"差值中位 " + num, lambda w: [w["cnn_repro"]]),
            ("共同场景", r"都有结果的 (\d+) 个场景上", lambda w: [w["n_com"]]),
            ("两组中位配对差", r"前者的中位配对差为 \$" + num + r"\$、\$" + num + r"\$ 与 \$" + num
             + r"\$\\,\\Brix\{\}，后者为 \$" + num + r"\$、\$" + num + r"\$ 与 \$" + num + r"\$",
             lambda w: [*w["pc"], *w["pl"]]),
            ("SVR+SB 落后", r"领先 SVR\+SB " + num + r"\\,\\Brix\{\} \[" + num + r", " + num + r"\]",
             lambda w: [w["svr"], w["svr_lo"], w["svr_hi"]]),
            ("未校正位移", r"逐场景变化中位 \$\+" + num + r"\$ 与 \$\+" + num + r"\$",
             lambda w: [w["plsr_shift"], w["cnn_shift"]]),
            ("同年噪声", r"在 (\d+) 个同年场景中.{0,80}?中位 " + num + r" 与 " + num + r"～" + num,
             lambda w: [w["n_same"], *w["noise_cnn"], *w["noise_phys"]]),
        ],
    }
    for lang, fname in (("en", "Supplementary_en.tex"), ("zh", "Supplementary_zh.tex")):
        text = _flat(fname)
        for tag, pat, exp in pats[lang]:
            m = re.search(pat, text)
            if not m:
                check(f"S2（{lang}）{tag} 可解析", "True", "False", fname)
                continue
            check(f"S2（{lang}）{tag}", " / ".join(m.groups()), " / ".join(map(str, exp(want))), src)
    # 旧说法：标准化器只在源域上拟合、目标域统计量不进入
    for fname, stale in (("Supplementary_en.tex", "so that target-domain statistics do not enter"),
                         ("Supplementary_zh.tex", "以避免目标域统计量进入训练流程")):
        check(f"{fname} 不再称目标域统计量不进入标准化", "False", str(stale in _tex(fname)), fname)


XL108 = OUT / "108_apple_unified_season_scaler.xlsx"
_WORD = {2: "two", 3: "three", 4: "four", 5: "five"}


def _sci(x: float) -> list[str]:
    """一位有效数字的科学计数：返回 [尾数, 指数的绝对值]，对应稿件里的 $m\\times10^{-e}$。"""
    m, e = f"{float(x):.0e}".split("e")
    return [m, str(-int(e))]


def check_unified_season_scaler() -> None:
    """统一基准苹果任务的逐季标准化改为扣除目标域重新拟合（108 号）：正文 §2.3/§3.1/§3.2/§4/局限/结论/摘要
    与补充材料 S2 统一基准段引用的数，中英各核一遍。"""
    src = "04outputs/108_apple_unified_season_scaler.xlsx"
    env = pd.read_excel(XL108, sheet_name="表c").set_index("n_cal")
    plat = pd.read_excel(XL108, sheet_name="表f：深度对照任务上平台差与口径差").set_index("方法").loc["全部深度方法"]
    reg = pd.read_excel(XL108, sheet_name="表i：五基准头条处方遗憾").set_index(["口径", "规则"])
    bj = pd.read_excel(XL108, sheet_name="表j：基准级均衡与苹果族均值")
    bj = bj[(bj["口径"] == "苹果换扣除目标域") & (bj["量"] == "基准级均衡均值差（深度−简单）")].set_index("n_cal")
    fk = pd.read_excel(XL108, sheet_name="表k：免选择比较（表fair口径）")
    fk = fk[fk["口径"] == "重建·扣除目标域"]
    rep = fk[fk["comparison"] == "representative pair"].iloc[0]
    fam = fk[fk["comparison"] == "family mean"].set_index("n")
    selfchk = pd.read_excel(XL108, sheet_name="表l：全季重建对正典（简单族）").iloc[0]
    metas = [json.loads(Path(f).read_text(encoding="utf-8"))
             for f in sorted(glob.glob(str(OUT / "108_shards" / "deep_no_target*.meta.json")))
             if not Path(f).name.startswith("._")]
    n_jobs = len([f for f in glob.glob(str(OUT / "108_shards" / "deep_no_target*_shard*.csv"))
                  if not Path(f).name.startswith("._")])
    n_gpu = sum(m["device"] == "cuda" for m in metas)

    def pct(x: float) -> str:
        return f"{100 * float(x):.1f}"

    def r(k: str, rule: str) -> Any:
        return reg.loc[(k, rule)]

    S, C = "苹果换扣除目标域", "正典"
    sep = [nc for nc in env.index if env.loc[nc, "wilcoxon_p"] < 0.05]
    rest = [nc for nc in env.index if nc not in sep]
    six = ["always_simple", "always_model_update", "always_target_only", "B_shift_unknown",
           "A_shift_known", "default_deep"]

    # 前提：稿件的定性说法
    check("108 包络分开的预算恰为 10 与 20 标签", "[10, 20]", str(sep), src + "[表c]")
    check("108 包络在 10、20 标签的 P ≤ 4e-4", "True", str(max(env.loc[sep, "wilcoxon_p"]) <= 4e-4), src + "[表c]")
    check("108 代表性配对 P<1e-4、族均值全网格 P<1e-5", "True",
          str(rep["wilcoxon_p"] < 1e-4 and fam["wilcoxon_p"].max() < 1e-5), src + "[表k]")
    check("108 表 regret 六行的排序与正典相同", str(sorted(six, key=lambda k: r(C, k)["遗憾中位"])),
          str(sorted(six, key=lambda k: r(S, k)["遗憾中位"])), src + "[表i]")
    check("108 40 标签处模型更新与目标自建都比最简单可用校正便宜", "True",
          str(r(S, "always_model_update")["n=40"] < r(S, "always_simple")["n=40"]
              and r(S, "always_target_only")["n=40"] < r(S, "always_simple")["n=40"]), src + "[表i]")
    others = [k for k in reg.loc[S].index if k != "always_simple"]
    check("108 最简单可用校正的汇总遗憾仍低于其余全部处方", "True",
          str(all(r(S, "always_simple")["遗憾中位"] < r(S, k)["遗憾中位"] for k in others)), src + "[表i]")
    check("108 最简单可用校正在 5–40 标签都在预言机 10% 以内", "True",
          str(max(r(S, "always_simple")[f"n={nc}"] for nc in (5, 10, 20, 40)) < 0.10), src + "[表i]")
    check("108 漂移类型已知规则不涉及苹果、数值不变", str(r(C, "A_shift_known")["遗憾中位"]),
          str(r(S, "A_shift_known")["遗憾中位"]), src + "[表i]")

    simple, mu, deep, to = (r(S, k) for k in ("always_simple", "always_model_update", "default_deep",
                                              "always_target_only"))
    b0, b20 = bj.loc[0], bj.loc[20]
    w: dict[str, Any] = {
        "n_task": str(int(env["n_task"].iloc[0])),
        "sep": [str(nc) for nc in sep], "d_sep": [_f(env.loc[nc, "cliffs_delta"], 2) for nc in sep],
        "d_rest_rng": [_f(min(env.loc[rest, "cliffs_delta"]), 2), _f(max(env.loc[rest, "cliffs_delta"]), 2)],
        "d_rest": [_f(env.loc[nc, "cliffs_delta"], 2) for nc in rest],
        "p_rest": f"{np.floor(min(env.loc[rest, 'wilcoxon_p']) * 100) / 100:.2f}",
        "rep": [_f(rep["delta_unpaired"], 2), str(int(rep["simple_win_pct"]))],
        "rep_p": _sci(rep["wilcoxon_p"]),
        "fam": [_f(fam.loc[n, "delta_unpaired"], 2) for n in (0, 5, 10, 20, 40)],
        "fam_rng": [_f(fam["delta_unpaired"].min(), 2), _f(fam["delta_unpaired"].max(), 2)],
        "fam_nr": [_f(fam["delta_unpaired_nrmsep"].min(), 2), _f(fam["delta_unpaired_nrmsep"].max(), 2)],
        "bench": [_f(b0["值"], 2), _f(-b0["CI下界"], 2), _f(b0["CI上界"], 2),
                  _f(b20["值"], 2), _f(-b20["CI下界"], 2), _f(b20["CI上界"], 2)],
        "bench_k": [str(int(b0["简单族更好的基准数"])), str(int(b20["简单族更好的基准数"]))],
        "s": [pct(simple["遗憾中位"]), pct(simple["CI下界"]), pct(simple["CI上界"])],
        "s_n": [pct(simple[f"n={nc}"]) for nc in (0, 5, 10, 20, 40)],
        "s_in": str(round(100 * simple["10%以内占比"])),
        "mu": [pct(mu["遗憾中位"]), pct(mu["CI下界"]), pct(mu["CI上界"])],
        "map": pct(r(S, "B_shift_unknown")["遗憾中位"]),
        "deep": [pct(deep["遗憾中位"]), pct(deep["CI下界"]), pct(deep["CI上界"])],
        "deep_in": str(round(100 * deep["10%以内占比"])), "to": pct(to["遗憾中位"]),
        "at40": [pct(mu["n=40"]), pct(to["n=40"]), pct(simple["n=40"])],
        "canon_s": pct(r(C, "always_simple")["遗憾中位"]), "canon_d": pct(r(C, "default_deep")["遗憾中位"]),
        "plat": [f"{plat['|平台差|中位']:.3f}", f"{plat['|口径差|中位']:.3f}"],
        "jobs": [str(n_jobs - n_gpu), str(n_jobs), str(n_gpu)],
        "chk": [*_sci(selfchk["最大绝对差"]), str(int(selfchk["行数"])),
                _WORD[int(selfchk["任务数"])]],
    }
    n = r"(\d+(?:\.\d+)?)"
    pats: dict[str, list[tuple[str, str, list[str]]]] = {
        "Elsevier_en.tex": [
            ("§2.3 任务数", r"refitted without the target domain \(the stricter of the case study's two refits\) "
             r"for all (\d+) apple tasks at five seeds \(script 108",
             [w["n_task"]]),
            ("§3.1 包络", r"they stay separated only at (\d+) and (\d+) labels \(\$\\delta=\+" + n + r"\$ and \$\+" + n
             + r"\$, \$P\\le4\\times10\^\{-4\}\$\)", [*w["sep"], *w["d_sep"]]),
            ("§3.1 免选择", r"the family mean stays at \$\+" + n + r"\$ to \$\+" + n + r"\$ over the budget grid "
             r"\(\$P<10\^\{-5\}\$\)", w["fam_rng"]),
            ("§3.1 基准级", r"\$\+" + n + r"\$ \(\[\$-" + n + r"\$, " + n
             + r"\]\) with the apple standardisation refitted without the target domain, and", w["bench"][:3]),
            ("§3.1 NRMSEP 重拟合", r"for origin/year \(\$\+" + n + r"\$ to \$\+" + n + r"\$ with the apple "
             r"standardisation refitted\)", w["fam_nr"]),
            ("§3.2 遗憾", r"the prescription costs " + n + r"\\% \(" + n + r"--" + n + r"\\%\) and "
             r"remains the cheapest, except at 40 labels, where model updating \(" + n + r"\\%\) and target-only "
             r"calibration \(" + n + r"\\%\) cost less than it \(" + n + r"\\%;", [*w["s"], *w["at40"]]),
            ("结论遗憾", r"regret, " + n + r"\\% \(" + n + r"\\% with the apple standardisation refitted",
             [w["canon_s"], w["s"][0]]),
            ("摘要", r"regret \(" + n + r"\\%; " + n + r"\\% with apple standardisation refitted without the target "
             r"domain; default deep model "
             + n + r"\\%\)", [w["canon_s"], w["s"][0], w["canon_d"]]),
        ],
        "Elsevier_zh.tex": [
            ("§2.3 任务数", r"在全部 (\d+) 个苹果任务、5 个种子上把(?:它|标准化)改为不含目标域重新拟合", [w["n_task"]]),
            ("§3.1 包络", r"只在 (\d+) 与 (\d+) 标签处仍被分开（\$\\delta=\+" + n + r"\$ 与 \$\+" + n
             + r"\$，\$P\\le4\\times10\^\{-4\}\$）", [*w["sep"], *w["d_sep"]]),
            ("§3.1 免选择", r"族均值在整个预算网格上仍为 \$\+" + n + r"\\sim\+" + n + r"\$（\$P<10\^\{-5\}\$）",
             w["fam_rng"]),
            ("§3.1 基准级", r"苹果标准化不含目标域重新拟合后为 \$\+" + n + r"\$（\[\$-" + n + r"\$, " + n
             + r"\]）", w["bench"][:3]),
            ("§3.1 NRMSEP 重拟合", r"（苹果标准化重新拟合后为 \$\+" + n + r"\\sim\+" + n + r"\$）", w["fam_nr"]),
            ("§3.2 遗憾", r"该处方为 " + n + r"\\%（" + n + r"\\%\$\\sim\$" + n
             + r"\\%），仍是最便宜的一条，只有 40 "
             r"个标签处例外：模型更新（" + n + r"\\%）与目标自建（" + n + r"\\%）比它（" + n + r"\\%）更便宜",
             [*w["s"], *w["at40"]]),
            ("结论遗憾", r"留一基准遗憾最低，为 " + n + r"\\%（苹果标准化重新拟合后为 " + n + r"\\%）",
             [w["canon_s"], w["s"][0]]),
            ("摘要", r"留一基准遗憾最低（" + n + r"\\%；苹果标准化不含目标域重新拟合时为 " + n + r"\\%；默认深度模型 " + n
             + r"\\%）",
             [w["canon_s"], w["s"][0], w["canon_d"]]),
        ],
        "Supplementary_en.tex": [
            ("任务数", r"without the whole target domain for all (\d+) tasks and reran", [w["n_task"]]),
            ("管线自检", r"largest difference \$(\d)\\times10\^\{-(\d+)\}\$ over the (\d+) runs of (\w+) "
             r"tasks checked",
             w["chk"]),
            ("平台作业", r"CPU workstation \((\d+) of the (\d+) task--seed jobs\) and a GPU server \((\d+)\)",
             w["jobs"]),
            ("平台差", r"changes a deep cell by a median " + n + r" in absolute value, against " + n
             + r" for the refit",
             w["plat"]),
            ("包络", r"separates the two families only at (\d+) and (\d+) labels \(\$\\delta=\+" + n
             + r"\$ and \$\+" + n
             + r"\$, \$P\\le4\\times10\^\{-4\}\$\), whereas at 0, 5 and 40 labels \$\\delta\$ is \$\+" + n
             + r"\$, \$\+" + n + r"\$ and \$\+" + n + r"\$ \(\$P\\ge" + n + r"\$\)",
             [*w["sep"], *w["d_sep"], *w["d_rest"], w["p_rest"]]),
            ("免选择", r"the representative pair at \$\+" + n + r"\$ \(simple better on (\d+)\\% of tasks, "
             r"\$P=(\d)\\times10\^\{-(\d+)\}\$\) and the family mean at \$\+" + n + r"\$, \$\+" + n
             + r"\$, \$\+" + n + r"\$, \$\+" + n + r"\$ and \$\+" + n + r"\$ at 0, 5, 10, 20 and 40 labels "
             r"\(all \$P<10\^\{-5\}\$; \$\+" + n + r"\$ to \$\+" + n + r"\$ on NRMSEP\)",
             [*w["rep"], *w["rep_p"], *w["fam"], *w["fam_nr"]]),
            ("基准级", r"becomes \$\+" + n + r"\$ \(95\\% CI \[\$-" + n + r"\$, " + n
             + r"\]\) at zero labels and \$\+" + n
             + r"\$ \(\[\$-" + n + r"\$, " + n + r"\]\) at 20, (\w+) and (\w+) of the five benchmarks",
             [*w["bench"], *(_WORD[int(k)] for k in w["bench_k"])]),
            ("遗憾", r"correction costs " + n + r"\\% \(" + n + r"--" + n + r"\\%; " + n + r", " + n + r", "
             + n + r", " + n + r" and " + n + r"\\% at 0, 5, 10, 20 and 40 labels; (\d+)\\% of cells within "
             r"10\\% of the oracle\), always "
             r"updating the model " + n + r"\\% \(" + n + r"--" + n + r"\\%\), the frequency map with the shift type "
             r"unknown " + n + r"\\%, the default deep model " + n + r"\\% \(" + n + r"--" + n + r"\\%\) and always "
             r"calibrating on the target alone " + n + r"\\%",
             [*w["s"], *w["s_n"], w["s_in"], *w["mu"], w["map"], *w["deep"], w["to"]]),
            ("40 标签", r"at 40 labels model updating \(" + n + r"\\%\) and target-only calibration \(" + n
             + r"\\%\) are then cheaper", w["at40"][:2]),
        ],
        "Supplementary_zh.tex": [
            ("任务数", r"在全部 (\d+) 个任务上把季节统计量改为扣除整个目标域后重新拟合", [w["n_task"]]),
            ("管线自检", r"抽查两个任务的 (\d+) 次运行，最大差 \$(\d)\\times10\^\{-(\d+)\}\$",
             [w["chk"][2], *w["chk"][:2]]),
            ("平台作业", r"CPU 工作站（(\d+) 个任务–种子作业中的 (\d+) 个）与一台 GPU 服务器（(\d+) 个）",
             [w["jobs"][1], w["jobs"][0], w["jobs"][2]]),
            ("平台差", r"变化（绝对值）中位 " + n + r"，同一批单元上口径变化为 " + n, w["plat"]),
            ("包络", r"只在 (\d+) 与 (\d+) 标签处把两族分开（\$\\delta=\+" + n + r"\$ 与 \$\+" + n
             + r"\$，\$P\\le4\\times10\^\{-4\}\$），0、5、40 标签处 \$\\delta\$ 为 \$\+" + n + r"\$、\$\+" + n
             + r"\$ 与 \$\+" + n + r"\$（\$P\\ge" + n + r"\$）", [*w["sep"], *w["d_sep"], *w["d_rest"], w["p_rest"]]),
            ("免选择", r"代表性配对为 \$\+" + n
             + r"\$（简单族在 (\d+)\\% 的任务上更优，\$P=(\d)\\times10\^\{-(\d+)\}\$），"
             r"族均值在 0、5、10、20、40 标签处为 \$\+" + n + r"\$、\$\+" + n + r"\$、\$\+" + n + r"\$、\$\+" + n
             + r"\$ 与 \$\+" + n + r"\$（均 \$P<10\^\{-5\}\$；在 NRMSEP 上为 \$\+" + n + r"\\sim\+" + n + r"\$）",
             [*w["rep"], *w["rep_p"], *w["fam"], *w["fam_nr"]]),
            ("基准级", r"在零标签处为 \$\+" + n + r"\$（95\\% CI \[\$-" + n + r"\$, " + n
             + r"\]），在 20 标签处为 \$\+" + n
             + r"\$（\[\$-" + n + r"\$, " + n + r"\]），5 个基准中仍分别有 (\d+) 个与 (\d+) 个偏向简单族",
             [*w["bench"], *w["bench_k"]]),
            ("遗憾", r"最简单可用校正为 " + n + r"\\%（" + n + r"\\%\$\\sim\$" + n
             + r"\\%；0、5、10、20、40 标签处分别为 " + n
             + r"\\%、" + n + r"\\%、" + n + r"\\%、" + n + r"\\%、" + n
             + r"\\%；(\d+)\\% 的单元落在预言机 10\\% 以内），"
             r"始终模型更新为 " + n + r"\\%（" + n + r"\\%\$\\sim\$" + n + r"\\%），漂移类型未知时把频率图当规则为 " + n
             + r"\\%，默认深度模型为 " + n + r"\\%（" + n + r"\\%\$\\sim\$" + n + r"\\%），始终目标自建为 " + n
             + r"\\%",
             [*w["s"], *w["s_n"], w["s_in"], *w["mu"], w["map"], *w["deep"], w["to"]]),
            ("40 标签", r"模型更新（" + n + r"\\%）与目标自建（" + n
             + r"\\%）此时比斜率/偏置校正更便宜", w["at40"][:2]),
        ],
    }
    check("108 结论「重拟合后基准级区间触及零」：零标签 CI 下界 ≤ 0", "True", str(float(b0["CI下界"]) <= 0),
          src + "[表j]")
    for fname, phrase in (("Elsevier_en.tex", "its interval touching zero once the apple standardisation is refitted"),
                          ("Elsevier_zh.tex", "苹果标准化不含目标域重新拟合后其区间触及零")):
        check(f"108（{fname}）结论定性句在", "True", str(phrase in _flat(fname)), fname)
    for fname, items in pats.items():
        text = _flat(fname)
        for tag, pat, exp in items:
            m = re.search(pat, text)
            if not m:
                check(f"108（{fname}）{tag} 可解析", "True", "False", fname)
                continue
            check(f"108（{fname}）{tag}", " / ".join(m.groups()), " / ".join(exp), src)
    # 旧说法：简单族汇总优势在零标签处成立、测试样本只进入 CORAL 的拟合
    for fname, stale in (("Elsevier_en.tex", "is therefore established at zero labels"),
                         ("Elsevier_en.tex", "the single exception being CORAL's transductive covariance estimate"),
                         ("Elsevier_zh.tex", "简单族的汇总优势在零标签处成立"),
                         ("Elsevier_zh.tex", "唯一例外是 CORAL 的直推式协方差估计")):
        check(f"{fname} 不再含「{stale[:20]}」", "False", str(stale in _flat(fname)), fname)


def check_deep_platform() -> None:
    """补充材料 S2：108 号深度重跑的 GPU 服务器作业数与 64 号深度主运行的平台，从分片元数据与调度脚本核。"""
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in (BASE / "04outputs/108_shards").glob("*.meta.json")
             if not p.name.startswith(".")]
    n_cuda = sum(1 for j in metas if j.get("device") == "cuda")
    for lang, name, pat in (("英", "Supplementary_en.tex", r"and a GPU server \((\d+)\), whereas the main deep runs were "
                                                             r"made on the CPUs of another server"),
                            ("中", "Supplementary_zh.tex", r"与一台 GPU 服务器（(\d+) 个）上完成，而主实验的深度运行在另一台服务器的 CPU 上")):
        m = _rx(f"S2 {lang} 深度平台句", pat, _flat(name), name)
        if m:
            check(f"S2 {lang} GPU 服务器作业数 = 108 号分片元数据 device=cuda 片数", m.group(1), str(n_cuda),
                  "04outputs/108_shards/*.meta.json")
    sh = (BASE / "02code/95_server_run_all.sh").read_text(encoding="utf-8")
    m64 = re.search(r"02code/64_deep_transfer_server\.py \\\s*\n\s*--benchmarks \"\$b\" --seeds \"\$s\" --rep 1 "
                    r"--device (\w+)", sh)
    check("S2 64 号深度主运行（95 号调度）的设备为 CPU", "cpu", m64.group(1) if m64 else "未找到",
          "02code/95_server_run_all.sh（64 号正典 25 分片由它跑出，14a2032 合并）")


def check_verdict_items() -> None:
    """第三轮独立验收处置后的三类事实：S2 写的 106 号方法与 106 号代码一致；正文文献按首次引用编号；
    摘要写的是「标准化不含目标域重新拟合」，不再是 target-blind 的口径（离群剔除仍看目标谱）。"""
    src106 = (BASE / "02code").glob("106_*.py")
    code = next(p for p in src106 if not p.name.startswith("._")).read_text(encoding="utf-8")
    mm = re.search(r"DEFAULT_METHODS = \[([^\]]*)\]", code)
    meths = re.findall(r"'([^']+)'", mm.group(1)) if mm else []
    for lang, name, pat, names in (
            ("英", "Supplementary_en.tex", r"reran the four methods it compares \(PLSR, SVR, the CNN and Phys\+BL, each "
                                           r"with and without slope/bias correction\)", ["PLSR", "SVR", "CNN", "Phys+BL"]),
            ("中", "Supplementary_zh.tex", r"重跑它所比较的四种方法（PLSR、SVR、CNN 与 Phys\+BL，各含与不含斜率/偏置校正）",
             ["PLSR", "SVR", "CNN", "Phys+BL"])):
        ok = re.search(pat, _flat(name)) is not None
        check(f"S2 {lang} 106 号重跑方法句在", "True", str(ok), name)
        check(f"S2 {lang} 106 号方法 = 106 号 DEFAULT_METHODS", str(names), str(meths), "02code/106_*.py DEFAULT_METHODS")
    for name in ("Elsevier_en.tex", "Elsevier_zh.tex"):
        t = _tex(name)
        i0 = t.index("\\begin{thebibliography}")
        body = re.sub(r"(?<!\\)%.*", "", t[:i0])
        order: list[str] = []
        for m in re.finditer(r"\\cite[pt]?\{([^}]*)\}", body):
            for k in m.group(1).split(","):
                if k.strip() and k.strip() not in order:
                    order.append(k.strip())
        items = re.findall(r"\\bibitem\{([^}]+)\}", t[i0:])
        check(f"{name} 文献按首次引用编号", "True", str(order == items), f"{name}（\\cite 首现顺序对 bibitem 顺序）")
    for name, bad in (("Elsevier_en.tex", "target-blind"), ("Elsevier_zh.tex", "不看目标谱")):
        t = _tex(name)
        ab = t[t.index("\\begin{abstract}"):t.index("\\end{abstract}")]
        check(f"{name} 摘要不用「{bad}」口径", "False", str(bad in ab), name)


def check_verdict4_items() -> None:
    """第四轮独立验收处置后的两类事实：表 1 与补充材料数据卡的波段数按 61 号载入的数据数；
    §2.3 与表 3 表注写明 δ 与 P 的单位（逐任务 RMSEP，合并了不同量纲分析物的基准另在 NRMSEP 上给）。"""
    spec = importlib.util.spec_from_file_location("bench61", BASE / "02code" / "61_benchmark_datasets.py")
    b61 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b61)
    src = "02code/61_benchmark_datasets.py（LOADERS 载入后各域 wl 长度）"
    nb = {}
    for key, name in (("corn", "Corn"), ("tablet", "Tablet"), ("mango", "Mango"), ("ossl_mir", "Soil (OSSL)")):
        lens = {len(d["wl"]) for d in b61.LOADERS[key]()["domains"].values()}
        nb[name] = "/".join(str(x) for x in sorted(lens))
    ap = b61.LOADERS["apple"]()["domains"]
    inst = {k.rsplit("_", 1)[1]: len(d["wl"]) for k, d in ap.items()}
    nb["Apple"] = f"{inst['A']}/{inst['B']}$^{{\\mathrm{{a}}}}$"
    for r in _table_rows(_tex("Elsevier_en.tex"), "tab:bench"):
        name = r[0].strip()
        if name in nb:
            check(f"表 1 {name} 波段数", r[5].strip(), nb[name], src)
    ok = re.search(rf"\\mathrm{{a}}\}}\$(\d+)\s*within instrument A; pairs involving instrument B use its (\d+) channels",
                   _flat("Elsevier_en.tex"))
    check("表 1 表注 仪器 A/B 通道数", f"{inst['A']}/{inst['B']}", f"{ok.group(1)}/{ok.group(2)}" if ok else "未解析", src)
    si = _flat("Supplementary_en.tex")
    for tag, pat, want in (("数据卡 芒果", r"vis--NIR 285--1200\\,nm \((\d+)\)", nb["Mango"]),
                           ("数据卡 苹果 2018", r"Apple 2018, instrument A.*?nm \((\d+)\)", str(inst["A"])),
                           ("数据卡 苹果 2019", r"Apple 2019, instrument A.*?nm \((\d+)\)", str(inst["A"]))):
        m = re.search(pat, si)
        check(f"SI {tag} 波段数", m.group(1) if m else "未解析", want, src)
    for name, pats in (("Elsevier_en.tex", (r"Cliff's \$\\delta\$ on per-task RMSEP",
                                            r"\$\\delta\$ and \$P\$ are on RMSEP")),
                       ("Elsevier_zh.tex", (r"Cliff's \$\\delta\$ 在逐任务 RMSEP 上计算",
                                            r"\$\\delta\$ 与 \$P\$ 在 RMSEP 上计算"))):
        tt = _flat(name)
        for pat in pats:
            check(f"{name} 检验单位说明在（{pat[:24]}…）", "True", str(re.search(pat, tt) is not None), name)


def _physbl_vs_simple(c62: pd.DataFrame) -> pd.DataFrame:
    """逐（漂移类型，预算）Phys+BL 胜过逐任务最优简单校正的任务占比（%）；n=0 用零样本，其上用微调。"""
    c64 = pd.read_excel(XL64, sheet_name="curves")
    med = pd.concat([c.groupby(["task_id", "shift_type", "method", "n_cal"], as_index=False)["rmsep"].median()
                     for c in (c62, c64)])
    rows = []
    for n in NCAL_GRID:
        at = med[med["n_cal"] == n]
        s = at[at["method"].isin(SIMPLE_FAM)].groupby(["task_id", "shift_type"])["rmsep"].min()
        for meth in ("physbl_zeroshot" if n == 0 else "physbl_ft", "cnn_zeroshot" if n == 0 else "cnn_finetune"):
            p = at[at["method"] == meth].set_index(["task_id", "shift_type"])["rmsep"]
            idx = s.index.intersection(p.index)
            d = pd.DataFrame({"s": s.loc[idx], "p": p.loc[idx]}).reset_index()
            for st, g in d.groupby("shift_type"):
                rows.append({"n": n, "shift": st, "method": meth, "tasks": len(g),
                             "win": 100 * float((g["p"] < g["s"]).mean())})
    return pd.DataFrame(rows)


def check_forensics_7550() -> None:
    """7550cf2 取证扫描后改稿的几处事实。"""
    c62 = pd.read_excel(XL62, sheet_name="curves")
    src = "04outputs/62_crossover_engine.xlsx + 64_deep_transfer_server.xlsx（逐任务中位 RMSEP）"
    full = _physbl_vs_simple(c62)
    pb = full[full["method"].isin(["physbl_zeroshot", "physbl_ft"])]
    en, zh = _flat("Elsevier_en.tex"), _flat("Elsevier_zh.tex")
    m = re.search(r"Phys\+BL beat the simple corrections consistently only on soil, on all (\d+) tasks at every\s*"
                  r"positive budget as the fine-tuned CNN did, and on at most ([\d.]+)\\% of the tasks at any budget", en)
    check("§3.1 Phys+BL 句可解析", "True", str(m is not None), "Elsevier_en.tex")
    if m:
        soil = full[(full["shift"] == "lab") & (full["n"] > 0)]
        check("§3.1 土壤任务数", m.group(1), str(int(soil["tasks"].max())), src)
        check("§3.1 土壤每个正预算 Phys+BL 与微调 CNN 全胜", "True",
              str(bool((soil["win"] == 100).all() and (soil["tasks"] == int(m.group(1))).all())), src)
        other = pb[pb["shift"] != "lab"]
        check("§3.1 其余漂移任一预算 Phys+BL 胜率上界（%）", m.group(2), f"{other['win'].max():.1f}", src)
    top = pb[pb["shift"] != "lab"]["win"].max()
    check("§3.1 中文 Phys+BL 句在（上界与数据一致）", "True",
          str("Phys+BL 只在土壤上稳定胜出简单校正" in zh and f"至多在 {top:.1f}\\% 的任务上更优" in zh), "Elsevier_zh.tex")
    five = _physbl_vs_simple(c62[c62["rep"] == 0])
    j = pb.merge(five[five["method"].isin(["physbl_zeroshot", "physbl_ft"])], on=["n", "shift", "method"],
                 suffixes=("", "_5"))
    si = _flat("Supplementary_en.tex")
    m = re.search(r"moves the share of tasks on which Phys\+BL beats the simple corrections by at most\s*"
                  r"(\d+) points and leaves every shift type and budget on the same side of 50\\%", si)
    check("S16.2 五次运行口径句可解析", "True", str(m is not None), "Supplementary_en.tex")
    if m:
        check("S16.2 五次运行口径 胜率最大变动（百分点）", m.group(1), str(round((j["win"] - j["win_5"]).abs().max())),
              src + "；经典方法只取 rep=0")
        same = bool((((j["win"] - 50) * (j["win_5"] - 50)) > 0).all())
        check("S16.2 五次运行口径 每类漂移每个预算在 50% 同侧", "True", str(same), src)
    x110 = OUT / "110_coral_matched_access.xlsx"
    s110 = "04outputs/110_coral_matched_access.xlsx[pairs]"
    rp = pd.read_excel(x110, sheet_name="repro").iloc[0]
    check("110 号复现 62 号零标签运行（未对齐行数）", "0", str(int(rp["rows_unmatched"])), "04outputs/110_coral_matched_access.xlsx[repro]")
    check("110 号复现 62 号零标签运行（最大绝对差 ≤1e-9）", "True", str(bool(rp["max_abs_diff"] <= 1e-9)),
          "04outputs/110_coral_matched_access.xlsx[repro]")
    pr = pd.read_excel(x110, sheet_name="pairs")
    m = re.search(r"individually, on (\d+)--(\d+)\\% of tasks under instrument shift\s*"
                  r"\(paired \$\\delta\$ \$\+([\d.]+)\$ to \$\+([\d.]+)\$, all \$P<10\^\{-6\}\$\) and on (\d+)--(\d+)\\% of the 126\s*"
                  r"tasks pooled \(all \$P<10\^\{-6\}\$\)", si)
    check("S16.2 CORAL 对四个深度方法句可解析", "True", str(m is not None), "Supplementary_en.tex")
    if m:
        ins = pr[(pr["scope"] == "instrument") & (pr["coral_variant"] == "coral")]
        allp = pr[(pr["scope"] == "all") & (pr["coral_variant"] == "coral")]
        check("S16.2 仪器 CORAL 胜率下界", m.group(1), str(round(ins["coral_win_pct"].min())), s110)
        check("S16.2 仪器 CORAL 胜率上界", m.group(2), str(round(ins["coral_win_pct"].max())), s110)
        _numchk("S16.2 仪器 配对 δ 下界", m.group(3), ins["paired_delta"].min(), s110)
        _numchk("S16.2 仪器 配对 δ 上界", m.group(4), ins["paired_delta"].max(), s110)
        check("S16.2 仪器 P 全部 <1e-6", "True", str(bool((ins["wilcoxon_p"] < 1e-6).all())), s110)
        check("S16.2 合并 CORAL 胜率下界", m.group(5), str(round(allp["coral_win_pct"].min())), s110)
        check("S16.2 合并 CORAL 胜率上界", m.group(6), str(round(allp["coral_win_pct"].max())), s110)
        check("S16.2 合并 P 全部 <1e-6", "True", str(bool((allp["wilcoxon_p"] < 1e-6).all())), s110)
        check("S16.2 合并任务数", "126", str(int(allp["n_tasks"].max())), s110)
    m = re.search(r"CORAL's task-level RMSEP rises by\s*a median of ([\d.]+)\\% and it still beats each deep method on "
                  r"(\d+)--(\d+)\\% of the instrument tasks \(paired\s*\$\\delta\$ \$\+([\d.]+)\$ to \$\+([\d.]+)\$\) "
                  r"and on (\d+)--(\d+)\\% of the 126 tasks \(all \$P<3\\times10\^\{-4\}\$\)", si)
    check("S16.2 CORAL 只用校准池句可解析", "True", str(m is not None), "Supplementary_en.tex")
    if m:
        acc = pd.read_excel(x110, sheet_name="coral_access")
        _numchk("S16.2 coral_pool 逐任务 RMSEP 中位上升（%）", m.group(1), acc["rel_change_pct"].median(),
                "04outputs/110_coral_matched_access.xlsx[coral_access]")
        ins = pr[(pr["scope"] == "instrument") & (pr["coral_variant"] == "coral_pool")]
        allp = pr[(pr["scope"] == "all") & (pr["coral_variant"] == "coral_pool")]
        check("S16.2 coral_pool 仪器胜率下界", m.group(2), str(round(ins["coral_win_pct"].min())), s110)
        check("S16.2 coral_pool 仪器胜率上界", m.group(3), str(round(ins["coral_win_pct"].max())), s110)
        _numchk("S16.2 coral_pool 仪器 δ 下界", m.group(4), ins["paired_delta"].min(), s110)
        _numchk("S16.2 coral_pool 仪器 δ 上界", m.group(5), ins["paired_delta"].max(), s110)
        check("S16.2 coral_pool 合并胜率下界", m.group(6), str(round(allp["coral_win_pct"].min())), s110)
        check("S16.2 coral_pool 合并胜率上界", m.group(7), str(round(allp["coral_win_pct"].max())), s110)
        check("S16.2 coral_pool 全部 P<3e-4", "True", str(bool((pd.concat([ins, allp])["wilcoxon_p"] < 3e-4).all())), s110)
        check("S16.2 coral_pool 仍胜过每个深度方法（胜率 >50%）", "True",
              str(bool((pd.concat([ins, allp])["coral_win_pct"] > 50).all())), s110)
    for name, bad in (("Elsevier_en.tex", "structure partly does"), ("Elsevier_en.tex", "explains where simple corrections win"),
                      ("Elsevier_zh.tex", "其结构则部分可以"), ("Elsevier_zh.tex", "能否解释简单校正在何处胜出")):
        check(f"{name} 不用「{bad}」", "False", str(bad in _flat(name)), name)
    for name, key in (("Elsevier_en.tex", r"standard-free\s*variants\\cite\{LiL2022SF,Fonseca2022,Lavoie2023\}"),
                      ("Elsevier_zh.tex", r"免标样变体\\cite\{LiL2022SF,Fonseca2022,Lavoie2023\}")):
        check(f"{name} Fonseca2022/Lavoie2023 挂在无标样变体上", "True", str(re.search(key, _flat(name)) is not None), name)
    for name, key in (("Supplementary_en.tex", "The season-wise outlier removal and standardisation precede this split"),
                      ("Supplementary_zh.tex", "逐季离群剔除与标准化在这一划分之前")):
        check(f"{name} S10 写明划分前的逐季预处理", "True", str(key in _flat(name)), name)


def check_split_disclosure() -> None:
    """§2.3：两族各自把目标域对半划分；芒果各季超过 800 条光谱，经典方法（62 号）每个域只用 800 条，深度方法（64 号）用全域。
    判据取自代码与 61 号载入的数据，不取自稿件。"""
    s62 = (BASE / "02code" / "62_crossover_engine.py").read_text(encoding="utf-8")
    s64 = (BASE / "02code" / "64_deep_transfer_server.py").read_text(encoding="utf-8")
    cap = int(re.search(r"^DOMAIN_CAP = (\d+)", s62, re.M).group(1))
    src = "02code/62_crossover_engine.py、64_deep_transfer_server.py 源码"
    check("62 号源域与目标域都按 DOMAIN_CAP 子采样", "True",
          str("if len(Xs_r) > DOMAIN_CAP:" in s62 and "if len(Xt_r) > DOMAIN_CAP:" in s62), src)
    check("64 号不做域子采样", "False", str("DOMAIN_CAP" in s64), src)
    halves = "test = perm[: nt // 2]"
    check("62、64 号都把目标域对半分为测试集与校准池", "True",
          str(s62.count(halves) == 1 and s64.count(halves) == 1 and "pool = perm[nt // 2:]" in s62
              and "pool = perm[nt // 2:]" in s64), src)
    spec = importlib.util.spec_from_file_location("bench61s", BASE / "02code" / "61_benchmark_datasets.py")
    b61 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b61)
    big: dict[str, list[int]] = {}
    for name in b61.LOADERS:
        b = b61.LOADERS[name]()
        sizes = set()
        for tk in b61.build_transfer_tasks(b):
            for key in ("src", "tgt"):
                d = b["domains"][tk[key]]
                sizes.add(int((np.isfinite(d["X"]).all(1) & np.isfinite(d["Y"][:, tk["prop_idx"]])).sum()))
        big[name] = sorted(sizes)
    s61 = "02code/61_benchmark_datasets.py（LOADERS 载入、按任务标签属性去缺失后的域大小）"
    check("只有芒果有超过 DOMAIN_CAP 的域", "['mango']", str([k for k, v in big.items() if max(v) > cap]), s61)
    check("芒果每个域都超过 DOMAIN_CAP", "True", str(min(big["mango"]) > cap), s61)
    en, zh = _flat("Elsevier_en.tex"), _flat("Elsevier_zh.tex")
    m = re.search(r"Each draw halves the target domain into a test set, fixed for every budget and method of a "
                  r"family within the draw \(the families draw separately\), .*? on mango the classical runs use "
                  r"(\d+) spectra per season, fewer than the deep models do\.", en)
    check("§2.3 划分与芒果子采样句可解析", "True", str(m is not None), "Elsevier_en.tex")
    if m:
        check("§2.3 经典方法每季用量", str(cap), m.group(1), src)
    check("Elsevier_zh.tex §2.3 划分与芒果子采样句在", "True",
          str("（对该次划分下同一族的全部预算与方法固定；两族各自划分）" in zh
              and f"芒果上经典方法的运行每季只用 {cap} 条光谱，少于深度模型" in zh), "Elsevier_zh.tex")
    for name, key in (("Supplementary_en.tex", f"taken from the {cap}-spectrum subsample on mango; script 110"),
                      ("Supplementary_zh.tex", f"芒果取自 {cap} 条子样本；脚本 110")):
        check(f"{name} S16.2 校准池的来源", "True", str(key in _flat(name)), name)
    for name, bad in (("Elsevier_en.tex", "The full source domain trains the source model"),
                      ("Elsevier_zh.tex", "源域全量训练源模型"),
                      ("Elsevier_en.tex", "-1$ on the same test samples"),
                      ("Elsevier_zh.tex", "-1$，同一批测试样本"),
                      ("Elsevier_en.tex", "the explanatory claim"),
                      ("Elsevier_zh.tex", "解释性主张"),
                      ("Elsevier_en.tex", "indicates why the simplest corrections"),
                      ("Supplementary_en.tex", "on the same zero-label runs"),
                      ("Supplementary_zh.tex", "在同一批零标签运行上"),
                      ("Supplementary_en.tex", "the spectra DANN and DeepCORAL use"),
                      ("Supplementary_zh.tex", "即 DANN 与 DeepCORAL 所用的那批"),
                      ("Elsevier_en.tex", "whose budget cells share one test set"),
                      ("Elsevier_zh.tex", "各预算单元共用一个测试集"),
                      ("Elsevier_en.tex", "simple corrections win where the shift is low-order"),
                      ("Elsevier_zh.tex", "简单校正在漂移为低阶时胜出")):
        check(f"{name} 不用「{bad}」", "False", str(bad in _flat(name)), name)
    title = re.search(r"\\subsection\{(Low-order shift structure as [^}]*)\}", en).group(1)
    r1 = _letter("Response_R1.tex")
    check("R1 引用的 4.1 标题与稿件一致", "True", str(f"4.1 {title};" in r1), "Elsevier_en.tex §4.1 标题")
    check("R1 (ii) 不写 demonstrated", "False", str("mechanistic explanation is now demonstrated" in r1), "Response_R1.tex")
    check("cover letter 不写 mechanistic explanation of why", "False",
          str("mechanistic explanation of why simple corrections win" in _letter("CoverLetter_R1.tex")), "CoverLetter_R1.tex")


def check_forensics_281614f() -> None:
    """281614f 确认扫描后改稿的几处事实。"""
    pp = pd.read_excel(OUT / "74_shift_structure_diagnosis.xlsx", sheet_name="per_pair")
    s74 = "04outputs/74_shift_structure_diagnosis.xlsx[per_pair]"
    en, zh = _flat("Elsevier_en.tex"), _flat("Elsevier_zh.tex")
    check("74 号域对数", "98", str(len(pp)), s74)
    pos = bool((pp["mmd2_raw"] > 0).all() and (pp["sw1_raw"] > 0).all() and (pp["dR_src"] > 0).all()
               and np.isfinite(pp["A_lowrank_oos"]).all())
    check("§2.4 两个诊断量的分母在全部域对上为正（含留出版本 A 有定义）", "True", str(pos), s74)
    check("§2.4 分母为正句在（中英）", "True",
          str("Both denominators are positive on all 98 pairs." in en and "两个分母在全部 98 个域对上都为正。" in zh), "Elsevier_en/zh.tex")
    si = _flat("Supplementary_en.tex")
    m = re.search(r"The raw distance is positive on every pair \(smallest MMD\$\^\{2\}\$ ([\d.]+), sliced Wasserstein "
                  r"([\d.]+)\).*?the smallest source median residual is \$([\d.]+)\\times10\^\{-(\d+)\}\$, so \$A\$ is "
                  r"defined on all 98 pairs \(([\d.]+)--([\d.]+)\)", si)
    check("S16.1 最小距离句可解析", "True", str(m is not None), "Supplementary_en.tex")
    if m:
        _numchk("S16.1 最小 MMD²", m.group(1), pp["mmd2_raw"].min(), s74)
        _numchk("S16.1 最小切片 Wasserstein", m.group(2), pp["sw1_raw"].min(), s74)
        _numchk("S16.1 最小源残差中位数（×10^-k）", m.group(3), pp["dR_src"].min() * 10 ** int(m.group(4)), s74)
        _numchk("S16.1 A 下界", m.group(5), pp["A_lowrank"].min(), s74)
        _numchk("S16.1 A 上界", m.group(6), pp["A_lowrank"].max(), s74)
    pr = pd.read_excel(OUT / "110_coral_matched_access.xlsx", sheet_name="pairs")
    s110 = "04outputs/110_coral_matched_access.xlsx[pairs]"
    allp = pr[(pr["scope"] == "all") & (pr["coral_variant"] == "coral_pool")]
    m = re.search(r"individually on 72--78\\% of the 126 tasks \(all \$P<10\^\{-6\}\$\), and on (\d+)--(\d+)\\% with CORAL "
                  r"restricted to a calibration pool \(all \$P<3\\times10\^\{-4\}\$; Supplementary Section~S16\)", en)
    check("§3.1 CORAL 只用校准池句可解析", "True", str(m is not None), "Elsevier_en.tex")
    if m:
        check("§3.1 只用校准池 合并胜率下界", m.group(1), str(round(allp["coral_win_pct"].min())), s110)
        check("§3.1 只用校准池 合并胜率上界", m.group(2), str(round(allp["coral_win_pct"].max())), s110)
        check("§3.1 只用校准池 合并 P 全部 <3e-4", "True", str(bool((allp["wilcoxon_p"] < 3e-4).all())), s110)
    lo, hi = round(allp["coral_win_pct"].min()), round(allp["coral_win_pct"].max())
    check("Elsevier_zh.tex §3.1 只用校准池句（数与 110 号一致）", "True",
          str(f"CORAL 也只用一个标定池时为 {lo}$\\sim${hi}\\%（均 $P<3\\times10^{{-4}}$" in zh), "Elsevier_zh.tex")
    for name, key in (("Supplementary_en.tex", "so this step, like CORAL, draws on unlabelled target spectra but never on target labels"),
                      ("Supplementary_zh.tex", "故这一步与 CORAL 一样用到无标签目标谱，但从不用目标标签")):
        check(f"{name} S2 写明逐季预处理用无标签目标谱、不用目标标签", "True", str(key in _flat(name)), name)
    r1 = _letter("Response_R1.tex")
    check("R1 Comment 5 所说 §3.2 的「designed for exactly this setting」在稿件中", "True",
          str("adds that CORAL is designed for exactly this setting" not in r1 or "CORAL, designed for exactly this setting," in en), "Response_R1.tex / Elsevier_en.tex")
    for name, bad in (("Elsevier_en.tex", "All methods share the split and seed of each task"),
                      ("Elsevier_zh.tex", "各方法在每个任务共享划分与种子")):
        check(f"{name} 不用「{bad}」", "False", str(bad in _flat(name)), name)


def check_correlations() -> None:
    """§3.4 引用的 Spearman 相关逐个对 74 号 correlations 表。"""
    full = _ci74_full()
    c = pd.DataFrame({k: dict(zip(("spearman_rho", "ci95_lo", "ci95_hi"), v)) for k, v in full.items()}).T
    src = "04outputs/74_shift_structure_diagnosis.xlsx[per_task] 重放 74 号 bootstrap 全精度"
    t = _flat("Elsevier_en.tex")
    want = [
        ("原始 MMD² 对 CORAL 增益", r"raw MMD\$\^\{2\}\$ is unrelated to the CORAL gain "
         r"\(Spearman \$\\rho=([\d.]+)\$, 95\\% CI "
         r"\$\[-([\d.]+), ([\d.]+)\]\$", ("mmd2_raw", "coral_gain")),
        ("原始 MMD² 对 20 标签差值", r"margin at 20 labels \(\$\\rho=([\d.]+)\$ \$\[-([\d.]+), ([\d.]+)\]\$\)",
         ("mmd2_raw", "margin20_deep_minus_simple")),
    ]
    for tag, pat, key in want:
        m = re.search(pat, t)
        if not m:
            raise SystemExit(f"§3.4 相关句没解析出来：{tag}")
        row = c.loc[key]
        check(f"§3.4 {tag} ρ", float(m.group(1)), round(float(row["spearman_rho"]), 2), src)
        check(f"§3.4 {tag} CI 下", -float(m.group(2)), round(float(row["ci95_lo"]), 2), src)
        check(f"§3.4 {tag} CI 上", float(m.group(3)), round(float(row["ci95_hi"]), 2), src)

    m = re.search(r"correlates with the CORAL gain \(\$\\rho=([\d.]+)\$ \$\[([\d.]+), "
                  r"([\d.]+)\]\$\), negatively with the error left after alignment "
                  r"\(\$\\rho=-([\d.]+)\$ \$\[-([\d.]+), -([\d.]+)\]\$\) and positively with "
                  r"the simple family's margin "
                  r"\(\$\\rho=([\d.]+)\$ \$\[([\d.]+), ([\d.]+)\]\$ at zero labels, "
                  r"\$([\d.]+)\$ \$\[([\d.]+), ([\d.]+)\]\$ at 20\)", t)
    if not m:
        raise SystemExit("§3.4 低阶份额三条相关没解析出来")
    trip = [("coral_gain", 0), ("nrmsep_coral0", 3), ("margin0_deep_minus_simple", 6),
            ("margin20_deep_minus_simple", 9)]
    for tgt, i in trip:
        row = c.loc[("f_coral_mmd2", tgt)]
        sign = -1.0 if tgt == "nrmsep_coral0" else 1.0
        check(f"§3.4 f_CORAL 对 {tgt} ρ", sign * float(m.group(i + 1)),
              round(float(row["spearman_rho"]), 2), src)
        lo, hi = float(m.group(i + 2)), float(m.group(i + 3))
        if tgt == "nrmsep_coral0":
            lo, hi = -lo, -hi
        check(f"§3.4 f_CORAL 对 {tgt} CI 下", lo, round(float(row["ci95_lo"]), 2), src)
        check(f"§3.4 f_CORAL 对 {tgt} CI 上", hi, round(float(row["ci95_hi"]), 2), src)

    m2 = re.search(r"the sliced-Wasserstein version gives "
                   r"\$\\rho=([\d.]+)\$, \$-([\d.]+)\$, \$([\d.]+)\$ and \$([\d.]+)\$", t)
    if not m2:
        raise SystemExit("§3.4 切片 Wasserstein 版本那句没解析出来")
    for claim, tgt, sign in zip(m2.groups(),
                                ("coral_gain", "nrmsep_coral0", "margin0_deep_minus_simple",
                                 "margin20_deep_minus_simple"), (1, -1, 1, 1), strict=True):
        check(f"§3.4 sw1 版 f_CORAL 对 {tgt} ρ", sign * float(claim),
              round(_rho_full("f_coral_sw1", tgt), 2), SRC74_TASK)

    m3 = re.search(r"It correlates with the zero-label error of direct "
                   r"transfer \(\$\\rho=([\d.]+)\$ \$\[([\d.]+), ([\d.]+)\]\$\) and with the "
                   r"CORAL gain \(\$\\rho=([\d.]+)\$ \$\[([\d.]+), ([\d.]+)\]\$\)", t)
    if not m3:
        raise SystemExit("§3.4 低秩指标 A 那句没解析出来")
    for i, tgt in ((0, "nrmsep_zero_shot0"), (3, "coral_gain")):
        row = c.loc[("A_lowrank", tgt)]
        check(f"§3.4 A 对 {tgt} ρ", float(m3.group(i + 1)),
              round(float(row["spearman_rho"]), 2), src)
        check(f"§3.4 A 对 {tgt} CI 下", float(m3.group(i + 2)),
              round(float(row["ci95_lo"]), 2), src)
        check(f"§3.4 A 对 {tgt} CI 上", float(m3.group(i + 3)),
              round(float(row["ci95_hi"]), 2), src)
    check("§3.4 A 对零标签误差差 0.008 未达预注册 0.5", "True",
          str(round(0.5 - float(c.loc[("A_lowrank", "nrmsep_zero_shot0"), "spearman_rho"]), 3)
              == 0.008), src)

    # §3.4 只报点估计、区间在补充材料 S16 的几条次要相关
    m4 = re.search(r"\$A\$ is unrelated to the error left after alignment and to the margin at 20 "
                   r"labels \(\$\\rho=(-?[\d.]+)\$ and \$([\d.]+)\$\), though it tracks the zero-label "
                   r"margin \(\$\\rho=([\d.]+)\$\)", t)
    if not m4:
        raise SystemExit("§3.4 A 的次要相关句没解析出来")
    for i, tgt in enumerate(("nrmsep_coral0", "margin20_deep_minus_simple", "margin0_deep_minus_simple"), 1):
        check(f"§3.4 A 对 {tgt} ρ", float(m4.group(i)), round(float(c.loc[("A_lowrank", tgt), "spearman_rho"]), 2), src)
    s16 = _flat("Supplementary_en.tex")
    rows16 = (
        (r"weakly related to the failure of direct transfer \(\$\\rho=([\d.]+)\$ \$\[(-?[\d.]+), (-?[\d.]+)\]\$\)",
         ("mmd2_raw", "nrmsep_zero_shot0")),
        (r"to the zero-label margin \(\$\\rho=([\d.]+)\$ \$\[(-?[\d.]+), (-?[\d.]+)\]\$\)\. Holding out",
         ("mmd2_raw", "margin0_deep_minus_simple")),
        (r"\$A\$ is unrelated to the error that remains after alignment \(\$\\rho=(-?[\d.]+)\$ "
         r"\$\[(-?[\d.]+), (-?[\d.]+)\]\$\)", ("A_lowrank", "nrmsep_coral0")),
        (r"to the margin at 20 labels \(\$\\rho=(-?[\d.]+)\$ \$\[(-?[\d.]+), (-?[\d.]+)\]\$\), while",
         ("A_lowrank", "margin20_deep_minus_simple")),
        (r"tracks the zero-label margin \(\$\\rho=(-?[\d.]+)\$ \$\[(-?[\d.]+), (-?[\d.]+)\]\$\)",
         ("A_lowrank", "margin0_deep_minus_simple")),
    )
    for pat, key in rows16:
        m = re.search(pat, s16)
        if not m:
            raise SystemExit(f"S16 相关句没解析出来：{key}")
        row = c.loc[key]
        for i, col in enumerate(("spearman_rho", "ci95_lo", "ci95_hi"), 1):
            check(f"S16 {key[0]} 对 {key[1]} {col}", float(m.group(i)), round(float(row[col]), 2), src)


def _letter(name: str) -> str:
    p = BASE / "06doc" / "02sub" / "revision" / name
    return re.sub(r"\s+", " ", p.read_text(encoding="utf-8"))


def _pages(pdf: Path) -> int:
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    m = re.search(r"^Pages:\s+(\d+)", out, re.M)
    if not m:
        raise SystemExit(f"读不出页数：{pdf}")
    return int(m.group(1))


XL47 = OUT / "47_public_only_replication.xlsx"
NCAL_GRID = (0, 5, 10, 20, 40)


def _grid(shift: str, kind: str) -> pd.DataFrame:
    """§3.1 与结论里的区间句取在整张 0--40 网格上，不是表 4/5 摆出的那几行。"""
    med = _task_medians()
    rows = []
    for n in NCAL_GRID:
        at = med[(med["shift_type"] == shift) & (med["n_cal"] == n)]
        agg = "min" if kind == "envelope" else "mean"
        s_ = getattr(at[at["method"].isin(SIMPLE_FAM)].groupby("task_id")[["rmsep", "nrmsep"]], agg)()
        d_ = getattr(at[at["method"].isin(DEEP_FAM)].groupby("task_id")[["rmsep", "nrmsep"]], agg)()
        idx = s_.index.intersection(d_.index)
        if len(idx) == 0:
            continue
        s_, d_ = s_.loc[idx], d_.loc[idx]
        rows.append({"n": n, "rmsep_s": float(s_["rmsep"].median()),
                     "rmsep_d": float(d_["rmsep"].median()),
                     "nrmsep_s": float(s_["nrmsep"].median()), "nrmsep_d": float(d_["nrmsep"].median()),
                     "delta": _cliff_unpaired(s_["rmsep"], d_["rmsep"]),
                     "delta_n": _cliff_unpaired(s_["nrmsep"], d_["nrmsep"]),
                     "win": float((s_["rmsep"] < d_["rmsep"]).mean() * 100),
                     "p": float(wilcoxon(s_["rmsep"].to_numpy(), d_["rmsep"].to_numpy()).pvalue),
                     "p_n": float(wilcoxon(s_["nrmsep"].to_numpy(), d_["nrmsep"].to_numpy()).pvalue)})
    return pd.DataFrame(rows)


def check_prose_ranges() -> None:
    r"""§3.1 的四句区间与结论那句效应量区间，逐个在整张预算网格上复算。

    2026-09-11 在这里查到过四处：仪器句的胜率/δ/P 三个区间只覆盖表 4 摆出的 0/20/40
    三档（全网格是 73--93\%、0.23--0.35、P≤0.003），产地/年句写 P≤0.003 而 n=5 处
    实为 0.0098（表 4 自己那一行就写着 0.010）。区间句此前没有任何活判据。
    """
    t = _flat("Elsevier_en.tex")
    src = "62/64 号 curves 在 0--40 全网格上重算"

    # 仪器漂移合并了玉米与药片两种量纲，区间只能在 NRMSEP 上给
    m = re.search(r"median NRMSEP ([\d.]+)--([\d.]+) against ([\d.]+)--([\d.]+), better on (\d+)\\%--(\d+)\\% "
                  r"of tasks, \$\\delta\$ ([\d.]+)--([\d.]+) on RMSEP and ([\d.]+)--([\d.]+) on NRMSEP,\s*"
                  r"Wilcoxon \$P\\le([\d.]+)\$", t)
    if not m:
        check("§3.1 仪器区间句可解析", "True", "False", "Elsevier_en.tex")
    else:
        g = _grid("instrument", "envelope")
        _numchk("§3.1 仪器 简单族 NRMSEP 下界", m.group(1), g["nrmsep_s"].min(), src)
        _numchk("§3.1 仪器 简单族 NRMSEP 上界", m.group(2), g["nrmsep_s"].max(), src)
        _numchk("§3.1 仪器 深度族 NRMSEP 下界", m.group(3), g["nrmsep_d"].min(), src)
        _numchk("§3.1 仪器 深度族 NRMSEP 上界", m.group(4), g["nrmsep_d"].max(), src)
        check("§3.1 仪器 胜率下界", m.group(5), str(round(g["win"].min())), src)
        check("§3.1 仪器 胜率上界", m.group(6), str(round(g["win"].max())), src)
        _numchk("§3.1 仪器 δ 下界", m.group(7), g["delta"].min(), src)
        _numchk("§3.1 仪器 δ 上界", m.group(8), g["delta"].max(), src)
        _numchk("§3.1 仪器 NRMSEP δ 下界", m.group(9), g["delta_n"].min(), src)
        _numchk("§3.1 仪器 NRMSEP δ 上界", m.group(10), g["delta_n"].max(), src)
        check("§3.1 仪器 P 上界成立（RMSEP）", "True", str(g["p"].max() <= float(m.group(11))),
              f"{src}（实际 Pmax={g['p'].max():.3g}）")
        check("§3.1 仪器 P 上界成立（NRMSEP）", "True", str(g["p_n"].max() <= float(m.group(11))),
              f"{src}（实际 Pmax={g['p_n'].max():.3g}）")

    m = re.search(r"under seasonal shift \(12 tasks\) likewise \((\d+)\\%--(\d+)\\% of tasks, "
                  r"\$\\delta\$ ([\d.]+)--([\d.]+), \$P\\le([\d.]+)\$\)", t)
    if not m:
        check("§3.1 季节区间句可解析", "True", "False", "Elsevier_en.tex")
    else:
        g = _grid("season", "envelope")
        check("§3.1 季节 胜率下界", m.group(1), str(round(g["win"].min())), src)
        check("§3.1 季节 胜率上界", m.group(2), str(round(g["win"].max())), src)
        _numchk("§3.1 季节 δ 下界", m.group(3), g["delta"].min(), src)
        _numchk("§3.1 季节 δ 上界", m.group(4), g["delta"].max(), src)
        check("§3.1 季节 P 上界成立", "True", str(g["p"].max() <= float(m.group(5))),
              f"{src}（实际 Pmax={g['p'].max():.3g}）")

    m = re.search(r"they are ahead at every positive budget \(\$\\delta\$ "
                  r"([\d.]+)--([\d.]+) at 5--40 labels, \$P\\le([\d.]+)\$\)", t)
    if not m:
        check("§3.1 产地/年区间句可解析", "True", "False", "Elsevier_en.tex")
    else:
        g = _grid("origin_year_instrument", "envelope")
        pos = g[g["n"] > 0]
        _numchk("§3.1 产地/年 δ 下界（正预算）", m.group(1), pos["delta"].min(), src)
        _numchk("§3.1 产地/年 δ 上界（正预算）", m.group(2), pos["delta"].max(), src)
        check("§3.1 产地/年 P 上界成立", "True", str(pos["p"].max() <= float(m.group(3))),
              f"{src}（实际 Pmax={pos['p'].max():.3g}）")
        check("§3.1 产地/年 每个正预算都领先", "True", str(bool((pos["delta"] > 0).all())), src)

    m = re.search(r"the deep family is better on all 12 tasks at every positive budget "
                  r"\(\$\\delta=-([\d.]+)\$ to \$-([\d.]+)\$", t)
    if not m:
        check("§3.1 土壤区间句可解析", "True", "False", "Elsevier_en.tex")
    else:
        pos = _grid("lab", "envelope")
        pos = pos[pos["n"] > 0]
        _numchk("§3.1 土壤 δ 绝对值下界", m.group(1), -pos["delta"].max(), src)
        _numchk("§3.1 土壤 δ 绝对值上界", m.group(2), -pos["delta"].min(), src)
        check("§3.1 土壤 每个正预算深度族全胜", "True", str(bool((pos["win"] == 0).all())), src)
    m = re.search(r"only at zero labels are the two not separated on RMSEP \(\$P=([\d.]+)\$; "
                  r"\$P=([\d.]+)\$ on NRMSEP\)", t)
    if not m:
        check("§3.1 土壤零标签句可解析", "True", "False", "Elsevier_en.tex")
    else:
        z0 = _grid("lab", "envelope")
        z0 = z0[z0["n"] == 0].iloc[0]
        _numchk("§3.1 土壤 零标签 P（RMSEP）", m.group(1), z0["p"], src)
        _numchk("§3.1 土壤 零标签 P（NRMSEP）", m.group(2), z0["p_n"], src)

    m = re.search(r"\(selection-free family-mean Cliff's \$\\delta\$ ([\d.]+)--([\d.]+), "
                  r"Wilcoxon \$P\\le([\d.]+)\$\)", t)
    if not m:
        check("结论 免选择区间句可解析", "True", "False", "Elsevier_en.tex")
    else:
        fam = pd.concat([_grid(s_, "family") for s_ in
                         ("instrument", "season", "origin_year_instrument")], ignore_index=True)
        _numchk("结论 族均值 δ 下界", m.group(1), fam["delta"].min(), src)
        _numchk("结论 族均值 δ 上界", m.group(2), fam["delta"].max(), src)
        check("结论 族均值 P 上界成立", "True", str(fam["p"].max() <= float(m.group(3))),
              f"{src}（实际 Pmax={fam['p'].max():.3g}）")

    # §3.1 免选择段把结论那句区间落在正文里（中英）
    fam = pd.concat([_grid(s_, "family") for s_ in
                     ("instrument", "season", "origin_year_instrument")], ignore_index=True)
    for lang, name, pat in (
            ("英", "Elsevier_en.tex", r"over the full budget grid the family mean gives \$\\delta\$ ([\d.]+)--([\d.]+) "
                                      r"\(\$P\\le([\d.]+)\$\) under instrument, seasonal and cross-origin shift"),
            ("中", "Elsevier_zh.tex", r"在整张预算网格上，族均值在仪器、季节与跨产地漂移下的 \$\\delta\$ 为 "
                                      r"([\d.]+)\$\\sim\$([\d.]+)（\$P\\le([\d.]+)\$")):
        m = _rx(f"§3.1 {lang} 全网格族均值句", pat, _flat(name), name)
        if m:
            _numchk(f"§3.1 {lang} 全网格族均值 δ 下界", m.group(1), fam["delta"].min(), src)
            _numchk(f"§3.1 {lang} 全网格族均值 δ 上界", m.group(2), fam["delta"].max(), src)
            check(f"§3.1 {lang} 全网格族均值 P 上界成立", "True", str(fam["p"].max() <= float(m.group(3))),
                  f"{src}（实际 Pmax={fam['p'].max():.3g}）")

    # §3.2 首句：CORAL 在仪器漂移下的中位 RPD 与达标份额（RPD = 1/NRMSEP）
    med = _task_medians()
    ins = med[(med["shift_type"] == "instrument") & (med["n_cal"] == 0)].set_index(["task_id", "method"])["nrmsep"]
    cor, zs = 1 / ins.xs("coral", level="method"), 1 / ins.xs("zero_shot", level="method")
    m = _rx("§3.2 CORAL 仪器 RPD 句", r"raises the median RPD of direct transfer from ([\d.]+) to ([\d.]+) under "
            r"instrument shift \((\d+)\\% of the (\d+) tasks individually reach RPD\$\\ge\$2 and (\d+)\\% reach 1\.4\)",
            t, "Elsevier_en.tex")
    if m:
        _numchk("§3.2 仪器 直接迁移中位 RPD", m.group(1), float(zs.median()), src)
        _numchk("§3.2 仪器 CORAL 中位 RPD", m.group(2), float(cor.median()), src)
        check("§3.2 仪器 CORAL RPD≥2 份额", m.group(3), str(round(100 * float((cor >= 2).mean()))), src)
        check("§3.2 仪器 任务数", m.group(4), str(len(cor)), src)
        check("§3.2 仪器 CORAL RPD≥1.4 份额", m.group(5), str(round(100 * float((cor >= 1.4).mean()))), src)

    # §4.2 那句「对齐后误差接近目标标准差」必须给得出两个具体值
    m = re.search(r"after alignment \(soil at NRMSEP ([\d.]+), apple at ([\d.]+)\)", t)
    if not m:
        check("§4.2 对齐后 NRMSEP 句可解析", "True", "False", "Elsevier_en.tex")
    else:
        rows = _table_rows(_tex("Elsevier_en.tex"), "tab:fair")
        for tag, shift, grp in (("土壤", "Lab (MIR)", 1), ("苹果", "Origin/year", 2)):
            cell = next(r[5] for r in rows if r[0].startswith("CORAL") and r[1] == shift and r[2] == "0")
            _numchk(f"§4.2 {tag} CORAL 后 NRMSEP", m.group(grp), float(cell.strip()),
                    "表 3 的 CORAL 对 DeepCORAL 零标签行")


def check_public_only() -> None:
    """§4.4 与补充材料 S15：只用四个公开基准时哪些解读变、哪些不变，逐格对 47 号。"""
    if not XL47.exists():
        check("47 号公共基准复算表存在", "True", "False", "04outputs/47_public_only_replication.xlsx")
        return
    src = "04outputs/47_public_only_replication.xlsx"
    reg = pd.read_excel(XL47, sheet_name="表c：留一基准遗憾")
    reg = reg[reg["n_cal"].astype(str) == "pooled"].set_index(["universe", "rule"])
    bs = pd.read_excel(XL47, sheet_name="表b2：基准等权bootstrap").set_index(["universe", "n_cal"])
    cor = pd.read_excel(XL47, sheet_name="表d：结构诊断相关")
    cor = cor[cor["predictor"] == "f_coral_mmd2"].set_index(["universe", "target"])
    lobo = pd.read_excel(XL47, sheet_name="表e：边际相关留一基准")
    lobo = lobo[lobo["predictor"] == "f_coral_mmd2"]

    t = _flat("Elsevier_en.tex")
    m = re.search(r"on the 54 public tasks the within-shift comparisons are unchanged", t)
    check("§4.4 指向公开基准复算", "True", str(bool(m)), "Elsevier_en.tex")
    m = re.search(r"the held-out regret of the simplest applicable correction is ([\d.]+)\\%, above "
                  r"always updating the model \(([\d.]+)\\%\)", t)
    if not m:
        check("§4.4 遗憾对比句可解析", "True", "False", "Elsevier_en.tex")
    else:
        _numchk("§4.4 公开基准 最简可用校正遗憾", m.group(1),
                float(reg.loc[("public4", "always_simple"), "median_regret"]) * 100, src)
        _numchk("§4.4 公开基准 始终模型更新遗憾", m.group(2),
                float(reg.loc[("public4", "always_model_update"), "median_regret"]) * 100, src)
        check("§4.4 公开基准上模型更新确实更便宜", "True",
              str(float(reg.loc[("public4", "always_model_update"), "median_regret"])
                  < float(reg.loc[("public4", "always_simple"), "median_regret"])), src)

    m = re.search(r"the benchmark-balanced zero-label\s+advantage, \$\+([\d.]+)\$ "
                  r"\[\$-([\d.]+)\$, ([\d.]+)\], is not separated from zero", t)
    if not m:
        check("§4.4 基准平衡句可解析", "True", "False", "Elsevier_en.tex")
    else:
        _numchk("§4.4 公开基准 零标签基准平衡均值", m.group(1),
                float(bs.loc[("public4", 0), "benchmark_balanced_mean_diff"]), src)
        _numchk("§4.4 公开基准 零标签区间下限", m.group(2),
                -float(bs.loc[("public4", 0), "boot95_lo"]), src)
        _numchk("§4.4 公开基准 零标签区间上限", m.group(3),
                float(bs.loc[("public4", 0), "boot95_hi"]), src)
        check("§4.4 公开基准零标签区间确实跨零", "True",
              str(float(bs.loc[("public4", 0), "boot95_lo"]) < 0), src)

    # §3.4 的留一陈述：零标签五中五、20 标签五中四
    m = re.search(r"keeping its sign when each benchmark is held out in all five cases at zero labels "
                  r"and in four of the five at 20", t)
    check("§3.4 留一句可解析", "True", str(bool(m)), "Elsevier_en.tex")
    for tag, target, want in (("零标签", "margin0_deep_minus_simple", 5),
                              ("20 标签", "margin20_deep_minus_simple", 4)):
        sub = lobo[lobo["target"] == target]
        check(f"§3.4 {tag}边际 留一基准保号（ρ>0）的次数", str(want),
              str(int((sub["rho_rest"] > 0).sum())), src)
        check(f"§3.4 {tag}边际 留一基准区间不含零的次数", str(want),
              str(int(sub["ci_excludes_0"].sum())), src)

    # 表 S6 逐格
    for stem in ("Supplementary_en.tex", "Supplementary_zh.tex"):
        rows = _table_rows(_tex(stem), "tab:si-public-only")
        data = [r for r in rows if len(r) == 3 and "multicolumn" not in r[0]]
        check(f"表 S6（{stem}）数据行 10 行", 10, len(data), stem)
    rows = [r for r in _table_rows(_tex("Supplementary_en.tex"), "tab:si-public-only")
            if len(r) == 3 and "multicolumn" not in r[0]]
    rule_of = {"Simplest applicable correction": "always_simple",
               "Always model update": "always_model_update",
               "Always target-only calibration": "always_target_only",
               "Frequency map, shift type unknown": "B_shift_unknown",
               "Frequency map, shift type known": "A_shift_known",
               "Default deep model": "default_deep"}
    for r in rows:
        name = r[0].strip()
        if name not in rule_of:
            continue
        for uni, cell in (("all5", r[1]), ("public4", r[2])):
            v = re.match(r"([\d.]+) \[([\d.]+), ([\d.]+)\]", cell.strip())
            if not v:
                check(f"表 S6 {name}（{uni}）单元格可解析", "True", "False", stem)
                continue
            row = reg.loc[(uni, rule_of[name])]
            _numchk(f"表 S6 {name}（{uni}）中位遗憾", v.group(1), float(row["median_regret"]) * 100, src)
            if uni == "all5":
                # 五基准列直接抄表 6（同一个量只留一个区间），区间由 check_si_crosschecks 逐格对表 6
                continue
            _numchk(f"表 S6 {name}（{uni}）区间下限", v.group(2), float(row["ci95_lo"]) * 100, src)
            _numchk(f"表 S6 {name}（{uni}）区间上限", v.group(3), float(row["ci95_hi"]) * 100, src)
    for label, target in (("Zero labels", "margin0_deep_minus_simple"),
                          ("20 labels", "margin20_deep_minus_simple")):
        hit = [r for r in rows if r[0].strip() == label]
        if len(hit) != 2:        # (a) 与 (c) 两个分块各有一行同名
            check(f"表 S6 「{label}」行出现两次", "2", str(len(hit)), stem)
            continue
        for uni, cell in (("all5", hit[1][1]), ("public4", hit[1][2])):
            v = re.match(r"([\d.]+) \[([-\d.]+), ([\d.]+)\]", cell.strip())
            if not v:
                check(f"表 S6 相关（{label},{uni}）单元格可解析", "True", "False", stem)
                continue
            row = cor.loc[(uni, target)]
            _numchk(f"表 S6 相关（{label},{uni}）ρ", v.group(1), float(row["spearman_rho"]), src)
            _numchk(f"表 S6 相关（{label},{uni}）下限", v.group(2), float(row["ci95_lo"]), src)
            _numchk(f"表 S6 相关（{label},{uni}）上限", v.group(3), float(row["ci95_hi"]), src)

    # 两封信里的补充材料节号范围必须等于补充材料实际最后一节
    n_sec = len(re.findall(r"\\section\{", _tex("Supplementary_en.tex")))
    for name in ("Response_R1.tex", "Response_R2.tex"):
        m = re.search(r"Supplementary Material S1--S(\d+)", _letter(name))
        if not m:
            check(f"{name}：补充材料节号范围句可解析", "True", "False", name)
        else:
            check(f"{name}：补充材料节号范围与实际节数一致", m.group(1), str(n_sec),
                  "06doc/01manuscript/Supplementary_en.tex（\\section 计数）")


def check_wordcount() -> None:
    """回复信与 cover letter 里的词数、页数与缩减幅度，逐个对 82 号与编译出的 PDF。

    判据取自信本身而不是手抄常量：这三处数字每改一次正文就会变，手抄必然过期。
    """
    script = BASE / "02code" / "82_main_text_wordcount.py"
    src = "02code/82_main_text_wordcount.py（同一口径跑两版）"
    res = subprocess.run(
        [sys.executable, str(script),
         str(BASE / "06doc/02sub/revision/_submitted_Elsevier_en.tex"),
         str(MS / "Elsevier_en.tex")],
        capture_output=True, text=True, check=True)
    counts = [int(line.split("\t")[0]) for line in res.stdout.strip().splitlines()]
    sub, now = counts
    cut = 1 - now / sub

    r1 = _letter("Response_R1.tex")
    m = re.search(r"about ([\d,]+) words in the body instead of about ([\d,]+)", r1)
    if not m:
        raise SystemExit("R1 里的词数句没解析出来")
    check("回复信 R1：修回稿正文词数", m.group(1), f"{round(now, -2):,.0f}", src)
    check("回复信 R1：投出稿正文词数", m.group(2), f"{round(sub, -2):,.0f}", src)
    m2 = re.search(r"and (\d+) pages instead of (\d+)", r1)
    if not m2:
        raise SystemExit("R1 里的页数句没解析出来")
    now_pdf = MS / "Elsevier_en.pdf"
    check("回复信 R1：修回稿页数", int(m2.group(1)), _pages(now_pdf),
          "06doc/01manuscript/Elsevier_en.pdf（pdfinfo）")

    r2 = _letter("Response_R2.tex")
    m3 = re.search(r"The body is about ([\d,]+) words instead of about ([\d,]+) \(same "
                   r"counting rule, prose only\) and (\d+) pages instead of (\d+)", r2)
    if not m3:
        raise SystemExit("R2 里的词数/页数句没解析出来")
    check("回复信 R2：修回稿词数与 R1 一致", m.group(1), m3.group(1), "两封信互校")
    check("回复信 R2：修回稿页数与 R1 一致", m2.group(1), m3.group(3), "两封信互校")

    cov = _letter("CoverLetter_R1.tex")
    m4 = re.search(r"about ([\d,]+) words in the body instead of about ([\d,]+) "
                   r"\((\d+)\\% shorter; (\d+) pages instead of (\d+)\)", cov)
    if not m4:
        raise SystemExit("cover letter 里的词数句没解析出来")
    check("cover letter：修回稿词数与 R1 一致", m.group(1), m4.group(1), "信与信互校")
    check("cover letter：缩减百分比", int(m4.group(3)), round(cut * 100),
          f"{src}（实际 {cut:.1%}）")
    check("cover letter：修回稿页数与 R1 一致", m2.group(1), m4.group(4), "信与信互校")

    # 两封回复信开头也各自写了缩减幅度，此前只有 cover letter 那一处有判据
    m5 = re.search(r"shortened by about (\d+)\\% in words and (\d+)\\% in pages", r1)
    if not m5:
        raise SystemExit("R1 里的缩减幅度句没解析出来")
    check("回复信 R1：词数缩减百分比", int(m5.group(1)), round(cut * 100), f"{src}（实际 {cut:.1%}）")
    pages_cut = 1 - _pages(now_pdf) / int(m2.group(2))
    check("回复信 R1：页数缩减百分比", int(m5.group(2)), round(pages_cut * 100),
          f"Elsevier_en.pdf {_pages(now_pdf)} 页对投出稿 {m2.group(2)} 页（实际 {pages_cut:.1%}）")
    # 投稿审计文档的页数/词数与四份稿同源——它是作者提交前读的那一份，不能停在旧数
    audit = (BASE / "06doc/02sub/00SUBMISSION_AUDIT.md").read_text(encoding="utf-8")
    m7 = re.search(r"返修后 (\d+) 页、正文 ([\d,]+) 词", audit)
    if not m7:
        check("投稿审计的页数/词数句可解析", "True", "False", "06doc/02sub/00SUBMISSION_AUDIT.md")
    else:
        check("投稿审计：修回稿页数", int(m7.group(1)), _pages(now_pdf),
              "06doc/01manuscript/Elsevier_en.pdf（pdfinfo）")
        check("投稿审计：修回稿正文词数", m7.group(2), f"{now:,d}", src)

    # 审稿意见追踪表 R2.3 那一行也写了词数/页数，同源判
    trk = (BASE / "06doc/02sub/0NREVIEWER_RESPONSE.md").read_text(encoding="utf-8")
    m8 = re.search(r"正文 ([\d,]+) 词 / (\d+) 页，相对投出稿 ([\d,]+) 词减 (\d+)%、(\d+) 页减 (\d+)%", trk)
    if not m8:
        check("R2.3 行的词数/页数句可解析", "True", "False", "06doc/02sub/0NREVIEWER_RESPONSE.md")
    else:
        check("R2.3 行：修回稿正文词数", m8.group(1), f"{now:,d}", src)
        check("R2.3 行：修回稿页数", int(m8.group(2)), _pages(now_pdf),
              "06doc/01manuscript/Elsevier_en.pdf（pdfinfo）")
        check("R2.3 行：投出稿正文词数", m8.group(3), f"{sub:,d}", src)
        check("R2.3 行：词数缩减百分比", int(m8.group(4)), round(cut * 100), f"{src}（实际 {cut:.1%}）")
        check("R2.3 行：投出稿页数", int(m8.group(5)), int(m2.group(2)), "回复信 R1 互校")
        check("R2.3 行：页数缩减百分比", int(m8.group(6)), round(pages_cut * 100),
              f"Elsevier_en.pdf {_pages(now_pdf)} 页对投出稿 {m2.group(2)} 页（实际 {pages_cut:.1%}）")

    m6 = re.search(r"has been shortened by about (\d+)\\% in words and (\d+)\\% in pages, "
                   r"and reorganised", r2)
    if not m6:
        raise SystemExit("R2 里的缩减幅度句没解析出来")
    check("回复信 R2：词数缩减百分比与 R1 一致", m5.group(1), m6.group(1), "两封信互校")
    check("回复信 R2：页数缩减百分比与 R1 一致", m5.group(2), m6.group(2), "两封信互校")
    # R2 开头的概述句也写了这两个百分比
    m6b = re.search(r"answered by a shortening of about (\d+)\\% in words and (\d+)\\% in "
                    r"pages, net of the new analyses", r2)
    if not m6b:
        raise SystemExit("R2 开头的缩减幅度句没解析出来")
    check("回复信 R2 开头：词数缩减百分比与 R1 一致", m5.group(1), m6b.group(1), "两封信互校")
    check("回复信 R2 开头：页数缩减百分比与 R1 一致", m5.group(2), m6b.group(2), "两封信互校")

    for script, claim in ((BASE / "02code/81_response_tex_to_txt.py",
                           "投稿系统用的 .txt 与回复信 .tex 同源"),
                          (BASE / "02code/69_build_revision_package.py",
                           "返修包派生件与活稿同期")):
        r = subprocess.run([sys.executable, str(script), "--check"],
                           capture_output=True, text=True)
        check(claim, 0, r.returncode,
              f"{script.relative_to(BASE)} --check：{r.stdout.strip().splitlines()[-1]}")

    for tag, pdf in (("正文中文稿", MS / "Elsevier_zh.pdf"),
                     ("补充材料英文", MS / "Supplementary_en.pdf"),
                     ("补充材料中文", MS / "Supplementary_zh.pdf")):
        check(f"{tag} PDF 与 .tex 同期编译", "True",
              str(pdf.stat().st_mtime >= pdf.with_suffix(".tex").stat().st_mtime),
              f"{pdf.relative_to(BASE)}")


def _bib_years(path: Path) -> list[int | None]:
    text = path.read_text(encoding="utf-8")
    items = re.findall(r"\\bibitem\{[^}]+\}(.*?)(?=\\bibitem|\\end\{thebibliography\})", text, re.S)
    out: list[int | None] = []
    for body in items:
        m = re.search(r"\b(18\d\d|19[5-9]\d|20[0-2]\d)\b", body)
        out.append(int(m.group(1)) if m else None)
    return out


def check_bibliography() -> None:
    years = _bib_years(MS / "Elsevier_en.tex")
    src = "06doc/01manuscript/Elsevier_en.tex（thebibliography）"
    check("回复信 R1.2：主文 54 条", 54, len(years), src)
    check("回复信 R1.2：>=2010 共 49 条", 49, sum(1 for y in years if y and y >= 2010), src)
    check("回复信 R1.2：>=2018 共 40 条", 40, sum(1 for y in years if y and y >= 2018), src)
    check("红线卡 D 条：近 5 年（>=2022）>=25", "True",
          str(sum(1 for y in years if y and y >= 2022) >= 25),
          f"{src}（实际 {sum(1 for y in years if y and y >= 2022)}）")
    check("红线卡 D 条：近 3 年（>=2024）>=10", "True",
          str(sum(1 for y in years if y and y >= 2024) >= 10),
          f"{src}（实际 {sum(1 for y in years if y and y >= 2024)}）")
    check("回复信 R1.2：2010 年前 5 条", 5, sum(1 for y in years if y and y < 2010), src)
    check("中英文献表条数一致", len(years), len(_bib_years(MS / "Elsevier_zh.tex")),
          "Elsevier_zh.tex")
    si = len(re.findall(r"\\bibitem", (MS / "Supplementary_en.tex").read_text(encoding="utf-8")))
    check("补充材料文献表 8 条", 8, si, "06doc/01manuscript/Supplementary_en.tex")

    def keys(text: str) -> set[str]:
        return set(re.findall(r"\\bibitem\{([^}]+)\}", text))
    now = keys((MS / "Elsevier_en.tex").read_text(encoding="utf-8"))
    sub = keys((BASE / "06doc/02sub/revision/_submitted_Elsevier_en.tex").read_text(encoding="utf-8"))
    check("回复信 R1.2：相对投出稿删 18 条", 18, len(sub - now), "投出稿 vs 修回稿 bibitem 集合")
    check("回复信 R1.2：相对投出稿增 16 条", 16, len(now - sub), "同上")
    check("回复信 R1.2：Storkey2009 不在修回稿（回复信已改口径）", "True",
          str("Storkey2009" not in now), src)
    cited = set()
    for grp in re.findall(r"\\cite\{([^}]+)\}", (MS / "Elsevier_en.tex").read_text(encoding="utf-8")):
        cited |= {k.strip() for k in grp.split(",")}
    check("无悬空引用（\\cite 全部有 bibitem）", "[]", str(sorted(cited - now)), src)
    check("无未引用条目", "[]", str(sorted(now - cited)), src)


XLDS = OUT / "62_ds_sensitivity.xlsx"
DS_BUDGETS = (5, 10, 20, 40)


def check_ds() -> None:
    """补充材料 S13 的 DS 对照段与 §4.4 那句：逐预算中位、三个 P、与 SBC/模型更新的对比。

    DS 是把 PDS 的窗口开到整条谱，两者共用同一份实现与同一个 Ridge 惩罚，所以这一段
    每个数字都能从 62 号的旁路工作簿重算：先在 5 个种子上取中位，再在 30 个配对仪器
    任务上取中位，与表 4 同一套聚合口径。
    """
    src = "62_ds_sensitivity.xlsx curves（5 种子取中位后在 30 个配对仪器任务上取中位）"
    cur = pd.read_excel(XLDS, sheet_name="curves")
    check("S13 DS 对照的配对仪器任务数", "30", str(cur["task_id"].nunique()), src)
    check("S13 DS 对照的种子数", "5", str(cur["seed"].nunique()), src)
    med = cur.groupby(["task_id", "method", "n_cal"])["rmsep"].median().reset_index()

    def at(method: str, n: int) -> pd.Series:
        s = med[(med["method"] == method) & (med["n_cal"] == n)]
        return s.set_index("task_id")["rmsep"].sort_index()

    t = _flat("Supplementary_en.tex")
    m = re.search(r"its median RMSEP is ([\d.]+), ([\d.]+), ([\d.]+) and ([\d.]+) at 5, 10, 20 "
                  r"and 40 target standards against ([\d.]+), ([\d.]+), ([\d.]+) and ([\d.]+) "
                  r"for PDS", t)
    if not m:
        check("S13 DS/PDS 中位句可解析", "True", "False", "Supplementary_en.tex")
    else:
        for k, n in enumerate(DS_BUDGETS):
            _numchk(f"S13 DS 中位 RMSEP n={n}", m.group(k + 1), float(at("ds", n).median()), src)
            _numchk(f"S13 PDS 中位 RMSEP n={n}", m.group(k + 5), float(at("pds", n).median()), src)
            check(f"S13 n={n} 处 DS 中位低于 PDS（「两者之中 DS 更优」）", "True",
                  str(float(at("ds", n).median()) < float(at("pds", n).median())), src)

    m = re.search(r"favours DS at 10, 20 and 40 standards \(\$P=([^$]+)\$, \$([^$]+)\$ and "
                  r"\$([^$]+)\$\) while the two are not separated at 5 \(\$P=([\d.]+)\$\)", t)
    if not m:
        check("S13 DS/PDS 配对 P 句可解析", "True", "False", "Supplementary_en.tex")
    else:
        for k, n in enumerate((10, 20, 40)):
            p = float(wilcoxon(at("ds", n).to_numpy(), at("pds", n).to_numpy()).pvalue)
            _scichk(f"S13 DS 对 PDS 配对 P n={n}", f"${m.group(k + 1)}$", p, src)
        p5 = float(wilcoxon(at("ds", 5).to_numpy(), at("pds", 5).to_numpy()).pvalue)
        _numchk("S13 DS 对 PDS 配对 P n=5", m.group(4), p5, src)
        check("S13 n=5 处两者不可分（P>0.05）", "True", str(p5 > 0.05), f"{src}（实际 P={p5:.3g}）")

    m = re.search(r"the slope/bias correction is at ([\d.]+)--([\d.]+) and model updating at "
                  r"([\d.]+)--([\d.]+)", t)
    if not m:
        check("S13 SBC/模型更新区间句可解析", "True", "False", "Supplementary_en.tex")
    else:
        sbc = [float(at("sbc", n).median()) for n in DS_BUDGETS]
        mu = [float(at("model_update", n).median()) for n in DS_BUDGETS]
        _numchk("S13 SBC 中位下界", m.group(1), min(sbc), src)
        _numchk("S13 SBC 中位上界", m.group(2), max(sbc), src)
        _numchk("S13 模型更新中位下界", m.group(3), min(mu), src)
        _numchk("S13 模型更新中位上界", m.group(4), max(mu), src)
        worst = max(max(sbc), max(mu))
        best_std = min(min(float(at("ds", n).median()) for n in DS_BUDGETS),
                       min(float(at("pds", n).median()) for n in DS_BUDGETS))
        check("S13 两种标准化在每个预算都劣于 SBC 与模型更新", "True", str(worst < best_std),
              f"{src}（标准化最好 {best_std:.3f}，经典最差 {worst:.3f}）")

    # SST/EPO 未跑不能只写「未予运行」：得写清这一缺口界定的是什么，以及要改变读数需满足什么
    check("S13 未跑成员的排除理由写到位（英）", "True",
          str("it would have to beat both the slope/bias correction and model updating "
              "on these tasks" in t), "Supplementary_en.tex")
    check("S13 未跑成员的排除理由写到位（中）", "True",
          str("它必须在这批任务上同时胜过斜率/偏置校正与模型更新" in _tex("Supplementary_zh.tex")),
          "Supplementary_zh.tex")

    # §4.4 第十条把 PDS 与 DS 并列、指向 S13；S13 写明 DS 是窗口开到全谱的 PDS
    main = _flat("Elsevier_en.tex")
    check("§4.4 第十条并列 PDS 与 DS 并指向 S13", "True",
          str("PDS and direct standardisation, informative mainly on the paired-instrument benchmarks, trail "
              "slope/bias correction and model updating there (Supplementary Section~S13)" in main),
          "Elsevier_en.tex")
    check("S13 写明 DS 是窗口开到全谱的 PDS", "True",
          str("with the window opened to the whole spectrum" in _flat("Supplementary_en.tex")),
          "Supplementary_en.tex")
    for name in ("Elsevier_zh.tex", "Supplementary_zh.tex"):
        check(f"{name} 已同步 DS 对照", "True", str("直接校正" in _flat(name)), name)


REV = BASE / "06doc" / "02sub" / "revision"


def _unbrace(s: str, cmds: tuple[str, ...]) -> str:
    """把 \\cmd{...} 换成花括号里的内容，按真正的配对扫描，支持任意层嵌套。

    正则版只能吃一层嵌套，latexdiff 给 \\cite 套的 \\DIFadd{\\mbox{\\cite{a,b}}\\hskip0pt}
    正好是两层，用正则会整段留在原地、被误判成内容差异。
    """
    pat = re.compile(r"\\(?:" + "|".join(cmds) + r")\{")
    while True:
        m = pat.search(s)
        if not m:
            return s
        i, depth = m.end(), 1
        while depth:
            ch = s[i]
            if ch == "\\":
                i += 2
                continue
            depth += (ch == "{") - (ch == "}")
            i += 1
        s = s[:m.start()] + s[m.end():i - 1] + s[i:]


def _strip_diff(marked: str) -> str:
    """把 latexdiff 标记稿还原成它所标记的那一版正文。"""
    s = re.sub(r"%DIF PREAMBLE EXTENSION ADDED BY LATEXDIFF.*?"
               r"%DIF END PREAMBLE EXTENSION ADDED BY LATEXDIFF", "", marked, flags=re.S)
    # \DIFdelbegin 与 \DIFdelendFL 会配对（浮动体里混用 FL 后缀），故两端都要容忍 FL
    s = re.sub(r"\\DIFdelbegin(?:FL)?\b.*?\\DIFdelend(?:FL)?\b", "", s, flags=re.S)
    s = re.sub(r"\\DIFaddbegin(?:FL)?\b|\\DIFaddend(?:FL)?\b", "", s)
    s = _unbrace(s, ("DIFaddFL", "DIFadd"))
    s = _unbrace(s.replace("\\hskip0pt", ""), ("mbox",))
    return s.replace("\\DIFOincludegraphics", "\\includegraphics")


def _tex_norm(s: str) -> str:
    """去注释、压空白，并抹掉标点旁的空格——latexdiff 拆包后会留下装饰性空格。"""
    s = "\n".join(r for line in s.split("\n")
                  for r in [re.sub(r"(?<!\\)%.*$", "", line)] if r.strip())
    s = re.sub(r"\s+", " ", s)
    return re.sub(r"(?<=[^0-9A-Za-z])\s+|\s+(?=[^0-9A-Za-z])", "", s).strip()


def check_marked_equivalence() -> None:
    """标记稿去掉标记之后，必须与干净修回稿逐字一致。

    审稿人看的是标记稿、编辑收的是干净稿，两者一旦分叉就是「改了但没标」或
    「标了但没改」。这是投稿包里唯一无法靠数字判据覆盖的一致性。
    """
    marked = (REV / "Elsevier_en_marked.tex").read_text(encoding="utf-8")
    clean = (REV / "_revised_Elsevier_en.tex").read_text(encoding="utf-8")
    a, b = _tex_norm(_strip_diff(marked)), _tex_norm(clean)
    check("标记稿去标记后与干净修回稿逐字一致", "True", str(a == b),
          f"06doc/02sub/revision/（标记稿 {len(a)} 字符，干净稿 {len(b)} 字符）")
    live = _tex_norm(_tex("Elsevier_en.tex"))
    check("干净修回稿与英文活稿逐字一致", "True", str(b == live),
          "_revised_Elsevier_en.tex 对 06doc/01manuscript/Elsevier_en.tex")


def check_note_and_counts() -> None:
    """表 3 表注的候选分母、§4.1 的基准计数、§4.5 的残差份额、§4.4 的局限条数。

    这几处都是「同一个量在别处已有权威出处、此处却抄成另一个数」的类型：表注的分母
    可以从 62/64 号 curves 逐预算数出来，残差份额在表 7 里印着，局限条数由序数词数出。
    """
    t = _flat("Elsevier_en.tex")
    c62, c64 = _curves()
    cur = pd.concat([c62, c64], ignore_index=True)
    src = "62/64 号 curves 逐预算数候选"

    m = re.search(r"The candidates at each budget are therefore the (\d+) label-free methods at \$n=0\$ and, "
                  r"at \$n\\ge5\$, (\d+) on the instrument benchmarks and (\d+) elsewhere "
                  r"\(of which the classical members are (\d+), (\d+) and (\d+)\)", t)
    if not m:
        check("表 2 表注的候选分母句可解析", "True", "False", "Elsevier_en.tex")
    else:
        def nmeth(bench: str, n: int) -> list[str]:
            s = cur[(cur["benchmark"] == bench) & (cur["n_cal"] == n)]
            return sorted(s["method"].unique())
        zero = nmeth("corn", 0)
        inst = nmeth("corn", 20)
        other = nmeth("mango", 20)
        check("表 2 表注 n=0 分母", m.group(1), str(len(zero)), src)
        check("表 2 表注 n>=5 仪器基准分母", m.group(2), str(len(inst)), src)
        check("表 2 表注 n>=5 其余基准分母", m.group(3), str(len(other)), src)
        check("表 2 表注 n=0 候选全为免标签", "True", str(set(zero) <= LABEL_FREE), src)
        check("表 2 表注 n>=5 候选全部吃标签", "True",
              str(not (set(inst) | set(other)) & LABEL_FREE), src)
        cls = {"zero_shot", "coral", "sbc", "pds", "model_update", "target_only"}
        check("表 2 表注 n=0 经典成员数", m.group(4), str(len(set(zero) & cls)), src)
        check("表 2 表注 n>=5 仪器经典成员数", m.group(5), str(len(set(inst) & cls)), src)
        check("表 2 表注 n>=5 其余经典成员数", m.group(6), str(len(set(other) & cls)), src)

    rows = _table_rows(_tex("Elsevier_en.tex"), "tab:struct")

    # §4.5 的苹果残差份额必须等于表 5 的 Residual 列
    apple = next(r for r in rows if r[0].strip() == "Apple")
    m = re.search(r"large residual\s+share \((\d+)\\% on apple\)", t)
    if not m:
        check("§4.5 苹果残差份额句可解析", "True", "False", "Elsevier_en.tex")
    else:
        check("§4.5 苹果残差份额与表 7 一致", m.group(1),
              str(round(float(apple[5]) * 100)), "表 5 tab:struct 的 Residual 列")

    # §4.4 承诺的局限条数与实际序数词数
    m = re.search(r"(Eight|Nine|Ten|Eleven) limitations bound these conclusions", t)
    if not m:
        check("§4.4 局限条数句可解析", "True", "False", "Elsevier_en.tex")
    else:
        ords = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh",
                "Eighth", "Ninth", "Tenth", "Eleventh"]
        seg = t[t.index("limitations bound these conclusions"):]
        seg = seg[:seg.index("\\subsection")] if "\\subsection" in seg else seg
        got = sum(1 for o in ords if re.search(r"\b" + o + r", ", seg))
        check("§4.4 局限条数与序数词数一致",
              {"Eight": 8, "Nine": 9, "Ten": 10, "Eleven": 11}[m.group(1)], got,
              "Elsevier_en.tex §4.4 的 First--Tenth")


def check_s14_apple_reference() -> None:
    """S14 那句「苹果对所有方法都难」：PLS 的苹果源域值与深度对 PLS 之比，全部从 80 号重算。

    这句是本轮把面板读数收窄回证据的地方：深度族在苹果源域内确实只到 0.92--0.95，但同一
    批划分上 PLS 也只到 0.80，所以「只有深度族在苹果失效」这个读法不成立，判据必须跟上。
    """
    long = pd.read_excel(OUT / "80_source_sanity_table.xlsx", sheet_name="long")
    med = long.groupby(["benchmark", "method"])["nrmsep_source"].median()
    src = "04outputs/80_source_sanity_table.xlsx[long] 的源域 NRMSEP 中位"
    si = _flat("Supplementary_en.tex")
    m = re.search(r"On the same splits a PLS regression reaches ([\d.]+), its weakest value on any "
                  r"benchmark.*?a deep-to-PLS ratio of ([\d.]+)--([\d.]+) against ([\d.]+)--([\d.]+) "
                  r"on corn and ([\d.]+)--([\d.]+) on mango", si)
    if not m:
        check("S14 苹果 PLS 对照句可解析", "True", "False", "Supplementary_en.tex")
        return
    _numchk("S14 苹果 PLS 源域 NRMSEP", m.group(1), float(med[("apple", "plsr")]), src)
    check("S14 苹果 PLS 值是各基准中最差", "True",
          str(float(med[("apple", "plsr")]) == max(float(med[(b, "plsr")]) for b in
              med.index.get_level_values(0).unique())), src)
    for bench, lo, hi in (("apple", 2, 3), ("corn", 4, 5), ("mango", 6, 7)):
        ratios = [float(med[(bench, d)]) / float(med[(bench, "plsr")]) for d in S14_DEEP]
        _numchk(f"S14 {bench} 深度对 PLS 之比下界", m.group(lo), min(ratios), src)
        _numchk(f"S14 {bench} 深度对 PLS 之比上界", m.group(hi), max(ratios), src)
    # 「领先最少」＝苹果的深度对 PLS 之比整段低于另外两个基准
    def ratio(b: str) -> list[float]:
        return [float(med[(b, d)]) / float(med[(b, "plsr")]) for d in S14_DEEP]

    check("S14 苹果是 PLS 领先三基准中领先最少的", "True",
          str(max(ratio("apple")) < min(min(ratio("corn")), min(ratio("mango")))), src)


def check_letter_bibliography() -> None:
    """回复信 R1 的文献时效计数，从主文文献表的年份逐条重算。"""
    t = _tex("Elsevier_en.tex")
    body = t[t.index("\\begin{thebibliography}"):t.index("\\end{thebibliography}")]
    years = []
    for it in re.split(r"\\bibitem\{", body)[1:]:
        ys = [int(y) for y in re.findall(r"\b(1[89]\d{2}|20[0-2]\d)\b", it)]
        years.append(max(ys) if ys else 0)
    src = f"Elsevier_en.tex 文献表 {len(years)} 条的年份"
    r1 = _letter("Response_R1.tex")
    m = re.search(r"(\d+) of the (\d+) entries are now from 2010 or later and (\d+) from 2018 "
                  r"or later\. (All four|Four of the five) earlier entries", r1)
    if not m:
        raise SystemExit("R1 的文献时效句没解析出来")
    check("回复信 R1：文献总条数", m.group(2), str(len(years)), src)
    check("回复信 R1：2010 及以后条数", m.group(1), str(sum(1 for y in years if y >= 2010)), src)
    check("回复信 R1：2018 及以后条数", m.group(3), str(sum(1 for y in years if y >= 2018)), src)
    check("回复信 R1：2010 以前的条数措辞", "All four" if sum(1 for y in years if y < 2010) == 4
          else f"{sum(1 for y in years if y < 2010)} 条", m.group(4), src)


def check_si_crosschecks() -> None:
    """补充材料里「同一个量在别处印着另一个数」的两处：表 S6 面板 b、S12 错分率。"""
    si = _flat("Supplementary_zh.tex"), _flat("Supplementary_en.tex")
    en = si[1]
    # 表 S6 面板 (b) 的五基准列必须与表 6 逐格一致
    tab6 = _table_rows(_tex("Elsevier_en.tex"), "tab:regret")
    pooled = {r[0].split(" (")[0]: r[6] for r in tab6}
    name_map = {
        "Simplest applicable correction": "Simplest applicable correction",
        "Always model update": "Always model update",
        "Always target-only calibration": "Always target-only calibration",
        "Frequency map, shift type unknown": "Frequency map as a rule, shift type unknown",
        "Frequency map, shift type known": "Frequency map as a rule, shift type known",
        "Default deep model": "Default deep model",
    }
    for si_name, t6_name in name_map.items():
        m = re.search(re.escape(si_name) + r" & ([\d.]+ \[[\d.]+, [\d.]+\]) &", en)
        if not m:
            check(f"表 S6 {si_name} 行可解析", "True", "False", "Supplementary_en.tex")
            continue
        check(f"表 S6 五基准列 {si_name} 与表 6 一致", pooled[t6_name], m.group(1),
              "Elsevier_en.tex 表 6 的汇总列")
    # S12 的错分率区间必须等于表 S5 里印着的那几个值
    m = re.search(r"misclassification rate still between (\d+)\\% and (\d+)\\%", en)
    if not m:
        check("S12 错分率句可解析", "True", "False", "Supplementary_en.tex")
    else:
        vals = [float(x) for x in re.findall(r"& ([0-9]{2}\.[0-9]) \\\\", en)]
        mis = [v for v in vals if 30 <= v <= 45]
        check("S12 错分率下界", m.group(1), str(math.floor(min(mis))), "表 S5 的 Misclass. 列")
        check("S12 错分率上界", m.group(2), str(math.ceil(max(mis))), "表 S5 的 Misclass. 列")


XL101 = OUT / "101_pair_level_inference.xlsx"


def check_script_refs() -> None:
    """四份稿与三封信里写的「script NN / 脚本 NN」必须真有那个脚本。

    这一类指针平时没人核：本轮就有过 02code 下两个脚本同占 85 号、返修包构建器改名 69 号，
    稿件里的编号若不跟着改，读者按图索骥会扑空，而编译与数字判据都不会出声。
    """
    import os
    have = {f.split("_")[0] for f in os.listdir(BASE / "02code")
            if f.endswith((".py", ".sh")) and f[:1].isdigit()}
    targets = [(MS / n) for n in ("Elsevier_en.tex", "Elsevier_zh.tex",
                                  "Supplementary_en.tex", "Supplementary_zh.tex")]
    targets += [(BASE / "06doc/02sub/revision" / n)
                for n in ("Response_R1.tex", "Response_R2.tex", "CoverLetter_R1.tex")]
    for f in targets:
        t = re.sub(r"\s+", " ", f.read_text(encoding="utf-8"))
        nums: set[str] = set()
        for m in re.finditer(r"(?:scripts?|脚本)\s*~?\s*((?:\d+)(?:\s*(?:,|、|and|与)\s*\d+)*)", t):
            nums |= set(re.findall(r"\d+", m.group(1)))
        missing = sorted(n for n in nums if n not in have and n.zfill(2) not in have)
        check(f"{f.name} 引用的脚本编号都存在", "[]", str(missing),
              f"02code/ 现有编号；该文件引用 {sorted(nums, key=int)}")


def check_letter_ordinals() -> None:
    """两封信按序数指认 §4.4 的某一条局限——局限一重排，这两处就静默指错。

    R1 指第六条（免标签胜出部分是可运行性之胜），R2 指第五条（留一基准遗憾不是事后数据）。
    这里不核措辞，只核那两个序位上的内容还是不是它们，以及信里写的序数词没被改动。
    """
    t = _flat("Elsevier_en.tex")
    i = t.index("\\subsection{Boundaries of identifiability")
    sec = t[i:t.index("\\subsection{Role of the apple case study")]
    ords = ["First,", "Second,", "Third,", "Fourth,", "Fifth,", "Sixth,", "Seventh,",
            "Eighth,", "Ninth,", "Tenth,"]
    at = {o: sec.find(" " + o) for o in ords}
    src = "06doc/01manuscript/Elsevier_en.tex（4.4 逐条序位）"
    check("§4.4 十条局限序数词齐全且递增", "True",
          str(all(at[o] >= 0 for o in ords)
              and [at[o] for o in ords] == sorted(at[o] for o in ords)), src)
    for word, key in (("Fifth,", "held-out regret is measured on a benchmark held out"),
                      ("Sixth,", "wins of applicability")):
        nxt = ords[ords.index(word) + 1]
        body = sec[at[word]:at[nxt]] if at[word] >= 0 and at[nxt] >= 0 else ""
        check(f"§4.4 第 {word[:-1]} 条仍是「{key}」那一条", "True", str(key in body), src)
    for letter, word in (("Response_R1.tex", "sixth limitation"),
                         ("Response_R2.tex", "fifth limitation")):
        check(f"{letter} 仍按序数指认 §4.4 的{word}", "True",
              str(f"Section 4.4, {word}" in _letter(letter)),
              f"06doc/02sub/revision/{letter}")


def check_highlights() -> None:
    """Highlights 是随稿上传的独立文件，正文改了它不会跟着改——所以它的数字也要活判。

    五条各 <=85 字符是 CILS 的硬格式要求；三个数（固定处方与默认深度模型的遗憾、五中之四）必须与正文同源。
    """
    f = BASE / "06doc/02sub/revision/Highlights.txt"
    lines = [ln for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
    src = "06doc/02sub/revision/Highlights.txt"
    check("Highlights 条数", "5", str(len(lines)), src)
    over = [ln[:30] for ln in lines if len(ln) > 85]
    check("Highlights 每条不超过 85 字符", "[]", str(over), f"{src}（最长 {max(map(len, lines))}）")
    main = re.sub(r"\s+", " ", _tex("Elsevier_en.tex"))
    joined = " ".join(lines)
    # 两个遗憾数取表 4 的汇总列（表 4 本身逐格对 73 号），不在这里手抄
    reg = {r[0].split(" (")[0]: r[6].split(" [")[0].strip() for r in _table_rows(_tex("Elsevier_en.tex"), "tab:regret")}
    for tag, key in (("固定处方遗憾%", "Simplest applicable correction"), ("默认深度模型遗憾%", "Default deep model")):
        pat = re.escape(reg.get(key, "未找到"))
        in_hl = bool(re.search(pat + r"\%", joined))
        in_main = bool(re.search(pat + r"\\%", main))
        check(f"Highlights 的{tag}与正文同源", "True", str(in_hl and in_main),
              f"{src} vs 06doc/01manuscript/Elsevier_en.tex")
    check("Highlights 的「五中之四」与正文同源", "True",
          str(("four of five benchmarks" in joined)
              and ("four of the five benchmarks" in main)),
          f"{src} vs 06doc/01manuscript/Elsevier_en.tex")


def check_budget_sensitivity() -> None:
    """S14 末尾的预算敏感性：表 S7 逐格 + 那一段的每个数，全部对 102 号的汇总表。

    这一段回答的是「源域内停在恒定预测器水平，究竟是迁移失败还是训练失败」。判读全靠
    三件事：苹果上逐位相同的运行数等于总运行数（早停先于上限触发，预算从来不是约束）、
    其余基准源域确有改善、以及目标域并不跟着改善。三件事各自都要能从数据重算出来。
    """
    xl = OUT / "102_budget_sensitivity.xlsx"
    if not xl.exists():
        check("102 号预算敏感性工作簿存在", "True", "False", str(xl.relative_to(BASE)))
        return
    d = pd.read_excel(xl, sheet_name="汇总").set_index("基准")
    src = "04outputs/102_budget_sensitivity.xlsx[汇总]"
    name = {"Corn": "corn", "Tablets": "tablet", "Mango": "mango",
            "Soil (MIR)": "ossl_mir", "Apple": "apple"}
    seen = []
    for row in _table_rows(_tex("Supplementary_en.tex"), "tab:si-budget"):
        b = name.get(row[0].strip())
        if b is None:
            check("表 S7 行名可识别", "True", f"False（{row[0]!r}）", "Supplementary_en.tex")
            continue
        seen.append(b)
        r = d.loc[b]
        same, n = (int(x) for x in row[1].split("/"))
        check(f"表 S7 {b} 逐位相同的运行数", same, int(r["逐位相同的运行数"]), src)
        check(f"表 S7 {b} 运行数", n, int(r["运行数"]), src)
        _numchk(f"表 S7 {b} 源域中位（基线）", row[2], float(r["源域中位_基线"]), src)
        _numchk(f"表 S7 {b} 源域中位（8000 步）", row[3], float(r["源域中位_8000步"]), src)
        _numchk(f"表 S7 {b} 目标域中位（基线）", row[5], float(r["目标域中位_基线"]), src)
        _numchk(f"表 S7 {b} 目标域中位（8000 步）", row[6], float(r["目标域中位_8000步"]), src)
        for tag, cell, val in (("源域", row[4], r["源域配对P"]),
                               ("目标域", row[7], r["目标域配对P"])):
            if row[1].split("/")[0] == row[1].split("/")[1]:
                check(f"表 S7 {b} {tag} P 为破折线（两上限逐位相同，无差异可检验）",
                      "True", str(cell.strip() in {"---", "--", "—"}), src)
            elif "times" in cell:
                _scichk(f"表 S7 {b} {tag} P", cell, float(val), src)
            else:
                _numchk(f"表 S7 {b} {tag} P", cell, float(val), src)
    check("表 S7 覆盖五个基准", str(sorted(name.values())), str(sorted(seen)),
          "Supplementary_en.tex")

    t = _flat("Supplementary_en.tex")
    m = re.search(r"every one of the (\d+) runs of CNN, Deep CORAL and Phys\+BL returns a "
                  r"bit-identical value", t)
    if not m:
        check("S14 苹果逐位相同句可解析", "True", "False", "Supplementary_en.tex")
    else:
        check("S14 苹果逐位相同的运行数", int(m.group(1)), int(d.loc["apple", "运行数"]), src)
        check("S14 苹果全部运行逐位相同", "True",
              str(int(d.loc["apple", "逐位相同的运行数"]) == int(d.loc["apple", "运行数"])), src)
    m = re.search(r"improve the source-domain fit, by ([\d.]+) to ([\d.]+) in median NRMSEP "
                  r"\(\$P\\le([\d.]+\\times10\^\{-?\d+\})\$\)", t)
    if not m:
        check("S14 其余四基准源域改善句可解析", "True", "False", "Supplementary_en.tex")
    else:
        other = d.drop(index="apple")
        gain = other["源域中位_基线"] - other["源域中位_8000步"]
        _numchk("S14 源域改善下界", m.group(1), float(gain.min()), src)
        _numchk("S14 源域改善上界", m.group(2), float(gain.max()), src)
        _scichk("S14 源域改善 P 上界", f"${m.group(3)}$", float(other["源域配对P"].max()), src)
        check("S14 其余四基准源域一律改善", "True", str(bool((gain > 0).all())), src)
        check("S14 四基准目标域无一改善", "True",
              str(bool(((other["目标域中位_8000步"] - other["目标域中位_基线"]) >= 0).all())),
              f"{src}（目标域中位一律不低于基线即未改善）")


def check_seed_completeness() -> None:
    """共同场景里不足五粒种子的格——S3 的披露与两条敏感性逐个复算。

    共同场景的取法是「每个方法-校正在该场景有行」，不是「每个方法-校正在该场景有 5 粒种子」，
    所以训练出非有限值、整格不落行的那些运行（只出现在 Phys 与 Phys+BL，各自连带抹掉对应的
    +SB 格）会让一部分格按 4 粒（个别 3 粒）种子取平均。两条敏感性把它一次做完：只留全齐的场景，
    以及把每一次缺失运行按稿件自己的发散阈值 10 \\Brix 计入。
    """
    raw = pd.read_excel(XL50, sheet_name="表h：全部原始结果")
    raw["方法_校正"] = raw["方法"] + raw["校正"].map({"裸": "", "+SB": " + SB"})
    mcs = sorted(raw["方法_校正"].unique())
    seeds = sorted(raw["种子"].unique())
    per = raw.groupby(["方法_校正", "场景"])["RMSE"].mean().reset_index()
    common: set[str] = set()
    for i, mc in enumerate(mcs):
        sc = set(per[per["方法_校正"] == mc]["场景"])
        common = sc if i == 0 else (common & sc)
    cnt = raw.groupby(["场景", "方法_校正"])["种子"].nunique().reset_index()
    ins = cnt[cnt["场景"].isin(common)]
    bad = ins[ins["种子"] < len(seeds)]
    full = sorted(set(common) - set(bad["场景"]))
    src = "04outputs/50_formal_multiseed_benchmark.xlsx[表h]"
    t = _flat("Supplementary_en.tex")

    m = re.search(r"Within that subset (\d[\d\s,]*) of the (\d[\d\s,]*) method--scenario cells "
                  r"\(([\d.]+)\\%, spread over (\d+) scenarios\) average fewer than five seeds, four in all but "
                  r"(\w+) of them: those runs produced non-finite values and hence returned no row at all, (\d+) of "
                  r"them for Phys and (\d+) for Phys\+BL", t)
    if not m:
        check("S3 缺种子格披露句可解析", "True", "False", "Supplementary_en.tex")
    else:
        def _int(x: str) -> int:
            return int(x.replace(",", "").replace("\\,", "").replace(" ", ""))
        check("S3 缺种子格数", _int(m.group(1)), len(bad), src)
        check("S3 方法-场景格总数", _int(m.group(2)), len(common) * len(mcs), src)
        _numchk("S3 缺种子格占比%", m.group(3), 100 * len(bad) / (len(common) * len(mcs)), src)
        check("S3 涉及场景数", int(m.group(4)), bad["场景"].nunique(), src)
        check("S3 少于四粒种子的格数", _WORDS.index(m.group(5)), int((bad["种子"] < len(seeds) - 1).sum()), src)
        for meth, got in (("Phys", m.group(6)), ("Phys+BL", m.group(7))):
            cells = bad[bad["方法_校正"] == meth]
            check(f"S3 {meth} 无结果产出的运行数", int(got), int((len(seeds) - cells["种子"]).sum()), src)
        check("S3 缺种子格只出现在 Phys 与 Phys+BL（含其 +SB）", "True",
              str(set(bad["方法_校正"]) <= {"Phys", "Phys + SB", "Phys+BL", "Phys+BL + SB"}), src)

    def stats(frame: pd.DataFrame, scen: list[str]) -> tuple[float, float, float, float]:
        piv = frame.pivot_table(index="场景", columns="方法_校正",
                                values="RMSE", aggfunc="mean").loc[scen]
        ph, cnn = piv["Phys+BL"], piv["CNN"]
        return (float(ph.median()), float(cnn.median()),
                float((100 * (cnn - ph) / cnn).median()),
                float(wilcoxon(ph.to_numpy(), cnn.to_numpy()).pvalue))

    filled = [{"场景": sc, "方法_校正": mc, "种子": sd, "RMSE": 10.0}
              for (sc, mc), g in raw.groupby(["场景", "方法_校正"])
              for sd in (set(seeds) - set(g["种子"]))]
    raw_filled = pd.concat([raw[["场景", "方法_校正", "种子", "RMSE"]],
                            pd.DataFrame(filled)], ignore_index=True)
    check("S3 按发散阈值补齐的运行数等于缺失运行数", int((len(seeds) - bad["种子"]).sum()),
          sum(1 for f in filled if f["场景"] in common), src)

    m = re.search(r"Restricted to the (\d+) scenarios whose cells are all complete, the median RMSE of Phys\+BL is "
                  r"([\d.]+)\\,\\Brix\{\} against the CNN's ([\d.]+)\\,\\Brix\{\} \(paired median improvement "
                  r"([\d.]+)\\%, \$P=([\d.]+\\times10\^\{-?\d+\})\$\); charging every missing run at the "
                  r"10\\,\\Brix\{\} divergence threshold instead gives ([\d.]+) against ([\d.]+)\\,\\Brix\{\} "
                  r"\(([\d.]+)\\%, \$P=([\d.]+\\times10\^\{-?\d+\})\$\)", t)
    if not m:
        check("S3 两条敏感性句可解析", "True", "False", "Supplementary_en.tex")
        return
    check("S3 全齐场景数", int(m.group(1)), len(full), src)
    a = stats(raw, full)
    _numchk("S3 全齐口径 Phys+BL 中位", m.group(2), a[0], src)
    _numchk("S3 全齐口径 CNN 中位", m.group(3), a[1], src)
    _numchk("S3 全齐口径 配对改进中位%", m.group(4), a[2], src)
    _scichk("S3 全齐口径 Wilcoxon P", f"${m.group(5)}$", a[3], src)
    b = stats(raw_filled, sorted(common))
    _numchk("S3 补齐口径 Phys+BL 中位", m.group(6), b[0], src)
    _numchk("S3 补齐口径 CNN 中位", m.group(7), b[1], src)
    _numchk("S3 补齐口径 配对改进中位%", m.group(8), b[2], src)
    _scichk("S3 补齐口径 Wilcoxon P", f"${m.group(9)}$", b[3], src)


def check_letter_quotes() -> None:
    """两封回复信里的 \\RevisedExcerpt 必须逐字出自现稿——这是最容易在改稿后静默失效的一处。

    审稿人拿到的是回复信，信里说「修回稿现在这么写」；正文改一个词，这句话就变成了不实陈述，
    而编译、闸门、数字判据没有一样会报错。此前靠 nature-response 的 check_package_consistency
    手动跑；现在进 83 号，与其余判据同批。

    另判每条引文里不得出现交叉引用命令：信是独立编译的文档，\\ref 在信里解析不出来，
    写成字面「Table~3」又与稿件逐字不符——两头都不成立，只能把引文切在引用之外。
    """
    main = re.sub(r"\s+", " ", _tex("Elsevier_en.tex"))
    for letter in ("Response_R1.tex", "Response_R2.tex"):
        text = _letter(letter)
        quotes = []
        for m in re.finditer(r"\\RevisedExcerpt\{", text):
            j, depth = m.end(), 1
            while depth:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            quotes.append(text[m.end():j - 1])
        src = f"06doc/02sub/revision/{letter} vs 06doc/01manuscript/Elsevier_en.tex"
        check(f"{letter} 有逐字引文", "True", str(bool(quotes)), src)
        for i, q in enumerate(quotes, 1):
            flat = re.sub(r"\s+", " ", q).strip()
            check(f"{letter} 引文 {i} 逐字出自现稿", "True", str(flat in main), src)
            check(f"{letter} 引文 {i} 不含交叉引用命令", "True",
                  str(not re.search(r"\\(ref|autoref|eqref)\b", q)), src)


def check_delivery_spec() -> None:
    """0NREVIEWER_RESPONSE.md §10 的验收表：每一格都对着交付物本身判。

    这张表是作者打开返修包时照着核的那一份，「验收只看文件」意味着它写错一个页数，
    验收就会在没有问题的地方停下来。所以页数对 PDF、节数与浮动体数对 .tex、词数对 82 号。
    """
    trk = (BASE / "06doc/02sub/0NREVIEWER_RESPONSE.md").read_text(encoding="utf-8")
    rev = BASE / "06doc/02sub/revision"
    src = "06doc/02sub/revision/*.pdf（pdfinfo）"

    def row(name: str) -> str:
        m = re.search(r"^\|\s*`" + re.escape(name) + r"`.*$", trk, re.M)
        if not m:
            check(f"§10 验收表有 {name} 一行", "True", "False", "0NREVIEWER_RESPONSE.md")
            return ""
        return m.group(0)

    for pdf, pat in (("Elsevier_en_clean.pdf", r"\| (\d+) 页、正文 ([\d,]+) 词"),
                     ("Elsevier_en_marked.pdf", r"\| (\d+) 页 0 错"),
                     ("Supplementary_en.pdf", r"\| (\d+) 页；S1--S(\d+)")):
        line = row(pdf)
        if not line:
            continue
        m = re.search(pat.replace("--", "\u2013"), line) or re.search(pat, line)
        if not m:
            check(f"§10 {pdf} 那一格可解析", "True", "False", "0NREVIEWER_RESPONSE.md")
            continue
        check(f"§10 {pdf} 页数", int(m.group(1)), _pages(rev / pdf), src)
        if pdf == "Elsevier_en_clean.pdf":
            res = subprocess.run(
                [sys.executable, str(BASE / "02code" / "82_main_text_wordcount.py"),
                 str(MS / "Elsevier_en.tex")], capture_output=True, text=True, check=True)
            check("§10 修回稿正文词数", m.group(2),
                  f"{int(res.stdout.split(chr(9))[0]):,d}", "02code/82_main_text_wordcount.py")
        if pdf == "Supplementary_en.pdf":
            check("§10 补充材料节数", m.group(2),
                  str(_tex("Supplementary_en.tex").count("\\section{")), "Supplementary_en.tex")

    line = row("Response_R1.pdf` / `Response_R2.pdf")
    m = re.search(r"R1 (\d+) 页 Comment 1[^0-9]+(\d+)；R2 (\d+) 页 Comment 1[^0-9]+(\d+)", line)
    if not m:
        check("§10 两封回复信那一格可解析", "True", "False", "0NREVIEWER_RESPONSE.md")
    else:
        check("§10 Response_R1 页数", int(m.group(1)), _pages(rev / "Response_R1.pdf"), src)
        check("§10 Response_R2 页数", int(m.group(3)), _pages(rev / "Response_R2.pdf"), src)
        for tag, n, letter in (("R1", m.group(2), "Response_R1.tex"),
                               ("R2", m.group(4), "Response_R2.tex")):
            got = len(re.findall(r"\\ReviewerComment", _letter(letter)))
            check(f"§10 {tag} 意见条数", n, str(got), f"06doc/02sub/revision/{letter}")

    line = row("CoverLetter_R1.pdf")
    m = re.search(r"\| (\d+) 页；稿号", line)
    if not m:
        check("§10 cover letter 那一格可解析", "True", "False", "0NREVIEWER_RESPONSE.md")
    else:
        check("§10 CoverLetter_R1 页数", int(m.group(1)), _pages(rev / "CoverLetter_R1.pdf"), src)

    line = row("Elsevier_zh_clean.pdf` / `Supplementary_zh.pdf")
    m = re.search(r"页数 (\d+) / (\d+)", line)
    if not m:
        check("§10 中文同步稿那一格可解析", "True", "False", "0NREVIEWER_RESPONSE.md")
    else:
        check("§10 Elsevier_zh_clean 页数", int(m.group(1)), _pages(rev / "Elsevier_zh_clean.pdf"), src)
        check("§10 Supplementary_zh 页数", int(m.group(2)), _pages(rev / "Supplementary_zh.pdf"), src)

    # 正文与补充材料的浮动体数、文献条数
    main, si = _tex("Elsevier_en.tex"), _tex("Supplementary_en.tex")
    line = row("Elsevier_en_clean.pdf")
    m = re.search(r"正文 (\d+) 表 (\d+) 图", line)
    if m:
        check("§10 正文表数", m.group(1), str(len(re.findall(r"\\label\{tab:", main))), "Elsevier_en.tex")
        check("§10 正文图数", m.group(2), str(len(re.findall(r"\\label\{fig:", main))), "Elsevier_en.tex")
    m = re.search(r"文献 (\d+) 条", line)
    if m:
        check("§10 文献条数", m.group(1), str(main.count("\\bibitem")), "Elsevier_en.tex")
    line = row("Supplementary_en.pdf")
    m = re.search(r"表 S1--S(\d+)、图 S1--S(\d+)", line.replace("\u2013", "--"))
    if m:
        check("§10 补充材料表数", m.group(1), str(len(re.findall(r"\\label\{tab:", si))), "Supplementary_en.tex")
        check("§10 补充材料图数", m.group(2), str(len(re.findall(r"\\label\{fig:", si))), "Supplementary_en.tex")


def check_pair_level() -> None:
    """§3.1 的第三项稳健性：把推断单位由任务换成源–目标域对之后还剩什么。

    这一段是本轮把「126 个任务只落在 98 个域对、21 个目标域上」这件事做成可核结果的地方，
    δ 区间、三个保号的 P 上界与仪器基准上失去显著性这一条，全部对 101 号的工作簿。
    """
    d = pd.read_excel(XL101)
    src = "04outputs/101_pair_level_inference.xlsx（域对级折叠后重算）"
    main = _flat("Elsevier_en.tex")
    check("§3.1 域对级一句指向 S16", "True",
          str("with the domain pair as the unit (98 pairs) every direction holds (script 101; "
              "Supplementary Section~S16)" in main), "Elsevier_en.tex")
    t = _flat("Supplementary_en.tex")
    m = re.search(r"The 126 tasks fall on (\d+) domain pairs and only (\d+) target domains", t)
    if not m:
        check("S16 域对级句可解析", "True", "False", "Supplementary_en.tex")
        return
    c62, c64 = _curves()
    cur = pd.concat([c62, c64], ignore_index=True)[["task_id"]].drop_duplicates()
    pair = cur["task_id"].str.split("|").str[:2].str.join("|")
    tgt = cur["task_id"].str.extract(r"\|[^>]*->([^|]*)\|")[0]
    check("§3.1 域对数", m.group(1), str(pair.nunique()), "62/64 号 curves 的 task_id")
    check("§3.1 目标域数", m.group(2), str(tgt.nunique()), "62/64 号 curves 的 task_id")

    m = re.search(r"with family-mean \$\\delta\$ of \$\+([\d.]+)\$ to \$\+([\d.]+)\$ for instrument, "
                  r"\$\+([\d.]+)\$ to \$\+([\d.]+)\$ for season, \$\+([\d.]+)\$ to \$\+([\d.]+)\$ "
                  r"for origin/year and \$-([\d.]+)\$ to \$-([\d.]+)\$ for soil at positive budgets", t)
    if not m:
        check("§3.1 域对级 δ 区间句可解析", "True", "False", "Elsevier_en.tex")
    else:
        spec = (("instrument", 1, 2, False), ("season", 3, 4, False),
                ("origin_year_instrument", 5, 6, False), ("lab", 7, 8, True))
        for shift, lo, hi, positive_only in spec:
            s = d[d["shift_type"] == shift]
            if positive_only:
                s = s[s["n_cal"] > 0]
            vals = s["delta_pair"].astype(float)
            want = (-vals.max(), -vals.min()) if positive_only else (vals.min(), vals.max())
            _numchk(f"§3.1 域对级 δ 下界 {shift}", m.group(lo), want[0], src)
            _numchk(f"§3.1 域对级 δ 上界 {shift}", m.group(hi), want[1], src)

    m = re.search(r"The significance survives for season \(\$P\\le([\d.]+)\$\), origin/year "
                  r"\(\$P<10\^\{-(\d+)\}\$\) and soil above zero labels \(\$P=([\d.]+)\$\), "
                  r"but not on the instrument benchmarks, which collapse to (\w+) "
                  r"pairs \(\$P=([\d.]+)\$ at every positive budget\)", t)
    if not m:
        check("§3.1 域对级显著性句可解析", "True", "False", "Elsevier_en.tex")
        return
    se = d[d["shift_type"] == "season"]["p_pair"].astype(float)
    check("§3.1 域对级 季节 P 上界成立", "True", str(se.max() <= float(m.group(1))),
          f"{src}（实际 Pmax={se.max():.4f}）")
    oy = d[d["shift_type"] == "origin_year_instrument"]["p_pair"].astype(float)
    check("§3.1 域对级 产地/年 P 上界成立", "True", str(oy.max() < 10.0 ** -int(m.group(2))),
          f"{src}（实际 Pmax={oy.max():.3g}）")
    lab = d[(d["shift_type"] == "lab") & (d["n_cal"] > 0)]["p_pair"].astype(float)
    _numchk("§3.1 域对级 土壤正预算 P", m.group(3), float(lab.max()), src)
    words = {"six": 6, "eight": 8, "twelve": 12, "twenty-one": 21}
    inst = d[d["shift_type"] == "instrument"]
    check("§3.1 仪器基准折成的域对数", str(words[m.group(4)]), str(int(inst["n_pair"].iloc[0])), src)
    pos = inst[inst["n_cal"] > 0]["p_pair"].astype(float)
    _numchk("§3.1 仪器基准正预算域对级 P", m.group(5), float(pos.max()), src)
    check("§3.1 仪器基准正预算域对级 P 处处相同", "True", str(pos.nunique() == 1), src)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="写入 refine-logs/forensics_receipts/")
    args = ap.parse_args()

    check_candidate_sets()
    check_s14()
    check_case_study()
    check_table4()
    check_table5()
    check_table6()
    check_tables_bilingual()
    check_universe_sensitivity()
    check_prescription_pairs_and_deep_level()
    check_runs_per_cell()
    check_protocol_constants()
    check_win_rates()
    check_bilingual_numbers()
    check_shift_structure()
    check_deep_platform()
    check_verdict_items()
    check_verdict4_items()
    check_forensics_7550()
    check_split_disclosure()
    check_forensics_281614f()
    check_correlations()
    check_structure_headlines_and_soil()
    check_season_scaler_s2()
    check_unified_season_scaler()
    check_prose_ranges()
    check_public_only()
    check_ds()
    check_marked_equivalence()
    check_note_and_counts()
    check_s14_apple_reference()
    check_letter_bibliography()
    check_si_crosschecks()
    check_pair_level()
    check_delivery_spec()
    check_budget_sensitivity()
    check_seed_completeness()
    check_letter_quotes()
    check_highlights()
    check_letter_ordinals()
    check_script_refs()
    check_wordcount()
    check_bibliography()

    n_fail = sum(1 for v, *_ in _rows if v == "FAIL")
    head = [
        "# 返修二轮新增数字确定性复算收据（冻结，写完不再改）",
        f"时间：{_dt.datetime.now().isoformat(timespec='seconds')}",
        "证据文件：" + "；".join(
            f"{p.relative_to(BASE)} sha256[:16]={sha16(p)}" for p in (XL62, XL63, XL64)),
        # 四份稿都要留哈希：只钉正文，补充材料改了收据也发现不了
        "被核稿件（此后任一份再改须另开收据）：" + "；".join(
            f"{n} sha256[:16]={sha16(MS / n)}"
            for n in ("Elsevier_en.tex", "Elsevier_zh.tex",
                      "Supplementary_en.tex", "Supplementary_zh.tex")),
        "被核回复信：" + "；".join(
            f"{n} sha256[:16]={sha16(BASE / '06doc/02sub/revision' / n)}"
            for n in ("Response_R1.tex", "Response_R2.tex", "CoverLetter_R1.tex")),
        f"判决：{'ALL PASS' if n_fail == 0 else f'{n_fail} FAIL'}（{len(_rows) - n_fail}/{len(_rows)}）",
        "",
    ]
    body = [f"{v}\t{c}\t{g}\t{s}" for v, c, g, s in _rows]
    text = "\n".join(head + body) + "\n"
    print(text)
    if args.write:
        dst = (BASE / "refine-logs" / "forensics_receipts"
               / f"{_dt.date.today().isoformat()}_revision_round2_numbers.txt")
        dst.write_text(text, encoding="utf-8")
        print(f"→ {dst.relative_to(BASE)}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
