"""
68_normalized_hierarchical_reanalysis.py: 归一化端点(NRMSEP)+基准级(层次)重分析

背景:
    外审第二轮最严格的一份意见提两点 CRITICAL,均为对已落地结果的重新分析
    (不重训任何模型):
      #1 主分析在异量纲任务上池化原始 RMSEP 不可比 → 改用无量纲 NRMSEP = RMSEP/σ_tgt
         = 1/RPD 复算简单 vs 深度族对比,检验结论是否依赖量纲。
      #3 有效独立单元远小于 126(基准内伪重复) → 以基准为聚类单位做层次/基准级推断,
         每基准等权,报 5 基准级配对差与基准级 bootstrap CI。

    σ_tgt(每任务目标域标签标准差)取自 63_crossover_analysis 的 y_std_tgt(126 任务全有,
    task_id 唯一)。NRMSEP 越小越好(= 1/RPD)。

    结论(见输出):两项重分析均支持与原始 RMSEP 相同的定性结论——简单族在仪器/季节基准
    NRMSEP 更小,苹果(产地/年)近零标签处无偏对比亦轻favor简单,土壤不可区分;基准级(n=5)
    配对差方向一致,但按基准等权后 P 值因 n=5 而弱,故只作"基准级方向一致"的保守陈述,
    不推小 P 值——与讨论局限段"层次推断受 n=5 限"一致。

运行方式:
    cd <项目根目录绝对路径>
    python 02code/68_normalized_hierarchical_reanalysis.py

输出文件:
    04outputs/68_normalized_hierarchical_reanalysis.xlsx
        — nrmsep_family(归一化族对比) + benchmark_level(基准级配对差+bootstrap)
    05logs/68_normalized_hierarchical_reanalysis_YY-MM-DD_HHMMSS.log
"""
import importlib.util
import os

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, ".."))
_OUT = os.path.join(_BASE, "04outputs")
_es = importlib.util.spec_from_file_location("export_utils", os.path.join(_CODE, "01_export_utils.py"))
assert _es is not None  # 对已知存在的 .py 文件 spec 恒非 None
assert _es.loader is not None  # 同理 loader 恒非 None
_eu = importlib.util.module_from_spec(_es)
_es.loader.exec_module(_eu)

SIMPLE = ["coral", "sbc"]
DEEP = ["cnn_zeroshot", "cnn_finetune", "dann", "deepcoral", "physbl_zeroshot", "physbl_ft"]
NCAL = [0, 5, 10, 20, 40]
SHIFTS = ["instrument", "season", "origin_year_instrument", "lab"]
RNG = np.random.default_rng(20060515)


def _cliff(diff: npt.NDArray[np.float64]) -> float:
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    return float((np.sum(d > 0) - np.sum(d < 0)) / len(d)) if len(d) else float("nan")


def _load_nrmsep() -> pd.DataFrame:
    """每方法每任务 NRMSEP = 跨种子中位 RMSEP / σ_tgt。"""
    cl = pd.read_excel(os.path.join(_OUT, "62_crossover_engine.xlsx"))
    dp = pd.read_excel(os.path.join(_OUT, "64_deep_transfer_server.xlsx"))
    allr = pd.concat([cl, dp], ignore_index=True)
    tm = allr.groupby(["benchmark", "shift_type", "task_id", "method", "n_cal"], as_index=False)["rmsep"].median()
    ns = pd.read_excel(os.path.join(_OUT, "63_crossover_analysis.xlsx"), sheet_name="nstar_per_task")
    sig = ns[["task_id", "y_std_tgt"]].drop_duplicates("task_id")
    tm = tm.merge(sig, on="task_id", how="left")
    tm["nrmsep"] = tm["rmsep"] / tm["y_std_tgt"]
    return tm


def _family_row(tm: pd.DataFrame, st: str, nc: int) -> dict[str, object] | None:
    sub = tm[(tm.shift_type == st) & (tm.n_cal == nc)]
    piv = sub.pivot_table(index="task_id", columns="method", values="nrmsep")
    sc = [m for m in SIMPLE if m in piv.columns]
    dc = [m for m in DEEP if m in piv.columns]
    if not sc or not dc:
        return None
    sm, dm = piv[sc].mean(axis=1).values, piv[dc].mean(axis=1).values
    m = np.isfinite(sm) & np.isfinite(dm)
    sm, dm = sm[m], dm[m]
    if len(sm) < 3:
        return None
    diff = dm - sm  # >0 表示简单法 NRMSEP 更小(更优)
    try:
        p = float(stats.wilcoxon(diff).pvalue)
    except ValueError:
        p = float("nan")
    return {"shift_type": st, "n_cal": nc, "n_task": len(sm),
            "simple_nrmsep_med": round(float(np.median(sm)), 3),
            "deep_nrmsep_med": round(float(np.median(dm)), 3),
            "cliffs_delta": round(_cliff(diff), 3),
            "simple_win_frac": round(float((diff > 0).mean()), 3),
            "wilcoxon_p": p}


def _benchmark_level(tm: pd.DataFrame, nc: int) -> tuple[pd.DataFrame, dict[str, object]]:
    """每基准的配对(深度族均值 - 简单族均值)NRMSEP 差,再按基准等权 bootstrap。"""
    rows = []
    for bench in sorted(tm["benchmark"].unique()):
        sub = tm[(tm.benchmark == bench) & (tm.n_cal == nc)]
        piv = sub.pivot_table(index="task_id", columns="method", values="nrmsep")
        sc = [m for m in SIMPLE if m in piv.columns]
        dc = [m for m in DEEP if m in piv.columns]
        if not sc or not dc:
            continue
        diff = (piv[dc].mean(axis=1) - piv[sc].mean(axis=1)).dropna()
        if len(diff) == 0:
            continue
        rows.append({"benchmark": bench, "n_task": len(diff),
                     "mean_diff_deep_minus_simple": round(float(diff.mean()), 3),
                     "simple_better": bool(diff.mean() > 0)})
    bdf = pd.DataFrame(rows)
    # 基准级 bootstrap:对 5 个基准级均值重采样(每基准等权)
    per_bench = bdf["mean_diff_deep_minus_simple"].values
    boots = [float(np.mean(RNG.choice(per_bench, len(per_bench), replace=True))) for _ in range(5000)]
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    summary: dict[str, object] = {"n_benchmarks": len(bdf),
               "benchmark_balanced_mean_diff": round(float(np.mean(per_bench)), 3),
               "boot95_lo": round(lo, 3), "boot95_hi": round(hi, 3),
               "n_benchmarks_simple_better": int(bdf["simple_better"].sum())}
    return bdf, summary


def main() -> None:
    logger = _eu.get_logger(os.path.basename(__file__).replace(".py", ""))
    tm = _load_nrmsep()
    fam = pd.DataFrame([r for st in SHIFTS for nc in NCAL if (r := _family_row(tm, st, nc)) is not None])
    logger.log("NRMSEP 归一化族对比:\n%s", fam.to_string(index=False))

    bench_rows, summaries = [], []
    for nc in [0, 20]:
        bdf, summ = _benchmark_level(tm, nc)
        bdf = bdf.assign(n_cal=nc)
        bench_rows.append(bdf)
        summaries.append({"n_cal": nc, **summ})
    bench = pd.concat(bench_rows, ignore_index=True)
    summ_df = pd.DataFrame(summaries)
    logger.log("基准级配对差(每基准):\n%s", bench.to_string(index=False))
    logger.log("基准级 bootstrap 汇总:\n%s", summ_df.to_string(index=False))

    path = _eu.write_script_workbook(__file__, {
        0: ("nrmsep_family", fam),
        "benchmark_level": bench,
        "benchmark_summary": summ_df,
    })
    logger.log("落盘: %s", path)
    print(fam.to_string(index=False))
    print("\n基准级配对差:\n", bench.to_string(index=False))
    print("\n基准级 bootstrap:\n", summ_df.to_string(index=False))
    print("\n注: cliffs_delta>0 / simple_win_frac>0.5 / mean_diff>0 均表示简单族 NRMSEP 更小(更优);"
          "NRMSEP=RMSEP/σ_tgt=1/RPD,无量纲可跨基准比;基准级 n=5 故只作方向一致的保守陈述。")


if __name__ == "__main__":
    main()
