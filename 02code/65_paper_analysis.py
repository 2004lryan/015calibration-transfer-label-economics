"""
65_paper_analysis.py: 合并 62 经典 + 64 深度,产论文核心表(清算/相图/负律/配对检验)。

统计(§7.2): 以任务为聚类单元的 cluster bootstrap 95%CI + 按任务配对 Wilcoxon
    signed-rank(简单 vs 深度) + Cliff's δ 效应量。
输入: 04outputs/62_crossover_engine.xlsx(classical) + 64_deep_transfer_server.xlsx(deep)
    + 63_crossover_analysis.xlsx(负律)
输出: 04outputs/65_paper_analysis.xlsx — reckoning/simple_vs_deep/phase/nstar_law/headline
    05logs/65_paper_analysis_*.log
运行: python 02code/65_paper_analysis.py
"""
import importlib.util
import os

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import wilcoxon

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, ".."))
_OUT = os.path.join(_BASE, "04outputs")
_es = importlib.util.spec_from_file_location("export_utils", os.path.join(_CODE, "01_export_utils.py"))
assert _es is not None  # 对已知存在的 .py 文件 spec 恒非 None
assert _es.loader is not None  # 同理 loader 恒非 None
_eu = importlib.util.module_from_spec(_es)
_es.loader.exec_module(_eu)

SIMPLE = ["coral", "sbc"]
CLASSIC_EXP = ["model_update", "target_only"]
DEEP = ["cnn_zeroshot", "cnn_finetune", "dann", "deepcoral", "physbl_zeroshot", "physbl_ft"]
NCAL = [0, 5, 10, 20, 40]
RNG = np.random.default_rng(20060515)


def cluster_boot_ci(vals: npt.ArrayLike, n: int = 2000) -> tuple[float, float]:
    """任务级 cluster bootstrap 95%CI(vals=每任务一值)。"""
    arr = np.asarray(vals, float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 3:
        return (np.nan, np.nan)
    bs = [np.median(RNG.choice(arr, len(arr), replace=True)) for _ in range(n)]
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def cliffs_delta(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    aa, bb = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in aa for y in bb)
    lt = sum((x < y) for x in aa for y in bb)
    return float((gt - lt) / (len(aa) * len(bb) + 1e-9))


def task_median(df: pd.DataFrame) -> pd.DataFrame:
    """每(任务,方法,n_cal)跨种子中位 RMSEP + RPD。"""
    g = df.groupby(["benchmark", "shift_type", "task_id", "method", "n_cal"]).agg(
        rmsep=("rmsep", "median")).reset_index()
    return g


def main() -> None:
    log = _eu.get_logger("65_paper_analysis")
    cl = pd.read_excel(os.path.join(_OUT, "62_crossover_engine.xlsx"), sheet_name="curves")
    feats = pd.read_excel(os.path.join(_OUT, "62_crossover_engine.xlsx"), sheet_name="features")[
        ["task_id", "y_std_tgt"]].drop_duplicates("task_id")
    dp = pd.read_excel(os.path.join(_OUT, "64_deep_transfer_server.xlsx"), sheet_name="curves")
    _eu.log_experiment_header(log, {"任务": "论文核心表", "经典行": len(cl), "深度行": len(dp)})

    a = pd.concat([task_median(cl), task_median(dp)], ignore_index=True)
    a = a.merge(feats, on="task_id", how="left")
    a["rpd"] = a["y_std_tgt"] / a["rmsep"]

    # ---- 表1 reckoning: shift×n_cal×方法组 中位RMSEP + CI ----
    reck = []
    for st in a["shift_type"].unique():
        for nc in NCAL:
            sub = a[(a["shift_type"] == st) & (a["n_cal"] == nc)]
            for grp, meths in [("简单经典", SIMPLE), ("经典昂贵", CLASSIC_EXP), ("深度物理", DEEP)]:
                # 每任务取该组最优(最小RMSEP)
                per_task = (sub[sub["method"].isin(meths)]
                            .groupby("task_id")["rmsep"].min())
                if len(per_task) == 0:
                    continue
                lo, hi = cluster_boot_ci(per_task.values)
                reck.append({"shift_type": st, "n_cal": nc, "group": grp, "n_task": len(per_task),
                                 "median_rmsep": round(float(per_task.median()), 3),
                                 "ci95": f"[{lo:.3f},{hi:.3f}]"})
    reck = pd.DataFrame(reck)

    # ---- 表2 simple_vs_deep: 每shift×n_cal 最优简单 vs 最优深度,配对Wilcoxon ----
    svd = []
    for st in a["shift_type"].unique():
        for nc in NCAL:
            sub = a[(a["shift_type"] == st) & (a["n_cal"] == nc)]
            simp = sub[sub["method"].isin(SIMPLE)].groupby("task_id")["rmsep"].min()
            deep = sub[sub["method"].isin(DEEP)].groupby("task_id")["rmsep"].min()
            common = simp.index.intersection(deep.index)
            if len(common) < 5:
                continue
            s, d = simp[common].values, deep[common].values
            try:
                _, p = wilcoxon(s, d)
            except Exception:
                p = np.nan
            svd.append({"shift_type": st, "n_cal": nc, "n_task": len(common),
                            "simple_med": round(float(np.median(s)), 3), "deep_med": round(float(np.median(d)), 3),
                            "simple_win_frac": round(float(np.mean(s < d)), 2),
                            "median_diff": round(float(np.median(d - s)), 3),
                            "cliffs_delta": round(cliffs_delta(d, s), 2),
                            "wilcoxon_p": None if np.isnan(p) else round(float(p), 4)})
    svd = pd.DataFrame(svd)

    # ---- 表3 phase: 各shift×n_cal 获胜方法(全方法) ----
    allm = SIMPLE + CLASSIC_EXP + DEEP + ["zero_shot", "pds"]
    phase = []
    for st in a["shift_type"].unique():
        for nc in [0, 5, 10, 20, 40]:
            sub = a[(a["shift_type"] == st) & (a["n_cal"] == nc) & (a["method"].isin(allm))]
            if len(sub) == 0:
                continue
            win = sub.loc[sub.groupby("task_id")["rmsep"].idxmin()]
            vc = win["method"].value_counts(normalize=True)
            phase.append({"shift_type": st, "n_cal": nc, "top_method": vc.index[0],
                              "top_frac": round(float(vc.iloc[0]), 2), "n_task": win["task_id"].nunique()})
    phase = pd.DataFrame(phase)

    # ---- 负律(从63) ----
    try:
        law = pd.read_excel(os.path.join(_OUT, "63_crossover_analysis.xlsx"), sheet_name="scaling_fit")
        lobo = pd.read_excel(os.path.join(_OUT, "63_crossover_analysis.xlsx"), sheet_name="lobo")
    except Exception:
        law, lobo = pd.DataFrame(), pd.DataFrame()

    # ---- headline 关键数字 ----
    hl = []
    for st in a["shift_type"].unique():
        d0 = a[(a["shift_type"] == st) & (a["n_cal"] == 0)]
        zs = d0[d0["method"] == "zero_shot"]["rpd"].median()
        co = d0[d0["method"] == "coral"]["rpd"].median()
        hl.append({"shift_type": st, "coral0_rpd": round(float(co), 2) if np.isfinite(co) else None,
                       "zeroshot0_rpd": round(float(zs), 2) if np.isfinite(zs) else None})
    hl = pd.DataFrame(hl)

    for nm, df in [("reckoning", reck), ("simple_vs_deep", svd), ("phase", phase), ("headline", hl)]:
        log.log(f"\n===== {nm} =====\n" + df.to_string(index=False))
    log.log("\n===== 负律 scaling_fit =====\n" + (law.to_string(index=False) if len(law) else "无"))

    _eu.write_script_workbook(__file__, {"reckoning": reck, "simple_vs_deep": svd, "phase": phase,
                                         "nstar_law": law, "lobo": lobo, "headline": hl})
    print("论文核心表 → 04outputs/65_paper_analysis.xlsx")
    print("\nsimple_vs_deep:\n", svd.to_string(index=False))
    print("\nheadline:\n", hl.to_string(index=False))


if __name__ == "__main__":
    main()
