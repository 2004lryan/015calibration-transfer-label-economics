"""
75_nrmsep_table_columns.py: 为表 2（oracle 包络）与表 3（无选择对比）补归一化端点 NRMSEP 列（返修 CILS R1.4）

【为什么必须做这个实验】
    审稿人 1：各基准用各自原单位报 RMSEP，跨基准、跨分析物难比，建议在汇总分析里并列 NRMSE 一类
    归一化指标。68 号已算族均 NRMSEP，但投出稿表 2（每任务最优简单 vs 最优深度的包络）与表 3
    （代表对 CORAL vs DeepCORAL、族均）报的都是原单位中位数。本脚本用同一批 62/64 曲线、同一口径
    （跨种子×重复中位；NRMSEP = RMSEP/σ_tgt = 1/RPD），逐行补出表 2、表 3 的 simple/deep NRMSEP 中位数，
    以及图 3（清算曲线）各面板的 NRMSEP 版曲线数据。不重训、不改任何结论。

运行方式:
    cd <项目根目录绝对路径>
    python 02code/75_nrmsep_table_columns.py

输出文件:
    04outputs/75_nrmsep_table_columns.xlsx — table2_envelope / table3_fair / reckoning_curves_nrmsep
    05logs/75_nrmsep_table_columns_YY-MM-DD_HHMMSS.log
"""

import importlib.util
import os

import numpy as np
import pandas as pd

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
# 投出稿表 2 的行（漂移类型 × 预算），原文顺序
TABLE2_ROWS = [
    ("instrument", 0),
    ("instrument", 20),
    ("instrument", 40),
    ("season", 10),
    ("season", 20),
    ("origin_year_instrument", 0),
    ("origin_year_instrument", 5),
    ("origin_year_instrument", 20),
    ("lab", 10),
]
TABLE3_FAMILY_ROWS = [
    ("instrument", 0),
    ("instrument", 20),
    ("season", 20),
    ("origin_year_instrument", 0),
    ("origin_year_instrument", 20),
    ("lab", 20),
]


def _load() -> pd.DataFrame:
    cl = pd.read_excel(os.path.join(_OUT, "62_crossover_engine.xlsx"), sheet_name="curves")
    dp = pd.read_excel(os.path.join(_OUT, "64_deep_transfer_server.xlsx"), sheet_name="curves")
    allr = pd.concat([cl, dp], ignore_index=True)
    tm = allr.groupby(["benchmark", "shift_type", "task_id", "method", "n_cal"], as_index=False)["rmsep"].median()
    ns = pd.read_excel(os.path.join(_OUT, "63_crossover_analysis.xlsx"), sheet_name="nstar_per_task")
    tm = tm.merge(ns[["task_id", "y_std_tgt"]].drop_duplicates("task_id"), on="task_id", how="left")
    tm["nrmsep"] = tm["rmsep"] / tm["y_std_tgt"]
    return tm


def _envelope(tm: pd.DataFrame, st: str, nc: int) -> dict[str, object]:
    sub = tm[(tm["shift_type"] == st) & (tm["n_cal"] == nc)]
    s = sub[sub["method"].isin(SIMPLE)].groupby("task_id")["nrmsep"].min()
    d = sub[sub["method"].isin(DEEP)].groupby("task_id")["nrmsep"].min()
    sr = sub[sub["method"].isin(SIMPLE)].groupby("task_id")["rmsep"].min()
    dr = sub[sub["method"].isin(DEEP)].groupby("task_id")["rmsep"].min()
    common = s.index.intersection(d.index)
    return {
        "shift_type": st,
        "n_cal": nc,
        "n_task": len(common),
        "simple_rmsep_med": round(float(sr[common].median()), 3),
        "deep_rmsep_med": round(float(dr[common].median()), 3),
        "simple_nrmsep_med": round(float(s[common].median()), 3),
        "deep_nrmsep_med": round(float(d[common].median()), 3),
        "simple_win_frac_nrmsep": round(float(np.mean(s[common].values < d[common].values)), 2),
    }


def _family_mean(tm: pd.DataFrame, st: str, nc: int) -> dict[str, object]:
    sub = tm[(tm["shift_type"] == st) & (tm["n_cal"] == nc)]
    s = sub[sub["method"].isin(SIMPLE)].groupby("task_id")["nrmsep"].mean()
    d = sub[sub["method"].isin(DEEP)].groupby("task_id")["nrmsep"].mean()
    sr = sub[sub["method"].isin(SIMPLE)].groupby("task_id")["rmsep"].mean()
    dr = sub[sub["method"].isin(DEEP)].groupby("task_id")["rmsep"].mean()
    common = s.index.intersection(d.index)
    return {
        "comparison": "Family mean",
        "shift_type": st,
        "n_cal": nc,
        "n_task": len(common),
        "simple_rmsep_med": round(float(sr[common].median()), 3),
        "deep_rmsep_med": round(float(dr[common].median()), 3),
        "simple_nrmsep_med": round(float(s[common].median()), 3),
        "deep_nrmsep_med": round(float(d[common].median()), 3),
    }


def _pair(tm: pd.DataFrame, st: str) -> dict[str, object]:
    sub = tm[(tm["shift_type"] == st) & (tm["n_cal"] == 0)]
    c = sub[sub["method"] == "coral"].set_index("task_id")
    d = sub[sub["method"] == "deepcoral"].set_index("task_id")
    common = c.index.intersection(d.index)
    return {
        "comparison": "CORAL vs. DeepCORAL",
        "shift_type": st,
        "n_cal": 0,
        "n_task": len(common),
        "simple_rmsep_med": round(float(c.loc[common, "rmsep"].median()), 3),
        "deep_rmsep_med": round(float(d.loc[common, "rmsep"].median()), 3),
        "simple_nrmsep_med": round(float(c.loc[common, "nrmsep"].median()), 3),
        "deep_nrmsep_med": round(float(d.loc[common, "nrmsep"].median()), 3),
    }


def _curves(tm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for st in ["instrument", "season", "origin_year_instrument", "lab"]:
        for nc in NCAL:
            sub = tm[(tm["shift_type"] == st) & (tm["n_cal"] == nc)]
            for grp, meths in [("best simple", SIMPLE), ("classical update", CLASSIC_EXP), ("best deep/physics", DEEP)]:
                g = sub[sub["method"].isin(meths)].groupby("task_id")["nrmsep"].min()
                if len(g) == 0:
                    continue
                rows.append(
                    {
                        "shift_type": st,
                        "n_cal": nc,
                        "group": grp,
                        "n_task": len(g),
                        "median_nrmsep": round(float(g.median()), 3),
                        "q25": round(float(g.quantile(0.25)), 3),
                        "q75": round(float(g.quantile(0.75)), 3),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    logger = _eu.get_logger(os.path.basename(__file__).replace(".py", ""))
    tm = _load()
    t2 = pd.DataFrame([_envelope(tm, st, nc) for st, nc in TABLE2_ROWS])
    t3 = pd.DataFrame(
        [_pair(tm, st) for st in ["instrument", "season", "origin_year_instrument", "lab"]]
        + [_family_mean(tm, st, nc) for st, nc in TABLE3_FAMILY_ROWS]
    )
    cv = _curves(tm)
    logger.log("表2 NRMSEP 列:\n%s", t2.to_string(index=False))
    logger.log("表3 NRMSEP 列:\n%s", t3.to_string(index=False))
    path = _eu.write_script_workbook(
        __file__, {0: ("table2_envelope", t2), "table3_fair": t3, "reckoning_curves_nrmsep": cv}
    )
    logger.log("落盘: %s", path)
    print(t2.to_string(index=False))
    print()
    print(t3.to_string(index=False))
    print()
    print(cv.to_string(index=False))


if __name__ == "__main__":
    main()
