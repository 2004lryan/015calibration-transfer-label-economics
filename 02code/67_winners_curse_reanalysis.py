"""
67_winners_curse_reanalysis.py: winner's curse 无偏复核——预先指定代表法与全族均值对比

背景:
    fusion 审稿一致指出:主表(reckoning/Table 3)的"最优简单 vs 最优深度"是每任务
    事后取族内 min(envelope),且简单族 2 法、深度族 6 法数量不等,min 选择对深度族更
    有利,构成 winner's curse。本脚本用同一批已落地结果(62 classical + 64 deep,不重训
    任何模型)重算两个不含 min 选择偏差的对比,检验"简单 >= 深度"是否稳健:
      (a) 预先指定代表对:CORAL vs DeepCORAL(同为二阶矩对齐,最公平的免标签同类对比);
      (b) 全族均值:每任务对族内所有方法取均值(而非 min)再配对。

    结论(见输出):两种无偏对比均保持甚至强化原结论——尤其 origin/year 近零标签处,
    envelope 显示深度微优(delta=-0.09),但代表对(delta=+0.28)与全族均值(delta=+0.25)
    均显示简单更优,证明 envelope 的近零标签深度优势是 6 法深度族的选择性假象。

运行方式:
    cd <项目根目录绝对路径>
    python 02code/67_winners_curse_reanalysis.py

输出文件:
    04outputs/67_winners_curse_reanalysis.xlsx — fair_comparison(族均值)+representative(代表对)
    05logs/67_winners_curse_reanalysis_YY-MM-DD_HHMMSS.log — 运行日志
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


def _cliff_paired(diff: npt.NDArray[np.float64]) -> float:
    """配对优势度:diff = deep - simple,>0 表示简单法更优(误差更小)。"""
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    return float((np.sum(d > 0) - np.sum(d < 0)) / len(d)) if len(d) else float("nan")


def _paired_row(simple_vals: npt.NDArray[np.float64], deep_vals: npt.NDArray[np.float64]) -> dict[str, float] | None:
    m = np.isfinite(simple_vals) & np.isfinite(deep_vals)
    sv, dv = simple_vals[m], deep_vals[m]
    if len(sv) < 3:
        return None
    diff = dv - sv
    try:
        p = float(stats.wilcoxon(diff).pvalue)
    except ValueError:
        p = float("nan")
    return {
        "n_task": len(sv),
        "simple_med": round(float(np.median(sv)), 3),
        "deep_med": round(float(np.median(dv)), 3),
        "cliffs_delta": round(_cliff_paired(diff), 3),
        "simple_win_frac": round(float((diff > 0).mean()), 3),
        "wilcoxon_p": p,
    }


def main() -> None:
    logger = _eu.get_logger(os.path.basename(__file__).replace(".py", ""))
    cl = pd.read_excel(os.path.join(_OUT, "62_crossover_engine.xlsx"))
    dp = pd.read_excel(os.path.join(_OUT, "64_deep_transfer_server.xlsx"))
    allr = pd.concat([cl, dp], ignore_index=True)
    # 每任务跨种子中位(与 65 号主分析同口径)
    tm = allr.groupby(["shift_type", "task_id", "method", "n_cal"], as_index=False)["rmsep"].median()

    fair_rows, rep_rows = [], []
    for st in SHIFTS:
        for nc in NCAL:
            sub = tm[(tm.shift_type == st) & (tm.n_cal == nc)]
            piv = sub.pivot_table(index="task_id", columns="method", values="rmsep")
            # (b) 全族均值(不取 min)
            sc = [m for m in SIMPLE if m in piv.columns]
            dc = [m for m in DEEP if m in piv.columns]
            if sc and dc:
                r = _paired_row(piv[sc].mean(axis=1).values, piv[dc].mean(axis=1).values)
                if r:
                    fair_rows.append({"shift_type": st, "n_cal": nc, **r})
            # (a) 预先指定代表对 CORAL vs DeepCORAL(免标签同类,仅 n_cal=0 有意义)
            if "coral" in piv.columns and "deepcoral" in piv.columns:
                r = _paired_row(piv["coral"].values, piv["deepcoral"].values)
                if r:
                    rep_rows.append({"shift_type": st, "n_cal": nc, "pair": "CORAL vs DeepCORAL", **r})

    fair = pd.DataFrame(fair_rows)
    rep = pd.DataFrame(rep_rows)
    logger.log("全族均值对比(无 min 选择偏差):\n%s", fair.to_string(index=False))
    logger.log("代表对 CORAL vs DeepCORAL:\n%s", rep.to_string(index=False))
    path = _eu.write_script_workbook(__file__, {0: ("fair_comparison", fair), "representative": rep})
    logger.log("落盘: %s", path)
    print(fair.to_string(index=False))
    print("\n", rep.to_string(index=False))
    print("\n注: cliffs_delta>0 与 simple_win_frac>0.5 表示简单法误差更小;两对比均无 min 选择偏差。")


if __name__ == "__main__":
    main()
