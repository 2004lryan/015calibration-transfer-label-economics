"""
63_crossover_analysis.py: 从 62 号曲线定位 n*、拟合标度律 n*≍C·D^α、LOBO 外推验证（v3 科学payoff）

面板生死点：n* 到底可不可预测？本脚本诚实回答——
    ① 每任务算多种 n* 定义(labels-to-usable / simple-suffices / source-vs-target)
    ② n* 与漂移特征(MMD/W1/维度/SNR)的相关 + log-log 标度律拟合(R², 指数α)
    ③ 留一基准外推(LOBO):拟合其余基准→预测留出基准 n*,报 MAE/R² + conformal 区间覆盖率
    ④ "谁在何处赢"相图数据
R² 高=律成立(够 npj/NatCommun);R² 低=诚实报告律不成立(退回强 CILS 清算稿)。

运行方式:
    python 02code/63_crossover_analysis.py            # 读 04outputs/62_crossover_engine.xlsx

输出文件:
    04outputs/63_crossover_analysis.xlsx  — nstar_per_task / scaling_fit / lobo / phase_diagram
    05logs/63_crossover_analysis_*.log
"""
import importlib.util
import os

import numpy as np
import numpy.typing as npt
import pandas as pd

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, ".."))
_OUT = os.path.join(_BASE, "04outputs")
_es = importlib.util.spec_from_file_location("export_utils", os.path.join(_CODE, "01_export_utils.py"))
assert _es is not None  # 对已知存在的 .py 文件 spec 恒非 None
assert _es.loader is not None  # 同理 loader 恒非 None
_eu = importlib.util.module_from_spec(_es)
_es.loader.exec_module(_eu)

SIMPLE = ["coral", "sbc"]                       # 便宜(零标签/2参数)
EXPENSIVE = ["model_update", "target_only"]     # 经典昂贵族(需较多标签/重训)
ALLM = ["zero_shot", "coral", "sbc", "pds", "target_only", "model_update"]
# 正文 2.3 节定义的 n*：最优简单方法追平最优深度/物理方法所需的最小预算。
# 两侧都按「该预算下可运行的成员」取，与表 4 的包络口径逐字一致：
#   简单侧 n=0 取 CORAL、n>0 取 SBC；深度侧 n=0 取四个零样本模型、n>0 取两个微调模型。
SIMPLE_ZERO, SIMPLE_POS = ["coral"], ["sbc"]
DEEP_ZERO = ["cnn_zeroshot", "physbl_zeroshot", "dann", "deepcoral"]
DEEP_POS = ["cnn_finetune", "physbl_ft"]
# 正文报告的预算网格。62/64 的曲线里还有 n_cal=80（只有大基准跑得到），但全文的表与图都止于 40，
# 故 n* 一律在这张网格上定位，40 标签内未发生交叉的任务记为右删失，而不是悄悄用 80 把它填上。
NSTAR_GRID = [0.0, 5.0, 10.0, 20.0, 40.0]
TOL_FRAC = 0.03      # simple 追平 expensive 的容差(×y_std)
USABLE_RPD = 2.0     # labels-to-usable 阈值


def best_at_budget(
    mat: dict[str, dict[float, float]], methods: list[str], ncal: float, grid: list[float]
) -> float:
    """给定预算 ncal,某方法集用 ≤ncal 标签能达到的最优 RMSEP。coral/zero_shot 视为 0 标签任意预算可用。"""
    vals = []
    for m in methods:
        if m not in mat:
            continue
        for n in grid:
            if (n <= ncal or (m in ("coral", "zero_shot") and n == 0)) and not np.isnan(mat[m].get(n, np.nan)):
                vals.append(mat[m][n])
    return min(vals) if vals else np.nan


def compute_nstar(task_df: pd.DataFrame, y_std: float, grid: list[float]) -> dict[str, float]:
    """task_df: 该任务 method×n_cal 均值 RMSEP 长表。返回多种 n* 定义。"""
    mat: dict[str, dict[float, float]] = {}
    for m in task_df["method"].unique():
        sub = task_df[task_df["method"] == m]
        mat[m] = dict(zip(sub["n_cal"], sub["rmsep"], strict=False))
    tol = TOL_FRAC * y_std
    out = {}
    # ① labels-to-usable: 最小 n 使任一方法 RPD>=USABLE_RPD
    n_usable = np.nan
    for n in grid:
        best = best_at_budget(mat, ALLM, n, grid)
        if not np.isnan(best) and y_std / best >= USABLE_RPD:
            n_usable = n
            break
    out["n_usable"] = n_usable
    # ② simple-suffices: 最小 n 使 best_simple(n) <= best_expensive(n)+tol
    n_simple = np.nan
    for n in grid:
        bs = best_at_budget(mat, SIMPLE, n, grid)
        be = best_at_budget(mat, EXPENSIVE, n, grid)
        if not np.isnan(bs) and not np.isnan(be) and bs <= be + tol:
            n_simple = n
            break
    out["n_simple_suffices"] = n_simple
    # ②' 正文口径的 n*：最优简单方法追平最优深度/物理方法（两侧都按该预算可运行的成员取）
    n_sd = np.nan
    for n in grid:
        smp = SIMPLE_ZERO if n == 0 else SIMPLE_POS
        dp = DEEP_ZERO if n == 0 else DEEP_POS
        bs = min([mat[m][n] for m in smp if m in mat and not np.isnan(mat[m].get(n, np.nan))], default=np.nan)
        bd = min([mat[m][n] for m in dp if m in mat and not np.isnan(mat[m].get(n, np.nan))], default=np.nan)
        if not np.isnan(bs) and not np.isnan(bd) and bs <= bd + tol:
            n_sd = n
            break
    out["n_simple_vs_deep"] = n_sd
    # ③ source-vs-target: 最小 n 使 target_only(n) <= best_source_informed(n)+tol
    n_svt = np.nan
    src_methods = ["sbc", "pds", "coral", "model_update"]
    for n in grid:
        if "target_only" not in mat:
            break
        to = mat["target_only"].get(n, np.nan)
        bsrc = best_at_budget(mat, src_methods, n, grid)
        if not np.isnan(to) and not np.isnan(bsrc) and to <= bsrc + tol:
            n_svt = n
            break
    out["n_source_vs_target"] = n_svt
    # 参考: 零标签/最优 RPD
    out["rpd_zeroshot"] = y_std / mat.get("zero_shot", {}).get(0, np.nan) if "zero_shot" in mat else np.nan
    out["rpd_coral0"] = y_std / mat.get("coral", {}).get(0, np.nan) if "coral" in mat else np.nan
    return out


def loglog_fit(
    X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64] | None, float, int]:
    """log y ~ [log X, 1] 最小二乘。返回 系数, R²。X:(n,k)>0, y:(n,)>0。"""
    ok = np.all(X > 0, 1) & (y > 0) & np.isfinite(y)
    if ok.sum() < X.shape[1] + 2:
        return None, np.nan, int(ok.sum())
    A = np.c_[np.log(X[ok]), np.ones(ok.sum())]
    b = np.log(y[ok])
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    pred = A @ coef
    ss_res = np.sum((b - pred) ** 2)
    ss_tot = np.sum((b - b.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    return coef, float(r2), int(ok.sum())


def main() -> None:
    log = _eu.get_logger("63_crossover_analysis")
    xls = os.path.join(_OUT, "62_crossover_engine.xlsx")
    curves = pd.read_excel(xls, sheet_name="curves")
    feats = pd.read_excel(xls, sheet_name="features")
    deep_xls = os.path.join(_OUT, "64_deep_transfer_server.xlsx")
    if os.path.exists(deep_xls):   # 深度曲线并入，n* 才能按正文 2.3 节的定义算
        curves = pd.concat([curves, pd.read_excel(deep_xls, sheet_name="curves")], ignore_index=True)
    else:
        log.log("⚠ 未找到 64 号深度曲线，n_simple_vs_deep 将全为 NaN")
    _eu.log_experiment_header(log, {"任务": "n*定位+标度律+LOBO", "曲线行": len(curves), "任务数": len(feats)})

    grid = sorted(curves["n_cal"].unique())
    # 跨种子×重复取中位，与表 4/表 5 的逐任务口径一致（此前用均值，两处对不上）
    mean_rmsep = curves.groupby(["task_id", "method", "n_cal"])["rmsep"].median().reset_index()

    feat_map = feats.set_index("task_id")
    recs = []
    for tid, g in mean_rmsep.groupby("task_id"):
        if tid not in feat_map.index:
            continue
        f = feat_map.loc[tid]
        y_std = float(f["y_std_tgt"])
        ns = compute_nstar(g, y_std, [n for n in NSTAR_GRID if n in grid])
        rec = dict(task_id=tid, benchmark=f["benchmark"], shift_type=f["shift_type"],
                   modality=f["modality"], mmd=f["mmd"], sliced_w1=f["sliced_w1"],
                   pca_dim_tgt=f["pca_dim_tgt"], y_std_tgt=y_std, y_mean_shift=f["y_mean_shift"],
                   n_tgt=f["n_tgt"], **ns)
        recs.append(rec)
    nstar = pd.DataFrame(recs)
    log.log(f"\nn* 台账({len(nstar)} 任务):\n" + nstar.groupby("shift_type")[
        ["n_usable", "n_simple_suffices", "n_source_vs_target", "rpd_zeroshot", "rpd_coral0"]]
        .mean().to_string())

    # 标度律拟合(对 n_source_vs_target,+1 避免 log0;分 pooled 与按 shift_type)
    fitrows = []
    for target_col in ["n_simple_vs_deep", "n_source_vs_target", "n_usable"]:
        for feat_set, cols in [("mmd", ["mmd"]), ("mmd+dim", ["mmd", "pca_dim_tgt"]),
                                ("w1+dim", ["sliced_w1", "pca_dim_tgt"])]:
            d = nstar.dropna(subset=[target_col, *cols])
            if len(d) < 8:
                continue
            X = d[cols].to_numpy(float)
            y = d[target_col].to_numpy(float) + 1.0
            coef, r2, n = loglog_fit(X, y)
            fitrows.append({"target": target_col, "features": feat_set, "n": n, "r2": r2,
                                "exponents": None if coef is None else np.round(coef[:-1], 3).tolist(),
                                "intercept": None if coef is None else round(float(coef[-1]), 3)})
    scaling = pd.DataFrame(fitrows)
    log.log("\n标度律拟合(log-log R²):\n" + scaling.to_string(index=False))

    # LOBO 外推: 留一基准,拟合其余→预测留出基准 n_source_vs_target
    lobo_rows = []
    for target_col in ["n_simple_vs_deep", "n_source_vs_target", "n_usable"]:
        cols = ["mmd", "pca_dim_tgt"]
        d = nstar.dropna(subset=[target_col, *cols]).copy()
        d["logy"] = np.log(d[target_col].to_numpy(float) + 1.0)
        for bmk in d["benchmark"].unique():
            tr = d[d["benchmark"] != bmk]
            te = d[d["benchmark"] == bmk]
            if len(tr) < 6 or len(te) < 2:
                continue
            Xtr = np.c_[np.log(tr[cols].to_numpy(float)), np.ones(len(tr))]
            coef, *_ = np.linalg.lstsq(Xtr, tr["logy"].to_numpy(), rcond=None)
            Xte = np.c_[np.log(te[cols].to_numpy(float)), np.ones(len(te))]
            pred = Xte @ coef
            resid_tr = tr["logy"].to_numpy() - Xtr @ coef
            q = np.quantile(np.abs(resid_tr), 0.9)     # conformal 90% 半宽(log域)
            true = te["logy"].to_numpy()
            cover = np.mean(np.abs(true - pred) <= q)
            mae_log = float(np.mean(np.abs(true - pred)))
            # 原尺度 n* MAE
            mae_n = float(np.mean(np.abs(np.exp(true) - np.exp(pred))))
            lobo_rows.append({"target": target_col, "held_out": bmk, "n_test": len(te),
                                  "mae_log": round(mae_log, 3), "mae_nstar": round(mae_n, 2),
                                  "conformal90_cover": round(float(cover), 2)})
    lobo = pd.DataFrame(lobo_rows)
    log.log("\nLOBO 留一基准外推:\n" + (lobo.to_string(index=False) if len(lobo) else "样本不足"))

    # 相图数据: 各 shift_type × n_cal 下 谁赢(最低RMSEP方法占比)
    win_rows = []
    for (st, nc), g in mean_rmsep.merge(
            nstar[["task_id", "shift_type"]], on="task_id").groupby(["shift_type", "n_cal"]):
        winners = g.loc[g.groupby("task_id")["rmsep"].idxmin()]
        vc = winners["method"].value_counts(normalize=True)
        win_rows.append(dict(shift_type=st, n_cal=nc, **{f"win_{m}": round(vc.get(m, 0), 2) for m in ALLM}))
    phase = pd.DataFrame(win_rows).sort_values(["shift_type", "n_cal"])
    log.log("\n相图(各 shift×n_cal 方法胜率):\n" + phase.to_string(index=False))

    # n* 台账：每种定义下取到各网格值的任务数、右删失（40 标签内未发生）与未定义的任务数，
    # 供正文如实交代回归所用的样本population（面板意见：不公开这一口径就不能用它下否定性结论）
    cnt_rows = []
    for col in ["n_simple_vs_deep", "n_simple_suffices", "n_source_vs_target", "n_usable"]:
        vc = nstar[col].value_counts(dropna=False)
        row: dict[str, object] = {"definition": col, "n_tasks": len(nstar)}
        for v in sorted([x for x in vc.index if pd.notna(x)]):
            row[f"n*={int(v)}"] = int(vc[v])
        row["undefined_or_censored"] = int(nstar[col].isna().sum())
        row["used_in_loglog_fit"] = int((nstar[col].notna() & (nstar[col] + 1 > 0)).sum())
        cnt_rows.append(row)
    nstar_counts = pd.DataFrame(cnt_rows).fillna(0)
    log.log("\nn* 台账（各定义下的取值分布）:\n" + nstar_counts.to_string(index=False))

    _eu.write_script_workbook(__file__, {"nstar_per_task": nstar, "scaling_fit": scaling,
                                         "lobo": lobo, "phase_diagram": phase,
                                         "nstar_counts": nstar_counts})
    print("=== 关键结论 ===")
    print(scaling.to_string(index=False))
    print("\nLOBO:\n" + (lobo.to_string(index=False) if len(lobo) else "样本不足"))


if __name__ == "__main__":
    main()
