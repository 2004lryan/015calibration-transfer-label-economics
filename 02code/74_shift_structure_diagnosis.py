"""
74_shift_structure_diagnosis.py: 五基准域漂移的低阶结构诊断 + 免标签低秩分解失稳指标推广到 126 任务（返修 CILS R1.1）

【为什么必须做这个实验】
    审稿人 1 的唯一 major：多基准主线与苹果案例像两篇文章，机理洞见最好在已引入的公开基准与方法上演示，
    而不是靠自建架构 + 内部数据。投出稿讨论第一段把"简单法在仪器/季节漂移下赢"归因于
    "这些漂移近似为特征空间的低阶（一阶、二阶矩）线性变换"，但这只是一个未经检验的解释。
    本脚本用与统一基准完全相同的数据管道（62 号：最近波长对齐 → SNV → 大域子采样 800），对全部
    126 个迁移任务（98 个有序域对）只用光谱、不用任何标签，回答两件事：
      ① 漂移里有多少能被低阶变换解释：固定核带宽/固定投影下，源–目标分布距离（MMD²、sliced-W1）
         在均值对齐、二阶矩对齐（CORAL，62 号同一实现）之后还剩多少。
           f_mean  = 1 − d(mean-aligned)/d(raw)   一阶可解释份额
           f_coral = 1 − d(CORAL)/d(raw)          一阶+二阶可解释份额
           residual = 1 − f_coral                 高阶残余份额
      ② 苹果案例的免标签"BL 型低秩可分解性失稳"指标 A（源域拟合 PCA-10，目标域相对重构残差中位数 /
         源域中位数，投出稿式 (1)(2)）推广到全部基准，另给源域留出版 A_oos 防过拟合。
    然后把这些免标签量与 62/64 号已落地的结果对上：CORAL 相对增益、零标签 NRMSEP、简单族−深度族
    的族均 NRMSEP 差（n=0、20）——Spearman，域对聚类 bootstrap 95% CI，留一基准的符号一致性。
    不重训任何模型；不引入新主张——它给讨论中既有的机理解释提供检验，并把苹果的免标签预警从单基准
    推广到五基准。

运行方式:
    cd <项目根目录绝对路径>
    python 02code/74_shift_structure_diagnosis.py

输出文件:
    04outputs/74_shift_structure_diagnosis.xlsx — by_benchmark / correlations / lobo / per_task / per_pair
    05logs/74_shift_structure_diagnosis_YY-MM-DD_HHMMSS.log
"""

import importlib.util
import os

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

F64 = npt.NDArray[np.float64]

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, ".."))
_OUT = os.path.join(_BASE, "04outputs")
_es = importlib.util.spec_from_file_location("export_utils", os.path.join(_CODE, "01_export_utils.py"))
assert _es is not None  # 对已知存在的 .py 文件 spec 恒非 None
assert _es.loader is not None  # 同理 loader 恒非 None
_eu = importlib.util.module_from_spec(_es)
_es.loader.exec_module(_eu)
_bs = importlib.util.spec_from_file_location("bench", os.path.join(_CODE, "61_benchmark_datasets.py"))
assert _bs is not None  # 同上
assert _bs.loader is not None
bench = importlib.util.module_from_spec(_bs)
_bs.loader.exec_module(bench)
_cs = importlib.util.spec_from_file_location("engine", os.path.join(_CODE, "62_crossover_engine.py"))
assert _cs is not None  # 同上
assert _cs.loader is not None
engine = importlib.util.module_from_spec(_cs)
_cs.loader.exec_module(engine)

SEED = 20060515
DOMAIN_CAP = 800  # 与 62 号一致
MMD_CAP = 300  # 距离估计子采样（三种对齐状态共用同一子样本）
NPROJ = 100
KPCA = 10  # 与投出稿式 (1)(2) 一致
NBOOT = 2000
SIMPLE = ["coral", "sbc"]
DEEP = ["cnn_zeroshot", "cnn_finetune", "dann", "deepcoral", "physbl_zeroshot", "physbl_ft"]
RNG = np.random.default_rng(SEED)


# ---------------- 免标签距离（固定带宽 / 固定投影，三种对齐状态可比） ----------------
def _mmd2_fixed(Xs: F64, Xt: F64, sigma: float) -> float:
    Z = np.vstack([Xs, Xt])
    d2 = np.sum((Z[:, None] - Z[None]) ** 2, -1)
    K = np.exp(-d2 / sigma)
    n = len(Xs)
    return float(K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean())


def _sw1_fixed(Xs: F64, Xt: F64, W: F64) -> float:
    Ps, Pt = Xs @ W, Xt @ W
    q = np.linspace(0, 1, 100)
    vals = [np.mean(np.abs(np.quantile(Ps[:, k], q) - np.quantile(Pt[:, k], q))) for k in range(W.shape[1])]
    return float(np.mean(vals))


def _lowrank_residual(Xfit: F64, Xeval: F64, k: int = KPCA) -> float:
    """源域拟合的 PCA-k 重构算子作用于 Xeval，返回相对重构残差的中位数（投出稿式 (1)）。"""
    mu = Xfit.mean(0)
    _u, _s, Vt = np.linalg.svd(Xfit - mu, full_matrices=False)
    V = Vt[:k].T
    Xc = Xeval - mu
    rec = Xc @ V @ V.T + mu
    num = np.linalg.norm(Xeval - rec, axis=1)
    den = np.linalg.norm(Xeval, axis=1) + 1e-12
    return float(np.median(num / den))


def _pair_metrics(Xs_r: F64, wls: F64, Xt_r: F64, wlt: F64, rng: np.random.Generator) -> dict[str, float]:
    okS, okT = np.isfinite(Xs_r).all(1), np.isfinite(Xt_r).all(1)
    Xs_r, Xt_r = Xs_r[okS], Xt_r[okT]
    if len(Xs_r) > DOMAIN_CAP:
        Xs_r = Xs_r[rng.choice(len(Xs_r), DOMAIN_CAP, replace=False)]
    if len(Xt_r) > DOMAIN_CAP:
        Xt_r = Xt_r[rng.choice(len(Xt_r), DOMAIN_CAP, replace=False)]
    Xs_a, Xt_a = engine.align_common(Xs_r, wls, Xt_r, wlt)
    Xs, Xt = engine.snv(Xs_a), engine.snv(Xt_a)
    # 三种对齐状态
    Xt_mean = Xt - Xt.mean(0) + Xs.mean(0)
    Xt_coral = engine.coral(Xs, Xt)
    # 共用子样本 + 固定带宽（median heuristic 只在 raw 上算一次）
    i_s = rng.choice(len(Xs), min(MMD_CAP, len(Xs)), replace=False)
    i_t = rng.choice(len(Xt), min(MMD_CAP, len(Xt)), replace=False)
    Z = np.vstack([Xs[i_s], Xt[i_t]])
    d2 = np.sum((Z[:, None] - Z[None]) ** 2, -1)
    sigma = float(np.median(d2[d2 > 0]) + 1e-12)
    W = rng.standard_normal((Xs.shape[1], NPROJ))
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-12
    out: dict[str, float] = {}
    for tag, Xtv in [("raw", Xt), ("mean", Xt_mean), ("coral", Xt_coral)]:
        out[f"mmd2_{tag}"] = _mmd2_fixed(Xs[i_s], Xtv[i_t], sigma)
        out[f"sw1_{tag}"] = _sw1_fixed(Xs, Xtv, W)
    for dist in ["mmd2", "sw1"]:
        d0 = max(out[f"{dist}_raw"], 1e-12)
        out[f"f_mean_{dist}"] = float(np.clip(1 - out[f"{dist}_mean"] / d0, 0, 1))
        out[f"f_coral_{dist}"] = float(np.clip(1 - out[f"{dist}_coral"] / d0, 0, 1))
        out[f"residual_{dist}"] = 1 - out[f"f_coral_{dist}"]
    # 低秩可分解性失稳指标 A（源内 vs 跨域），另给源域留出版
    rs = _lowrank_residual(Xs, Xs)
    rt = _lowrank_residual(Xs, Xt)
    out["dR_src"], out["dR_tgt"], out["A_lowrank"] = rs, rt, rt / max(rs, 1e-12)
    perm = rng.permutation(len(Xs))
    nfit = int(0.7 * len(Xs))
    rs_oos = _lowrank_residual(Xs[perm[:nfit]], Xs[perm[nfit:]])
    rt_oos = _lowrank_residual(Xs[perm[:nfit]], Xt)
    out["A_lowrank_oos"] = rt_oos / max(rs_oos, 1e-12)
    out["n_src"], out["n_tgt"], out["p_bands"] = float(len(Xs)), float(len(Xt)), float(Xs.shape[1])
    return out


# ---------------- 结果侧：62/64 已落地曲线 ----------------
def _task_outcomes() -> pd.DataFrame:
    cl = pd.read_excel(os.path.join(_OUT, "62_crossover_engine.xlsx"), sheet_name="curves")
    dp = pd.read_excel(os.path.join(_OUT, "64_deep_transfer_server.xlsx"), sheet_name="curves")
    allr = pd.concat([cl, dp], ignore_index=True)
    tm = allr.groupby(["benchmark", "shift_type", "task_id", "method", "n_cal"], as_index=False)["rmsep"].median()
    ns = pd.read_excel(os.path.join(_OUT, "63_crossover_analysis.xlsx"), sheet_name="nstar_per_task")
    tm = tm.merge(ns[["task_id", "y_std_tgt"]].drop_duplicates("task_id"), on="task_id", how="left")
    tm["nrmsep"] = tm["rmsep"] / tm["y_std_tgt"]
    rows = []

    def get(g: pd.DataFrame, m: str, nc: int) -> float:
        v = g[(g["method"] == m) & (g["n_cal"] == nc)]["nrmsep"]
        return float(v.iloc[0]) if len(v) else float("nan")

    def fam(g: pd.DataFrame, meths: list[str], nc: int) -> float:
        v = g[g["method"].isin(meths) & (g["n_cal"] == nc)]["nrmsep"]
        return float(v.mean()) if len(v) else float("nan")

    for tid, g in tm.groupby("task_id"):
        z0, c0 = get(g, "zero_shot", 0), get(g, "coral", 0)
        rows.append(
            {
                "task_id": tid,
                "benchmark": g["benchmark"].iloc[0],
                "shift_type": g["shift_type"].iloc[0],
                "nrmsep_zero_shot0": z0,
                "nrmsep_coral0": c0,
                "coral_gain": (1 - c0 / z0) if np.isfinite(z0) and np.isfinite(c0) and z0 > 0 else np.nan,
                "nrmsep_sbc20": get(g, "sbc", 20),
                "margin0_deep_minus_simple": fam(g, DEEP, 0) - fam(g, SIMPLE, 0),
                "margin20_deep_minus_simple": fam(g, DEEP, 20) - fam(g, SIMPLE, 20),
            }
        )
    return pd.DataFrame(rows)


def _spearman_ci(x: F64, y: F64, cluster: npt.NDArray[np.str_]) -> tuple[float, float, float, float, int]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, cluster = x[ok], y[ok], cluster[ok]
    if len(x) < 5:
        return float("nan"), float("nan"), float("nan"), float("nan"), len(x)
    rho, p = stats.spearmanr(x, y)
    cl = np.unique(cluster)
    idx = {c: np.where(cluster == c)[0] for c in cl}
    boots = []
    for _ in range(NBOOT):
        pick = RNG.choice(cl, len(cl), replace=True)
        ii = np.concatenate([idx[c] for c in pick])
        if len(np.unique(x[ii])) < 3 or len(np.unique(y[ii])) < 3:
            continue
        boots.append(stats.spearmanr(x[ii], y[ii])[0])
    lo, hi = (np.percentile(boots, 2.5), np.percentile(boots, 97.5)) if boots else (np.nan, np.nan)
    return float(rho), float(lo), float(hi), float(p), len(x)


def main() -> None:
    logger = _eu.get_logger(os.path.basename(__file__).replace(".py", ""))
    # 1) 逐域对免标签指标
    pair_rows = []
    for name in ["corn", "tablet", "mango", "ossl_mir", "apple"]:
        if name not in bench.LOADERS:
            logger.log("跳过 %s（无缓存）", name)
            continue
        b = bench.LOADERS[name]()
        keys = list(b["domains"].keys())
        for s in keys:
            for t in keys:
                if s == t:
                    continue
                rng = np.random.default_rng(abs(hash((name, s, t))) % (2**32))  # 逐对固定，跨运行可复现
                m = _pair_metrics(
                    b["domains"][s]["X"], b["domains"][s]["wl"], b["domains"][t]["X"], b["domains"][t]["wl"], rng
                )
                pair_rows.append(
                    {
                        "benchmark": name,
                        "shift_type": b["shift_type"],
                        "src": s,
                        "tgt": t,
                        "pair": f"{name}|{s}->{t}",
                        **m,
                    }
                )
        logger.log("[%s] %d 域对完成", name, sum(r["benchmark"] == name for r in pair_rows))
    pairs = pd.DataFrame(pair_rows)

    # 2) 接到任务（task_id = bench|src->tgt|prop）
    outc = _task_outcomes()
    outc["pair"] = outc["task_id"].str.rsplit("|", n=1).str[0]
    per_task = outc.merge(pairs.drop(columns=["benchmark", "shift_type"]), on="pair", how="left")
    miss = per_task["A_lowrank"].isna().sum()
    logger.log("任务=%d，未匹配到域对的任务=%d", len(per_task), int(miss))

    # 3) 按基准汇总（中位数）
    cols = [
        "f_mean_mmd2",
        "f_coral_mmd2",
        "residual_mmd2",
        "f_mean_sw1",
        "f_coral_sw1",
        "residual_sw1",
        "A_lowrank",
        "A_lowrank_oos",
        "mmd2_raw",
        "coral_gain",
        "nrmsep_zero_shot0",
        "nrmsep_coral0",
        "margin0_deep_minus_simple",
        "margin20_deep_minus_simple",
    ]
    by_bench = per_task.groupby(["benchmark", "shift_type"])[cols].median().round(4).reset_index()
    by_bench.insert(2, "n_task", per_task.groupby(["benchmark", "shift_type"]).size().values)
    by_bench.insert(3, "n_pair", per_task.groupby(["benchmark", "shift_type"])["pair"].nunique().values)
    logger.log("按基准中位数:\n%s", by_bench.to_string(index=False))

    # 4) 相关（域对聚类 bootstrap）
    cl = per_task["pair"].values.astype(str)
    preds = ["f_coral_mmd2", "f_coral_sw1", "residual_mmd2", "A_lowrank", "A_lowrank_oos", "mmd2_raw"]
    targets = [
        "coral_gain",
        "nrmsep_zero_shot0",
        "nrmsep_coral0",
        "margin0_deep_minus_simple",
        "margin20_deep_minus_simple",
    ]
    crows = []
    for pr in preds:
        for tg in targets:
            rho, lo, hi, p, n = _spearman_ci(per_task[pr].values.astype(float), per_task[tg].values.astype(float), cl)
            crows.append(
                {
                    "predictor": pr,
                    "target": tg,
                    "n": n,
                    "spearman_rho": round(rho, 3),
                    "ci95_lo": round(lo, 3),
                    "ci95_hi": round(hi, 3),
                    "p_nominal": p,
                    "ci_excludes_0": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
                }
            )
    corr = pd.DataFrame(crows)
    logger.log("相关:\n%s", corr.to_string(index=False))

    # 5) 留一基准：去掉一个基准后的 rho（符号一致性）+ 基准内 rho（n≥12）
    lrows = []
    for pr in ["f_coral_mmd2", "A_lowrank"]:
        for tg in ["coral_gain", "nrmsep_zero_shot0"]:
            for B in sorted(per_task["benchmark"].unique()):
                sub = per_task[per_task["benchmark"] != B]
                rho, lo, hi, p, n = _spearman_ci(
                    sub[pr].values.astype(float), sub[tg].values.astype(float), sub["pair"].values.astype(str)
                )
                within = per_task[per_task["benchmark"] == B]
                wr = stats.spearmanr(within[pr], within[tg], nan_policy="omit")[0] if len(within) >= 12 else np.nan
                lrows.append(
                    {
                        "predictor": pr,
                        "target": tg,
                        "held_out": B,
                        "n_rest": n,
                        "rho_rest": round(float(rho), 3),
                        "ci95_lo": round(float(lo), 3),
                        "ci95_hi": round(float(hi), 3),
                        "rho_within_heldout": round(float(wr), 3) if np.isfinite(wr) else np.nan,
                        "n_within": len(within),
                    }
                )
    lobo = pd.DataFrame(lrows)
    logger.log("留一基准:\n%s", lobo.to_string(index=False))

    path = _eu.write_script_workbook(
        __file__,
        {
            0: ("by_benchmark", by_bench),
            "correlations": corr,
            "lobo": lobo,
            "per_task": per_task,
            "per_pair": pairs,
        },
    )
    logger.log("落盘: %s", path)
    print(by_bench.to_string(index=False))
    print("\n相关（域对聚类 bootstrap 95% CI）:\n", corr.to_string(index=False))
    print("\n留一基准:\n", lobo.to_string(index=False))
    print(
        "\n注: f_coral=1−d(CORAL 对齐后)/d(raw)，越大说明漂移越能被一阶+二阶矩变换解释；"
        "A_lowrank>1 表示目标域偏离源域低秩加性结构；全部只用光谱不用标签。"
        "相关的结果侧来自 62/64 号已落地曲线（跨种子中位、NRMSEP）。"
    )


if __name__ == "__main__":
    main()
