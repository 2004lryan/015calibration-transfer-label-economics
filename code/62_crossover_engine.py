"""
62_crossover_engine.py: 标签预算交叉点引擎（idea v3 科学核心，经典方法本机版）

对 61 号注册的每个迁移任务，扫目标标签预算 n_cal，跑经典标定转移方法全套，
产 RMSEP/RPD 曲线 + 每任务漂移特征(MMD/Wasserstein/维度/SNR)，供 63 号定位 n*、拟合标度律、画相图。
深度方法(DANN/DeepCORAL/Phys+BL)在服务器 torch 版(64 号)补齐。

方法(经典,留出评估,无 oracle,REP 随机重复):
    zero_shot  源模型直接预测(n_cal=0)
    coral      免标签二阶矩对齐(n_cal=0)
    sbc        斜率/偏置校正(n_cal 标签)
    pds        Piecewise Direct Standardization(仅配对基准,n_cal 配对转移样本)
    target_only仅目标 PLS(n_cal 标签)
    model_update 源+n_cal目标 合并重训(n_cal 标签)

评估: 目标域固定留出一半做测试,另一半池随机抽 n_cal,REP 次;源模型用整个源域训练。
      跨长度域(苹果A↔B 1213 vs 229)按最近波长对齐到公共网格。

运行方式:
    python 02code/62_crossover_engine.py [--benchmarks corn,tablet,mango,apple] [--rep 10]

输出文件:
    04outputs/62_crossover_engine.xlsx  — curves(长表) / features(每任务) / summary
    05logs/62_crossover_engine_*.log
"""
import argparse
import importlib.util
import os
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 光谱/算子矩阵统一别名（float64 二维/一维数组），与论文数学符号 X/Xs/Xt/K/C/W 对应
F64 = npt.NDArray[np.float64]

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, ".."))
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

SEED = 20060515
SEEDS = [20060515, 20041210, 19810915, 2023, 2024]   # CLAUDE.md §5.3 固定 5 种子(Formal)
NCAL_GRID = [0, 5, 10, 20, 40, 80]
REP = 10
DOMAIN_CAP = 800     # 大域子采样上限(crossover 测量 800 样本已足,提速 mango/OSSL)


def snv(X: F64) -> F64:
    out: F64 = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
    return out


def rmse(a: F64, b: F64) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def align_common(Xs: F64, wls: F64, Xt: F64, wlt: F64) -> tuple[F64, F64]:
    """跨长度域按最近波长对齐到较短网格。返回对齐后 (Xs2, Xt2)。"""
    if Xs.shape[1] == Xt.shape[1]:
        return Xs, Xt
    if len(wls) <= len(wlt):
        ref, other, Xo, Xr = wls, wlt, Xt, Xs
        sel = [int(np.argmin(np.abs(other - w))) for w in ref]
        return Xr, Xo[:, sel]
    else:
        ref, other, Xo, Xr = wlt, wls, Xs, Xt
        sel = [int(np.argmin(np.abs(other - w))) for w in ref]
        return Xo[:, sel], Xr


def sel_k(X: F64, y: F64, rng: np.random.Generator, kmax: int = 15) -> int:
    best_k, best_e = 5, np.inf
    hi = min(kmax, X.shape[1], len(y) - 2)
    for k in range(2, max(3, hi)):
        es = []
        for a, b in KFold(3, shuffle=True, random_state=SEED).split(X):
            if len(a) <= k:
                continue
            p = Pipeline([("s", StandardScaler()), ("p", PLSRegression(k))]).fit(X[a], y[a])
            es.append(rmse(p.predict(X[b]).ravel(), y[b]))
        if es and np.mean(es) < best_e:
            best_e, best_k = float(np.mean(es)), k
    return best_k


def fit_pls(X: F64, y: F64, k: int) -> Any:
    return Pipeline([("s", StandardScaler()), ("p", PLSRegression(k))]).fit(X, y)


def coral(Xs: F64, Xt: F64, eps: float = 1e-2) -> F64:
    mus, mut = Xs.mean(0), Xt.mean(0)
    Cs = np.cov(Xs, rowvar=False) + eps * np.eye(Xs.shape[1])
    Ct = np.cov(Xt, rowvar=False) + eps * np.eye(Xt.shape[1])
    def ms(C: F64, inv: bool) -> F64:
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-10, None)
        w = 1 / np.sqrt(w) if inv else np.sqrt(w)
        msout: F64 = (V * w) @ V.T
        return msout
    res: F64 = (Xt - mut) @ ms(Ct, True) @ ms(Cs, False) + mus
    return res


def pds(Xs_cal: F64, Xt_cal: F64, Xt_test: F64, hw: int = 7, alpha: float = 1.0) -> F64:
    p = Xs_cal.shape[1]
    out = np.zeros_like(Xt_test)
    for j in range(p):
        lo, hi = max(0, j - hw), min(p, j + hw + 1)
        reg = Ridge(alpha=alpha).fit(Xt_cal[:, lo:hi], Xs_cal[:, j])
        out[:, j] = reg.predict(Xt_test[:, lo:hi])
    return out


# ---------------- 漂移特征(每任务一次,无标签) ----------------
def mmd2_rbf(Xs: F64, Xt: F64, rng: np.random.Generator, cap: int = 200) -> float:
    if len(Xs) > cap:
        Xs = Xs[rng.choice(len(Xs), cap, replace=False)]
    if len(Xt) > cap:
        Xt = Xt[rng.choice(len(Xt), cap, replace=False)]
    Z = np.vstack([Xs, Xt])
    d2 = np.sum((Z[:, None] - Z[None]) ** 2, -1)
    sig = np.median(d2[d2 > 0]) + 1e-12
    K = np.exp(-d2 / sig)
    n = len(Xs)
    Kss, Ktt, Kst = K[:n, :n], K[n:, n:], K[:n, n:]
    return float(Kss.mean() + Ktt.mean() - 2 * Kst.mean())


def sliced_w1(Xs: F64, Xt: F64, rng: np.random.Generator, nproj: int = 50) -> float:
    p = Xs.shape[1]
    W = rng.standard_normal((p, nproj))
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-12
    Ps, Pt = Xs @ W, Xt @ W
    vals = []
    for k in range(nproj):
        a, b = np.sort(Ps[:, k]), np.sort(Pt[:, k])
        q = np.linspace(0, 1, 100)
        vals.append(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))
    return float(np.mean(vals))


def pca_dim(X: F64) -> float:
    Xc = X - X.mean(0)
    ev = np.linalg.svd(Xc, compute_uv=False) ** 2
    ev = ev / (ev.sum() + 1e-12)
    return float((ev.sum() ** 2) / (np.sum(ev ** 2) + 1e-12))   # participation ratio


def task_features(Xs: F64, Xt: F64, ys: F64, yt: F64, rng: np.random.Generator) -> dict[str, float]:
    return {
        "mmd": mmd2_rbf(Xs, Xt, rng), "sliced_w1": sliced_w1(Xs, Xt, rng),
        "pca_dim_src": pca_dim(Xs), "pca_dim_tgt": pca_dim(Xt),
        "y_std_src": float(ys.std()), "y_std_tgt": float(yt.std()),
        "y_mean_shift": float(abs(ys.mean() - yt.mean())),
        "n_src": len(Xs), "n_tgt": len(Xt),
    }


def run_task(
    dom_s: dict[str, Any], dom_t: dict[str, Any], prop_idx: int, paired: bool, rng: np.random.Generator
) -> tuple[list[tuple[int, str, int, float]] | None, dict[str, float] | None]:
    Xs_r, ys = dom_s["X"], dom_s["Y"][:, prop_idx]
    Xt_r, yt = dom_t["X"], dom_t["Y"][:, prop_idx]
    okS = np.isfinite(Xs_r).all(1) & np.isfinite(ys)
    okT = np.isfinite(Xt_r).all(1) & np.isfinite(yt)
    Xs_r, ys, Xt_r, yt = Xs_r[okS], ys[okS], Xt_r[okT], yt[okT]
    if len(Xs_r) > DOMAIN_CAP:                      # 大域子采样提速
        i = rng.choice(len(Xs_r), DOMAIN_CAP, replace=False)
        Xs_r, ys = Xs_r[i], ys[i]
    if len(Xt_r) > DOMAIN_CAP:
        i = rng.choice(len(Xt_r), DOMAIN_CAP, replace=False)
        Xt_r, yt = Xt_r[i], yt[i]
    Xs_a, Xt_a = align_common(Xs_r, dom_s["wl"], Xt_r, dom_t["wl"])
    Xs, Xt = snv(Xs_a), snv(Xt_a)
    if len(Xs) < 20 or len(Xt) < 24:
        return None, None
    feats = task_features(Xs, Xt, ys, yt, rng)

    ks = sel_k(Xs, ys, rng)
    src = fit_pls(Xs, ys, ks)
    # CORAL 零标签,每任务恒定,循环外算一次
    try:
        pred_coral = src.predict(coral(Xs, Xt)).ravel()
    except Exception:
        pred_coral = None
    rows = []
    nt = len(Xt)
    for r in range(REP):
        perm = rng.permutation(nt)
        test = perm[: nt // 2]
        pool = perm[nt // 2:]
        yhat0 = src.predict(Xt[test]).ravel()
        ytest = yt[test]
        # n_cal=0 方法
        rows.append((0, "zero_shot", r, rmse(yhat0, ytest)))
        if pred_coral is not None:
            rows.append((0, "coral", r, rmse(pred_coral[test], ytest)))
        for nc in NCAL_GRID:
            if nc == 0:
                continue
            if nc > len(pool):
                continue
            cal = rng.choice(pool, nc, replace=False)
            # SBC
            pc = src.predict(Xt[cal]).ravel()
            co = np.linalg.lstsq(np.c_[pc, np.ones(len(pc))], yt[cal], rcond=None)[0]
            rows.append((nc, "sbc", r, rmse(co[0] * yhat0 + co[1], ytest)))
            # PDS(配对)
            if paired:
                try:
                    Xt_map = pds(Xs[cal], Xt[cal], Xt[test])
                    rows.append((nc, "pds", r, rmse(src.predict(Xt_map).ravel(), ytest)))
                except Exception:
                    pass
            # target_only
            if nc >= 5:
                kt = sel_k(Xt[cal], yt[cal], rng, kmax=min(10, nc - 1))
                to = fit_pls(Xt[cal], yt[cal], max(1, kt))
                rows.append((nc, "target_only", r, rmse(to.predict(Xt[test]).ravel(), ytest)))
            # model_update
            Xu = np.vstack([Xs, Xt[cal]])
            yu = np.concatenate([ys, yt[cal]])
            ku = sel_k(Xu, yu, rng)
            mu = fit_pls(Xu, yu, ku)
            rows.append((nc, "model_update", r, rmse(mu.predict(Xt[test]).ravel(), ytest)))
    feats["y_std_test_ref"] = float(np.mean([yt.std()]))
    return rows, feats


def main() -> None:
    global REP
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default="corn,tablet,mango,apple")
    ap.add_argument("--rep", type=int, default=5)
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = ap.parse_args()
    REP = args.rep
    seeds = [int(s) for s in args.seeds.split(",")]

    log = _eu.get_logger("62_crossover_engine")
    _eu.log_experiment_header(log, {"任务": "标签预算交叉点引擎(经典)", "基准": args.benchmarks,
                                    "n_cal网格": NCAL_GRID, "REP": REP, "种子": seeds})
    curve_rows, feat_rows = [], []
    names = args.benchmarks.split(",")
    seen_feat = set()
    for name in names:
        b = bench.LOADERS[name]()
        tasks = bench.build_transfer_tasks(b)
        log.log(f"[{name}] {len(tasks)} 任务 × {len(seeds)} 种子")
        for ti, tk in enumerate(tasks):
            tid = f"{name}|{tk['src']}->{tk['tgt']}|{tk['prop_name']}"
            for seed in seeds:
                rng = np.random.default_rng(seed)
                rows, feats = run_task(b["domains"][tk["src"]], b["domains"][tk["tgt"]],
                                       tk["prop_idx"], tk["paired"], rng)
                if rows is None or feats is None:
                    continue
                for nc, meth, r, val in rows:
                    curve_rows.append({"task_id": tid, "benchmark": name, "shift_type": tk["shift_type"],
                                           "modality": tk["modality"], "prop": tk["prop_name"],
                                           "n_cal": nc, "method": meth, "seed": seed, "rep": r, "rmsep": val})
                if tid not in seen_feat:   # 漂移特征每任务记一次(种子间不变)
                    fr = dict(task_id=tid, benchmark=name, shift_type=tk["shift_type"], modality=tk["modality"],
                              prop=tk["prop_name"], paired=tk["paired"], **feats)
                    fr["same_instrument"] = tk.get("same_instrument", np.nan)
                    feat_rows.append(fr)
                    seen_feat.add(tid)
            if (ti + 1) % 10 == 0:
                log.log(f"    {name} {ti+1}/{len(tasks)} 完成")
    curves = pd.DataFrame(curve_rows)
    feats = pd.DataFrame(feat_rows)
    # 汇总: 每任务每方法每 n_cal 的 RMSEP 均值
    summ = (curves.groupby(["benchmark", "shift_type", "method", "n_cal"])["rmsep"]
            .mean().reset_index())
    log.log(f"\n完成任务数={feats.shape[0]}  曲线行数={curves.shape[0]}")
    log.log("\n===== 各 shift_type × 方法 × n_cal 平均 RMSEP(前40行) =====\n"
            + summ.head(40).to_string(index=False))
    _eu.write_script_workbook(__file__, {"curves": curves, "features": feats, "summary": summ})
    print(f"完成任务数={feats.shape[0]}  曲线行数={curves.shape[0]}  → 04outputs/62_crossover_engine.xlsx")


if __name__ == "__main__":
    main()
