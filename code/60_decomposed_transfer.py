"""
60_decomposed_transfer.py: 9 域跨域 SSC 迁移——按漂移类型分解 + 标定策略对照 + 本地样本预算曲线

核心问题:
    跨域近红外 SSC 迁移能否达到可用精度(RPD≥1.4)，取决于「漂移类型」与「用什么标定策略」。
    2018/2019/2025 三年 9 个产地-年域（仪器 A=2018/2019 原生 1213 波段；仪器 B=2025 为 229）。

标定策略（全部现实协议、留出评估、无 oracle）:
    · 零标签           源域模型直接预测目标域
    · 斜率/偏置校正     用 n_cal 目标标定样本拟合一元线性，校正源域预测
    · 模型更新(主)     源域样本 + n_cal 目标标定样本并入，重训 PLSR —— 本文推荐
    · 池化+更新         同仪器其余各域合并为源 + n_cal 本地，重训（留一域外验证）

数据: 原始谱(05data 多年库) → SSC>20 越界剔除 + 逐位重复光谱剔除 → SNV → PLS(主成分源域CV选)。
      同仪器对用原生波段；跨仪器对(涉 2025)降到 229 公共网格。RPD=目标 std/RMSE，可用阈值 1.4。

运行方式:
    cd /Volumes/Lin Ryan/01_科研竞赛/02论文竞赛/015苹果SSC迁移
    python 02code/60_decomposed_transfer.py

输出文件:
    04outputs/60_decomposed_transfer.xlsx  — feasibility/by_pair/by_shift/ncal_curve/pooled 五表
    05logs/60_decomposed_transfer_YY-MM-DD_HHMMSS.log
"""
import os
import sys
import importlib.util

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, ".."))
_OUT = os.path.join(_BASE, "04outputs")

_es = importlib.util.spec_from_file_location("export_utils", os.path.join(_CODE, "01_export_utils.py"))
_eu = importlib.util.module_from_spec(_es)
_es.loader.exec_module(_eu)
_ds = importlib.util.spec_from_file_location("dp", os.path.join(_CODE, "02_data_processing.py"))
dp = importlib.util.module_from_spec(_ds)
_ds.loader.exec_module(dp)

MY = os.path.join(os.path.dirname(_eu.DATA_DIR), "..", "..", "05data", "001_apple_hyperspectral_multiyear")
MY = os.path.normpath(MY)
RAW_XLSX = {2018: os.path.join(MY, "01新疆-2山东-3陕西苹果光谱数据(含糖度）2018.xlsx"),
            2019: os.path.join(MY, "1新疆-2山东-3陕西苹果光谱数据(含糖度）2019.xlsx")}
SPEC25 = os.path.join(MY, "新疆-山东-甘肃苹果光谱数据2025.csv")
SUGAR25 = os.path.join(MY, "新疆-山东-甘肃苹果糖度数据2025.csv")

INSTRUMENT = {2018: "A", 2019: "A", 2025: "B"}
NCALS = [20, 30, 50, 80]
REP = 10
SEED = 20060515
USABLE = 1.4


def snv(A):
    return (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-8)


def rpd(pred, y):
    return y.std() / np.sqrt(np.mean((pred - y) ** 2))


def fit(X, y, k):
    return Pipeline([("s", StandardScaler()), ("p", PLSRegression(k))]).fit(X, y)


def sel_k(X, y, kmax=22):
    best_k, best_e = 8, np.inf
    for k in range(2, min(kmax, X.shape[1], len(y) - 2)):
        es = [np.sqrt(np.mean((fit(X[a], y[a], k).predict(X[b]).ravel() - y[b]) ** 2))
              for a, b in KFold(3, shuffle=True, random_state=SEED).split(X)]
        if np.mean(es) < best_e:
            best_e, best_k = np.mean(es), k
    return best_k


def apply_filters(X, y, reg):
    """L1 硬过滤：SSC>20 越界 + 逐位重复光谱整组剔除（对齐 01_export_utils）。"""
    X, y, reg = np.asarray(X, float), np.asarray(y, float), np.asarray(reg)
    keep = y <= 20.0
    X, y, reg = X[keep], y[keep], reg[keep]
    key = np.array(["|".join(f"{v:.6g}" for v in row) for row in X])
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    keep = cnt[inv] == 1
    return X[keep], y[keep], reg[keep]


def align229(X, wl, ref_wl):
    sel = [int(np.argmin(np.abs(wl - w))) for w in ref_wl]
    return X[:, sel]


def load_domains(log):
    doms, ref_wl = {}, None
    for yr in (2025, 2018, 2019):        # 先 2025 定 229 参考网格
        if yr == 2025:
            X, y, reg, wl = dp.load_csv_data_2025(SPEC25, SUGAR25)
            ref_wl = np.asarray(wl, float)
        else:
            X, y, reg, wl = dp.load_excel_data(RAW_XLSX[yr])
        wl = np.asarray(wl, float)
        X, y, reg = apply_filters(X, y, reg)
        for org in np.unique(reg):
            if org == "未知":
                continue
            m = reg == org
            doms[(yr, org)] = dict(X=X[m], y=y[m], wl=wl)
    for d in doms.values():
        d["X229"] = align229(snv(d["X"]), d["wl"], ref_wl)   # 229 版预先 SNV
        d["Xn"] = snv(d["X"])                                  # 原生 SNV
    log(f"域: {sorted((y, o) for y, o in doms)}")
    return doms


def transfer(src, tgt, same_inst, ncal, rng, rep=REP):
    key = "Xn" if (same_inst and src["Xn"].shape[1] == tgt["Xn"].shape[1]) else "X229"
    Xs, ys, Xt, yt = src[key], src["y"], tgt[key], tgt["y"]
    k = src.setdefault(f"k_{key}", sel_k(Xs, ys))
    predt = fit(Xs, ys, k).predict(Xt).ravel()
    zero = rpd(predt, yt)
    n = len(yt)
    if n <= ncal + 8:
        return zero, np.nan, np.nan
    sb, mu = [], []
    for _ in range(rep):
        idx = rng.permutation(n)
        cal, ev = idx[:ncal], idx[ncal:]
        a, b = np.polyfit(predt[cal], yt[cal], 1)
        sb.append(rpd(a * predt[ev] + b, yt[ev]))
        m2 = fit(np.vstack([Xs, Xt[cal]]), np.concatenate([ys, yt[cal]]), k)
        mu.append(rpd(m2.predict(Xt[ev]).ravel(), yt[ev]))
    return zero, float(np.mean(sb)), float(np.mean(mu))


def shift_type(sy, so, ty, to):
    if INSTRUMENT[sy] != INSTRUMENT[ty]:
        return "跨仪器(涉2025)"
    if so == to:
        return "同产地·跨年"
    if sy == ty:
        return "跨产地·同年"
    return "跨产地·跨年·同仪器"


def main():
    log = _eu.get_logger("60_decomposed_transfer").log
    doms = load_domains(log)
    keys = sorted(doms, key=lambda k: (k[0], k[1]))
    rng = np.random.default_rng(SEED)

    # 域内可行性（原生 SNV，5 折）
    feas = []
    for (yr, org) in keys:
        X, y = doms[(yr, org)]["Xn"], doms[(yr, org)]["y"]
        pr = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X):
            k = sel_k(X[tr], y[tr])
            pr[te] = fit(X[tr], y[tr], k).predict(X[te]).ravel()
        feas.append(dict(域=f"{yr}_{org}", 仪器=INSTRUMENT[yr], 波段=X.shape[1], n=len(y),
                         SSC_std=round(y.std(), 3), 域内RMSE=round(np.sqrt(np.mean((pr - y) ** 2)), 3),
                         域内RPD=round(rpd(pr, y), 3)))
    feas_df = pd.DataFrame(feas)
    log("域内可行性(原生SNV):\n" + feas_df.to_string(index=False))

    # 逐对迁移 × n_cal
    rows = []
    for (sy, so) in keys:
        for (ty, to) in keys:
            if (sy, so) == (ty, to):
                continue
            si = INSTRUMENT[sy] == INSTRUMENT[ty]
            rec = dict(源=f"{sy}_{so}", 目标=f"{ty}_{to}", 漂移类型=shift_type(sy, so, ty, to))
            for nc in NCALS:
                z, s, m = transfer(doms[(sy, so)], doms[(ty, to)], si, nc, rng)
                rec["零标签RPD"] = round(z, 3)
                rec[f"斜率偏置_ncal{nc}"] = round(s, 3) if s == s else np.nan
                rec[f"模型更新_ncal{nc}"] = round(m, 3) if m == m else np.nan
            rows.append(rec)
    pair_df = pd.DataFrame(rows)

    # 按漂移类型汇总（n_cal=50 模型更新）
    col = "模型更新_ncal50"
    agg = pair_df.groupby("漂移类型").agg(
        对数=("源", "size"),
        零标签RPD中位=("零标签RPD", "median"),
        模型更新RPD中位=(col, "median"),
        模型更新RPD最好=(col, "max"),
        达可用对数=(col, lambda s: int((s >= USABLE).sum())),
    ).reset_index().round(3)
    order = {"同产地·跨年": 0, "跨产地·同年": 1, "跨产地·跨年·同仪器": 2, "跨仪器(涉2025)": 3}
    agg = agg.sort_values("漂移类型", key=lambda s: s.map(order)).reset_index(drop=True)
    log("\n按漂移类型(n_cal=50,模型更新):\n" + agg.to_string(index=False))

    # n_cal 曲线（各漂移类型模型更新中位 RPD）
    curve = []
    for st in order:
        sub = pair_df[pair_df["漂移类型"] == st]
        curve.append(dict(漂移类型=st, **{f"ncal{nc}": round(sub[f"模型更新_ncal{nc}"].median(), 3) for nc in NCALS}))
    curve_df = pd.DataFrame(curve)
    log("\nn_cal 曲线(模型更新中位RPD):\n" + curve_df.to_string(index=False))

    # 池化+更新（同仪器其余各域合训 + 本地，留一域外验证）
    pooled = []
    for tk in keys:
        ty, to = tk
        srcs = [k for k in keys if k != tk and INSTRUMENT[k[0]] == INSTRUMENT[ty]]
        if not srcs:
            continue
        dimt = doms[tk]["Xn"].shape[1]
        key = "Xn" if all(doms[k]["Xn"].shape[1] == dimt for k in srcs) else "X229"
        Xs = np.vstack([doms[k][key] for k in srcs])
        ys = np.concatenate([doms[k]["y"] for k in srcs])
        Xt, yt = doms[tk][key], doms[tk]["y"]
        k = sel_k(Xs, ys)
        z = rpd(fit(Xs, ys, k).predict(Xt).ravel(), yt)
        mu = []
        n = len(yt)
        for _ in range(REP):
            idx = rng.permutation(n)
            cal, ev = idx[:50], idx[50:]
            m2 = fit(np.vstack([Xs, Xt[cal]]), np.concatenate([ys, yt[cal]]), k)
            mu.append(rpd(m2.predict(Xt[ev]).ravel(), yt[ev]))
        pooled.append(dict(目标域=f"{ty}_{to}", 仪器=INSTRUMENT[ty], 源域数=len(srcs),
                           池化零标签RPD=round(z, 3), 池化更新RPD_ncal50=round(float(np.mean(mu)), 3),
                           可用=int(np.mean(mu) >= USABLE)))
    pooled_df = pd.DataFrame(pooled)
    log("\n池化+本地更新(留一域外, n_cal=50):\n" + pooled_df.to_string(index=False))

    _eu.write_script_workbook(__file__, {
        "feasibility": feas_df, "by_shift": agg, "ncal_curve": curve_df,
        "pooled": pooled_df, "by_pair": pair_df})
    log(f"\n✓ 已写 {os.path.join(_OUT, '60_decomposed_transfer.xlsx')}")


if __name__ == "__main__":
    sys.exit(main())
