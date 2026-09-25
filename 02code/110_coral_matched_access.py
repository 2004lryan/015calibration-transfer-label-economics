"""
110_coral_matched_access.py: 零标签比较里 CORAL 的无标签目标谱用量与深度自适应方法对齐后的敏感性

【为什么做】表 2 里各零标签方法看到的无标签目标谱不一样：CORAL 用目标域全部光谱（含测试半，直推式），
DANN 与 DeepCORAL 只用校准池，CNN 与 Phys+BL 不用。补充材料 S16.2 用「同一批零标签运行上 CORAL
逐个胜过四个深度方法」说明代表性配对不是挑出来的；CORAL 的领先里有多少来自它多看了测试半的
无标签谱，此前没有拆开。本脚本在 62 号的每一次零标签运行上，另算一个只用校准池估计目标域
矩的 CORAL（coral_pool），与 DANN、DeepCORAL 的无标签目标谱用量相同，再与 64 号的四个深度方法逐任务比较。

【怎么保证与正典同一批运行】62 号的 run_task 里消耗随机数的只有：大域子采样、漂移特征（MMD 抽样与
切片 Wasserstein 的随机方向）、每次重复的 rng.permutation，以及每个预算的 rng.choice 抽校准集；
sel_k 用固定 random_state 的 KFold，不消耗 rng。本脚本按同一顺序消耗随机数（校准集抽取只模拟、不拟合），
因此每次重复的测试半与校准池与 62 号逐次相同。自检：本脚本重算的 zero_shot 与 coral（全部目标谱）
必须与 62 号 curves 表逐值一致，否则直接报错退出。

运行方式（项目根目录）:
    python 02code/110_coral_matched_access.py

输出文件:
    04outputs/110_coral_matched_access.xlsx — runs（每次运行三种零标签读数）/ repro（与 62 号的逐值自检）/
        pairs（CORAL 两种用量各对四个深度方法：胜率、配对 δ、Wilcoxon P，仪器与五基准合并）/
        coral_access（逐任务 coral_pool 相对 coral 的变化）
"""

import importlib.util
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import wilcoxon

F64 = npt.NDArray[np.float64]

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, ".."))
_OUT = os.path.join(_BASE, "04outputs")


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_CODE, fname))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m62 = _load("engine62", "62_crossover_engine.py")
bench = m62.bench
_eu = m62._eu

BENCHMARKS = ["corn", "tablet", "mango", "ossl_mir", "apple"]
REP = 5                                   # 62 号正典运行的 --rep
DEEP = ["cnn_zeroshot", "dann", "deepcoral", "physbl_zeroshot"]


def coral_fit_apply(Xs: F64, Xt_est: F64, Xt_apply: F64, eps: float = 1e-2) -> F64:
    """与 62 号 coral() 同一公式，只是目标域均值与协方差由 Xt_est 估计，再作用到 Xt_apply。"""
    mus, mut = Xs.mean(0), Xt_est.mean(0)
    Cs = np.cov(Xs, rowvar=False) + eps * np.eye(Xs.shape[1])
    Ct = np.cov(Xt_est, rowvar=False) + eps * np.eye(Xt_est.shape[1])

    def ms(C: F64, inv: bool) -> F64:
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-10, None)
        w = 1 / np.sqrt(w) if inv else np.sqrt(w)
        out: F64 = (V * w) @ V.T
        return out
    res: F64 = (Xt_apply - mut) @ ms(Ct, True) @ ms(Cs, False) + mus
    return res


def run_task_zero(dom_s: dict, dom_t: dict, prop_idx: int, rng: np.random.Generator) -> list[tuple] | None:
    """62 号 run_task 的零标签部分；随机数消耗顺序与之逐步相同。"""
    Xs_r, ys = dom_s["X"], dom_s["Y"][:, prop_idx]
    Xt_r, yt = dom_t["X"], dom_t["Y"][:, prop_idx]
    okS = np.isfinite(Xs_r).all(1) & np.isfinite(ys)
    okT = np.isfinite(Xt_r).all(1) & np.isfinite(yt)
    Xs_r, ys, Xt_r, yt = Xs_r[okS], ys[okS], Xt_r[okT], yt[okT]
    if len(Xs_r) > m62.DOMAIN_CAP:
        i = rng.choice(len(Xs_r), m62.DOMAIN_CAP, replace=False)
        Xs_r, ys = Xs_r[i], ys[i]
    if len(Xt_r) > m62.DOMAIN_CAP:
        i = rng.choice(len(Xt_r), m62.DOMAIN_CAP, replace=False)
        Xt_r, yt = Xt_r[i], yt[i]
    Xs_a, Xt_a = m62.align_common(Xs_r, dom_s["wl"], Xt_r, dom_t["wl"])
    Xs, Xt = m62.snv(Xs_a), m62.snv(Xt_a)
    if len(Xs) < 20 or len(Xt) < 24:
        return None
    m62.task_features(Xs, Xt, ys, yt, rng)         # 只为按原顺序消耗随机数
    ks = m62.sel_k(Xs, ys, rng)
    src = m62.fit_pls(Xs, ys, ks)
    try:
        pred_coral = src.predict(m62.coral(Xs, Xt)).ravel()
    except Exception:
        pred_coral = None
    rows = []
    nt = len(Xt)
    for r in range(REP):
        perm = rng.permutation(nt)
        test = perm[: nt // 2]
        pool = perm[nt // 2:]
        ytest = yt[test]
        rows.append((r, "zero_shot", m62.rmse(src.predict(Xt[test]).ravel(), ytest)))
        if pred_coral is not None:
            rows.append((r, "coral", m62.rmse(pred_coral[test], ytest)))
            try:
                pp = src.predict(coral_fit_apply(Xs, Xt[pool], Xt[test])).ravel()
                rows.append((r, "coral_pool", m62.rmse(pp, ytest)))
            except Exception:
                pass
        for nc in m62.NCAL_GRID:                   # 各预算的校准集抽取：只消耗随机数
            if nc == 0 or nc > len(pool):
                continue
            rng.choice(pool, nc, replace=False)
    return rows


def pair_stats(a: pd.Series, b: pd.Series) -> dict:
    """a 为 CORAL、b 为深度方法的逐任务 RMSEP 中位数；胜率与配对 δ（正值偏向 CORAL）。"""
    idx = a.index.intersection(b.index)
    x, y = a.loc[idx].to_numpy(), b.loc[idx].to_numpy()
    win = float(np.mean(x < y))
    lose = float(np.mean(x > y))
    p = float(wilcoxon(x, y).pvalue) if len(idx) > 0 and np.any(x != y) else float("nan")
    return {"n_tasks": len(idx), "coral_win_pct": 100 * win, "paired_delta": win - lose, "wilcoxon_p": p}


_BENCH_CACHE: dict = {}


def _job(args: tuple[str, int]) -> list[dict]:
    """一个（基准，任务序号）的 5 粒种子；每个进程各自载入一次基准数据。"""
    name, ti = args
    if name not in _BENCH_CACHE:
        b = bench.LOADERS[name]()
        _BENCH_CACHE[name] = (b, bench.build_transfer_tasks(b))
    b, tasks = _BENCH_CACHE[name]
    tk = tasks[ti]
    tid = f"{name}|{tk['src']}->{tk['tgt']}|{tk['prop_name']}"
    rows = []
    for seed in m62.SEEDS:
        rng = np.random.default_rng(seed)
        out = run_task_zero(b["domains"][tk["src"]], b["domains"][tk["tgt"]], tk["prop_idx"], rng)
        for r, meth, val in out or []:
            rows.append({"task_id": tid, "benchmark": name, "shift_type": tk["shift_type"],
                         "method": meth, "seed": seed, "rep": r, "rmsep": val})
    return rows


def main() -> None:
    log = _eu.get_logger("110_coral_matched_access")
    _eu.log_experiment_header(log, {"任务": "零标签 CORAL 的无标签目标谱用量对齐（校准池）", "基准": BENCHMARKS,
                                    "种子": m62.SEEDS, "REP": REP})
    jobs = []
    for name in BENCHMARKS:
        n = len(bench.build_transfer_tasks(bench.LOADERS[name]()))
        log.log(f"[{name}] {n} 任务 × {len(m62.SEEDS)} 种子")
        jobs += [(name, ti) for ti in range(n)]
    rows = []
    with ProcessPoolExecutor(max_workers=4) as ex:   # 各任务互不依赖，随机数按（任务，种子）各自重置
        for k, part in enumerate(ex.map(_job, jobs), 1):
            rows += part
            if k % 20 == 0:
                log.log(f"    {k}/{len(jobs)} 任务完成")
    runs = pd.DataFrame(rows)

    # 自检：zero_shot 与 coral 与 62 号正典逐值一致
    c62 = pd.read_excel(os.path.join(_OUT, "62_crossover_engine.xlsx"), sheet_name="curves")
    c62 = c62[(c62["n_cal"] == 0) & c62["method"].isin(["zero_shot", "coral"])]
    key = ["task_id", "method", "seed", "rep"]
    mg = runs[runs["method"].isin(["zero_shot", "coral"])].merge(c62[key + ["rmsep"]], on=key, how="outer",
                                                                 suffixes=("_110", "_62"), indicator=True)
    mg["abs_diff"] = (mg["rmsep_110"] - mg["rmsep_62"]).abs()
    n_only = int((mg["_merge"] != "both").sum())
    max_diff = float(mg["abs_diff"].max())
    repro = pd.DataFrame([{"rows_both": int((mg["_merge"] == "both").sum()), "rows_unmatched": n_only,
                           "max_abs_diff": max_diff}])
    log.log(f"\n自检：与 62 号 curves 对齐 {repro.iloc[0].to_dict()}")
    if n_only or not max_diff <= 1e-9:
        raise SystemExit("与 62 号正典的零标签运行不一致：划分复现失败，结果作废")

    # 逐任务中位数，与 64 号四个深度方法逐一比较
    med = runs.groupby(["task_id", "shift_type", "method"])["rmsep"].median().unstack("method")
    d64 = pd.read_excel(os.path.join(_OUT, "64_deep_transfer_server.xlsx"), sheet_name="curves")
    d64 = d64[(d64["n_cal"] == 0) & d64["method"].isin(DEEP)]
    dmed = d64.groupby(["task_id", "method"])["rmsep"].median().unstack("method")
    shift = med.reset_index(level="shift_type")["shift_type"]
    med = med.reset_index(level="shift_type", drop=True)
    pairs = []
    for scope, ids in (("instrument", shift[shift == "instrument"].index), ("all", shift.index)):
        for cm in ("coral", "coral_pool"):
            for dm in DEEP:
                s = pair_stats(med.loc[med.index.intersection(ids), cm].dropna(), dmed[dm].dropna())
                pairs.append({"scope": scope, "coral_variant": cm, "deep": dm, **s})
    pairs = pd.DataFrame(pairs)
    acc = med[["coral", "coral_pool"]].dropna().copy()
    acc["shift_type"] = shift.loc[acc.index]
    acc["rel_change_pct"] = 100 * (acc["coral_pool"] / acc["coral"] - 1)
    log.log("\n===== CORAL 两种用量对四个深度方法（逐任务中位 RMSEP）=====\n" + pairs.to_string(index=False))
    log.log("\n===== coral_pool 相对 coral 的逐任务变化（%）=====\n"
            + acc.groupby("shift_type")["rel_change_pct"].describe().to_string())
    _eu.write_script_workbook(__file__, {"runs": runs, "repro": repro, "pairs": pairs,
                                         "coral_access": acc.reset_index()})
    print("→ 04outputs/110_coral_matched_access.xlsx")


if __name__ == "__main__":
    main()
