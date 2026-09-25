#!/usr/bin/env python3
"""89_source_reference_baselines.py — 与 88 号同一批划分上的**参照基线**。

为什么必须有这一支：88 号回答了"深度模型在源域内学到了吗"，但只有它自己的数字无法判读——
源域 NRMSEP≈1 究竟是模型没学到，还是该源域在这个样本量下本来就学不出关系？本脚本在**逐位
相同的划分**上补两条参照：

    (i) PLSR（成分数由源域训练集内 5 折交叉验证选出），代表"这批数据在同样输入、同样划分下
        能学到什么"；
    (ii) 常数预测器（源域训练集的均值），代表 NRMSEP=1 的字面含义。

判读：若 PLSR 的源域留出 NRMSEP 明显小于 1 而深度模型≈1，则深度模型确实没有训练起来；若两者
都≈1，则该源域在此划分下本就不可学，深度族的目标域表现不能单独归咎于训练预算。

划分与 88 号逐位一致（同 seed、同 permutation、同 SNV/共同波段对齐），因此两张表可按
(benchmark, src, prop, seed) 直接拼接。

用法::

    python3 02code/89_source_reference_baselines.py --benchmarks corn,tablet,mango,ossl_mir,apple
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, ".."))


def _load(mod_file: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, os.path.join(_CODE, mod_file))
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fit_plsr(x_tr: npt.NDArray[np.floating[Any]], y_tr: npt.NDArray[np.floating[Any]],
              max_lv: int, seed: int) -> PLSRegression:
    """源域训练集内 5 折 CV 选成分数，绝不看留出集或目标域。"""
    upper = int(min(max_lv, x_tr.shape[1], max(1, len(x_tr) - 2)))
    best_lv, best_err = 1, float("inf")
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for lv in range(1, upper + 1):
        errs: list[float] = []
        for tr_i, va_i in kf.split(x_tr):
            if len(tr_i) <= lv:
                continue
            mdl = PLSRegression(n_components=lv).fit(x_tr[tr_i], y_tr[tr_i])
            pred = np.asarray(mdl.predict(x_tr[va_i])).ravel()
            errs.append(float(np.sqrt(np.mean((pred - y_tr[va_i]) ** 2))))
        if errs and float(np.mean(errs)) < best_err:
            best_lv, best_err = lv, float(np.mean(errs))
    return PLSRegression(n_components=best_lv).fit(x_tr, y_tr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks", default="corn,tablet,mango,ossl_mir,apple")
    ap.add_argument("--per-bench", type=int, default=9,
                    help="每个基准最多取几组 (源域, 属性)，在源域之间轮转取，与 88 号一致（默认 9）")
    ap.add_argument("--seeds", default="20060515,20041210,19810915,2023,2024")
    ap.add_argument("--max-lv", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(_BASE, "04outputs", "89_source_reference_baselines.xlsx"))
    args = ap.parse_args()

    deep = _load("64_deep_transfer_server.py", "deep64")
    bench_mod = _load("61_benchmark_datasets.py", "bench61")
    bench_mod.PROC = os.path.join(_BASE, "03data", "processed", "benchmarks")
    seeds = [int(s) for s in args.seeds.split(",")]

    rows: list[dict[str, Any]] = []
    for name in args.benchmarks.split(","):
        if name not in bench_mod.LOADERS:
            print(f"⚠ {name} 未注册，跳过")
            continue
        b = bench_mod.LOADERS[name]()
        tasks = bench_mod.build_transfer_tasks(b)
        # 在源域之间轮转取样：任务表按源域成组排列，直接取前 N 个会让证据全来自第一个源域
        # （玉米的前三条曾全是 m5）。先每个源域各取一个属性，再回头取第二个。
        seen: set[tuple[str, int]] = set()
        by_src: dict[str, list[Any]] = {}
        order: list[str] = []
        for tk in tasks:
            key = (str(tk["src"]), int(tk["prop_idx"]))
            if key in seen:
                continue
            seen.add(key)
            s = str(tk["src"])
            if s not in by_src:
                by_src[s] = []
                order.append(s)
            by_src[s].append(tk)
        combos: list[Any] = []
        i = 0
        while len(combos) < args.per_bench and any(by_src[s] for s in order):
            s = order[i % len(order)]
            if by_src[s]:
                combos.append(by_src[s].pop(0))
            i += 1
        for tk in combos:
            ds, dt = b["domains"][tk["src"]], b["domains"][tk["tgt"]]
            xs, ys = ds["X"], ds["Y"][:, tk["prop_idx"]]
            xt, yt = dt["X"], dt["Y"][:, tk["prop_idx"]]
            ok_s = np.isfinite(xs).all(1) & np.isfinite(ys)
            ok_t = np.isfinite(xt).all(1) & np.isfinite(yt)
            xs, ys, xt, yt = xs[ok_s], ys[ok_s], xt[ok_t], yt[ok_t]
            xs, xt = deep.align_common(xs, ds["wl"], xt, dt["wl"])
            xs, xt = deep.snv(xs).astype(np.float32), deep.snv(xt).astype(np.float32)
            if len(xs) < 24 or len(xt) < 24:
                continue
            for seed in seeds:
                rng = np.random.default_rng(seed)                 # 与 88 号逐位一致
                ps = rng.permutation(len(xs))
                n_hold = max(6, len(xs) // 5)
                s_te, s_tr = ps[:n_hold], ps[n_hold:]
                pt = rng.permutation(len(xt))
                t_te = pt[: len(xt) // 2]
                sd_s = float(np.std(ys[s_te])) or float("nan")
                sd_t = float(np.std(yt[t_te])) or float("nan")
                pls = _fit_plsr(xs[s_tr], ys[s_tr], args.max_lv, seed)
                const = float(np.mean(ys[s_tr]))
                for meth in ("plsr", "source_mean"):
                    if meth == "plsr":
                        p_s = np.asarray(pls.predict(xs[s_te])).ravel()
                        p_t = np.asarray(pls.predict(xt[t_te])).ravel()
                    else:
                        p_s = np.full(len(s_te), const)
                        p_t = np.full(len(t_te), const)
                    rows.append({
                        "benchmark": name, "src": str(tk["src"]), "prop": tk.get("prop", tk["prop_idx"]),
                        "seed": seed, "method": meth,
                        "n_src_train": len(s_tr), "n_src_test": len(s_te), "n_tgt_test": len(t_te),
                        "n_components": int(pls.n_components) if meth == "plsr" else 0,
                        "nrmsep_source": float(np.sqrt(np.mean((p_s - ys[s_te]) ** 2))) / sd_s,
                        "nrmsep_target": float(np.sqrt(np.mean((p_t - yt[t_te]) ** 2))) / sd_t,
                    })
                print(f"  {name} src={tk['src']} prop={tk.get('prop', tk['prop_idx'])} seed={seed} "
                      f"PLSR 源域 {rows[-2]['nrmsep_source']:.3f} 目标域 {rows[-2]['nrmsep_target']:.3f}"
                      f"　常数 源域 {rows[-1]['nrmsep_source']:.3f} 目标域 {rows[-1]['nrmsep_target']:.3f}",
                      flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        print("没有可用组合")
        return 1
    with pd.ExcelWriter(args.out) as w:
        df.to_excel(w, sheet_name="reference", index=False)
    print("\n按基准 × 方法汇总（中位 NRMSEP）")
    print(df.groupby(["benchmark", "method"])[["nrmsep_source", "nrmsep_target"]].median().round(3).to_string())
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
