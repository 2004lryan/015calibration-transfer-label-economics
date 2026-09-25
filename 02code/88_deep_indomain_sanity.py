#!/usr/bin/env python3
"""88_deep_indomain_sanity.py — 深度基线在**源域内**是否真的学到了东西。

为什么必须有这一支：统一基准只记录目标域误差，而深度族的目标域 NRMSEP 中位在 1.0 附近
（NRMSEP $=1$ 即"预测目标域均值"的水平）。仅凭这一点，"深度方法迁移失败"与"深度方法压根没
训练起来"是观测等价的，审稿人有充分理由按后者驳回全文的核心对比。本脚本把两者分开：

    用与 64 号完全相同的架构、超参与预处理，在**源域内部**留出一块从不参与训练、也不参与
    早停的测试集，报告同一个源模型在「源域留出集」与「目标域测试集」上的 NRMSEP。

四个深度源模型全部纳入（CNN、Phys+BL、DANN、Deep CORAL）。后两个训练时要看目标域的无标签
谱，只喂给它们目标域中不参与评测的那一半，否则源域阳性对照会借到评测样本的分布信息。

判读：源域 NRMSEP 明显小于 1 ⇒ 模型确实学到了源域的谱–含量关系，目标域 NRMSEP≈1 就是**迁移
失败**而非训练失败；若源域 NRMSEP 也≈1，则本文对深度族的结论只能限定为"在该训练预算下"。

按 (基准, 源域, 属性) 去重，因为同一个源模型服务于以它为源的全部迁移任务。

本脚本单进程单线程（BLAS 线程数在调度脚本里锁成 1，torch 也只在一个核上跑），给它整机
也快不了。正式跑法是**一个基准一个进程**，各写各的 xlsx，由 80 号合并：

    python3 02code/88_deep_indomain_sanity.py --device cpu --benchmarks corn \
        --out 04outputs/88_deep_indomain_sanity_corn.xlsx

五个基准之间没有任何共享状态（各自加载数据、各自训练、各自写盘），拆开不改变任何一个
数字。80 号默认读的就是这五份分基准文件；不带 --out 跑全部基准会写出一份合并名的 xlsx，
那份**不在 80 号的默认输入里**。

用法::

    python3 02code/88_deep_indomain_sanity.py --benchmarks corn,tablet,mango --per-bench 3 --seeds 20060515,2023
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from typing import Any

import numpy as np
import pandas as pd

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, ".."))


def _load(mod_file: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, os.path.join(_CODE, mod_file))
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks", default="corn,tablet,mango,ossl_mir,apple")
    ap.add_argument("--per-bench", type=int, default=9,
                    help="每个基准最多取几组 (源域, 属性)，在源域之间轮转取；默认 9 = 苹果的源域个数，"
                         "取到这个数则 5 个基准的每个源域都至少覆盖一次")
    ap.add_argument("--seeds", default="20060515,20041210,19810915,2023,2024")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--train-steps", type=int, default=None,
                    help="覆盖 64 号的固定优化预算（默认 None 即沿用 TRAIN_STEPS=2000）。"
                         "用它跑预算敏感性时必须对全部五个基准用同一个值，"
                         "逐基准调参会破坏「同一固定预算、不逐基准调参」的协议。")
    ap.add_argument("--out", default=os.path.join(_BASE, "04outputs", "88_deep_indomain_sanity.xlsx"))
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
            Xs, ys = ds["X"], ds["Y"][:, tk["prop_idx"]]
            Xt, yt = dt["X"], dt["Y"][:, tk["prop_idx"]]
            okS = np.isfinite(Xs).all(1) & np.isfinite(ys)
            okT = np.isfinite(Xt).all(1) & np.isfinite(yt)
            Xs, ys, Xt, yt = Xs[okS], ys[okS], Xt[okT], yt[okT]
            Xs, Xt = deep.align_common(Xs, ds["wl"], Xt, dt["wl"])
            Xs, Xt = deep.snv(Xs).astype(np.float32), deep.snv(Xt).astype(np.float32)
            if len(Xs) < 24 or len(Xt) < 24:
                continue
            for seed in seeds:
                rng = np.random.default_rng(seed)
                ps = rng.permutation(len(Xs))
                n_hold = max(6, len(Xs) // 5)          # 源域留出测试集：不训练、不早停
                s_te, s_tr = ps[:n_hold], ps[n_hold:]
                pt = rng.permutation(len(Xt))
                t_te = pt[: len(Xt) // 2]              # 与 64 号一致的目标域对半留出
                sd_s = float(np.std(ys[s_te])) or float("nan")
                sd_t = float(np.std(yt[t_te])) or float("nan")
                # 目标域的无标签谱：域自适应模型训练时可见，但只给未参与评测的那一半，
                # 否则源域阳性对照会借到评测样本的分布信息。
                t_un = pt[len(Xt) // 2:]
                kw = {"seed": seed}
                if args.train_steps is not None:
                    kw["steps"] = args.train_steps
                fitted = {
                    "cnn": (deep.train_cnn(Xs[s_tr], ys[s_tr], args.device, **kw),
                            deep.predict_cnn),
                    "physbl": (deep.train_physbl(Xs[s_tr], ys[s_tr], args.device, **kw),
                               deep.predict_physbl),
                    "dann": (deep.train_dann(Xs[s_tr], ys[s_tr], Xt[t_un], args.device, **kw),
                             deep.predict_head),
                    "deepcoral": (deep.train_deepcoral(Xs[s_tr], ys[s_tr], Xt[t_un], args.device,
                                                       **kw),
                                  deep.predict_head),
                }
                for meth, (mod, pred) in fitted.items():
                    rows.append({
                        "benchmark": name, "src": str(tk["src"]), "prop": tk.get("prop", tk["prop_idx"]),
                        "seed": seed, "method": meth,
                        "n_src_train": len(s_tr), "n_src_test": len(s_te), "n_tgt_test": len(t_te),
                        "nrmsep_source": deep.rmse(pred(mod, Xs[s_te], args.device), ys[s_te]) / sd_s,
                        "nrmsep_target": deep.rmse(pred(mod, Xt[t_te], args.device), yt[t_te]) / sd_t,
                    })
                last = rows[-len(fitted):]
                print("  {} src={} prop={} seed={} 源域 {} 目标域 {}".format(
                    name, tk["src"], tk.get("prop", tk["prop_idx"]), seed,
                    "/".join(f"{r['nrmsep_source']:.3f}" for r in last),
                    "/".join(f"{r['nrmsep_target']:.3f}" for r in last)), flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        print("没有可用组合")
        return 1
    with pd.ExcelWriter(args.out) as w:
        df.to_excel(w, sheet_name="indomain", index=False)
    print("\n按基准 × 方法汇总（中位 NRMSEP）")
    summary = (df.groupby(["benchmark", "method"])[["nrmsep_source", "nrmsep_target"]]
               .median().round(3))
    print(summary.to_string())
    print("\n全体中位：源域 {:.3f}　目标域 {:.3f}　（{} 次训练）".format(
        df["nrmsep_source"].median(), df["nrmsep_target"].median(), len(df)))
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
