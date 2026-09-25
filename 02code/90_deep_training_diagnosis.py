#!/usr/bin/env python3
"""90_deep_training_diagnosis.py — 深度族源域内不学习的根因定位与修复候选。

背景：89 号在与 88 号**逐位相同**的划分上给出参照——PLSR 的源域留出 NRMSEP 为
0.31(corn)/0.70(tablet)/0.39(mango)/0.32(ossl_mir)/0.71(apple)，而 88 号里 64 号的
CNN 与 Phys+BL 源域留出 NRMSEP 多在 1.00 附近（=常数预测器）。同数据同划分下差这么多，
说明不是"这批数据学不出关系"，而是深度侧的实现或训练预算有问题。

64 号实现里有两处足以单独造成该现象：
  (1) 编码器末端是 ``AdaptiveAvgPool1d(1)``——把整条谱在波长轴上求平均，位置信息几乎全丢，
      而 PLSR 用的正是波长位置上的形状；
  (2) 训练是**全批量**、每个 epoch 只走一次 ``opt.step()``，400 个 epoch 即 400 次参数更新，
      对随机初始化的卷积网络远不足以离开"输出≈均值"的初始状态。

本脚本在同一批划分上跑"改进配方"（池化到 16 个 bin 后展平 + 小批量训练 + 更长早停），
与 88 号的现配方、89 号的 PLSR 并排比较，用来判定要不要重跑深度臂。

用法::

    python3 02code/90_deep_training_diagnosis.py --benchmarks corn,tablet,mango --per-bench 3
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, ".."))
F64 = npt.NDArray[np.float32]


def _load(mod_file: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, os.path.join(_CODE, mod_file))
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Encoder2(nn.Module):
    """与 64 号同构，唯一改动：末端自适应池化到 BINS 个 bin 后展平，保留波长位置信息。"""

    BINS = 16

    def __init__(self, feat: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 7, padding=3), nn.ReLU(), nn.AdaptiveAvgPool1d(self.BINS))
        self.fc = nn.Linear(32 * self.BINS, feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.unsqueeze(1))
        out: torch.Tensor = torch.relu(self.fc(h.flatten(1)))
        return out


class CNNReg2(nn.Module):
    def __init__(self, feat: int = 64) -> None:
        super().__init__()
        self.enc = Encoder2(feat)
        self.head = nn.Linear(feat, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.head(self.enc(x)).squeeze(-1)
        return out


class PhysBL2(nn.Module):
    def __init__(self, p: int, k: int = 8, feat: int = 64) -> None:
        super().__init__()
        self.enc = Encoder2(feat)
        self.to_b = nn.Linear(feat, k)
        self.a = nn.Parameter(torch.randn(k, p) * 0.01)
        self.reg = nn.Linear(k, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = F.softplus(self.to_b(self.enc(x)))
        recon = b @ F.softplus(self.a)
        y = self.reg(b).squeeze(-1)
        return y, recon, b


def _train(model: nn.Module, x_np: F64, y_np: F64, device: str, seed: int, *,
           lam: float = 0.0, epochs: int = 300, batch: int = 32, lr: float = 1e-3,
           patience: int = 40) -> nn.Module:
    """小批量训练 + 源域内 1/6 留出早停（与 64 号同口径的验证集比例）。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x_np))
    nv = max(4, len(x_np) // 6)
    vi, ti = idx[:nv], idx[nv:]
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    ymu, ysd = y[ti].mean(), y[ti].std() + 1e-8
    yn = (y - ymu) / ysd
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.MSELoss()
    best, best_state, bad = 1e9, None, 0
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(ti), device=device)
        for s in range(0, len(ti), batch):
            sel = torch.as_tensor(ti, device=device)[order[s:s + batch]]
            opt.zero_grad()
            out = model(x[sel])
            if lam > 0:
                pred, recon, _ = out
                loss = lossf(pred, yn[sel]) + lam * lossf(recon, x[sel])
            else:
                loss = lossf(out, yn[sel])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            ov = model(x[vi])
            pv = ov[0] if lam > 0 else ov
            vl = float(lossf(pv, yn[vi]).item())
        if vl < best - 1e-5:
            best, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.ymu, model.ysd = ymu, ysd
    return model


def _predict(model: nn.Module, x_np: F64, device: str, *, phys: bool) -> npt.NDArray[np.float32]:
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(x_np, dtype=torch.float32, device=device))
        pred = out[0] if phys else out
        p = pred * model.ysd + model.ymu
    arr: npt.NDArray[np.float32] = p.cpu().numpy()
    return arr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmarks", default="corn,tablet,mango,ossl_mir,apple")
    ap.add_argument("--per-bench", type=int, default=3)
    ap.add_argument("--seeds", default="20060515,2023")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(_BASE, "04outputs", "90_deep_training_diagnosis.xlsx"))
    args = ap.parse_args()

    deep = _load("64_deep_transfer_server.py", "deep64")
    bench_mod = _load("61_benchmark_datasets.py", "bench61")
    bench_mod.PROC = os.path.join(_BASE, "03data", "processed", "benchmarks")
    seeds = [int(s) for s in args.seeds.split(",")]

    rows: list[dict[str, Any]] = []
    for name in args.benchmarks.split(","):
        if name not in bench_mod.LOADERS:
            continue
        b = bench_mod.LOADERS[name]()
        seen: set[tuple[str, int]] = set()
        combos = []
        for tk in bench_mod.build_transfer_tasks(b):
            key = (tk["src"], tk["prop_idx"])
            if key in seen:
                continue
            seen.add(key)
            combos.append(tk)
            if len(combos) >= args.per_bench:
                break
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
                rng = np.random.default_rng(seed)
                ps = rng.permutation(len(xs))
                n_hold = max(6, len(xs) // 5)
                s_te, s_tr = ps[:n_hold], ps[n_hold:]
                pt = rng.permutation(len(xt))
                t_te = pt[: len(xt) // 2]
                sd_s = float(np.std(ys[s_te])) or float("nan")
                sd_t = float(np.std(yt[t_te])) or float("nan")
                for meth in ("cnn_v2", "physbl_v2"):
                    phys = meth == "physbl_v2"
                    mdl: nn.Module = (PhysBL2(xs.shape[1]) if phys else CNNReg2()).to(args.device)
                    _train(mdl, xs[s_tr], ys[s_tr], args.device, seed, lam=0.3 if phys else 0.0)
                    rows.append({
                        "benchmark": name, "src": str(tk["src"]), "prop": tk.get("prop", tk["prop_idx"]),
                        "seed": seed, "method": meth,
                        "n_src_train": len(s_tr), "n_src_test": len(s_te), "n_tgt_test": len(t_te),
                        "nrmsep_source": float(np.sqrt(np.mean(
                            (_predict(mdl, xs[s_te], args.device, phys=phys) - ys[s_te]) ** 2))) / sd_s,
                        "nrmsep_target": float(np.sqrt(np.mean(
                            (_predict(mdl, xt[t_te], args.device, phys=phys) - yt[t_te]) ** 2))) / sd_t,
                    })
                print(f"  {name} src={tk['src']} prop={tk.get('prop', tk['prop_idx'])} seed={seed} "
                      f"改进配方 源域 {rows[-2]['nrmsep_source']:.3f}/{rows[-1]['nrmsep_source']:.3f} "
                      f"目标域 {rows[-2]['nrmsep_target']:.3f}/{rows[-1]['nrmsep_target']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        print("没有可用组合")
        return 1
    with pd.ExcelWriter(args.out) as w:
        df.to_excel(w, sheet_name="improved", index=False)
    print("\n按基准 × 方法汇总（中位 NRMSEP）")
    print(df.groupby(["benchmark", "method"])[["nrmsep_source", "nrmsep_target"]].median().round(3).to_string())
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
