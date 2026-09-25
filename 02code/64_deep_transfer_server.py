"""
64_deep_transfer_server.py: 深度标定转移方法(服务器 torch/4090)——为"简单 vs 深度"清算提供公平深度基线

方法(与 62 号经典曲线同 schema,可合并):
    cnn_zeroshot / cnn_finetune   1D-CNN 回归(源域训练; 零标签 / n_cal 微调)
    physbl_zeroshot / physbl_ft   可微 Beer-Lambert 分解(编码器→丰度b→共享分量A→重构+回归; 源训练/微调)
    deepcoral                     共享编码器 + 源/目标特征二阶矩对齐损失(无标签目标) + n_cal 头微调
    dann                          域对抗(GRL) + n_cal 头微调

公平性(§7.4): 与经典基线同任务/同 n_cal 网格/同 5 种子/同预处理(SNV)/同留出评估。
每(task,seed)源模型训练一次,跨 n_cal 复用微调。CUDA 确定性。

⚠ 2026-09-09 修正的深度侧训练缺陷（88/89/90 号脚本诊断，详见各脚本文档）：
  原实现有两处使深度基线在**源域内部**就退化成常数预测器——
  (1) 编码器末端 AdaptiveAvgPool1d(1) 把整条谱在波长轴上求平均，位置信息几乎全丢，
      而 PLSR 用的正是波长位置上的形状；
  (2) 全批量训练、每个 epoch 只走一次 opt.step()，400 个 epoch 即 400 次参数更新，
      不足以让随机初始化的卷积网络离开"输出≈均值"的初始状态。
  实测（89 号，同一批划分、同样输入）：PLSR 源域留出 NRMSEP 为 0.31(corn)/0.70(tablet)/
  0.39(mango)/0.32(ossl_mir)/0.71(apple)，而原配方的 CNN 与 Phys+BL 多在 1.00 附近。
  现改为：编码器池化到 ENCODER_BINS 个 bin 后展平；训练按**固定优化步数预算**走小批量，
  各基准的更新次数因此可比，不再随样本量而变。用 Encoder(bins=1) 与 steps=None 可复现旧配方。

运行方式(服务器):
    python 02code/64_deep_transfer_server.py --data-dir 03data/processed/benchmarks \
        --benchmarks corn,tablet,mango,apple,ossl_mir --device cuda --seeds 20060515,20041210,19810915,2023,2024

输出文件:
    04outputs/64_deep_transfer_server.xlsx  — curves(与62同schema) / budget / compute
    05logs/64_deep_transfer_server_*.log
"""
import argparse
import importlib.util
import os
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

# 光谱矩阵别名（float64 数组），与论文数学符号 X/Xs/Xt 对应
F64 = npt.NDArray[np.float64]

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, ".."))
_es = importlib.util.spec_from_file_location("export_utils", os.path.join(_CODE, "01_export_utils.py"))
assert _es is not None  # 对已知存在的 .py 文件 spec 恒非 None
assert _es.loader is not None  # 同理 loader 恒非 None
_eu = importlib.util.module_from_spec(_es)
_es.loader.exec_module(_eu)

NCAL_GRID = [0, 5, 10, 20, 40, 80]
DEF_SEEDS = [20060515, 20041210, 19810915, 2023, 2024]

# 训练预算：以**优化步数**而非 epoch 计，使样本量不同的基准拿到同样多的参数更新。
ENCODER_BINS = 16      # 编码器末端自适应池化的 bin 数（1 = 旧配方的全局平均池化）
TRAIN_STEPS = 2000     # 源模型（CNN / Phys+BL）与域自适应模型（DANN / DeepCORAL）的更新次数
BATCH = 32             # 小批量大小；样本不足时退化为全批量
CORAL_BATCH = 128      # Deep CORAL 的批要大于特征维(64)，否则批内协方差估计过噪
EVAL_EVERY = 25        # 每多少步在源域留出集上评一次（早停用）
PATIENCE_EVALS = 20    # 连续多少次评估无改善即早停（= 500 步）


def snv(X: F64) -> F64:
    out: F64 = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
    return out


def rmse(a: npt.NDArray[Any], b: npt.NDArray[Any]) -> float:
    # 预测值为 float32、参考值为 float64,故用 NDArray[Any] 兼容两种浮点精度
    return float(np.sqrt(np.mean((a - b) ** 2)))


def align_common(Xs: F64, wls: F64, Xt: F64, wlt: F64) -> tuple[F64, F64]:
    if Xs.shape[1] == Xt.shape[1]:
        return Xs, Xt
    if len(wls) <= len(wlt):
        sel = [int(np.argmin(np.abs(wlt - w))) for w in wls]
        return Xs, Xt[:, sel]
    sel = [int(np.argmin(np.abs(wls - w))) for w in wlt]
    return Xs[:, sel], Xt


# ----------------------------- 模型 -----------------------------
class Encoder(nn.Module):
    """1D-CNN 编码器。自适应池化到固定 bin 数,既与输入波段数无关,又保留波长位置信息。

    bins=1 即旧配方的全局平均池化(整条谱求平均),保留该取值仅为复现旧结果。
    """
    def __init__(self, feat: int = 64, bins: int = ENCODER_BINS) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 7, padding=3), nn.ReLU(), nn.AdaptiveAvgPool1d(bins))
        self.fc = nn.Linear(32 * bins, feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.unsqueeze(1))
        out: torch.Tensor = torch.relu(self.fc(h.flatten(1)))
        return out


class CNNReg(nn.Module):
    def __init__(self, feat: int = 64) -> None:
        super().__init__()
        self.enc = Encoder(feat)
        self.head = nn.Linear(feat, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.head(self.enc(x)).squeeze(-1)
        return out


class PhysBL(nn.Module):
    """编码器→丰度 b(≥0,K维)→共享分量谱 A(K×p,≥0)物理重构 x̂=bA + 回归 y=w·b。"""
    def __init__(self, p: int, K: int = 8, feat: int = 64) -> None:
        super().__init__()
        self.enc = Encoder(feat)
        self.to_b = nn.Linear(feat, K)
        self.A = nn.Parameter(torch.randn(K, p) * 0.01)
        self.reg = nn.Linear(K, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = F.softplus(self.to_b(self.enc(x)))
        recon = b @ F.softplus(self.A)
        y = self.reg(b).squeeze(-1)
        return y, recon, b


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lamb: float) -> torch.Tensor:
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, g: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lamb * g, None


def _batches(n: int, batch: int, steps: int, seed: int) -> Any:
    """按固定步数预算产出小批量下标。样本不足一个批时退化为全批量。"""
    g = np.random.default_rng(seed)
    order = g.permutation(n)
    pos = 0
    for _ in range(steps):
        if batch >= n:
            yield np.arange(n)
            continue
        if pos + batch > n:
            order = g.permutation(n)
            pos = 0
        yield order[pos:pos + batch]
        pos += batch


def train_cnn(Xs: F64, ys: F64, device: str, steps: int = TRAIN_STEPS, lr: float = 1e-3,
              wd: float = 1e-4, seed: int = 0, patience: int = PATIENCE_EVALS,
              batch: int = BATCH) -> Any:
    torch.manual_seed(seed)
    n = len(Xs)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    nv = max(4, n // 6)
    vi, ti = idx[:nv], idx[nv:]
    m = CNNReg().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    X = torch.tensor(Xs, dtype=torch.float32, device=device)
    Y = torch.tensor(ys, dtype=torch.float32, device=device)
    ymu, ysd = Y[ti].mean(), Y[ti].std() + 1e-8
    Yn = (Y - ymu) / ysd
    lossf = nn.MSELoss()
    best: float = 1e9
    best_state: dict[str, Any] | None = None
    bad = 0
    for step, sel in enumerate(_batches(len(ti), batch, steps, seed), 1):
        m.train()
        opt.zero_grad()
        lossf(m(X[ti[sel]]), Yn[ti[sel]]).backward()
        opt.step()
        if step % EVAL_EVERY:
            continue
        m.eval()
        with torch.no_grad():
            vl = lossf(m(X[vi]), Yn[vi]).item()
        if vl < best - 1e-5:
            best, best_state, bad = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        m.load_state_dict(best_state)
    m.ymu, m.ysd = ymu, ysd
    return m


def predict_cnn(m: Any, X: F64, device: str) -> npt.NDArray[np.float32]:
    m.eval()
    with torch.no_grad():
        p = m(torch.tensor(X, dtype=torch.float32, device=device)) * m.ysd + m.ymu
    out: npt.NDArray[np.float32] = p.cpu().numpy()
    return out


def train_physbl(Xs: F64, ys: F64, device: str, steps: int = TRAIN_STEPS, lr: float = 1e-3,
                 lam: float = 0.3, seed: int = 0, patience: int = PATIENCE_EVALS,
                 batch: int = BATCH) -> Any:
    torch.manual_seed(seed)
    p = Xs.shape[1]
    n = len(Xs)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    nv = max(4, n // 6)
    vi, ti = idx[:nv], idx[nv:]
    m = PhysBL(p).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    X = torch.tensor(Xs, dtype=torch.float32, device=device)
    Y = torch.tensor(ys, dtype=torch.float32, device=device)
    ymu, ysd = Y[ti].mean(), Y[ti].std() + 1e-8
    Yn = (Y - ymu) / ysd
    mse = nn.MSELoss()
    best: float = 1e9
    best_state: dict[str, Any] | None = None
    bad = 0
    for step, sel in enumerate(_batches(len(ti), batch, steps, seed), 1):
        m.train()
        opt.zero_grad()
        sub = ti[sel]
        yp, recon, _ = m(X[sub])
        (mse(yp, Yn[sub]) + lam * mse(recon, X[sub])).backward()
        opt.step()
        if step % EVAL_EVERY:
            continue
        m.eval()
        with torch.no_grad():
            yv, _, _ = m(X[vi])
            vl = mse(yv, Yn[vi]).item()
        if vl < best - 1e-5:
            best, best_state, bad = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        m.load_state_dict(best_state)
    m.ymu, m.ysd = ymu, ysd
    return m


def predict_physbl(m: Any, X: F64, device: str) -> npt.NDArray[np.float32]:
    m.eval()
    with torch.no_grad():
        yp, _, _ = m(torch.tensor(X, dtype=torch.float32, device=device))
    out: npt.NDArray[np.float32] = (yp * m.ysd + m.ymu).cpu().numpy()
    return out


def finetune(m: Any, Xc: F64, yc: F64, device: str, epochs: int = 80,
             lr: float = 5e-4, physbl: bool = False) -> Any:
    if len(Xc) == 0:
        return m
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    Xt_ = torch.tensor(Xc, dtype=torch.float32, device=device)
    yt_ = torch.tensor(yc, dtype=torch.float32, device=device)
    mse = nn.MSELoss()
    for _ in range(epochs):
        m.train()
        opt.zero_grad()
        if physbl:
            yp, recon, _ = m(Xt_)
            loss = mse(yp, (yt_ - m.ymu) / m.ysd) + 0.3 * mse(recon, Xt_)
        else:
            loss = mse(m(Xt_), (yt_ - m.ymu) / m.ysd)
        loss.backward()
        opt.step()
    return m


def train_dann(Xs: F64, ys: F64, Xt_un: F64, device: str,
               steps: int = TRAIN_STEPS, lr: float = 1e-3, seed: int = 0,
               batch: int = BATCH) -> dict[str, Any]:
    torch.manual_seed(seed)
    enc = Encoder().to(device)
    head = nn.Linear(64, 1).to(device)
    dom = nn.Linear(64, 1).to(device)
    opt = torch.optim.Adam(
        list(enc.parameters()) + list(head.parameters()) + list(dom.parameters()),
        lr=lr, weight_decay=1e-4)
    Xs_ = torch.tensor(Xs, dtype=torch.float32, device=device)
    ys_ = torch.tensor(ys, dtype=torch.float32, device=device)
    Xu_ = torch.tensor(Xt_un, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xs_))
    nv = max(4, len(Xs_) // 6)
    vi, ti = idx[:nv], idx[nv:]          # 源域留出集：只用来早停，不进任何损失
    ymu, ysd = ys_[ti].mean(), ys_[ti].std() + 1e-8
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    best: float = 1e9
    best_state: dict[str, Any] | None = None
    bad = 0
    src_batches = _batches(len(ti), batch, steps, seed)
    tgt_batches = _batches(len(Xu_), batch, steps, seed + 1)
    for step, (sj, ui) in enumerate(zip(src_batches, tgt_batches, strict=False), 1):
        si = ti[sj]
        lamb = 0.5 * (2 / (1 + np.exp(-10 * step / steps)) - 1)
        enc.train()
        opt.zero_grad()
        fs = enc(Xs_[si])
        fu = enc(Xu_[ui])
        reg = mse(head(fs).squeeze(-1), (ys_[si] - ymu) / ysd)
        fs_grl = GRL.apply(fs, lamb)  # type: ignore[no-untyped-call]  # Function.apply 上游 untyped
        fu_grl = GRL.apply(fu, lamb)  # type: ignore[no-untyped-call]  # 同上
        ds = bce(dom(fs_grl).squeeze(-1), torch.ones(len(si), device=device))
        du = bce(dom(fu_grl).squeeze(-1), torch.zeros(len(ui), device=device))
        (reg + ds + du).backward()
        opt.step()
        if step % EVAL_EVERY:
            continue
        enc.eval()
        with torch.no_grad():
            vl = float(mse(head(enc(Xs_[vi])).squeeze(-1), (ys_[vi] - ymu) / ysd).item())
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {"enc": {k: v.clone() for k, v in enc.state_dict().items()},
                          "head": {k: v.clone() for k, v in head.state_dict().items()}}
        else:
            bad += 1
            if bad >= PATIENCE_EVALS:
                break
    if best_state is not None:
        enc.load_state_dict(best_state["enc"])
        head.load_state_dict(best_state["head"])
    return {"enc": enc, "head": head, "ymu": ymu, "ysd": ysd}


def train_deepcoral(Xs: F64, ys: F64, Xt_un: F64, device: str, steps: int = TRAIN_STEPS,
                    lr: float = 1e-3, beta: float = 1.0, seed: int = 0,
                    batch: int = CORAL_BATCH) -> dict[str, Any]:
    torch.manual_seed(seed)
    enc = Encoder().to(device)
    head = nn.Linear(64, 1).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
    Xs_ = torch.tensor(Xs, dtype=torch.float32, device=device)
    ys_ = torch.tensor(ys, dtype=torch.float32, device=device)
    Xu_ = torch.tensor(Xt_un, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xs_))
    nv = max(4, len(Xs_) // 6)
    vi, ti = idx[:nv], idx[nv:]          # 源域留出集：只用来早停，不进任何损失
    ymu, ysd = ys_[ti].mean(), ys_[ti].std() + 1e-8
    mse = nn.MSELoss()
    best: float = 1e9
    best_state: dict[str, Any] | None = None
    bad = 0
    src_batches = _batches(len(ti), batch, steps, seed)
    tgt_batches = _batches(len(Xu_), batch, steps, seed + 1)
    for step, (sj, ui) in enumerate(zip(src_batches, tgt_batches, strict=False), 1):
        si = ti[sj]
        enc.train()
        opt.zero_grad()
        fs, fu = enc(Xs_[si]), enc(Xu_[ui])
        reg = mse(head(fs).squeeze(-1), (ys_[si] - ymu) / ysd)
        cs = torch.cov(fs.T)
        ct = torch.cov(fu.T)
        coral = ((cs - ct) ** 2).sum() / (4 * fs.shape[1] ** 2)
        (reg + beta * coral).backward()
        opt.step()
        if step % EVAL_EVERY:
            continue
        enc.eval()
        with torch.no_grad():
            vl = float(mse(head(enc(Xs_[vi])).squeeze(-1), (ys_[vi] - ymu) / ysd).item())
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {"enc": {k: v.clone() for k, v in enc.state_dict().items()},
                          "head": {k: v.clone() for k, v in head.state_dict().items()}}
        else:
            bad += 1
            if bad >= PATIENCE_EVALS:
                break
    if best_state is not None:
        enc.load_state_dict(best_state["enc"])
        head.load_state_dict(best_state["head"])
    return {"enc": enc, "head": head, "ymu": ymu, "ysd": ysd}


def predict_head(mod: dict[str, Any], X: F64, device: str) -> npt.NDArray[np.float32]:
    mod["enc"].eval()
    with torch.no_grad():
        p = mod["head"](mod["enc"](torch.tensor(X, dtype=torch.float32, device=device))).squeeze(-1)
    out: npt.NDArray[np.float32] = (p * mod["ysd"] + mod["ymu"]).cpu().numpy()
    return out


def run_task(
    bench: dict[str, Any], tk: dict[str, Any], seed: int, device: str, rep: int = 3
) -> list[tuple[int, str, int, int, float]]:
    ds, dt = bench["domains"][tk["src"]], bench["domains"][tk["tgt"]]
    Xs, ys = ds["X"], ds["Y"][:, tk["prop_idx"]]
    Xt, yt = dt["X"], dt["Y"][:, tk["prop_idx"]]
    okS = np.isfinite(Xs).all(1) & np.isfinite(ys)
    okT = np.isfinite(Xt).all(1) & np.isfinite(yt)
    Xs, ys, Xt, yt = Xs[okS], ys[okS], Xt[okT], yt[okT]
    Xs, Xt = align_common(Xs, ds["wl"], Xt, dt["wl"])
    Xs, Xt = snv(Xs).astype(np.float32), snv(Xt).astype(np.float32)
    if len(Xs) < 20 or len(Xt) < 24:
        return []
    rng = np.random.default_rng(seed)
    src_cnn = train_cnn(Xs, ys, device, seed=seed)
    src_phys = train_physbl(Xs, ys, device, seed=seed)
    rows = []
    nt = len(Xt)
    for r in range(rep):
        perm = rng.permutation(nt)
        test = perm[: nt // 2]
        pool = perm[nt // 2:]
        dann = train_dann(Xs, ys, Xt[pool], device, seed=seed + r)
        dcoral = train_deepcoral(Xs, ys, Xt[pool], device, seed=seed + r)
        rows.append((0, "cnn_zeroshot", rmse(predict_cnn(src_cnn, Xt[test], device), yt[test])))
        rows.append((0, "physbl_zeroshot", rmse(predict_physbl(src_phys, Xt[test], device), yt[test])))
        rows.append((0, "dann", rmse(predict_head(dann, Xt[test], device), yt[test])))
        rows.append((0, "deepcoral", rmse(predict_head(dcoral, Xt[test], device), yt[test])))
        for nc in NCAL_GRID:
            if nc == 0 or nc > len(pool):
                continue
            cal = rng.choice(pool, nc, replace=False)
            import copy
            fc = finetune(copy.deepcopy(src_cnn), Xt[cal], yt[cal], device)
            rows.append((nc, "cnn_finetune", rmse(predict_cnn(fc, Xt[test], device), yt[test])))
            fp = finetune(copy.deepcopy(src_phys), Xt[cal], yt[cal], device, physbl=True)
            rows.append((nc, "physbl_ft", rmse(predict_physbl(fp, Xt[test], device), yt[test])))
    return [(nc, meth, seed, r % rep, val) for r, (nc, meth, val) in enumerate(rows)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(_BASE, "03data", "processed", "benchmarks"))
    ap.add_argument("--benchmarks", default="corn,tablet,mango,apple,ossl_mir")
    ap.add_argument("--seeds", default=",".join(map(str, DEF_SEEDS)))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--rep", type=int, default=3)
    ap.add_argument("--tag", default=None,
                    help="给定则把本次结果写成 04outputs/64_shards/<tag>.csv，不动正式工作簿")
    ap.add_argument("--merge", action="store_true",
                    help="合并 04outputs/64_shards/*.csv 为正式工作簿（不跑实验）")
    args = ap.parse_args()

    shard_dir = os.path.join(_BASE, "04outputs", "64_shards")
    if args.merge:
        import glob
        import re
        # 只收 <基准>_<种子>.csv 形式的正式分片，冒烟/试跑产物（如 smoke_tablet.csv）不进合并
        files = sorted(f for f in glob.glob(os.path.join(shard_dir, "*.csv"))
                       if re.fullmatch(r"[a-z_]+_\d{4,8}\.csv", os.path.basename(f)))
        if not files:
            raise SystemExit(f"没有可合并的分片: {shard_dir}")
        merged = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        _eu.write_script_workbook(__file__, {"curves": merged})
        print(f"合并 {len(files)} 个分片 → 行数={len(merged)} → 04outputs/64_deep_transfer_server.xlsx")
        return

    device = _eu.get_device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]
    log = _eu.get_logger("64_deep_transfer_server")
    _eu.log_experiment_header(log, {"任务": "深度标定转移基线", "device": str(device),
                                    "seeds": seeds, "基准": args.benchmarks, "n_cal": NCAL_GRID})

    _bs = importlib.util.spec_from_file_location("bench", os.path.join(_CODE, "61_benchmark_datasets.py"))
    assert _bs is not None  # 对已知存在的 .py 文件 spec 恒非 None
    assert _bs.loader is not None  # 同理 loader 恒非 None
    bmod = importlib.util.module_from_spec(_bs)
    _bs.loader.exec_module(bmod)
    bmod.PROC = args.data_dir  # type: ignore[attr-defined]  # 动态加载模块,PROC 为其运行期全局,静态不可见

    curve_rows = []
    for name in args.benchmarks.split(","):
        if name not in bmod.LOADERS:
            log.log(f"⚠ {name} 未注册,跳过")
            continue
        b = bmod.LOADERS[name]()
        tasks = bmod.build_transfer_tasks(b)
        log.log(f"[{name}] {len(tasks)} 任务 × {len(seeds)} 种子")
        for ti, tk in enumerate(tasks):
            for seed in seeds:
                task_rows = run_task(b, tk, seed, device, rep=args.rep)
                for nc, meth, sd, r, val in task_rows:
                    curve_rows.append({
                        "task_id": f"{name}|{tk['src']}->{tk['tgt']}|{tk['prop_name']}",
                        "benchmark": name, "shift_type": tk["shift_type"], "modality": tk["modality"],
                        "prop": tk["prop_name"], "n_cal": nc, "method": meth,
                        "seed": sd, "rep": r, "rmsep": val})
            if (ti + 1) % 5 == 0:
                log.log(f"    {name} {ti+1}/{len(tasks)}")
    curves = pd.DataFrame(curve_rows)
    if args.tag:
        os.makedirs(shard_dir, exist_ok=True)
        out = os.path.join(shard_dir, f"{args.tag}.csv")
        curves.to_csv(out, index=False)
        print(f"深度曲线行数={len(curves)} → {out}")
        return
    _eu.write_script_workbook(__file__, {"curves": curves})
    print(f"深度曲线行数={len(curves)} → 04outputs/64_deep_transfer_server.xlsx")


if __name__ == "__main__":
    main()
