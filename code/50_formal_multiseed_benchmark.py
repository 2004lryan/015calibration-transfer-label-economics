"""
50_formal_multiseed_benchmark.py: 正式跨域迁移基准（5 方法 × 2 校正 × 全量场景 × 5 种子）

【本脚本存在的理由】
18_exp（表1/表2 的来源）与 46_exp（表3 的来源）此前均为 **单种子 seed=42**，且 46 只跑了
100 个场景。两处口径不一：表间无法互相印证，且不满足 01CLAUDE.md 第五章第 3 条
"Formal 阶段 = CUDA + 全 5 种子"。本脚本把两者合并为一次运行：

  · 方法：PLSR / SVR / CNN / CNN+MMD / Phys+BL      —— 与 18_exp 逐字节一致
  · 校正：{裸, +slope/bias}                          —— 46_exp 的核心对照
  · 场景：04_migration_scenarios.json 全量 602 个
  · 种子：Formal 5 粒 [20060515, 20041210, 19810915, 2023, 2024]（01CLAUDE.md §5.3）
          外加 seed=42 作为与旧 18_exp 的复现锚点（不参与正式统计）

于是表1/表2/表3 全部落在同一划分、同一组种子、同一份预测上，内部完全自洽。

【统计口径（01CLAUDE.md §7.2，实验前预注册，禁止 post-hoc 切换）】
  · 聚类单位 = 场景。同一场景的 5 个种子是重复测量，**不是**独立样本。
    先按场景对种子取均值 → 得到每场景一个值 → n_primary = 场景数。
  · 配对检验：Wilcoxon signed-rank（主，RMSE 分布重尾非正态）；paired t 仅作辅助报告。
  · 多比较校正：Holm（族内，见 HYPOTHESES）。
  · 效应量：Cliff's δ（主，非参数）+ Cohen's d（辅）。
  · 置信区间：cluster bootstrap（按场景有放回重采样，≥1000 次），禁止 i.i.d. bootstrap。

运行方式:
    # 单机全量（慢，仅调试用）
    python 02code/50_formal_multiseed_benchmark.py --device cuda --epochs 200 --seed 20060515

    # 分片并行（推荐；由 51_run_formal_benchmark.sh 调度）
    python 02code/50_formal_multiseed_benchmark.py --device cuda --seed 20060515 \
        --shard_id 0 --n_shards 16

    # 全部分片跑完后合并 + 统计检验 + 出唯一 xlsx
    python 02code/50_formal_multiseed_benchmark.py --merge

输出文件:
    04outputs/50_shards/seed<S>_shard<NN>.csv        — 分片原始结果（流式中间产物）
    04outputs/50_formal_multiseed_benchmark.xlsx     — 唯一正式产出（多 sheet）
    05logs/50_formal_multiseed_benchmark_<时间戳>.log — 运行日志
"""

import os
import sys
import json
import time
import glob
import argparse
import platform
import warnings
import importlib.util
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

warnings.filterwarnings('ignore')

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE_DIR, '..'))
_OUT = os.path.join(_BASE, '04outputs')
_LOG = os.path.join(_BASE, '05logs')
_SHARD_DIR = os.path.join(_OUT, '50_shards')

FILE_STEM = '50_formal_multiseed_benchmark'

# ─── 复用 18_exp 的数据加载与物理模型（同一来源，保证可比）────────────────────
_spec1 = importlib.util.spec_from_file_location(
    "export_utils", os.path.join(_CODE_DIR, "01_export_utils.py"))
_mod1 = importlib.util.module_from_spec(_spec1)
_spec1.loader.exec_module(_mod1)

_spec2 = importlib.util.spec_from_file_location(
    "physics_model", os.path.join(_CODE_DIR, "13_model_physics_informed.py"))
_mod2 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_mod2)

get_device           = _mod1.get_device
load_year_region_map = _mod1.load_year_region_map
AVAILABLE_YEARS      = _mod1.AVAILABLE_YEARS
extract_domain_keys  = _mod1.extract_domain_keys
merge_domain_payload = _mod1.merge_domain_payload
PhysicsInformedDomainAdaptation = _mod2.PhysicsInformedDomainAdaptation


# ═══════════════════════════════════════════════════════════════════════════
# 配置：与 18_exp_fc_unified_comparison.py 的 CONFIG 逐项一致（不得改动）
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = {
    'input_dim':            229,
    'n_components_bl':      3,
    'hidden_dim':           64,
    'batch_size':           16,
    'learning_rate':        1e-3,
    'weight_decay':         1e-5,
    'early_stop_patience':  20,
    'validation_freq':      5,
    'max_grad_norm':        1.0,
    'lambda_bl':            0.5,
    'lambda_mmd_cnn':       0.1,
    'grade_threshold':      13.0,
}

# 01CLAUDE.md §5.3：Formal 阶段固定 5 种子，双轨锚定
FORMAL_SEEDS = [20060515, 20041210, 19810915, 2023, 2024]
ANCHOR_SEED = 42          # 与旧 18_exp 的复现锚点，不参与正式统计

# 7 个方法一次跑完，令表1（方法对比）、表2（消融）、表3（SB 校正）同源同划分同种子。
#   CNN / CNN+MMD / CNN+BL 共用 18_exp 的 BaselineCNN 主干；Phys / Phys+BL 共用物理架构。
#   于是"BL 损失"这一变量在两种架构上都被干净地隔离出来。
METHODS = ['PLSR', 'SVR', 'CNN', 'CNN+MMD', 'CNN+BL', 'Phys', 'Phys+BL']
MIGRATION_TYPES = ['1源→1目标', '1源→多目标', '多源→1目标', '多源→多目标']

# 预注册假设族（Holm 在族内校正；实验前锁定，禁止事后增删）
HYPOTHESES = [
    # —— 表1/表3 的核心主张 ——
    ('H1', 'Phys+BL',      'CNN',       '物理先验相对深度基线的增益（论文原主张）'),
    ('H2', 'Phys+BL + SB', 'PLSR + SB', '校正后物理先验能否守住对化学计量学基线的优势'),
    ('H3', 'Phys+BL + SB', 'CNN + SB',  '校正后物理先验能否守住对深度基线的优势'),
    ('H4', 'Phys+BL',      'CNN+MMD',   '物理先验相对深度域适应基线的增益'),
    ('H5', 'Phys+BL + SB', 'Phys+BL',   'slope/bias 校正对 Phys+BL 自身的增益'),
    # —— 表2 的架构-约束协同设计消融（BL 损失在两种架构上的效果相反）——
    ('H6', 'CNN+BL',       'CNN',       'BL 损失加在通用 CNN 架构上是否有害'),
    ('H7', 'Phys+BL',      'Phys',      'BL 损失加在物理架构上是否有益'),
]


def make_logger(tag=''):
    os.makedirs(_LOG, exist_ok=True)
    stamp = datetime.now().strftime('%y-%m-%d_%H%M%S')
    path = os.path.join(_LOG, f'{FILE_STEM}_{stamp}{tag}.log')

    def log(msg):
        line = f'[{datetime.now().strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    return log


def set_seed(seed):
    """六路同步。CUDA 通路开启确定性算法（01CLAUDE.md §5.4）。"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def enforce_determinism(log):
    """CUBLAS 工作区必须在首次 CUDA 调用前设置，故在 main 入口调用一次。"""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    try:
        torch.use_deterministic_algorithms(True)
        log('确定性算法: torch.use_deterministic_algorithms(True) 已开启')
    except Exception as e:                       # 某些算子无确定性实现时降级并记录
        log(f'确定性算法: 降级为 cudnn.deterministic（原因: {e}）')


def cap_gpu_memory(fraction, log):
    """
    显存硬上限：这台卡是共享的，另有他人项目在跑。
    本实验模型极小（单进程实测 ~0.6 GB），故给每个进程设死上限，
    超限即 OOM 报错而不是把整张卡吃满——宁可自己失败，不可拖垮邻居。
    """
    if not torch.cuda.is_available() or fraction <= 0:
        return
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    log(f'显存上限: 本进程 ≤ {fraction*100:.1f}% × {total:.1f} GB = {fraction*total:.2f} GB')


# ═══════════════════════════════════════════════════════════════════════════
# 模型（与 18_exp 逐字节一致）
# ═══════════════════════════════════════════════════════════════════════════

class BaselineCNN(nn.Module):
    def __init__(self, input_dim=229, hidden_dims=(64, 32), latent_dim=4, dropout=0.3):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(1, hidden_dims[0], kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            self.feature_dim = self.feat_extractor(dummy).view(1, -1).shape[1]

        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, latent_dim),
        )
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def extract_features(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.feat_extractor(x).view(x.size(0), -1)
        return self.encoder(h)

    def forward(self, x):
        return self.regressor(self.extract_features(x)).squeeze(-1)


class BaselineCNN_BL(BaselineCNN):
    """
    CNN+BL：在 BaselineCNN 上挂一个 BL 光谱重构解码头（推理时不参与，接口与 CNN 完全一致）。

    【与旧 19_exp 的差别，必须说明】
    旧 19_exp 自带一份 BaselineCNN，其 regressor 是 Linear(latent,16)+Dropout+Linear(16,1)，
    与 18_exp 的 Linear(latent,32)+Linear(32,1) **并不相同**——尽管其 docstring 声称"完全相同"。
    19_exp 内部 CNN 与 CNN+BL 共用那份改动过的主干，所以它的消融本身是公平的；
    但它的 CNN 数字与 18_exp 的 CNN 数字来自两套不同架构，并列进同一张表会误导读者。
    本脚本统一继承 18_exp 的 BaselineCNN，令 CNN / CNN+MMD / CNN+BL 共用同一主干，
    消融差异因此只可能来自 BL 损失本身。
    """
    def __init__(self, input_dim=229, hidden_dims=(64, 32), latent_dim=4, dropout=0.3):
        super().__init__(input_dim, hidden_dims, latent_dim, dropout)
        self.bl_decoder = nn.Sequential(          # 与 19_exp 的解码头一致
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, input_dim),
            nn.Softplus(),                        # 输出为正，log 才有定义
        )

    def bl_loss(self, x):
        """BL 重构损失：log 空间 MSE，与 PhysicsInformedDA 的定义相同。"""
        if x.dim() == 3:
            x = x.squeeze(1)
        x_recon = self.bl_decoder(self.extract_features(x))
        eps = 1e-6
        return nn.functional.mse_loss(torch.log(x_recon + eps), torch.log(x.abs() + eps))


def mmd_loss(z_src, z_tgt):
    """与 18_exp 一致：隐空间均值嵌入的 L2 距离（线性核 MMD 的一阶形式）。"""
    return torch.norm(z_src.mean(0) - z_tgt.mean(0), p=2)


# ═══════════════════════════════════════════════════════════════════════════
# 数据划分 / 指标 / 校正
# ═══════════════════════════════════════════════════════════════════════════

def split_data(X_src, y_src, X_tgt, y_tgt):
    """与 18_exp 的 split_data 逐行一致：源域 100%，目标域 60/20/20。"""
    scaler = StandardScaler()
    X_src_s = scaler.fit_transform(X_src)
    X_tgt_s = scaler.transform(X_tgt)

    n_tgt = len(X_tgt_s)
    idx = np.random.permutation(n_tgt)
    s1 = int(n_tgt * 0.6)
    s2 = int(n_tgt * 0.8)

    return {
        'X_src':       X_src_s,
        'y_src':       y_src,
        'X_tgt_train': X_tgt_s[idx[:s1]],
        'y_tgt_train': y_tgt[idx[:s1]],
        'X_tgt_val':   X_tgt_s[idx[s1:s2]],
        'y_tgt_val':   y_tgt[idx[s1:s2]],
        'X_tgt_test':  X_tgt_s[idx[s2:]],
        'y_tgt_test':  y_tgt[idx[s2:]],
    }


def to_loader(X, y, batch_size, shuffle=True, device='cpu'):
    Xt = torch.FloatTensor(X).to(device)
    yt = torch.FloatTensor(y).reshape(-1, 1).to(device)
    return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle)


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    rpd = float(np.std(y_true) / rmse) if rmse > 0 else 0.0
    iqr = float(np.percentile(y_true, 75) - np.percentile(y_true, 25))
    rpiq = iqr / rmse if rmse > 0 else 0.0
    return {'RMSE': rmse, 'R2': r2, 'RPD': rpd, 'RPIQ': rpiq, 'Bias': bias}


def grade_error_rate(y_true, y_pred, threshold=13.0):
    true_grade = (np.asarray(y_true) >= threshold).astype(int)
    pred_grade = (np.asarray(y_pred) >= threshold).astype(int)
    return float(np.mean(true_grade != pred_grade))


def slope_bias_correction(y_val_true, y_val_pred, y_test_pred):
    """
    post-hoc 斜率/偏置校正（标定转移的经典手段，对任何模型的预测都适用）：
    在目标域 val（20% 标签）上拟合 y_true ≈ a·y_pred + b，用 (a,b) 校正 test 预测。
    """
    y_val_pred = np.asarray(y_val_pred).flatten()
    y_val_true = np.asarray(y_val_true).flatten()
    if len(y_val_pred) < 3 or np.std(y_val_pred) < 1e-8:
        return np.asarray(y_test_pred).flatten()   # 退化，放弃校正
    coef = np.polyfit(y_val_pred, y_val_true, deg=1)
    return coef[0] * np.asarray(y_test_pred).flatten() + coef[1]


# ═══════════════════════════════════════════════════════════════════════════
# 各方法：统一返回 (val_pred, test_pred)，以便施加同样的 slope/bias 校正
# ═══════════════════════════════════════════════════════════════════════════

def run_plsr(splits):
    pls = PLSRegression(
        n_components=min(10, splits['X_src'].shape[1] - 1, len(splits['X_src']) - 1))
    pls.fit(splits['X_src'], splits['y_src'])
    return (pls.predict(splits['X_tgt_val']).flatten(),
            pls.predict(splits['X_tgt_test']).flatten())


def run_svr(splits):
    svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
    svr.fit(splits['X_src'], splits['y_src'])
    return (svr.predict(splits['X_tgt_val']),
            svr.predict(splits['X_tgt_test']))


def _predict_np(model, X, device, physics=False):
    loader = to_loader(X, np.zeros(len(X)), CONFIG['batch_size'],
                       shuffle=False, device=device)
    model.eval()
    outs = []
    with torch.no_grad():
        for Xb, _ in loader:
            out = model(Xb)
            out = out['y_pred'] if physics else out
            outs.append(out.cpu().numpy())
    return np.concatenate(outs).flatten()


def _train_loop(model, splits, device, epochs, step_fn):
    """CNN / CNN+MMD / Phys+BL 共用的早停训练循环（与 18_exp 的调度一致）。"""
    optim = Adam(model.parameters(), lr=CONFIG['learning_rate'],
                 weight_decay=CONFIG['weight_decay'])
    sched = CosineAnnealingLR(optim, T_max=epochs)

    src_loader = to_loader(splits['X_src'],       splits['y_src'],       CONFIG['batch_size'], device=device)
    tgt_loader = to_loader(splits['X_tgt_train'], splits['y_tgt_train'], CONFIG['batch_size'], device=device)
    val_loader = to_loader(splits['X_tgt_val'],   splits['y_tgt_val'],   CONFIG['batch_size'], shuffle=False, device=device)

    is_phys = isinstance(model, PhysicsInformedDomainAdaptation)
    best_val, best_st, patience = float('inf'), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for (Xs, ys), (Xt, _) in zip(src_loader, tgt_loader):
            optim.zero_grad()
            loss = step_fn(model, Xs, ys, Xt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
            optim.step()
        sched.step()

        if epoch % CONFIG['validation_freq'] == 0:
            model.eval()
            with torch.no_grad():
                vp, vt = [], []
                for Xv, yv in val_loader:
                    out = model(Xv)
                    vp.append((out['y_pred'] if is_phys else out).cpu().numpy())
                    vt.append(yv.cpu().numpy())
            val_rmse = np.sqrt(mean_squared_error(
                np.concatenate(vt).flatten(), np.concatenate(vp).flatten()))
            if val_rmse < best_val:
                best_val, patience = val_rmse, 0
                best_st = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= CONFIG['early_stop_patience']:
                    break

    if best_st is not None:
        model.load_state_dict(best_st)
    return (_predict_np(model, splits['X_tgt_val'],  device, physics=is_phys),
            _predict_np(model, splits['X_tgt_test'], device, physics=is_phys))


def run_cnn(splits, device, epochs, lambda_mmd=0.0, lambda_bl=0.0):
    """通用 CNN 通路：lambda_mmd>0 → CNN+MMD；lambda_bl>0 → CNN+BL；都为 0 → 裸 CNN。"""
    cls = BaselineCNN_BL if lambda_bl > 0 else BaselineCNN
    model = cls(input_dim=CONFIG['input_dim']).to(device)
    crit = nn.MSELoss()

    def step(m, Xs, ys, Xt):
        loss = crit(m(Xs), ys.squeeze(1))
        if lambda_mmd > 0:
            loss = loss + lambda_mmd * mmd_loss(
                m.extract_features(Xs), m.extract_features(Xt))
        if lambda_bl > 0:                          # 与 19_exp 一致：源域 + 目标域都算重构
            loss = loss + lambda_bl * (m.bl_loss(Xs) + m.bl_loss(Xt))
        return loss

    return _train_loop(model, splits, device, epochs, step)


def run_phys(splits, device, epochs, lambda_bl):
    """物理架构通路：lambda_bl=0 → Phys（无 BL 约束，20_exp 的对照）；=0.5 → Phys+BL。"""
    model = PhysicsInformedDomainAdaptation(
        n_bands=CONFIG['input_dim'],
        n_components=CONFIG['n_components_bl'],
        hidden_dim=CONFIG['hidden_dim'],
    ).to(device)

    def step(m, Xs, ys, Xt):
        results = m(Xs, ys, Xt)
        total, _ = m.compute_total_loss(
            results, lambda_bl=lambda_bl, lambda_mmd=0.0)
        return total

    return _train_loop(model, splits, device, epochs, step)


# ═══════════════════════════════════════════════════════════════════════════
# 运行模式：跑一个 (seed, shard) 分片
# ═══════════════════════════════════════════════════════════════════════════

def run_shard(args, log):
    device = get_device(args.device)
    if args.device == 'cuda' and device != 'cuda':
        raise RuntimeError('01CLAUDE.md §5.3：Formal 阶段 CUDA 强制，但当前不可用')
    enforce_determinism(log)
    cap_gpu_memory(args.gpu_mem_frac, log)

    log('=' * 78)
    log(f'50 - Formal 基准 | seed={args.seed} | shard {args.shard_id}/{args.n_shards}')
    log(f'设备={device} | epochs={args.epochs} | 方法={METHODS}')
    log('=' * 78)

    active_methods = ([m.strip() for m in args.methods.split(',')]
                      if args.methods else list(METHODS))
    unknown = [m for m in active_methods if m not in METHODS]
    if unknown:
        raise ValueError(f'未知方法 {unknown}，可选：{METHODS}')
    log(f'本次运行方法: {active_methods}')

    set_seed(args.seed)
    year_region_data = load_year_region_map(years=AVAILABLE_YEARS, target_wavelengths='2025')

    with open(os.path.join(_CODE_DIR, '04_migration_scenarios.json'), 'r', encoding='utf-8') as f:
        all_scenarios = json.load(f)

    scenario_list = list(all_scenarios.items())
    if args.max_scenarios:
        scenario_list = scenario_list[:args.max_scenarios]
    # stride 切分：保证每个分片的迁移类型分布与全量一致
    scenario_list = scenario_list[args.shard_id::args.n_shards]
    log(f'本分片场景数: {len(scenario_list)}')

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    rows = []
    t0 = time.time()
    for sc_idx, (scenario_id, config) in enumerate(scenario_list, 1):
        try:
            src_payload = merge_domain_payload(year_region_data, extract_domain_keys(config['source']))
            tgt_payload = merge_domain_payload(year_region_data, extract_domain_keys(config['target']))
        except KeyError:
            continue

        X_src, y_src = src_payload['X'], src_payload['y']
        X_tgt, y_tgt = tgt_payload['X'], tgt_payload['y']
        if len(y_tgt) < 15 or len(y_src) < 10:
            continue

        # 划分与初始化都由同一 seed 驱动：种子间差异 = 划分方差 + 初始化方差
        set_seed(args.seed)
        splits = split_data(X_src, y_src, X_tgt, y_tgt)

        y_val_true, y_test_true = splits['y_tgt_val'], splits['y_tgt_test']
        if len(y_test_true) < 5 or len(y_val_true) < 3:
            continue

        preds = {}
        all_runners = [
            ('PLSR',    lambda: run_plsr(splits)),
            ('SVR',     lambda: run_svr(splits)),
            ('CNN',     lambda: run_cnn(splits, device, args.epochs)),
            ('CNN+MMD', lambda: run_cnn(splits, device, args.epochs,
                                        lambda_mmd=CONFIG['lambda_mmd_cnn'])),
            ('CNN+BL',  lambda: run_cnn(splits, device, args.epochs,
                                        lambda_bl=CONFIG['lambda_bl'])),
            ('Phys',    lambda: run_phys(splits, device, args.epochs, lambda_bl=0.0)),
            ('Phys+BL', lambda: run_phys(splits, device, args.epochs,
                                         lambda_bl=CONFIG['lambda_bl'])),
        ]
        runners = [(n, f) for n, f in all_runners if n in active_methods]
        for name, fn in runners:
            try:
                set_seed(args.seed)          # 每个方法同起点，消除方法间的初始化差异
                preds[name] = fn()
            except Exception as e:
                log(f'  [{name}] {scenario_id}: {e}')

        for method, (val_pred, test_pred) in preds.items():
            if test_pred is None or len(test_pred) == 0:
                continue
            base = {'场景': scenario_id, '迁移类型': config.get('type', '未知'),
                    '方法': method, '种子': args.seed,
                    'n_test': len(y_test_true), 'n_src': len(y_src), 'n_tgt': len(y_tgt)}
            rows.append({**base, '校正': '裸',
                         '品质误判率': grade_error_rate(y_test_true, test_pred, CONFIG['grade_threshold']),
                         **compute_metrics(y_test_true, test_pred)})
            test_sb = slope_bias_correction(y_val_true, val_pred, test_pred)
            rows.append({**base, '校正': '+SB',
                         '品质误判率': grade_error_rate(y_test_true, test_sb, CONFIG['grade_threshold']),
                         **compute_metrics(y_test_true, test_sb)})

        if sc_idx % 5 == 0 or sc_idx == len(scenario_list):
            el = time.time() - t0
            log(f'  进度 {sc_idx}/{len(scenario_list)} | 已用 {el/60:.1f} min '
                f'| 预计剩余 {el/sc_idx*(len(scenario_list)-sc_idx)/60:.1f} min')

    df = pd.DataFrame(rows)
    os.makedirs(_SHARD_DIR, exist_ok=True)
    out = os.path.join(_SHARD_DIR, f'seed{args.seed}_shard{args.shard_id:02d}.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')

    wall = time.time() - t0
    peak_mem = (torch.cuda.max_memory_allocated() / 1024**3) if device == 'cuda' else 0.0
    meta = {'seed': args.seed, 'shard_id': args.shard_id, 'n_shards': args.n_shards,
            'n_scenarios': len(scenario_list), 'n_rows': len(df),
            'wall_clock_s': round(wall, 1), 'peak_gpu_mem_GB': round(peak_mem, 3),
            'device': device, 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else '',
            'epochs': args.epochs, 'torch': torch.__version__,
            'platform': platform.platform()}
    with open(out.replace('.csv', '.meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log(f'\n已保存: {out} | {len(df)} 行 | wall {wall/60:.1f} min | 峰值显存 {peak_mem:.2f} GB')


# ═══════════════════════════════════════════════════════════════════════════
# 统计工具（01CLAUDE.md §7.2）
# ═══════════════════════════════════════════════════════════════════════════

def cliffs_delta(x, y):
    """非参数效应量：δ = P(X>Y) − P(X<Y)。|δ|<.147 可忽略, <.33 小, <.474 中, 否则大。"""
    x, y = np.asarray(x), np.asarray(y)
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    d = (gt - lt) / (len(x) * len(y))
    a = abs(d)
    mag = '可忽略' if a < 0.147 else ('小' if a < 0.33 else ('中' if a < 0.474 else '大'))
    return float(d), mag


def cohens_d_paired(x, y):
    diff = np.asarray(x) - np.asarray(y)
    sd = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd > 1e-12 else 0.0


def holm(pvals):
    """Holm-Bonferroni 逐步校正，返回与输入同序的校正后 p。"""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)          # 保证单调不减
        adj[i] = min(1.0, running)
    return adj


def cluster_bootstrap_ci(per_scenario, n_boot=1000, ci=0.95, seed=12345):
    """
    按场景（聚类单位）有放回重采样 → 均值的 CI。
    per_scenario: 1-D array，每个元素是一个场景的（跨种子平均后的）指标值。
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(per_scenario, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float('nan'), float('nan')
    n = len(v)
    means = np.array([v[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    return lo, hi


# ═══════════════════════════════════════════════════════════════════════════
# 合并模式：读所有分片 → 统计检验 → 唯一 xlsx
# ═══════════════════════════════════════════════════════════════════════════

def do_merge(args, log):
    files = sorted(glob.glob(os.path.join(_SHARD_DIR, 'seed*_shard*.csv')))
    if not files:
        log('!! 未找到任何分片结果')
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['方法_校正'] = df['方法'] + df['校正'].map({'裸': '', '+SB': ' + SB'})

    seeds_found = sorted(df['种子'].unique())
    log(f'读入 {len(files)} 个分片 | {len(df)} 行 | 种子 {seeds_found}')

    formal = df[df['种子'].isin(FORMAL_SEEDS)].copy()
    anchor = df[df['种子'] == ANCHOR_SEED].copy()
    n_scen = formal['场景'].nunique()
    n_seed = formal['种子'].nunique()
    log(f'正式统计集: {n_scen} 场景 × {n_seed} 种子 | 锚点集(seed42): {anchor["场景"].nunique()} 场景')

    # ── 每场景先对种子取均值：这是唯一合法的配对单位 ────────────────────────
    METRICS = ['RMSE', 'R2', 'RPD', 'RPIQ', 'Bias', '品质误判率']
    per_scen = (formal.groupby(['方法_校正', '场景'])[METRICS]
                      .mean().reset_index())

    # sheet 1：按方法汇总（mean ± std over 场景 + cluster bootstrap CI）
    summary = []
    for mc, g in per_scen.groupby('方法_校正'):
        lo, hi = cluster_bootstrap_ci(g['RMSE'].values, n_boot=args.n_boot)
        # 种子间方差：同场景内跨种子的 RMSE 标准差，再对场景取平均
        seed_sd = (formal[formal['方法_校正'] == mc]
                   .groupby('场景')['RMSE'].std(ddof=1).mean())
        summary.append({
            '方法': mc,
            'RMSE_mean': g['RMSE'].mean(), 'RMSE_std': g['RMSE'].std(ddof=1),
            'RMSE_median': g['RMSE'].median(),
            'RMSE_CI95_low': lo, 'RMSE_CI95_high': hi,
            '种子间SD_mean': seed_sd,
            'R2_mean': g['R2'].mean(), 'RPD_mean': g['RPD'].mean(),
            'RPIQ_mean': g['RPIQ'].mean(), 'Bias_mean': g['Bias'].mean(),
            '品质误判率_mean': g['品质误判率'].mean(),
            'R2大于0场景数': int((g['R2'] > 0).sum()),
            'n_scenarios': len(g),
        })
    summary = pd.DataFrame(summary).sort_values('RMSE_mean').reset_index(drop=True)

    # sheet 1b：共同场景子集（消除幸存者偏差）
    #   深度模型会在少数场景上抛异常/发散而缺行（旧表1 的 n 就是 602/602/575/571/569）。
    #   在"各自幸存的场景集"上取均值不可横比——缺的往往正是最难的场景。
    #   故额外报一份所有方法 × 全部种子都跑通的场景交集上的汇总。
    all_mc = sorted(per_scen['方法_校正'].unique())
    common = None
    for mc in all_mc:
        s = set(per_scen[per_scen['方法_校正'] == mc]['场景'])
        common = s if common is None else (common & s)
    common = sorted(common) if common else []
    log(f'共同场景（所有方法均跑通）: {len(common)} / {n_scen}')

    summary_common = []
    if common:
        sub = per_scen[per_scen['场景'].isin(common)]
        for mc, g in sub.groupby('方法_校正'):
            lo, hi = cluster_bootstrap_ci(g['RMSE'].values, n_boot=args.n_boot)
            summary_common.append({
                '方法': mc,
                'RMSE_mean': g['RMSE'].mean(), 'RMSE_std': g['RMSE'].std(ddof=1),
                'RMSE_median': g['RMSE'].median(),
                'RMSE_CI95_low': lo, 'RMSE_CI95_high': hi,
                'R2_mean': g['R2'].mean(), 'RPD_mean': g['RPD'].mean(),
                'RPIQ_mean': g['RPIQ'].mean(), 'Bias_mean': g['Bias'].mean(),
                '品质误判率_mean': g['品质误判率'].mean(),
                'R2大于0场景数': int((g['R2'] > 0).sum()),
                'n_scenarios': len(g),
            })
    summary_common = (pd.DataFrame(summary_common).sort_values('RMSE_mean').reset_index(drop=True)
                      if summary_common else pd.DataFrame())

    # sheet 2：按迁移类型
    type_map = formal[['场景', '迁移类型']].drop_duplicates().set_index('场景')['迁移类型']
    per_scen['迁移类型'] = per_scen['场景'].map(type_map)
    by_type = per_scen.pivot_table(index='方法_校正', columns='迁移类型',
                                   values='RMSE', aggfunc='mean').reset_index()

    # sheet 3：预注册配对检验（Wilcoxon 主 + paired t 辅；Holm 族内校正）
    tests = []
    for hid, a_name, b_name, desc in HYPOTHESES:
        A = per_scen[per_scen['方法_校正'] == a_name].set_index('场景')['RMSE']
        B = per_scen[per_scen['方法_校正'] == b_name].set_index('场景')['RMSE']
        common = A.index.intersection(B.index)
        a, b = A.loc[common].values, B.loc[common].values
        if len(common) < 5:
            log(f'  [{hid}] 配对场景不足（{len(common)}），跳过')
            continue
        try:
            w_stat, p_w = stats.wilcoxon(a, b)
        except ValueError:                              # 全零差值
            w_stat, p_w = float('nan'), 1.0
        t_stat, p_t = stats.ttest_rel(a, b)
        d, mag = cliffs_delta(a, b)
        lo, hi = cluster_bootstrap_ci(a - b, n_boot=args.n_boot)
        tests.append({
            '假设': hid, 'A': a_name, 'B': b_name, '含义': desc,
            'n_场景': len(common),
            'A_RMSE_mean': a.mean(), 'B_RMSE_mean': b.mean(),
            '差值_A减B': a.mean() - b.mean(),
            '相对改进_%': (b.mean() - a.mean()) / b.mean() * 100,
            '差值_CI95_low': lo, '差值_CI95_high': hi,
            'Wilcoxon_p': p_w, 'paired_t_p': p_t,
            'Cliffs_delta': d, '效应量级': mag,
            'Cohens_d': cohens_d_paired(a, b),
        })
    tests = pd.DataFrame(tests)
    if not tests.empty:
        tests['Wilcoxon_p_holm'] = holm(tests['Wilcoxon_p'].values)
        tests['显著性'] = pd.cut(tests['Wilcoxon_p_holm'],
                                 bins=[-1, 0.001, 0.01, 0.05, 1],
                                 labels=['***', '**', '*', 'ns'])
        # 列序：把校正后 p 放到原始 p 旁边
        cols = list(tests.columns)
        cols.insert(cols.index('paired_t_p'), cols.pop(cols.index('Wilcoxon_p_holm')))
        tests = tests[cols]

    # sheet 4：seed=42 锚点（与旧 18_exp 对齐性核验，不参与正式统计）
    #   18_exp 在 CNN / CNN+MMD / Phys+BL 前各重置一次 set_seed(42)，本脚本同构；
    #   故 seed=42 一轮应逐位复现旧数字。偏差 >0.05 即说明两套代码已不等价，须排查。
    #   CNN+BL / Phys 无旧值可比：它们原属 19/20_exp，主干架构与 18_exp 不同（见 BaselineCNN_BL 注释）。
    OLD_18EXP_RMSE = {'PLSR': 3.987, 'SVR': 3.065, 'CNN': 2.577,
                      'CNN+MMD': 2.642, 'Phys+BL': 1.976}
    anchor_sum = pd.DataFrame()
    if not anchor.empty:
        anchor_sum = (anchor[anchor['校正'] == '裸']
                      .groupby('方法')
                      .agg(RMSE_mean=('RMSE', 'mean'), RMSE_median=('RMSE', 'median'),
                           R2_mean=('R2', 'mean'), RPD_mean=('RPD', 'mean'),
                           Bias_mean=('Bias', 'mean'),
                           品质误判率=('品质误判率', 'mean'),
                           n_场景=('RMSE', 'size'))
                      .reset_index())
        anchor_sum['旧18exp_RMSE'] = anchor_sum['方法'].map(OLD_18EXP_RMSE)
        anchor_sum['偏差'] = anchor_sum['RMSE_mean'] - anchor_sum['旧18exp_RMSE']
        anchor_sum['复现'] = np.where(
            anchor_sum['旧18exp_RMSE'].isna(), '—(新增方法)',
            np.where(anchor_sum['偏差'].abs() < 0.05, '✅一致', '⚠️不一致'))
        anchor_sum = anchor_sum.sort_values('RMSE_mean')

    # sheet 5：算力账（01CLAUDE.md §12.1）
    metas = []
    for mf in sorted(glob.glob(os.path.join(_SHARD_DIR, '*.meta.json'))):
        with open(mf, 'r', encoding='utf-8') as f:
            metas.append(json.load(f))
    compute = pd.DataFrame(metas)
    if not compute.empty:
        gpu_h = compute['wall_clock_s'].sum() / 3600
        compute_total = pd.DataFrame([{
            '总_wall_clock_h': round(compute['wall_clock_s'].max() / 3600, 2),   # 并行下取最长分片
            '累计_GPU_hours': round(gpu_h, 2),
            '峰值显存_GB': compute['peak_gpu_mem_GB'].max(),
            'GPU型号': compute['gpu'].iloc[0] if 'gpu' in compute else '',
            '分片数': len(compute),
            '场景-方法-种子 组合数': int(compute['n_rows'].sum() / 2),   # 每组合 2 行（裸/+SB）
        }])
    else:
        compute_total = pd.DataFrame()

    os.makedirs(_OUT, exist_ok=True)
    xlsx = os.path.join(_OUT, f'{FILE_STEM}.xlsx')
    with pd.ExcelWriter(xlsx) as wr:
        summary.to_excel(wr, sheet_name='表a：按方法汇总', index=False)
        summary_common.to_excel(wr, sheet_name='表a2：共同场景子集', index=False)
        by_type.to_excel(wr, sheet_name='表b：按迁移类型', index=False)
        tests.to_excel(wr, sheet_name='表c：预注册配对检验', index=False)
        anchor_sum.to_excel(wr, sheet_name='表d：seed42复现锚点', index=False)
        compute_total.to_excel(wr, sheet_name='表e：算力账', index=False)
        compute.to_excel(wr, sheet_name='表f：分片明细', index=False)
        per_scen.to_excel(wr, sheet_name='表g：每场景跨种子均值', index=False)
        df.to_excel(wr, sheet_name='表h：全部原始结果', index=False)
    log(f'\n已保存: {xlsx}')

    log('\n=== 表a：按方法汇总（各自幸存场景集；RMSE 升序，CI 为 cluster bootstrap）===')
    log('\n' + summary.round(4).to_string(index=False))
    if not summary_common.empty:
        log(f'\n=== 表a2：共同场景子集（n={len(common)}，消除幸存者偏差，此表方可横比）===')
        log('\n' + summary_common.round(4).to_string(index=False))
    if not tests.empty:
        log('\n=== 表c：预注册配对检验（Holm 校正后）===')
        log('\n' + tests.round(5).to_string(index=False))
    if not anchor_sum.empty:
        log('\n=== 表d：seed=42 锚点（须逐位复现旧 18_exp，否则两套代码不等价）===')
        log('\n' + anchor_sum.round(4).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--seed', type=int, default=FORMAL_SEEDS[0])
    ap.add_argument('--shard_id', type=int, default=0)
    ap.add_argument('--n_shards', type=int, default=1)
    ap.add_argument('--max_scenarios', type=int, default=None, help='仅调试用')
    ap.add_argument('--merge', action='store_true', help='合并分片 + 统计检验 + 出 xlsx')
    ap.add_argument('--n_boot', type=int, default=1000, help='cluster bootstrap 次数')
    ap.add_argument('--gpu_mem_frac', type=float, default=0.04,
                    help='本进程显存占比上限（卡是共享的，默认 4%% ≈ 1 GB）')
    ap.add_argument('--methods', default=None,
                    help='逗号分隔的方法子集（默认全部 7 个）。'
                         '算力受限时可先跑表1/表3 所需的 5 个：'
                         'PLSR,SVR,CNN,CNN+MMD,Phys+BL')
    ap.add_argument('--shard_dir', default=None,
                    help='分片 CSV 的读写目录（默认 04outputs/50_shards）。'
                         '跑消融等独立子实验时另指定目录，避免与主实验分片混淆、'
                         '尤其 seed=42 在两处都是锚点时的文件名碰撞。')
    args = ap.parse_args()

    # 分片目录可覆盖：run_shard / do_merge / 元数据汇总都读同一个 module 级 _SHARD_DIR
    if args.shard_dir:
        global _SHARD_DIR
        _SHARD_DIR = (args.shard_dir if os.path.isabs(args.shard_dir)
                      else os.path.join(_BASE, args.shard_dir))

    tag = '' if args.merge else f'_seed{args.seed}_shard{args.shard_id:02d}'
    log = make_logger('_merge' if args.merge else tag)

    if args.merge:
        do_merge(args, log)
    else:
        run_shard(args, log)


if __name__ == '__main__':
    main()
