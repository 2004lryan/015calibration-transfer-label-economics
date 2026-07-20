"""
46_slopebias_unified_comparison.py：统一数据划分下的 slope/bias 校正对照实验

【为什么必须做这个实验】
论文 Table `tbl:label_budget_full` 的核心结论——"有 20% 目标标签时 PLSR+slope/bias
(1.57) 优于 Phys+BL (1.976)"——建立在两个缺陷之上：

  缺陷 1（对比不对称）：18_exp 脚本产出 Phys+BL 的裸预测；44_exp 脚本只给 PLSR/SVR
      施加了 slope/bias 后处理校正，**从未对 Phys+BL 施加同样的校正**。
      而 slope/bias 是通用后处理，任何模型的预测都能套。

  缺陷 2（划分不一致）：18_exp 用 np.random.permutation（legacy MT19937）划分；
      44_exp 用 np.random.default_rng(42)（PCG64）划分。同样写 seed=42，但 RNG 算法
      不同，实测 20% 测试集仅 1/10 样本重合 —— 两组数字根本不在同一测试集上。

本脚本在**统一划分**（严格复用 18_exp 的 set_seed(42)+split_data）上，对同一批场景
跑完整的 2×4 对照：

    PLSR / SVR / CNN / Phys+BL   ×   {裸, +slope-bias 校正}

回答三个问题：
  Q1 Phys+BL 裸结果能否复现 18_exp 的 1.976（脚本正确性锚点）
  Q2 Phys+BL + SB 能否反超 PLSR + SB
  Q3 校正后物理架构相对 CNN 是否仍有优势（否则先验就没价值了）

运行方式:
    python 46_slopebias_unified_comparison.py --device auto --epochs 200 --max_scenarios 12
    python 46_slopebias_unified_comparison.py --device auto --epochs 200          # 全量

输出:
    04outputs/46_slopebias_unified_comparison.xlsx
    05logs/46_slopebias_unified_comparison.log
"""

import os
import sys
import json
import time
import argparse
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

warnings.filterwarnings('ignore')

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE_DIR, '..'))
_OUT = os.path.join(_BASE, '04outputs')
_LOG = os.path.join(_BASE, '05logs')

# ─── 复用工具模块与物理模型（与 18_exp 完全一致的来源）──────────────────────
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

FILE_STEM = '46_slopebias_unified_comparison'

# ═══════════════════════════════════════════════════════════════════════════
# 配置：与 18_exp_fc_unified_comparison.py 的 CONFIG 逐项一致（不得改动）
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = {
    'random_seed':          42,
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
    'grade_threshold':      13.0,
}

MIGRATION_TYPES = ['1源→1目标', '1源→多目标', '多源→1目标', '多源→多目标']
BASE_METHODS = ['PLSR', 'SVR', 'CNN', 'Phys+BL']


def log(msg):
    line = f'[{datetime.now().strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    os.makedirs(_LOG, exist_ok=True)
    with open(os.path.join(_LOG, f'{FILE_STEM}.log'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════
# CNN 基线（与 18_exp 的 BaselineCNN 完全相同）
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


def to_loader(X, y, batch_size, shuffle=True, device='cpu'):
    Xt = torch.FloatTensor(X).to(device)
    yt = torch.FloatTensor(y).reshape(-1, 1).to(device)
    return DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=shuffle)


def split_data(X_src, y_src, X_tgt, y_tgt):
    """与 18_exp 的 split_data 逐行一致：源域 100%，目标域 60/20/20。"""
    scaler = StandardScaler()
    X_src_s = scaler.fit_transform(X_src)
    X_tgt_s = scaler.transform(X_tgt)

    n_tgt = len(X_tgt_s)
    idx   = np.random.permutation(n_tgt)
    s1    = int(n_tgt * 0.6)
    s2    = int(n_tgt * 0.8)

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


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2    = r2_score(y_true, y_pred)
    bias  = float(np.mean(y_pred - y_true))
    rpd   = float(np.std(y_true) / rmse) if rmse > 0 else 0.0
    iqr   = float(np.percentile(y_true, 75) - np.percentile(y_true, 25))
    rpiq  = iqr / rmse if rmse > 0 else 0.0
    return {'RMSE': rmse, 'R2': r2, 'RPD': rpd, 'RPIQ': rpiq, 'Bias': bias}


def grade_error_rate(y_true, y_pred, threshold=13.0):
    true_grade = (np.asarray(y_true) >= threshold).astype(int)
    pred_grade = (np.asarray(y_pred) >= threshold).astype(int)
    return float(np.mean(true_grade != pred_grade))


def slope_bias_correction(y_val_true, y_val_pred, y_test_pred):
    """
    post-hoc 斜率/偏置校正（与 44_exp 的实现一致）：
    在目标域 val（20% 标签）上拟合 y_true ≈ a·y_pred + b，用 (a,b) 校正 test 预测。
    这是标定转移的经典手段，对任何模型的预测都适用。
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


def run_cnn(splits, device, epochs):
    """与 18_exp 的 run_cnn(lambda_mmd=0) 一致，但额外返回 val 预测。"""
    model = BaselineCNN(input_dim=CONFIG['input_dim']).to(device)
    optim = Adam(model.parameters(), lr=CONFIG['learning_rate'],
                 weight_decay=CONFIG['weight_decay'])
    sched = CosineAnnealingLR(optim, T_max=epochs)
    crit  = nn.MSELoss()

    src_loader = to_loader(splits['X_src'],       splits['y_src'],       CONFIG['batch_size'], device=device)
    tgt_loader = to_loader(splits['X_tgt_train'], splits['y_tgt_train'], CONFIG['batch_size'], device=device)
    val_loader = to_loader(splits['X_tgt_val'],   splits['y_tgt_val'],   CONFIG['batch_size'], shuffle=False, device=device)

    best_val, best_st, patience = float('inf'), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for (Xs, ys), (Xt, _) in zip(src_loader, tgt_loader):
            optim.zero_grad()
            loss = crit(model(Xs), ys.squeeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
            optim.step()
        sched.step()

        if epoch % CONFIG['validation_freq'] == 0:
            model.eval()
            with torch.no_grad():
                vp = [model(Xv).cpu().numpy() for Xv, _ in val_loader]
                vt = [yv.cpu().numpy()        for _, yv in val_loader]
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
    return (_predict_np(model, splits['X_tgt_val'],  device, physics=False),
            _predict_np(model, splits['X_tgt_test'], device, physics=False))


def run_phys_bl(splits, device, epochs):
    """与 18_exp 的 run_phys_bl 一致，但额外返回 best-model 的 val 预测。"""
    model = PhysicsInformedDomainAdaptation(
        n_bands=CONFIG['input_dim'],
        n_components=CONFIG['n_components_bl'],
        hidden_dim=CONFIG['hidden_dim'],
    ).to(device)

    optim = Adam(model.parameters(), lr=CONFIG['learning_rate'],
                 weight_decay=CONFIG['weight_decay'])
    sched = CosineAnnealingLR(optim, T_max=epochs)

    src_loader = to_loader(splits['X_src'],       splits['y_src'],       CONFIG['batch_size'], device=device)
    tgt_loader = to_loader(splits['X_tgt_train'], splits['y_tgt_train'], CONFIG['batch_size'], device=device)
    val_loader = to_loader(splits['X_tgt_val'],   splits['y_tgt_val'],   CONFIG['batch_size'], shuffle=False, device=device)

    best_val, best_st, patience = float('inf'), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for (Xs, ys), (Xt, _) in zip(src_loader, tgt_loader):
            optim.zero_grad()
            results = model(Xs, ys, Xt)
            total_loss, _ = model.compute_total_loss(
                results, lambda_bl=CONFIG['lambda_bl'], lambda_mmd=0.0)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
            optim.step()
        sched.step()

        if epoch % CONFIG['validation_freq'] == 0:
            model.eval()
            with torch.no_grad():
                vp, vt = [], []
                for Xv, yv in val_loader:
                    vp.append(model(Xv)['y_pred'].cpu().numpy())
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
    return (_predict_np(model, splits['X_tgt_val'],  device, physics=True),
            _predict_np(model, splits['X_tgt_test'], device, physics=True))


# ═══════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════

def main(args):
    log('=' * 78)
    log('46 - 统一划分下的 slope/bias 校正对照实验')
    log('=' * 78)
    device = get_device(args.device)
    log(f'设备: {device} | epochs: {args.epochs} | 场景上限: {args.max_scenarios or "全部"}')

    set_seed(CONFIG['random_seed'])
    year_region_data = load_year_region_map(years=AVAILABLE_YEARS, target_wavelengths='2025')

    with open(os.path.join(_CODE_DIR, '04_migration_scenarios.json'), 'r', encoding='utf-8') as f:
        all_scenarios = json.load(f)

    scenario_list = list(all_scenarios.items())
    if args.max_scenarios and args.max_scenarios < len(scenario_list):
        sampled, per_type = [], max(1, args.max_scenarios // len(MIGRATION_TYPES))
        for mtype in MIGRATION_TYPES:
            pool = [(k, v) for k, v in scenario_list if v.get('type') == mtype]
            sampled.extend(pool[:per_type])
        scenario_list = sampled[:args.max_scenarios]
    log(f'场景数: {len(scenario_list)}')

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

        # ── 统一划分：与 18_exp 逐字节一致 ─────────────────────────────
        set_seed(CONFIG['random_seed'])
        splits = split_data(X_src, y_src, X_tgt, y_tgt)

        y_val_true, y_test_true = splits['y_tgt_val'], splits['y_tgt_test']
        if len(y_test_true) < 5 or len(y_val_true) < 3:
            continue

        preds = {}
        try:
            preds['PLSR'] = run_plsr(splits)
        except Exception as e:
            log(f'  [PLSR] {scenario_id}: {e}')
        try:
            preds['SVR'] = run_svr(splits)
        except Exception as e:
            log(f'  [SVR] {scenario_id}: {e}')
        try:
            set_seed(CONFIG['random_seed'])
            preds['CNN'] = run_cnn(splits, device, args.epochs)
        except Exception as e:
            log(f'  [CNN] {scenario_id}: {e}')
        try:
            set_seed(CONFIG['random_seed'])
            preds['Phys+BL'] = run_phys_bl(splits, device, args.epochs)
        except Exception as e:
            log(f'  [Phys+BL] {scenario_id}: {e}')

        for method, (val_pred, test_pred) in preds.items():
            if test_pred is None or len(test_pred) == 0:
                continue
            # 裸版
            rows.append({
                '场景': scenario_id, '迁移类型': config.get('type', '未知'),
                '方法': method, '校正': '裸',
                '品质误判率': grade_error_rate(y_test_true, test_pred, CONFIG['grade_threshold']),
                **compute_metrics(y_test_true, test_pred),
            })
            # +slope/bias 校正版（用同一份 20% target val 标签）
            test_sb = slope_bias_correction(y_val_true, val_pred, test_pred)
            rows.append({
                '场景': scenario_id, '迁移类型': config.get('type', '未知'),
                '方法': method, '校正': '+SB',
                '品质误判率': grade_error_rate(y_test_true, test_sb, CONFIG['grade_threshold']),
                **compute_metrics(y_test_true, test_sb),
            })

        if sc_idx % 10 == 0 or sc_idx == len(scenario_list):
            el = time.time() - t0
            log(f'  进度 {sc_idx}/{len(scenario_list)} | 已用 {el/60:.1f} min '
                f'| 预计剩余 {el/sc_idx*(len(scenario_list)-sc_idx)/60:.1f} min')

    df = pd.DataFrame(rows)
    if df.empty:
        log('!! 无有效结果')
        return

    df['方法_校正'] = df['方法'] + df['校正'].map({'裸': '', '+SB': ' + SB'})

    by_method = (df.groupby('方法_校正')
                   .agg(RMSE_mean=('RMSE', 'mean'), RMSE_median=('RMSE', 'median'),
                        R2_mean=('R2', 'mean'), RPD_mean=('RPD', 'mean'),
                        RPIQ_mean=('RPIQ', 'mean'), Bias_mean=('Bias', 'mean'),
                        误判率=('品质误判率', 'mean'),
                        R2大于0场景数=('R2', lambda s: int((s > 0).sum())),
                        n=('RMSE', 'size'))
                   .reset_index()
                   .sort_values('RMSE_mean'))

    by_type = (df.pivot_table(index='方法_校正', columns='迁移类型',
                              values='RMSE', aggfunc='mean')
                 .reset_index())

    os.makedirs(_OUT, exist_ok=True)
    xlsx = os.path.join(_OUT, f'{FILE_STEM}.xlsx')
    with pd.ExcelWriter(xlsx) as wr:
        by_method.to_excel(wr, sheet_name='按方法汇总', index=False)
        by_type.to_excel(wr, sheet_name='按迁移类型', index=False)
        df.to_excel(wr, sheet_name='全部原始结果', index=False)
    log(f'\n已保存: {xlsx}')

    log('\n=== 按方法汇总（RMSE 升序）===')
    log('\n' + by_method.to_string(index=False))
    log('\n=== 按迁移类型的平均 RMSE ===')
    log('\n' + by_type.to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='auto')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--max_scenarios', type=int, default=None)
    main(ap.parse_args())
