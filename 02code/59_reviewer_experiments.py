"""
59_reviewer_experiments.py：Round 2 审稿意见驱动的补充实验（严格零标签 + 经典基线）

回应两条审稿意见：
  · CRITICAL 3（严格零标签）：主试验中深度模型用 20% 目标域标签早停，PLSR/SVR 用 0，
    标签预算不对称。本脚本增设**严格零标签协议**：深度模型改用**源域**留出集早停，
    目标域仅提供无标签光谱（供 MMD/重构），全程不使用任何目标域标签；测试集与主试验一致。
    据此可公平回答"在真正零目标标签下深度模型是否仍优于化学计量学基线"。
  · MAJOR 3（经典化学计量学基线）：主试验未系统比较常见预处理。本脚本补
    PLSR+SNV、PLSR+MSC、PLSR+一阶导、PLSR+二阶导（均为直接迁移，0 目标标签）。

复用 50 号已验证的模型代码（导入），仅改数据划分与新增预处理。可分片并行。

运行方式:
    python 02code/59_reviewer_experiments.py --seed 42 --shard_id 0 --n_shards 24 [--zero_epochs 200]
    python 02code/59_reviewer_experiments.py --merge
    python 02code/59_reviewer_experiments.py --merge --override_dir 04outputs/59_shards_physfix   # 补跑个别方法后

输出:
    04outputs/59_shards/seed<S>_shard<NN>.csv
    04outputs/59_reviewer_experiments.xlsx（--merge）
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, '..'))
_OUT = os.path.join(_BASE, '04outputs')
_SHARD = os.path.join(_OUT, '59_shards')

# ── 导入 50 号（复用模型/训练/数据入口/CONFIG）─────────────────────────
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'benchmark50', os.path.join(_CODE, '50_formal_multiseed_benchmark.py'))
_b50 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b50)

CONFIG = _b50.CONFIG
set_seed = _b50.set_seed
compute_metrics = _b50.compute_metrics
grade_error_rate = _b50.grade_error_rate
run_cnn, run_phys = _b50.run_cnn, _b50.run_phys
load_year_region_map = _b50.load_year_region_map
AVAILABLE_YEARS = _b50.AVAILABLE_YEARS
from sklearn.preprocessing import StandardScaler


# ── 经典预处理（逐样本，直接迁移，0 目标标签）─────────────────────────
def snv(X):
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True)
    return (X - m) / np.maximum(s, 1e-8)


def msc(X, ref=None):
    ref = X.mean(axis=0) if ref is None else ref
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        a, b = np.polyfit(ref, X[i], 1)          # X_i ≈ a*ref + b
        a = np.clip(a, 0.1, 10.0)                 # 斜率近零时除法会爆炸，clip 到物理合理范围
        out[i] = (X[i] - b) / a
    return out, ref


def sg_deriv(X, order):
    return savgol_filter(X, window_length=11, polyorder=3, deriv=order, axis=1)


def plsr_with_prep(X_src, y_src, X_tgt_test, prep, ref_holder):
    """预处理 → 源域标准化(仅去均值口径与主线一致) → PLSR-10 → 目标测试。0 目标标签。"""
    if prep == 'SNV':
        Xs, Xt = snv(X_src), snv(X_tgt_test)
    elif prep == 'MSC':
        Xs, ref = msc(X_src); Xt, _ = msc(X_tgt_test, ref)
    elif prep == 'SG1':
        Xs, Xt = sg_deriv(X_src, 1), sg_deriv(X_tgt_test, 1)
    elif prep == 'SG2':
        Xs, Xt = sg_deriv(X_src, 2), sg_deriv(X_tgt_test, 2)
    else:
        Xs, Xt = X_src, X_tgt_test
    sc = StandardScaler().fit(Xs)
    m = PLSRegression(n_components=min(10, Xs.shape[1])).fit(sc.transform(Xs), y_src)
    return m.predict(sc.transform(Xt)).ravel()


# ── 严格零标签数据划分：源域 train/val 早停，目标域无标签+测试 ──────────
def split_strict_zero(X_src, y_src, X_tgt, y_tgt, val_frac=0.15):
    """
    源域 → 训练(1−val) + 留出验证(val，用于早停)；
    目标域 → 60% 无标签光谱（供 MMD/重构，标签不用）+ 20% 测试（与主试验同口径）。
    把源域验证集放进 'X_tgt_val'/'y_tgt_val' 槽，_train_loop 即以源域早停，无需改模型代码。
    """
    scaler = StandardScaler().fit(X_src)
    Xs, Xt = scaler.transform(X_src), scaler.transform(X_tgt)
    ns = len(Xs)
    si = np.random.permutation(ns)
    cut = int(ns * (1 - val_frac))
    # 目标域仍按 60/20/20，只取无标签 train 与 test（与主试验 test 同口径）
    nt = len(Xt)
    ti = np.random.permutation(nt)
    t1, t2 = int(nt * 0.6), int(nt * 0.8)
    return {
        'X_src': Xs[si[:cut]], 'y_src': y_src[si[:cut]],
        'X_tgt_train': Xt[ti[:t1]], 'y_tgt_train': y_tgt[ti[:t1]],   # 标签不用于训练
        'X_tgt_val': Xs[si[cut:]], 'y_tgt_val': y_src[si[cut:]],     # 源域留出 → 早停
        'X_tgt_test': Xt[ti[t2:]], 'y_tgt_test': y_tgt[ti[t2:]],
    }


def run_shard(args):
    set_seed(args.seed)
    data = load_year_region_map(years=AVAILABLE_YEARS, target_wavelengths='2025')
    with open(os.path.join(_CODE, '04_migration_scenarios.json'), encoding='utf-8') as f:
        scen = json.load(f)
    items = list(scen.items())[args.shard_id::args.n_shards]

    def dom_X(dcfg):
        Xs, ys = [], []
        for d in dcfg['domains']:
            k = (d['year'], d['region'])
            Xs.append(np.asarray(data[k]['X'])); ys.append(np.asarray(data[k]['y']))
        return np.vstack(Xs), np.concatenate(ys)

    rows = []
    for sid, cfg in items:
        try:
            X_src, y_src = dom_X(cfg['source']); X_tgt, y_tgt = dom_X(cfg['target'])
        except Exception:
            continue
        base = {'场景': sid, '迁移类型': cfg.get('type', '?'), '种子': args.seed}
        # 1) 经典基线（源域全量拟合 → 目标测试；0 目标标签；用原始光谱）
        set_seed(args.seed)
        _sp_raw = _split_raw(X_src, y_src, X_tgt, y_tgt)
        if len(_sp_raw['y_tgt_test']) < 5:
            continue
        for prep in ([] if args.methods else ['SNV', 'MSC', 'SG1', 'SG2']):
            try:
                pred = plsr_with_prep(_sp_raw['X_src'], _sp_raw['y_src'], _sp_raw['X_tgt_test'], prep, None)
                rows.append({**base, '方法': f'PLSR+{prep}', '协议': '经典基线(0标签)',
                             '品质误判率': grade_error_rate(_sp_raw['y_tgt_test'], pred, CONFIG['grade_threshold']),
                             **compute_metrics(_sp_raw['y_tgt_test'], pred)})
            except Exception as e:
                pass
        if args.classical_only:
            continue
        # 2) 严格零标签深度模型（源域早停）
        set_seed(args.seed)
        spz = split_strict_zero(X_src, y_src, X_tgt, y_tgt)
        if len(spz['y_tgt_test']) < 5:
            continue
        deep = [('CNN', lambda: run_cnn(spz, args.device, args.zero_epochs)),
                ('CNN+MMD', lambda: run_cnn(spz, args.device, args.zero_epochs, lambda_mmd=0.1)),
                ('Phys+BL', lambda: run_phys(spz, args.device, args.zero_epochs, lambda_bl=0.5))]
        if args.methods:
            # 只补跑点名的深度方法：经典基线整段跳过，划分与种子序列照旧，场景集合不变
            deep = [(n, f) for n, f in deep if n in args.methods.split(',')]
        for name, fn in deep:
            try:
                set_seed(args.seed)
                _, test_pred = fn()
                if test_pred is None or len(test_pred) == 0:
                    continue
                rows.append({**base, '方法': name, '协议': '严格零标签(源域早停)',
                             '品质误判率': grade_error_rate(spz['y_tgt_test'], test_pred, CONFIG['grade_threshold']),
                             **compute_metrics(spz['y_tgt_test'], test_pred)})
            except Exception as e:
                print(f'  [{name}] {sid}: {e}')

    os.makedirs(_SHARD, exist_ok=True)
    out = os.path.join(_SHARD, f'seed{args.seed}_shard{args.shard_id:02d}.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'saved {out} | {len(rows)} 行')


def _split_raw(X_src, y_src, X_tgt, y_tgt):
    """经典基线用：不标准化，直接给原始光谱 + 目标 20% 测试（同口径）。"""
    nt = len(X_tgt); ti = np.random.permutation(nt); t2 = int(nt * 0.8)
    return {'X_src': X_src, 'y_src': y_src,
            'X_tgt_test': X_tgt[ti[t2:]], 'y_tgt_test': y_tgt[ti[t2:]]}


def do_merge(override_dir=None):
    files = sorted(glob.glob(os.path.join(_SHARD, 'seed*_shard*.csv')))
    if not files:
        raise SystemExit('无分片')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # 覆盖：只补跑了个别方法时，覆盖目录里出现的（协议, 方法）整体替换原分片里的同名组合
    if override_dir:
        odir = override_dir if os.path.isabs(override_dir) else os.path.join(_BASE, override_dir)
        ofiles = sorted(glob.glob(os.path.join(odir, 'seed*_shard*.csv')))
        if not ofiles:
            raise SystemExit(f'覆盖目录 {odir} 里没有分片')
        ov = pd.concat([pd.read_csv(f) for f in ofiles], ignore_index=True)
        pairs = set(map(tuple, ov[['协议', '方法']].drop_duplicates().to_numpy()))
        hit = df[['协议', '方法']].apply(tuple, axis=1).isin(pairs)
        df = pd.concat([df[~hit], ov], ignore_index=True)
        print(f'覆盖：{odir} 的 {len(ofiles)} 个分片 | {len(ov)} 行 | {sorted(pairs)} '
              f'整体替换原分片中的 {int(hit.sum())} 行')
    def _agg(d):
        return d.groupby(['协议', '方法']).agg(
            RMSE_mean=('RMSE', 'mean'), RMSE_median=('RMSE', 'median'),
            R2_median=('R2', 'median'), RPD_mean=('RPD', 'mean'),
            误判率_mean=('品质误判率', 'mean'), n=('RMSE', 'size')).reset_index()

    # 按方法×协议聚合（各方法各自跑通的场景）
    agg = _agg(df)

    # 共同场景子集：深度模型在个别场景上不收敛，缺的往往正是最难的场景，
    # 在「各自幸存的集合」上取的均值不可横比。与 50 号同一口径，另报一份交集。
    common = None
    for _, g in df.groupby('方法'):
        s_ = set(g['场景'])
        common = s_ if common is None else (common & s_)
    common = sorted(common) if common else []
    agg_common = _agg(df[df['场景'].isin(common)]) if common else pd.DataFrame()

    with pd.ExcelWriter(os.path.join(_OUT, '59_reviewer_experiments.xlsx')) as w:
        agg.to_excel(w, sheet_name='汇总', index=False)
        if not agg_common.empty:
            agg_common.to_excel(w, sheet_name='表a2：共同场景子集', index=False)
        df.to_excel(w, sheet_name='原始', index=False)
    print(agg.to_string(index=False))
    print(f'\n共同场景（全部方法均跑通）: {len(common)} / {df["场景"].nunique()}')
    if not agg_common.empty:
        print(agg_common.to_string(index=False))
    print('\n已保存 04outputs/59_reviewer_experiments.xlsx')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--shard_id', type=int, default=0)
    ap.add_argument('--n_shards', type=int, default=1)
    ap.add_argument('--zero_epochs', type=int, default=200)
    ap.add_argument('--classical_only', action='store_true')
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--methods', default=None,
                    help='逗号分隔的深度方法子集（CNN / CNN+MMD / Phys+BL）；给出时跳过经典基线，'
                         '用于只补跑个别方法而不动其余分片')
    ap.add_argument('--shard_dir', default=None,
                    help='分片 CSV 的读写目录（默认 04outputs/59_shards）；补跑时另指定，避免覆盖原分片')
    ap.add_argument('--override_dir', default=None,
                    help='合并时的覆盖分片目录：其中出现的（协议, 方法）整体替换原分片里的同名组合'
                         '（如 04outputs/59_shards_physfix）')
    args = ap.parse_args()
    if args.shard_dir:
        global _SHARD
        _SHARD = args.shard_dir if os.path.isabs(args.shard_dir) else os.path.join(_BASE, args.shard_dir)
    if args.merge:
        do_merge(args.override_dir)
    else:
        run_shard(args)


if __name__ == '__main__':
    main()
