"""
106_season_scaler_sensitivity.py: 年份级逐波段标准化的敏感性

【为什么做】02 号在每个年份内按全部产地拟合逐波段 StandardScaler，发生在 50 号划分之前，
所以目标年份的无标签光谱——包括目标域测试集的光谱——参与了标准化统计量的拟合。这一步不能
直接去掉：2018/2019 的光谱是原始强度（约 10^1–10^4.8），2025 是反射率（0.1–1），而全部跨年
场景都含 2025，去掉之后源域与目标域根本不在同一量纲上。本脚本把年份级标准化挪进场景循环，
按三种口径重新拟合每个年份的均值与标准差：
  all        该年份全部光谱（与 02 号相同；作为同一平台上的对照）
  no_test    扣除目标域测试集的光谱
  no_target  扣除目标域的全部光谱（目标域所在年份须还有别的产地；602 个场景中 564 个可定义）
其余一切——离群剔除、小波、SG、划分、种子、模型、slope/bias 校正——与 50 号逐项相同，
相关函数直接从 50 号导入。

数据：02 号加 --no_season_scaler 生成（输出停在 SG 平滑之后），经 PROJECT_DATA_DIR 指给本脚本。

运行方式（项目根目录；分片为 stride 切分，每个场景、每个方法开跑前都重置同一粒种子，
分片数不影响结果）:
    PROJECT_DATA_DIR=<无年份标准化的数据目录> python 02code/106_season_scaler_sensitivity.py \
        --seed 20060515 --shard_id 0 --n_shards 60 --shard_dir 04outputs/106_shards
    python 02code/106_season_scaler_sensitivity.py --merge --shard_dir 04outputs/106_shards

输出文件:
    <shard_dir>/seed<S>_shard<NN>.csv / .meta.json    — 分片原始结果
    04outputs/106_season_scaler_sensitivity.xlsx       — 合并与对比（--merge）
    05logs/106_season_scaler_sensitivity_<时间戳><标签>.log
"""

import argparse
import glob
import importlib.util
import json
import os
import platform
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE_DIR, '..'))
_OUT = os.path.join(_BASE, '04outputs')
_LOG = os.path.join(_BASE, '05logs')
FILE_STEM = '106_season_scaler_sensitivity'

_spec = importlib.util.spec_from_file_location(
    'bench50', os.path.join(_CODE_DIR, '50_formal_multiseed_benchmark.py'))
assert _spec is not None  # 对已知存在的 .py 文件 spec 恒非 None
assert _spec.loader is not None  # 同理 loader 恒非 None
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

VARIANTS = ['all', 'no_test', 'no_target']
DEFAULT_METHODS = ['PLSR', 'SVR', 'CNN', 'Phys+BL']

F64 = npt.NDArray[np.float64]
Base = tuple[int, F64, F64, F64]   # 一个年份的 (行数, 中心 c, (x−c) 列和, (x−c)² 列和)
Logger = Callable[[str], None]


def make_logger(tag: str = '') -> Logger:
    os.makedirs(_LOG, exist_ok=True)
    stamp = datetime.now().strftime('%y-%m-%d_%H%M%S')
    path = os.path.join(_LOG, f'{FILE_STEM}_{stamp}{tag}.log')

    def log(msg: str) -> None:
        line = f'[{datetime.now().strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    return log


def season_bases() -> dict[int, Base]:
    """每个年份拟合统计量的底数：02 号拟合 StandardScaler 时用的同一批行——CSV 的全部行，
    含读数入口（01_export_utils）按 SSC 越界、逐位重复光谱剔掉的少数几行；列取对齐到 2025
    波长网格后的那些列。记下行数、中心 c 与 (x−c)、(x−c)² 的列和，扣除时只减去被扣行的贡献。"""
    U = B._mod1
    ref = U.get_reference_wavelengths(U.REFERENCE_YEAR)
    bases: dict[int, Base] = {}
    for year in B.AVAILABLE_YEARS:
        df = pd.read_csv(os.path.join(U.DATA_DIR, U.PROCESSED_DATA_FILES[year]))
        cols, _ = U.align_spectrum_columns(df, ref)
        P = df[cols].to_numpy(np.float64)
        c = P.mean(axis=0)
        D = P - c
        bases[year] = (len(P), c, D.sum(axis=0), (D ** 2).sum(axis=0))
    return bases


def fit_stats(bases: dict[int, Base], X_excl: npt.NDArray[Any],
              excl_years: npt.NDArray[Any]) -> dict[int, tuple[F64, F64]]:
    """扣除 X_excl 这些行之后每个年份的均值与标准差（ddof=0，与 StandardScaler 一致）。"""
    stats: dict[int, tuple[F64, F64]] = {}
    for y, (n, c, s1, s2) in bases.items():
        m = excl_years == y
        k = int(m.sum())
        if k:
            D = X_excl[m] - c
            s1, s2 = s1 - D.sum(axis=0), s2 - (D ** 2).sum(axis=0)
        d = s1 / (n - k)
        sd = np.sqrt(np.maximum(s2 / (n - k) - d ** 2, 0.0))
        sd[sd == 0] = 1.0
        stats[y] = (c + d, sd)
    return stats


def season_standardize(X: npt.NDArray[Any], row_years: npt.NDArray[Any],
                       stats: dict[int, tuple[F64, F64]]) -> npt.NDArray[np.float32]:
    Z = np.empty_like(X, dtype=np.float64)
    for y, (mu, sd) in stats.items():
        m = row_years == y
        if m.any():
            Z[m] = (X[m] - mu) / sd
    return Z.astype(np.float32)        # 50 号以 float32 读入已标准化的数据，这里保持同一精度


def run_shard(args: argparse.Namespace, log: Logger) -> None:
    device = B.get_device(args.device)
    B.enforce_determinism(log)
    methods = [m.strip() for m in args.methods.split(',')]
    variants = [v.strip() for v in args.variants.split(',')]
    assert all(m in B.METHODS for m in methods), methods
    assert all(v in VARIANTS for v in variants), variants

    log('=' * 78)
    log(f'106 - 年份级标准化敏感性 | seed={args.seed} | shard {args.shard_id}/{args.n_shards}')
    log(f'设备={device} | epochs={args.epochs} | 方法={methods} | 口径={variants}')
    log(f'数据目录={B._mod1.DATA_DIR}')
    log('=' * 78)

    B.set_seed(args.seed)
    yrd = B.load_year_region_map(years=B.AVAILABLE_YEARS, target_wavelengths='2025',
                                 dtype=np.float64)
    bases = season_bases()
    origins: dict[int, set[str]] = {}
    for y, r in yrd:
        origins.setdefault(y, set()).add(r)

    with open(os.path.join(_CODE_DIR, '04_migration_scenarios.json'), encoding='utf-8') as f:
        scenario_list = list(json.load(f).items())
    if args.max_scenarios:
        scenario_list = scenario_list[:args.max_scenarios]
    scenario_list = scenario_list[args.shard_id::args.n_shards]
    log(f'本分片场景数: {len(scenario_list)}')

    rows, t0 = [], time.time()
    for sc_idx, (scenario_id, config) in enumerate(scenario_list, 1):
        try:
            src = B.merge_domain_payload(yrd, B.extract_domain_keys(config['source']))
            tgt = B.merge_domain_payload(yrd, B.extract_domain_keys(config['target']))
        except KeyError:
            continue
        y_src, y_tgt = src['y'].astype(np.float32), tgt['y'].astype(np.float32)
        if len(y_tgt) < 15 or len(y_src) < 10:
            continue
        src_years, tgt_years = src['years'], tgt['years']
        tgt_keys = set(tgt['domain_keys'])
        # no_target 只在目标域所在的每个年份都还有别的产地时可定义
        no_target_ok = all(origins[y] - {r for yy, r in tgt_keys if yy == y}
                           for y in {yy for yy, _ in tgt_keys})

        # 与 50 号 split_data 同一粒种子、同一次 permutation，先取出测试集下标
        n_tgt = len(y_tgt)
        B.set_seed(args.seed)
        idx = np.random.permutation(n_tgt)
        test_idx = idx[int(n_tgt * 0.8):]

        for variant in variants:
            if variant == 'all':
                keep = np.zeros(n_tgt, dtype=bool)
            elif variant == 'no_test':
                keep = np.zeros(n_tgt, dtype=bool)
                keep[test_idx] = True
            else:
                if not no_target_ok:
                    continue
                keep = np.ones(n_tgt, dtype=bool)
            stats = fit_stats(bases, tgt['X'][keep], tgt_years[keep])
            X_src = season_standardize(src['X'], src_years, stats)
            X_tgt = season_standardize(tgt['X'], tgt_years, stats)

            B.set_seed(args.seed)
            splits = B.split_data(X_src, y_src, X_tgt, y_tgt)
            y_val_true, y_test_true = splits['y_tgt_val'], splits['y_tgt_test']
            if len(y_test_true) < 5 or len(y_val_true) < 3:
                continue

            # 这些 lambda 只在本轮循环内（下面的 for name in methods）立即调用，不会带着 splits 留到下一轮，
            # 故 B023（lambda 不绑定循环变量）在此不构成问题
            all_runners: dict[str, Callable[[], Any]] = {
                'PLSR':    lambda: B.run_plsr(splits),  # noqa: B023
                'SVR':     lambda: B.run_svr(splits),  # noqa: B023
                'CNN':     lambda: B.run_cnn(splits, device, args.epochs),  # noqa: B023
                'CNN+MMD': lambda: B.run_cnn(splits, device, args.epochs,  # noqa: B023
                                             lambda_mmd=B.CONFIG['lambda_mmd_cnn']),
                'CNN+BL':  lambda: B.run_cnn(splits, device, args.epochs,  # noqa: B023
                                             lambda_bl=B.CONFIG['lambda_bl']),
                'Phys':    lambda: B.run_phys(splits, device, args.epochs, lambda_bl=0.0),  # noqa: B023
                'Phys+BL': lambda: B.run_phys(splits, device, args.epochs,  # noqa: B023
                                              lambda_bl=B.CONFIG['lambda_bl']),
            }
            for name in methods:
                try:
                    B.set_seed(args.seed)
                    val_pred, test_pred = all_runners[name]()
                except Exception as e:
                    log(f'  [{variant}/{name}] {scenario_id}: {e}')
                    continue
                if test_pred is None or len(test_pred) == 0:
                    continue
                base = {'场景': scenario_id, '迁移类型': config.get('type', '未知'),
                        '方法': name, '种子': args.seed, '标准化口径': variant,
                        'n_test': len(y_test_true), 'n_src': len(y_src), 'n_tgt': n_tgt}
                rows.append({**base, '校正': '裸',
                             '品质误判率': B.grade_error_rate(y_test_true, test_pred,
                                                         B.CONFIG['grade_threshold']),
                             **B.compute_metrics(y_test_true, test_pred)})
                test_sb = B.slope_bias_correction(y_val_true, val_pred, test_pred)
                rows.append({**base, '校正': '+SB',
                             '品质误判率': B.grade_error_rate(y_test_true, test_sb,
                                                         B.CONFIG['grade_threshold']),
                             **B.compute_metrics(y_test_true, test_sb)})

        if sc_idx % 5 == 0 or sc_idx == len(scenario_list):
            el = time.time() - t0
            log(f'  进度 {sc_idx}/{len(scenario_list)} | 已用 {el/60:.1f} min '
                f'| 预计剩余 {el/sc_idx*(len(scenario_list)-sc_idx)/60:.1f} min')

    df = pd.DataFrame(rows)
    os.makedirs(args.shard_dir, exist_ok=True)
    out = os.path.join(args.shard_dir, f'seed{args.seed}_shard{args.shard_id:02d}.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')
    meta = {'seed': args.seed, 'shard_id': args.shard_id, 'n_shards': args.n_shards,
            'n_scenarios': len(scenario_list), 'n_rows': len(df),
            'methods': methods, 'variants': variants,
            'wall_clock_s': round(time.time() - t0, 1), 'device': device,
            'epochs': args.epochs, 'torch': torch.__version__, 'platform': platform.platform()}
    with open(out.replace('.csv', '.meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f'已保存: {out} | {len(df)} 行 | wall {meta["wall_clock_s"]/60:.1f} min')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu', choices=['auto', 'cuda', 'mps', 'cpu'])
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--seed', type=int, default=B.FORMAL_SEEDS[0])
    ap.add_argument('--shard_id', type=int, default=0)
    ap.add_argument('--n_shards', type=int, default=1)
    ap.add_argument('--max_scenarios', type=int, default=None, help='仅调试用')
    ap.add_argument('--methods', default=','.join(DEFAULT_METHODS))
    ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--shard_dir', default='04outputs/106_shards')
    ap.add_argument('--merge', action='store_true')
    args = ap.parse_args()
    if not os.path.isabs(args.shard_dir):
        args.shard_dir = os.path.join(_BASE, args.shard_dir)

    if args.merge:
        log = make_logger('_merge')
        do_merge(args, log)
    else:
        log = make_logger(f'_seed{args.seed}_shard{args.shard_id:02d}')
        run_shard(args, log)


def do_merge(args: argparse.Namespace, log: Logger) -> None:
    """三种口径逐场景配对：各方法中位 RMSE、§3.5 的四个比较、口径相对 all 的位移；
    另把 all 口径与正典分片（同一种子）对照，量出跨平台的浮点差异。"""
    files = sorted(glob.glob(os.path.join(args.shard_dir, 'seed*_shard*.csv')))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    log(f'分片 {len(files)} 个，{len(df)} 行，场景 {df["场景"].nunique()} 个')
    df['方法_校正'] = df['方法'] + np.where(df['校正'] == '+SB', ' + SB', '')
    wide = df.pivot_table(index=['标准化口径', '场景'], columns='方法_校正', values='RMSE')
    cols = list(wide.columns)
    # 共同场景：该口径下全部方法×校正都有值；三口径再取交集（no_target 只在可定义的场景上）
    ok = {v: set(wide.loc[v].dropna().index) for v in wide.index.get_level_values(0).unique()}
    common_all = ok['all'] & ok['no_test']
    common3 = common_all & ok.get('no_target', set())
    log(f'all∩no_test 共同场景 {len(common_all)}；三口径共同场景 {len(common3)}')

    rng = np.random.default_rng(20060515)

    def boot_median(x: Any, n: int = 2000) -> tuple[float, float]:
        x = np.asarray(x, float)
        bs = np.median(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
        return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    comps = [('未校正 Phys+BL − CNN', 'Phys+BL', 'CNN'),
             ('校正后 PLSR+SB − Phys+BL+SB', 'PLSR + SB', 'Phys+BL + SB'),
             ('校正后 SVR+SB − Phys+BL+SB', 'SVR + SB', 'Phys+BL + SB'),
             ('校正后 CNN+SB − Phys+BL+SB', 'CNN + SB', 'Phys+BL + SB')]
    rows_med, rows_cmp, rows_shift = [], [], []
    for scope, sc_set in (('all∩no_test', common_all), ('三口径共同', common3)):
        sc = sorted(sc_set)
        for v in ('all', 'no_test', 'no_target'):
            if scope == 'all∩no_test' and v == 'no_target':
                continue
            W = wide.loc[v].loc[sc]
            for c in cols:
                rows_med.append({'场景集': scope, '口径': v, '方法_校正': c, 'n': len(sc),
                                 '中位RMSE': round(float(W[c].median()), 4),
                                 '均值RMSE': round(float(W[c].mean()), 4)})
            for lab, a, b in comps:
                d = W[a] - W[b]
                lo, hi = boot_median(d)
                rows_cmp.append({'场景集': scope, '口径': v, '比较': lab, 'n': len(d),
                                 '配对差中位': round(float(d.median()), 4), 'CI下界': round(lo, 4),
                                 'CI上界': round(hi, 4), '配对差均值': round(float(d.mean()), 4),
                                 '前者更低的场景占比': round(float((d < 0).mean()), 3)})
            if v != 'all':
                A = wide.loc['all'].loc[sc]
                for c in cols:
                    d = W[c] - A[c]
                    rows_shift.append({'场景集': scope, '口径': v, '方法_校正': c, 'n': len(d),
                                       '相对all配对差中位': round(float(d.median()), 4),
                                       '相对all配对差均值': round(float(d.mean()), 4),
                                       '|差|中位': round(float(d.abs().median()), 4)})
    sheets = {'表a：各口径中位与均值RMSE': pd.DataFrame(rows_med),
              '表b：§3.5四个比较': pd.DataFrame(rows_cmp),
              '表c：口径相对all的位移': pd.DataFrame(rows_shift)}

    # all 口径对正典（50 号分片 + Phys/Phys+BL 修正后补跑分片，同一种子）
    seed = int(df['种子'].iloc[0])
    canon = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(_OUT, '50_shards', f'seed{seed}_shard*.csv'))])
    fix = pd.concat([pd.read_csv(f)
                     for f in glob.glob(os.path.join(_OUT, '50_shards_physfix', f'seed{seed}_shard*.csv'))])
    canon = pd.concat([canon[~canon['方法'].isin(fix['方法'].unique())], fix])
    m = df[df['标准化口径'] == 'all'].merge(canon, on=['场景', '方法', '校正'], suffixes=('', '_正典'))
    rows_can = []
    for (meth, cor), g in m.groupby(['方法', '校正']):
        d = (g['RMSE'] - g['RMSE_正典']).abs()
        rows_can.append({'方法': meth, '校正': cor, '配对场景': len(g),
                         '|差|中位': float(d.median()), '|差|P90': float(d.quantile(0.9)),
                         '|差|最大': float(d.max()), '差<1e-6 的比例': round(float((d < 1e-6).mean()), 3)})
    sheets['表d：all口径对正典（跨平台浮点差）'] = pd.DataFrame(rows_can)

    # 同年场景里两层标准化互相抵消，重拟合只在浮点舍入上改变输入：那里的 |差| 就是网络对舍入级扰动的训练噪声，
    # 跨年场景的位移要拿它当参照
    lw = df.pivot_table(index=['场景', '方法_校正'], columns='标准化口径', values='RMSE').reset_index()
    lw['同年'] = lw['场景'].map(lambda sc: ('2018' in sc) != ('2025' in sc))
    rows_noise = []
    for v in ('no_test', 'no_target'):
        for (same, c), g in lw.groupby(['同年', '方法_校正']):
            d = (g[v] - g['all']).abs().dropna()
            rows_noise.append({'口径': v, '场景类型': '同年' if same else '跨年', '方法_校正': c,
                               '场景数': int(g['场景'].nunique()), '配对数': len(d),
                               '|差|中位': round(float(d.median()), 4)})
    sheets['表e：同年舍入噪声与跨年位移'] = pd.DataFrame(rows_noise)
    sheets['表f：场景计数'] = pd.DataFrame([{
        '场景总数': int(df['场景'].nunique()),
        'no_target可定义场景数': int(df.loc[df['标准化口径'] == 'no_target', '场景'].nunique()),
        'all∩no_test共同场景': len(common_all), '三口径共同场景': len(common3),
        '同年场景数': int(lw.loc[lw['同年'], '场景'].nunique()), '种子': seed}])
    # 各季未做年份级标准化的光谱数值范围（S2 节据此说明各季原始信号单位不同）；
    # 只有 PROJECT_DATA_DIR 指向 02 号 --no_season_scaler 生成的目录时才算得出，合并时须带上它
    dd = os.environ.get('PROJECT_DATA_DIR')
    if dd:
        rows_rng = []
        for year in (2018, 2019, 2025):
            raw = pd.read_csv(os.path.join(dd, f'02_data_{year}.csv'))
            P = raw[[c for c in raw.columns if c.startswith('波段')]].to_numpy(float)
            rows_rng.append({'年份': year, '行数': len(P), '波段数': P.shape[1],
                             '最小': float(P.min()), '最大': float(P.max())})
        sheets['表g：各季未标准化光谱范围'] = pd.DataFrame(rows_rng)
    else:
        log('未设 PROJECT_DATA_DIR：不写表g（各季未标准化光谱范围）')
    path = os.path.join(_OUT, f'{FILE_STEM}.xlsx')
    with pd.ExcelWriter(path) as w:
        for k, v in sheets.items():
            v.to_excel(w, sheet_name=k[:31], index=False)
    for k, v in sheets.items():
        log(f'== {k}\n{v.to_string(index=False)}')
    log(f'已写 {path}')


if __name__ == '__main__':
    main()
