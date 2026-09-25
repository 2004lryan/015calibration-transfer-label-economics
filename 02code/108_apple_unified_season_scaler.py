"""
108_apple_unified_season_scaler.py: 统一基准苹果任务上年份级逐波段标准化的敏感性

【为什么做】统一基准的苹果域由 61 号 load_apple 读入 02 号的输出（02_data_{年}.csv），那份光谱
已在每个年份内按全部产地做过逐波段标准化，随后才进入 62/64 号的 SNV 与迁移。与 106 号（案例研究）
同理，这一步用到了目标年份的无标签光谱，包括目标域本身。本脚本对 72 个苹果任务按两种口径重拟合
年份统计量，再原样调用 62 号（简单族与经典昂贵方法）与 64 号（深度族）的 run_task：
  all        该年份全部光谱（与 02 号相同；简单族应在浮点舍入内复现正典结果，作为管线自检）
  no_target  扣除目标域的全部光谱（苹果每季三个产地，72 个任务都可定义）
年份统计量的底数与 02 号拟合 StandardScaler 时相同：CSV 的全部行（含读数入口按 SSC 越界、
逐位重复光谱剔掉的少数几行）、全部波段。

数据：02 号加 --no_season_scaler 生成的目录，经 PROJECT_DATA_DIR 指给本脚本（必须在导入 61 号之前设好）。

运行方式（项目根目录）:
    PROJECT_DATA_DIR=<无年份标准化的数据目录> python 02code/108_apple_unified_season_scaler.py \
        --family simple --variant no_target --seeds 20060515,... --shard 0 --n_shards 4
    python 02code/108_apple_unified_season_scaler.py --analyze

输出文件:
    04outputs/108_shards/<family>_<variant>[<tag>]_shard<NN>.csv — 与 62/64 号 curves 同列，另加 variant
    04outputs/108_apple_unified_season_scaler.xlsx          — 最优简单 vs 最优深度（同 65 号口径）三方对照
"""

import argparse
import glob
import importlib.util
import json
import os
import platform
import sys
import time
import types
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, '..'))
_OUT = os.path.join(_BASE, '04outputs')
SHARD_DIR = os.path.join(_OUT, '108_shards')
FILE_STEM = '108_apple_unified_season_scaler'
SEEDS = [20060515, 20041210, 19810915, 2023, 2024]

F64 = npt.NDArray[np.float64]
Base = tuple[int, F64, F64, F64]   # 一个年份的 (行数, 中心 c, (x−c) 列和, (x−c)² 列和)


def _load(name: str, fname: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, os.path.join(_CODE, fname))
    assert spec is not None  # 对已知存在的 .py 文件 spec 恒非 None
    assert spec.loader is not None  # 同理 loader 恒非 None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def season_bases(data_dir: str) -> dict[int, Base]:
    """每个年份：CSV 全部行、全部波段列上的行数、中心 c 与 (x−c)、(x−c)² 的列和。"""
    bases: dict[int, Base] = {}
    for year in (2018, 2019, 2025):
        df = pd.read_csv(os.path.join(data_dir, f'02_data_{year}.csv'))
        bc = [c for c in df.columns if c.startswith('波段')]
        P = df[bc].to_numpy(float)
        c = P.mean(axis=0)
        D = P - c
        bases[year] = (len(P), c, D.sum(axis=0), (D ** 2).sum(axis=0))
    return bases


def season_stats(base: Base, X_excl: npt.NDArray[Any] | None = None) -> tuple[F64, F64]:
    n, c, s1, s2 = base
    k = 0
    if X_excl is not None and len(X_excl):
        D = X_excl - c
        s1, s2, k = s1 - D.sum(axis=0), s2 - (D ** 2).sum(axis=0), len(X_excl)
    d = s1 / (n - k)
    sd = np.sqrt(np.maximum(s2 / (n - k) - d ** 2, 0.0))
    sd[sd == 0] = 1.0
    return c + d, sd


def task_domains(raw: dict[str, Any], bases: dict[int, Base], src: str, tgt: str,
                 variant: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """返回该任务下按口径标准化后的源域、目标域（其余字段照抄）。"""
    ys, yt = int(src.split('_')[0]), int(tgt.split('_')[0])
    excl = raw[tgt]['X'] if variant == 'no_target' else None
    out: dict[str, dict[str, Any]] = {}
    for key, year in ((src, ys), (tgt, yt)):
        mu, sd = season_stats(bases[year], excl if year == yt else None)
        out[key] = {**raw[key], 'X': (raw[key]['X'] - mu) / sd}
    return out[src], out[tgt]


def run(args: argparse.Namespace) -> None:
    data_dir = os.environ.get('PROJECT_DATA_DIR')
    if not data_dir:
        raise SystemExit('须设 PROJECT_DATA_DIR 指向无年份标准化的数据目录')
    bench = _load('bench61', '61_benchmark_datasets.py')
    raw_b = bench.load_apple()
    raw = raw_b['domains']
    bases = season_bases(data_dir)
    tasks = bench.build_transfer_tasks(raw_b)[args.shard::args.n_shards]
    seeds = [int(s) for s in args.seeds.split(',')]
    rows = []
    t0 = time.time()
    versions = {'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pd.__version__}
    if args.family == 'simple':
        E = _load('eng62', '62_crossover_engine.py')
        E.REP = args.rep  # type: ignore[attr-defined]  # 动态加载模块，REP 为其运行期全局，静态不可见
        for tk in tasks:
            tid = f"apple|{tk['src']}->{tk['tgt']}|{tk['prop_name']}"
            ds, dt = task_domains(raw, bases, tk['src'], tk['tgt'], args.variant)
            for seed in seeds:
                res, _ = E.run_task(ds, dt, tk['prop_idx'], tk['paired'], np.random.default_rng(seed))
                for nc, meth, r, val in res or []:
                    rows.append({'task_id': tid, 'benchmark': 'apple', 'shift_type': tk['shift_type'],
                                 'n_cal': nc, 'method': meth, 'seed': seed, 'rep': r, 'rmsep': val})
            print(f'[simple/{args.variant}] {tid} 完成', flush=True)
    else:
        import torch
        torch.set_num_threads(1)
        versions['torch'] = torch.__version__
        D = _load('deep64', '64_deep_transfer_server.py')
        for tk in tasks:
            tid = f"apple|{tk['src']}->{tk['tgt']}|{tk['prop_name']}"
            ds, dt = task_domains(raw, bases, tk['src'], tk['tgt'], args.variant)
            b = {'domains': {tk['src']: ds, tk['tgt']: dt}}
            for seed in seeds:
                for nc, meth, sd, r, val in D.run_task(b, tk, seed, args.device, rep=args.rep):
                    rows.append({'task_id': tid, 'benchmark': 'apple', 'shift_type': tk['shift_type'],
                                 'n_cal': nc, 'method': meth, 'seed': sd, 'rep': r, 'rmsep': val})
            print(f'[deep/{args.variant}] {tid} 完成', flush=True)
    os.makedirs(SHARD_DIR, exist_ok=True)
    out = os.path.join(SHARD_DIR, f'{args.family}_{args.variant}{args.tag}_shard{args.shard:02d}.csv')
    pd.DataFrame(rows).assign(variant=args.variant).to_csv(out, index=False, encoding='utf-8-sig')
    # 运行平台随分片记下：同一口径的分片可能分在本机与服务器、CPU 与 GPU 上跑
    meta = {'family': args.family, 'variant': args.variant, 'seeds': seeds, 'shard': args.shard,
            'n_shards': args.n_shards, 'n_rows': len(rows),
            'device': args.device if args.family == 'deep' else 'cpu',
            'wall_clock_s': round(time.time() - t0, 1), 'platform': platform.platform(), **versions}
    with open(out.replace('.csv', '.meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f'已保存 {out}：{len(rows)} 行')


def analyze() -> None:
    """最优简单 vs 最优深度（65 号口径）：正典、正典限同一种子、重建·全季、重建·扣除目标域四列对照。
    深度族只重跑了一粒种子时，正典深度也限到这粒种子，口径之间才是同种子配对。"""
    A = _load('ana65', '65_paper_analysis.py')
    canon_s = pd.read_excel(os.path.join(_OUT, '62_crossover_engine.xlsx'), sheet_name='curves')
    canon_d = pd.read_excel(os.path.join(_OUT, '64_deep_transfer_server.xlsx'), sheet_name='curves')
    canon_s = canon_s[canon_s.benchmark == 'apple']
    canon_d = canon_d[canon_d.benchmark == 'apple']
    shards = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(SHARD_DIR, '*.csv')))],
                       ignore_index=True)
    simple_methods = ['zero_shot', 'coral', 'sbc', 'target_only', 'model_update']
    s_nt = shards[(shards.variant == 'no_target') & shards.method.isin(simple_methods)]
    d_all = shards[(shards.variant == 'all') & shards.method.isin(A.DEEP)]
    d_nt = shards[(shards.variant == 'no_target') & shards.method.isin(A.DEEP)]
    deep_seeds = sorted(set(d_nt.seed)) if len(d_nt) else sorted(set(canon_d.seed))

    def svd(cur: pd.DataFrame) -> pd.DataFrame:
        a = A.task_median(cur)
        out = []
        for nc in A.NCAL:
            sub = a[a['n_cal'] == nc]
            s = sub[sub['method'].isin(A.SIMPLE)].groupby('task_id')['rmsep'].min()
            d = sub[sub['method'].isin(A.DEEP)].groupby('task_id')['rmsep'].min()
            com = s.index.intersection(d.index)
            s, d = s[com].to_numpy(), d[com].to_numpy()
            p = float(A.wilcoxon(s, d).pvalue)
            out.append({'n_cal': nc, 'n_task': len(com), 'simple_med': round(float(np.median(s)), 3),
                        'deep_med': round(float(np.median(d)), 3),
                        'simple_win_frac': round(float(np.mean(s < d)), 2),
                        'cliffs_delta': round(A.cliffs_delta(d, s), 2), 'wilcoxon_p': p})
        return pd.DataFrame(out)

    configs = {'正典（稿件所用，深度 5 种子）': pd.concat([canon_s, canon_d])}
    cd = canon_d[canon_d.seed.isin(deep_seeds)]
    if len(deep_seeds) < len(SEEDS):
        configs[f'正典·深度限种子 {deep_seeds}'] = pd.concat([canon_s, cd])
    if len(d_all):
        configs['重建·全季（简单族取正典）'] = pd.concat([canon_s, d_all])
    if len(s_nt) and len(d_nt):
        configs['重建·扣除目标域'] = pd.concat([s_nt, d_nt])
    sheets: dict[str, pd.DataFrame] = {}
    for i, (name, cur) in enumerate(configs.items()):
        t = svd(cur)
        t.insert(0, '口径', name)
        sheets[f'表{chr(97 + i)}'] = t
    # 简单族各方法的中位 RMSEP 位移（扣除目标域 − 正典），按任务配对
    if len(s_nt):
        a0 = A.task_median(canon_s).set_index(['task_id', 'method', 'n_cal'])['rmsep']
        a1 = A.task_median(s_nt).set_index(['task_id', 'method', 'n_cal'])['rmsep']
        k = a0.index.intersection(a1.index)
        d = (a1[k] - a0[k]).rename('diff').reset_index()
        sheets['表e：简单族位移'] = (d.groupby(['method', 'n_cal'])['diff']
                                   .agg(['median', 'mean', 'count']).round(4).reset_index())
    # 深度族：同一批对照任务上，重建·全季对正典（同种子）是跨平台差（正典由 95 号以 --device cpu 在另一台服务器上跑、此处在本机 CPU 上跑），
    # 扣除目标域对重建·全季（都在 CPU 上）才是年份统计量本身的影响；两者并排，后者要与前者比大小
    if len(d_all):
        keys = ['task_id', 'method', 'n_cal', 'seed']
        m = d_all.merge(cd, on=keys, suffixes=('', '_c')).merge(d_nt, on=keys, suffixes=('', '_nt'))
        m['平台差'] = m['rmsep'] - m['rmsep_c']
        m['口径差'] = m['rmsep_nt'] - m['rmsep']
        rows_f = []
        for meth, g in [('全部深度方法', m), *list(m.groupby('method'))]:
            rows_f.append({'方法': meth, '配对行数': len(g), '任务数': g['task_id'].nunique(),
                           '平台差中位': round(float(g['平台差'].median()), 4),
                           '|平台差|中位': round(float(g['平台差'].abs().median()), 4),
                           '口径差中位': round(float(g['口径差'].median()), 4),
                           '|口径差|中位': round(float(g['口径差'].abs().median()), 4)})
        sheets['表f：深度对照任务上平台差与口径差'] = pd.DataFrame(rows_f)
    # 深度族位移（扣除目标域 − 正典同种子），72 个任务；含跨平台差，大小以表f的平台差为参照
    if len(d_nt):
        b0 = A.task_median(cd).set_index(['task_id', 'method', 'n_cal'])['rmsep']
        b1 = A.task_median(d_nt).set_index(['task_id', 'method', 'n_cal'])['rmsep']
        k = b0.index.intersection(b1.index)
        d = (b1[k] - b0[k]).rename('diff').reset_index()
        sheets['表g：深度族位移（含平台差）'] = (d.groupby(['method', 'n_cal'])['diff']
                                           .agg(['median', 'mean', 'count']).round(4).reset_index())
    # 处方遗憾（73 号口径，只在苹果任务上）：固定处方（n=0 CORAL，n>0 SBC）与默认深度模型
    # （n=0 cnn_zeroshot，n>0 cnn_finetune）相对每任务最优方法的遗憾 RMSEP/RMSEP(oracle)−1；
    # 候选集为该预算下有曲线的全部方法。正典与扣除目标域都取深度同一粒种子，两者才可比
    universe = simple_methods + list(A.DEEP)
    presc: dict[str, Callable[[int], str]] = {
        '固定处方（CORAL/SBC）': lambda nc: 'coral' if nc == 0 else 'sbc',
        '默认深度（CNN）': lambda nc: 'cnn_zeroshot' if nc == 0 else 'cnn_finetune'}
    rows_h = []
    for name, cur in ((f'正典·深度限种子 {deep_seeds}', pd.concat([canon_s, cd])),
                      ('重建·扣除目标域', pd.concat([s_nt, d_nt]))):
        if name == '重建·扣除目标域' and not (len(s_nt) and len(d_nt)):
            continue
        tm = A.task_median(cur[cur.method.isin(universe) & cur.n_cal.isin(A.NCAL)])
        tm = tm[tm.task_id.isin(set(d_nt.task_id)) if len(d_nt) else tm.task_id.notna()]
        for nc in A.NCAL:
            w = tm[tm.n_cal == nc].pivot_table(index='task_id', columns='method', values='rmsep')
            best = w.min(axis=1)
            for pname, pick in presc.items():
                reg = (w[pick(nc)] / best - 1).dropna()
                rows_h.append({'口径': name, 'n_cal': nc, '处方': pname, 'n_task': len(reg),
                               '遗憾中位': round(float(reg.median()), 4),
                               '在oracle 5%以内占比': round(float((reg <= 0.05).mean()), 3)})
            rows_h.append({'口径': name, 'n_cal': nc, '处方': 'oracle 构成',
                           'n_task': len(w), '遗憾中位': float('nan'), '在oracle 5%以内占比': float('nan'),
                           'oracle 构成': '; '.join(f'{k} {v}' for k, v in
                                                    w.idxmin(axis=1).value_counts().items())})
    sheets['表h：苹果任务的处方遗憾'] = pd.DataFrame(rows_h)
    # 头条处方遗憾（73 号口径：五基准、fig4 候选集、留一基准）：苹果任务整体换成扣除目标域口径后重算。
    # 深度族须补齐 5 粒种子才与正典同口径；正典那一列应逐位复现 73 号的 always_simple 2.78%
    if len(s_nt) and len(d_nt) and len(deep_seeds) == len(SEEDS):
        L = _load('lobo73', '73_heuristic_map_lobo.py')
        cl = pd.read_excel(os.path.join(_OUT, '62_crossover_engine.xlsx'), sheet_name='curves')
        dp = pd.read_excel(os.path.join(_OUT, '64_deep_transfer_server.xlsx'), sheet_name='curves')
        sig = pd.read_excel(os.path.join(_OUT, '63_crossover_analysis.xlsx'),
                            sheet_name='nstar_per_task')[['task_id', 'y_std_tgt']].drop_duplicates('task_id')

        class _Mute:  # 73 号的 _run_universe 要一个带 log 的对象，这里不需要它的日志
            def log(self, *a: Any, **k: Any) -> None:
                pass

        rows_i = []
        for name, allr in (('正典', pd.concat([cl, dp])),
                           ('苹果换扣除目标域', pd.concat([cl[cl.benchmark != 'apple'], dp[dp.benchmark != 'apple'],
                                                     s_nt, d_nt]))):
            allr = allr[allr.method.isin(L.UNIVERSE) & allr.n_cal.isin(L.NCAL)]
            tm = allr.groupby(['benchmark', 'shift_type', 'task_id', 'method', 'n_cal'],
                              as_index=False)['rmsep'].median().merge(sig, on='task_id', how='left')
            tm['nrmsep'] = tm['rmsep'] / tm['y_std_tgt']
            pt, *_ = L._run_universe(tm, 'fig4', _Mute())
            L.RNG = np.random.default_rng(20060515)  # type: ignore[attr-defined]  # 与 73 号 main 同序抽样，置信区间才逐位可比
            by_n = L._summarise(pt, ['universe', 'rule', 'n_cal'])
            L._summarise(pt, ['universe', 'rule', 'benchmark', 'n_cal'])
            sm = L._summarise(pt, ['universe', 'rule'])
            # 表 regret 的六行，外加 C 规则与基准等权的 B 规则
            for r in sm[sm.rule.isin(['always_simple', 'always_model_update', 'always_target_only',
                                      'B_shift_unknown', 'A_shift_known', 'default_deep',
                                      'C_best_fixed_prescription', 'B_shift_unknown_bal'])].itertuples(index=False):
                per_n = by_n[by_n.rule == r.rule].set_index('n_cal')['median_regret']
                rows_i.append({'口径': name, '规则': r.rule, '单元数': r.n_task, '遗憾中位': r.median_regret,
                               'CI下界': r.ci95_lo, 'CI上界': r.ci95_hi, '10%以内占比': r.within10_frac,
                               **{f'n={nc}': per_n.get(nc, float('nan')) for nc in L.NCAL}})
        sheets['表i：五基准头条处方遗憾'] = pd.DataFrame(rows_i)
        # 基准级均衡（68 号口径：每基准的深度族均值减简单族均值 NRMSEP，基准等权 bootstrap），
        # 以及苹果的族均值 NRMSEP 对比（68 号 _family_row）；正典列应逐位复现 68 号
        H = _load('hier68', '68_normalized_hierarchical_reanalysis.py')
        rows_j = []
        for name, allr in (('正典', pd.concat([pd.read_excel(os.path.join(_OUT, '62_crossover_engine.xlsx')),
                                              pd.read_excel(os.path.join(_OUT, '64_deep_transfer_server.xlsx'))])),
                           ('苹果换扣除目标域', pd.concat([cl[cl.benchmark != 'apple'], dp[dp.benchmark != 'apple'],
                                                     s_nt, d_nt]))):
            tm = allr.groupby(['benchmark', 'shift_type', 'task_id', 'method', 'n_cal'],
                              as_index=False)['rmsep'].median().merge(sig, on='task_id', how='left')
            tm['nrmsep'] = tm['rmsep'] / tm['y_std_tgt']
            H.RNG = np.random.default_rng(20060515)  # type: ignore[attr-defined]  # 与 68 号 main 同序（先 0 后 20 标签）
            for nc in (0, 20):
                bdf, summ = H._benchmark_level(tm, nc)
                apple = bdf.loc[bdf.benchmark == 'apple', 'mean_diff_deep_minus_simple']
                rows_j.append({'口径': name, 'n_cal': nc, '量': '基准级均衡均值差（深度−简单）',
                               '值': summ['benchmark_balanced_mean_diff'], 'CI下界': summ['boot95_lo'],
                               'CI上界': summ['boot95_hi'], '简单族更好的基准数': summ['n_benchmarks_simple_better'],
                               '苹果基准均值差': float(apple.iloc[0]) if len(apple) else float('nan')})
            for nc in A.NCAL:
                fr = H._family_row(tm, 'origin_year_instrument', nc)
                if fr is not None:
                    rows_j.append({'口径': name, 'n_cal': nc, '量': '苹果族均值 NRMSEP δ',
                                   '值': fr['cliffs_delta'], 'CI下界': float('nan'), 'CI上界': float('nan'),
                                   '简单族更好的基准数': float('nan'), '苹果基准均值差': float('nan'),
                                   'P': fr['wilcoxon_p'], '简单族胜率': fr['simple_win_frac']})
        sheets['表j：基准级均衡与苹果族均值'] = pd.DataFrame(rows_j)
        # 免选择比较（86 号口径，即稿件表 fair：非配对 Cliff's δ，代表性配对 CORAL vs DeepCORAL 与族均值）；
        # 正典那几行应逐位复现表 fair 的 Origin/year 行
        R = _load('fair86', '86_selection_free_effect_sizes.py')
        rows_k = []
        for name, cur in (('正典', pd.concat([canon_s, canon_d])), ('重建·扣除目标域', pd.concat([s_nt, d_nt]))):
            med = cur.groupby(['task_id', 'shift_type', 'method', 'n_cal'], as_index=False)['rmsep'].median()
            med = med.merge(sig, on='task_id', how='left')
            med['nrmsep'] = med['rmsep'] / med['y_std_tgt']
            t = R.rows(med)
            t.insert(0, '口径', name)
            rows_k.append(t)
        sheets['表k：免选择比较（表fair口径）'] = pd.concat(rows_k, ignore_index=True)
    # 管线自检：全部光谱口径重建的季节统计量下，简单族与正典 62 号的同一 (任务, 方法, 预算, 种子, 重复) 相比
    s_all = shards[(shards.variant == 'all') & shards.method.isin(simple_methods)]
    if len(s_all):
        keys = ['task_id', 'method', 'n_cal', 'seed', 'rep']
        m = s_all.merge(canon_s[[*keys, 'rmsep']], on=keys, suffixes=('', '_c'))
        sheets['表l：全季重建对正典（简单族）'] = pd.DataFrame([{
            '任务数': s_all['task_id'].nunique(), '行数': len(s_all), '配上正典的行数': len(m),
            '最大绝对差': float((m['rmsep'] - m['rmsep_c']).abs().max())}])
    path =os.path.join(_OUT, f'{FILE_STEM}.xlsx')
    with pd.ExcelWriter(path) as w:
        for k, v in sheets.items():
            v.to_excel(w, sheet_name=k[:31], index=False)
    for k, v in sheets.items():
        print(f'== {k}\n{v.to_string(index=False)}')
    print(f'已写 {path}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', choices=['simple', 'deep'])
    ap.add_argument('--variant', choices=['all', 'no_target'])
    ap.add_argument('--seeds', default=','.join(map(str, SEEDS)))
    ap.add_argument('--rep', type=int, default=None, help='默认同正典：简单族 5、深度族 1')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--n_shards', type=int, default=1)
    ap.add_argument('--device', default='cpu', help='深度族的训练设备（cpu 或 cuda）；简单族忽略')
    ap.add_argument('--tag', default='', help='分片文件名附加段（同一口径按种子分批跑时区分文件，如 _s2023）')
    ap.add_argument('--analyze', action='store_true')
    args = ap.parse_args()
    if args.analyze:
        analyze()
        return
    if args.rep is None:
        args.rep = 5 if args.family == 'simple' else 1
    run(args)


if __name__ == '__main__':
    main()
