"""
72_convergence_survivorship.py: 收敛率不等 → "幸存者比较"敏感性检验（返修 T1）

【为什么必须做这个实验】
表 6（tab:apple5）里各方法的场景覆盖数**不等**：PLSR/SVR 602，CNN 572，CNN+MMD 568，
Phys+BL 568。缺的 30~34 个场景是深度模型**没跑出来**的（BatchNorm 末批为 1、数值发散），
它们被整体剔除后才算的均值。审稿人的质疑很直接：

    "未收敛的场景恰恰可能是最难的场景。把它们从深度/物理方法里删掉、却留在 PLSR/SVR 里，
     '未校正 Phys+BL 最优' 就成了幸存者比较。"

稿件 §2.6 只声明了收敛场景数，§3.5 把配对比较限定在双方均收敛的场景，
但**从未报告"把缺失场景按最坏情形补齐后排名是否翻转"**。本脚本补这一步。

【检验设计（预先写定）】
  四种支撑集口径，同一批原始结果，只改"缺失怎么处理"：
    S0 已发表口径   各方法在**各自可用场景**上聚合（= 表 6 现状，用于复现锚点）
    S1 共同支撑     只留 5 个方法**全部**跑出结果的场景（最保守的等支撑比较）
    S2 最坏观测补齐  缺失格 ← 该场景中**其它方法的最大 RMSE**（"跑不出来至少和最差的一样差"）
    S3 发散阈值补齐  缺失格 ← 10 °Brix（稿件 §3.6 自己对"发散"的定义值，等于判其失败）
    S4 秩口径       场景内对方法排名，缺失judged为**最差名次** → 比较平均名次（完全非参、无需编造数值）

  两条核心排序（就是被质疑的那两条）：
    ① 未校正：Phys+BL 是否仍为 RMSE 最低
    ② SB 校正后：PLSR+SB 是否仍低于 Phys+BL+SB
  配对检验：共同支撑集上的双侧 Wilcoxon（场景为单位，跨 5 种子先取均值）

  判定：四种口径下两条排序都不翻转 → 结论对幸存者偏差稳健；
        任一口径翻转 → 必须在稿中如实报告翻转条件。

运行方式:
    cd <项目根目录>
    <venv>/bin/python 02code/72_convergence_survivorship.py

输出文件:
    04outputs/72_convergence_survivorship.xlsx
    05logs/72_convergence_survivorship_<时间戳>.log
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, '..'))
_OUT = os.path.join(_BASE, '04outputs')
_LOG = os.path.join(_BASE, '05logs')
FILE_STEM = '72_convergence_survivorship'

FORMAL_SEEDS = [20060515, 20041210, 19810915, 2023, 2024]
METHODS = ['PLSR', 'SVR', 'CNN', 'CNN+MMD', 'Phys+BL']
DIVERGENCE_RMSE = 10.0    # 与 §3.6 对"发散"的定义一致


def make_logger() -> Callable[..., None]:
    os.makedirs(_LOG, exist_ok=True)
    path = os.path.join(_LOG, f'{FILE_STEM}_{datetime.now():%y-%m-%d_%H%M%S}.log')

    def log(msg: object = '') -> None:
        print(msg, flush=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(str(msg) + '\n')
    return log


def per_scenario_matrix(raw: pd.DataFrame, correction: str) -> pd.DataFrame:
    """场景 × 方法 的 RMSE 矩阵（跨种子先取均值）。缺失 = 该方法在该场景没有结果。"""
    sub = raw[raw['校正'] == correction]
    per = sub.groupby(['场景', '方法'], as_index=False)['RMSE'].mean()
    return per.pivot(index='场景', columns='方法', values='RMSE').reindex(columns=METHODS)


def summarise(mat: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for m in METHODS:
        col = mat[m].dropna()
        rows.append({'口径': label, '方法': m, 'n场景': len(col),
                     'RMSE_mean': float(col.mean()), 'RMSE_median': float(col.median())})
    return pd.DataFrame(rows)


def orderings(bare: pd.DataFrame, sb: pd.DataFrame, label: str,
              log: Callable[..., None]) -> dict[str, Any]:
    """两条核心排序在该口径下是否成立。均值与中位数分开判——稿中两处都用到。"""
    out: dict[str, Any] = {'口径': label}
    for stat_name, fn in (('mean', np.nanmean), ('median', np.nanmedian)):
        b = {m: float(fn(bare[m].to_numpy(dtype=np.float64))) for m in METHODS}
        s = {m: float(fn(sb[m].to_numpy(dtype=np.float64))) for m in METHODS}
        best_bare = min(b, key=lambda k: b[k])
        out[f'未校正最优_{stat_name}'] = best_bare
        out[f'Phys+BL未校正_{stat_name}'] = b['Phys+BL']
        out[f'CNN未校正_{stat_name}'] = b['CNN']
        out[f'排序①成立_{stat_name}'] = bool(best_bare == 'Phys+BL')
        out[f'PLSR+SB_{stat_name}'] = s['PLSR']
        out[f'Phys+BL+SB_{stat_name}'] = s['Phys+BL']
        out[f'排序②成立_{stat_name}'] = bool(s['PLSR'] < s['Phys+BL'])
        log(f'  [{stat_name}] 未校正最优 = {best_bare:8s} '
            f'(Phys+BL {b["Phys+BL"]:.3f} / CNN {b["CNN"]:.3f} / PLSR {b["PLSR"]:.3f})  '
            f'排序①{"✅" if best_bare == "Phys+BL" else "🔴翻转"}')
        log(f'  [{stat_name}] 校正后 PLSR+SB {s["PLSR"]:.3f} vs Phys+BL+SB {s["Phys+BL"]:.3f}  '
            f'排序②{"✅" if s["PLSR"] < s["Phys+BL"] else "🔴翻转"}')
    return out


def main() -> None:
    log = make_logger()
    log('=' * 88)
    log('72 - 收敛率不等的幸存者偏差敏感性（返修 T1）')
    log('=' * 88)

    raw = pd.read_excel(os.path.join(_OUT, '50_formal_multiseed_benchmark.xlsx'),
                        sheet_name='表h：全部原始结果')
    raw = raw[raw['种子'].isin(FORMAL_SEEDS)]

    bare0 = per_scenario_matrix(raw, '裸')
    sb0 = per_scenario_matrix(raw, '+SB')
    all_scen = bare0.index
    log(f'\n总场景数 {len(all_scen)}')

    # ── 1. 缺失结构诊断：缺的是不是"更难的"场景 ────────────────────────
    log('\n' + '=' * 88)
    log('诊断一：缺失是否与难度相关（若缺失场景系统性更难 → 幸存者偏差是实质性的）')
    log('=' * 88)
    seedcnt = (raw[raw['校正'] == '裸'].groupby(['方法', '场景'])['种子'].nunique()
               .rename('种子数').reset_index())
    diag_rows = []
    log(f'{"方法":9s} {"覆盖场景":>7s} {"缺失":>5s} {"缺失场景 PLSR 中位 RMSE":>22s} '
        f'{"其余场景":>9s} {"Mann-Whitney P":>15s}')
    for m in METHODS:
        present = bare0[m].dropna().index
        missing = all_scen.difference(present)
        ref = bare0['PLSR']
        a = ref.loc[missing].dropna().to_numpy(dtype=np.float64)
        b = ref.loc[present].dropna().to_numpy(dtype=np.float64)
        if len(a) >= 3:
            p = float(stats.mannwhitneyu(a, b, alternative='two-sided').pvalue)
            med_a, med_b = float(np.median(a)), float(np.median(b))
        else:
            p, med_a, med_b = float('nan'), float('nan'), float(np.median(b))
        log(f'{m:9s} {len(present):7d} {len(missing):5d} {med_a:22.3f} {med_b:9.3f} {p:15.3g}')
        n_partial = int(((seedcnt['方法'] == m) & (seedcnt['种子数'] < 5)).sum())
        diag_rows.append({'方法': m, '覆盖场景': len(present), '缺失场景': len(missing),
                          '缺失场景_PLSR中位RMSE': med_a, '其余场景_PLSR中位RMSE': med_b,
                          'MannWhitney_P': p, '种子不全的场景数': n_partial})
    diag = pd.DataFrame(diag_rows)
    log('\n种子不全（<5 种子）的场景数：'
        + ', '.join(f'{r["方法"]} {r["种子不全的场景数"]}' for _, r in diag.iterrows()))

    # ── 2. 四种支撑集口径 ──────────────────────────────────────────────
    common = bare0.dropna().index.intersection(sb0.dropna().index)
    log(f'\n共同支撑集（5 方法全部跑出）场景数：{len(common)}')

    variants: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    variants['S0 已发表口径'] = (bare0, sb0)
    variants['S1 共同支撑'] = (bare0.loc[common], sb0.loc[common])

    def impute_worst_observed(mat: pd.DataFrame) -> pd.DataFrame:
        out = mat.copy()
        worst = mat.max(axis=1)
        for m in METHODS:
            out[m] = out[m].fillna(worst)
        return out

    def impute_fixed(mat: pd.DataFrame, value: float) -> pd.DataFrame:
        return mat.fillna(value)

    variants['S2 最坏观测补齐'] = (impute_worst_observed(bare0), impute_worst_observed(sb0))
    variants['S3 发散阈值补齐'] = (impute_fixed(bare0, DIVERGENCE_RMSE),
                              impute_fixed(sb0, DIVERGENCE_RMSE))

    summ = []
    ords = []
    for label, (b, s) in variants.items():
        log('\n' + '-' * 88)
        log(f'{label}   (未校正 n={len(b)} 场景)')
        log('-' * 88)
        summ.append(summarise(b, label + '·未校正'))
        summ.append(summarise(s, label + '·+SB'))
        ords.append(orderings(b, s, label, log))
    summary = pd.concat(summ, ignore_index=True)
    order_df = pd.DataFrame(ords)

    # ── 3. S4 秩口径（缺失 = 场景内最差名次，不编造任何数值）───────────
    log('\n' + '-' * 88)
    log('S4 秩口径：场景内给方法排名，缺失记最差名次（完全非参，不需要补任何数值）')
    log('-' * 88)
    rank_rows = []
    for tag, mat in (('未校正', bare0), ('+SB', sb0)):
        r = mat.rank(axis=1, method='average', na_option='bottom')
        means = {m: float(r[m].mean()) for m in METHODS}
        best = min(means, key=lambda k: means[k])
        log(f'  [{tag}] 平均名次（1=最好）: '
            + '  '.join(f'{m} {means[m]:.2f}' for m in METHODS)
            + f'   → 最优 = {best}')
        for m in METHODS:
            rank_rows.append({'校正': tag, '方法': m, '平均名次': means[m]})
    ranks = pd.DataFrame(rank_rows)
    rank_bare = ranks[ranks['校正'] == '未校正'].set_index('方法')['平均名次']
    rank_sb = ranks[ranks['校正'] == '+SB'].set_index('方法')['平均名次']
    s4_ord1 = bool(rank_bare.idxmin() == 'Phys+BL')
    s4_ord2 = bool(rank_sb['PLSR'] < rank_sb['Phys+BL'])
    log(f'  排序①（未校正 Phys+BL 最优）{"✅" if s4_ord1 else "🔴翻转"}  |  '
        f'排序②（PLSR+SB < Phys+BL+SB）{"✅" if s4_ord2 else "🔴翻转"}')

    # ── 4. 共同支撑集上的配对检验 ──────────────────────────────────────
    log('\n' + '=' * 88)
    log('诊断二：共同支撑集上的配对 Wilcoxon（场景为单位）')
    log('=' * 88)
    pair_rows = []
    for tag, mat, a, b_ in (('未校正 Phys+BL vs CNN', bare0.loc[common], 'Phys+BL', 'CNN'),
                            ('+SB PLSR vs Phys+BL', sb0.loc[common], 'PLSR', 'Phys+BL')):
        x = mat[a].to_numpy(dtype=np.float64)
        y = mat[b_].to_numpy(dtype=np.float64)
        w = stats.wilcoxon(x, y)
        d = float(np.median(x - y))
        win = float(np.mean(x < y) * 100)
        log(f'  {tag:28s} 中位差 {d:+.3f} °Brix | {a} 更优占比 {win:.1f}% | '
            f'Wilcoxon P = {w.pvalue:.3g} | n = {len(x)}')
        pair_rows.append({'对比': tag, '中位差': d, f'{a}更优占比%': win,
                          'Wilcoxon_P': float(w.pvalue), 'n': len(x)})
    pairs = pd.DataFrame(pair_rows)

    # ── 5. 结论 ────────────────────────────────────────────────────────
    log('\n' + '=' * 88)
    log('结论')
    log('=' * 88)
    cols1 = [c for c in order_df.columns if c.startswith('排序①成立')]
    cols2 = [c for c in order_df.columns if c.startswith('排序②成立')]
    ok1 = bool(order_df[cols1].to_numpy().all()) and s4_ord1
    ok2 = bool(order_df[cols2].to_numpy().all()) and s4_ord2
    # 措辞纪律（外部审计 E3）：S0–S3 才有均值/中位数两种统计量；S4 只有平均名次。
    # 不得笼统写成「S0–S4 × 均值与中位数全不翻转」——S4 根本没有均值/中位数这一维。
    log(f'  排序①（未校正 Phys+BL 最优）：S0–S3 的均值与中位数两种统计量下全部成立，'
        f'且 S4 的平均名次口径下也成立 → {ok1}')
    log(f'  排序②（SB 校正后 PLSR 反超）：同上口径 → {ok2}')
    if ok1 and ok2:
        verdict = '✅ 两条核心排序对幸存者偏差稳健（含最坏情形补齐与秩口径），可如实写入稿中'
    elif ok2 and not ok1:
        verdict = ('⚠ 排序②稳健，但排序①（未校正 Phys+BL 最优）在某些口径下翻转 → '
                   '须在稿中写明该结论条件于收敛场景')
    else:
        verdict = '🔴 核心排序在保守口径下翻转 → 必须在稿中如实报告翻转条件，不得只报 S0'
    log(f'\n  {verdict}')

    os.makedirs(_OUT, exist_ok=True)
    out = os.path.join(_OUT, f'{FILE_STEM}.xlsx')
    with pd.ExcelWriter(out) as w:
        diag.to_excel(w, sheet_name='表a：缺失结构诊断', index=False)
        summary.to_excel(w, sheet_name='表b：四口径汇总', index=False)
        order_df.to_excel(w, sheet_name='表c：核心排序是否翻转', index=False)
        ranks.to_excel(w, sheet_name='表d：S4秩口径', index=False)
        pairs.to_excel(w, sheet_name='表e：共同支撑配对检验', index=False)
        pd.DataFrame([{'判定': verdict, '排序①全稳': ok1, '排序②全稳': ok2,
                       '共同支撑场景数': len(common)}]).to_excel(
            w, sheet_name='表f：结论', index=False)
    log(f'\n已保存: {out}')


if __name__ == '__main__':
    main()
