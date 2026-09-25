"""
58_cluster_robust_and_strata.py：审稿意见驱动的稳健性重分析（不新增模型运行）

针对 Round 2 审稿两条方法学意见：
  · CRITICAL 4：602 场景来自 6 域的非空组合，样本/目标集合大量复用，
    按场景聚类的 P 值可能过乐观。→ 改按**唯一目标集合**（62 个）聚类，
    对配对 RMSE 差值做 cluster bootstrap，给出更保守的 95% CI 与显著性。
  · MAJOR 1：602 场景中多数为跨年（含跨仪器），不宜统称"跨产地"。
    → 按"同年跨产地 / 跨年(含跨仪器)"分层报告核心方法，验证否定性结论是否跨层稳健。

仅重分析 50 号既有结果（表g：每场景跨种子均值），不重跑任何模型。

运行方式:
    python 02code/58_cluster_robust_and_strata.py

输出:
    04outputs/58_cluster_robust_and_strata.xlsx / .md
"""

import os
import numpy as np
import pandas as pd

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, '..'))
_OUT = os.path.join(_BASE, '04outputs')


def parse_scenario(sc):
    """从 '2018_山东→2018_新疆' 解析目标集合与是否同年。"""
    src, tgt = sc.split('→')
    src_years = set(x.split('_')[0] for x in src.split('+'))
    tgt_years = set(x.split('_')[0] for x in tgt.split('+'))
    within = (src_years == tgt_years and len(src_years) == 1)
    return tgt, within


def cluster_bootstrap_ci(values, clusters, n_boot=10000, seed=12345):
    """按 cluster 重采样的配对差值 95% CI（聚类单位 = 唯一目标集合）。"""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'v': values, 'c': clusters}).dropna()
    uniq = df['c'].unique()
    by_c = {c: df[df['c'] == c]['v'].values for c in uniq}
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        vals = np.concatenate([by_c[c] for c in pick])
        means[b] = vals.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(df['v'].mean())


def main():
    g = pd.read_excel(os.path.join(_OUT, '50_formal_multiseed_benchmark.xlsx'),
                      sheet_name='表g：每场景跨种子均值')
    g['目标集合'] = g['场景'].map(lambda s: parse_scenario(s)[0])
    g['同年'] = g['场景'].map(lambda s: parse_scenario(s)[1])
    # 案例研究的聚合量一律取七个未校正方法都有结果的共同场景（补充材料 S3）：
    # 各方法在各自跑出的场景上算，难场景没跑完的方法会显得更好，跨方法也不可比。
    bare7 = ['PLSR', 'SVR', 'CNN', 'CNN+MMD', 'CNN+BL', 'Phys', 'Phys+BL']
    wide = g.pivot(index='场景', columns='方法_校正', values='RMSE')
    g = g[g['场景'].isin(wide[bare7].dropna().index)].copy()

    rep = ['# 稳健性重分析（cluster-robust + 分层）\n',
           f'唯一目标集合数 = {g["目标集合"].nunique()}；'
           f'同年跨产地场景 = {g[g["同年"]]["场景"].nunique()}，'
           f'跨年(含跨仪器)场景 = {g[~g["同年"]]["场景"].nunique()}。\n']

    # ── 1. 按唯一目标集合聚类的配对差值 95% CI ──────────────────────
    def paired_diff(a_name, b_name):
        a = g[g['方法_校正'] == a_name].set_index('场景')
        b = g[g['方法_校正'] == b_name].set_index('场景')
        common = a.index.intersection(b.index)
        d = a.loc[common, 'RMSE'].values - b.loc[common, 'RMSE'].values
        clusters = a.loc[common, '目标集合'].values
        return d, clusters

    hyps = [('H1', 'Phys+BL', 'CNN', '零标签下 Phys+BL 优于 CNN'),
            ('H2', 'Phys+BL + SB', 'PLSR + SB', '校正后 Phys+BL 劣于 PLSR'),
            ('H3', 'Phys+BL + SB', 'CNN + SB', '校正后 Phys+BL 劣于 CNN')]
    rep.append('## 1. 按唯一目标集合（n=62）聚类的配对 RMSE 差值 95% CI\n')
    rep.append('| 假设 | 含义 | 差值(A−B) | 目标集合聚类 95% CI | 显著 |')
    rep.append('|---|---|---|---|---|')
    rows = []
    for hid, a, b, desc in hyps:
        d, cl = paired_diff(a, b)
        lo, hi, mean = cluster_bootstrap_ci(d, cl)
        sig = '是' if (lo > 0 or hi < 0) else '否'
        rep.append(f'| {hid} | {desc} | {mean:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {sig} |')
        rows.append(dict(假设=hid, 含义=desc, 差值=mean, CI下界=lo, CI上界=hi, 目标集合聚类显著=sig))
    # 结论句由实算结果生成：硬编码「均不变」会在数据换版后变成假陈述。
    n_sig = sum(1 for r in rows if r['目标集合聚类显著'] == '是')
    if n_sig == len(rows):
        verdict1 = '三个核心假设的方向与显著性均不变。'
    else:
        lost = '、'.join(r['假设'] for r in rows if r['目标集合聚类显著'] != '是')
        verdict1 = (f'{n_sig}/{len(rows)} 个核心假设的方向与显著性不变；'
                    f'{lost} 的 95% CI 在该聚类单位下跨 0。')
    rep.append(f'\n> 把聚类单位从 602 场景放粗到 62 个唯一目标集合（远更保守）后，{verdict1}\n')

    # ── 2. 分层：同年跨产地 vs 跨年 ─────────────────────────────────
    rep.append('## 2. 分层稳健性（同年跨产地 vs 跨年含跨仪器）\n')
    rep.append('| 分层 | 场景数 | 方法 | 均值RMSE | 中位RMSE |')
    rep.append('|---|---|---|---|---|')
    strata_rows = []
    for lab, sub in [('同年跨产地', g[g['同年']]), ('跨年(含跨仪器)', g[~g['同年']])]:
        n_sc = sub['场景'].nunique()
        for m in ['PLSR', 'CNN', 'Phys+BL', 'Phys+BL + SB', 'PLSR + SB']:
            r = sub[sub['方法_校正'] == m]['RMSE']
            if len(r):
                rep.append(f'| {lab} | {n_sc} | {m} | {r.mean():.3f} | {r.median():.3f} |')
                strata_rows.append(dict(分层=lab, 场景数=n_sc, 方法=m,
                                        均值RMSE=r.mean(), 中位RMSE=r.median()))
    # 同上：两层的排序判定与难度对比都现算，不写死。
    def _m(lab, meth):
        hit = [r['均值RMSE'] for r in strata_rows if r['分层'] == lab and r['方法'] == meth]
        return hit[0] if hit else float('nan')

    ok_both = all(
        _m(lab, 'Phys+BL') < min(_m(lab, 'PLSR'), _m(lab, 'CNN'))
        and _m(lab, 'PLSR + SB') < _m(lab, 'Phys+BL + SB')
        for lab in ('同年跨产地', '跨年(含跨仪器)')
    )
    head = ('两层中核心否定性结论均成立：未校正时 Phys+BL 最优，校正后 PLSR+SB 优于 Phys+BL+SB。'
            if ok_both else
            '两层的排序并不一致，逐层数字见上表。')
    rep.append(f"\n> {head}绝对难度跨年层更高"
               f"（PLSR 均值 {_m('同年跨产地', 'PLSR'):.2f} → {_m('跨年(含跨仪器)', 'PLSR'):.2f}）。\n")

    with pd.ExcelWriter(os.path.join(_OUT, '58_cluster_robust_and_strata.xlsx')) as w:
        pd.DataFrame(rows).to_excel(w, sheet_name='目标集合聚类', index=False)
        pd.DataFrame(strata_rows).to_excel(w, sheet_name='分层', index=False)
    with open(os.path.join(_OUT, '58_cluster_robust_and_strata.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(rep) + '\n')
    print('\n'.join(rep))
    print('\n已保存: 04outputs/58_cluster_robust_and_strata.xlsx / .md')


if __name__ == '__main__':
    main()
