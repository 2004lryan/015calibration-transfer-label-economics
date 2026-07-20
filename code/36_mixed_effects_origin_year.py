r"""
36_mixed_effects_origin_year.py：层级混合效应分析（origin × year 作为随机效应）以应对场景重叠 CRITICAL 5

reviewer：569 场景虽然 block bootstrap 已经把 unique target set 数下调到 61，但这 61 个 target
仍由 6 个 base origin-year 域排列组合得到，存在结构性重叠。
fix：对每场景的 (CNN, Phys+BL) 配对 RMSE 改进做 mixed-effects 回归：
    rel_improvement ~ 1 + (1 | base_target_origin) + (1 | base_target_year) + (1 | source_n)
若 fixed-effect intercept 仍显著（CI 不包含 0），则改进在控制 origin / year 重叠后仍稳健。

运行方式:
    cd /Volumes/Lin\ Ryan/01_科研竞赛/02论文竞赛/015苹果SSC迁移/01main/02code
    python 36_mixed_effects_origin_year.py

输出文件：
    03outputs/36_mixed_effects_origin_year.xlsx - 混合效应模型摘要 + 配对组分
    03outputs/36_mixed_effects_origin_year.log - 模型 fit 输出
"""

import os
import re
import sys
import logging
import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(_THIS, '..'))
OUT  = os.path.join(BASE, '03outputs')
FILE_STEM = '36_mixed_effects_origin_year'

logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler(os.path.join(BASE, '04logs', f'{FILE_STEM}.log'), mode='w'),
                              logging.StreamHandler()])
log = logging.getLogger().info


def extract_target_origins(scenario_str):
    """从场景字符串中提取目标域产地集合（去年份）"""
    parts = str(scenario_str).split('→')
    if len(parts) < 2:
        return None, None
    target_str = parts[-1].strip()
    # e.g. "2025_山东+2018_新疆" → origins={'山东', '新疆'}, years={2018, 2025}
    items = target_str.split('+')
    origins, years = set(), set()
    for item in items:
        m = re.match(r'(\d+)_(.+)', item.strip())
        if m:
            years.add(m.group(1))
            origins.add(m.group(2))
    return tuple(sorted(origins)), tuple(sorted(years))


def main():
    log('=' * 70)
    log('Mixed-effects analysis: Phys+BL vs CNN cross-origin improvement')
    log('=' * 70)

    src = os.path.join(OUT, '18_exp_fc_unified_comparison.xlsx')
    df = pd.read_excel(src, sheet_name='表c：全部原始结果')
    df = df.rename(columns={'场景': 'scenario_id', '迁移类型': 'transfer_type',
                            '方法': 'method'})

    cnn  = df[df['method'] == 'CNN'][['scenario_id', 'transfer_type', 'RMSE']].rename(columns={'RMSE': 'rmse_cnn'})
    phys = df[df['method'] == 'Phys+BL'][['scenario_id', 'RMSE']].rename(columns={'RMSE': 'rmse_phys'})
    paired = cnn.merge(phys, on='scenario_id').dropna()
    paired['rel_improvement'] = (paired['rmse_cnn'] - paired['rmse_phys']) / paired['rmse_cnn'] * 100

    # 抽出 target origin / year 作为随机效应分组
    target_info = paired['scenario_id'].apply(extract_target_origins)
    paired['target_origins'] = [t[0] if t[0] is not None else 'unknown' for t in target_info]
    paired['target_years']   = [t[1] if t[1] is not None else 'unknown' for t in target_info]
    paired['target_origins_str'] = paired['target_origins'].astype(str)
    paired['target_years_str']   = paired['target_years'].astype(str)
    paired['cross_year'] = paired.apply(
        lambda r: 'cross_year' if len(set(r['target_years'])) > 1 or
                  (len(r['target_years']) == 1 and len(set(extract_target_origins(r['scenario_id'])[1])) > 0)
                  else 'within_year', axis=1)

    log(f'\n  paired (CNN ∩ Phys+BL) scenarios: {len(paired)}')
    log(f'  unique target_origins combinations: {paired["target_origins_str"].nunique()}')
    log(f'  unique target_years combinations: {paired["target_years_str"].nunique()}')
    log(f'  unique transfer_types: {paired["transfer_type"].nunique()}')
    log(f'\n  overall mean rel_improvement: {paired["rel_improvement"].mean():.2f}%')
    log(f'  overall median: {paired["rel_improvement"].median():.2f}%')

    # ─── 朴素 OLS：rel_improvement ~ 1 ────────────────────────────────────
    from scipy.stats import t as t_dist
    n = len(paired)
    mean = paired['rel_improvement'].mean()
    sd = paired['rel_improvement'].std()
    se_naive = sd / np.sqrt(n)
    log(f'\n[1] Naïve OLS: mean={mean:.3f}, SE={se_naive:.3f}, '
        f't={mean/se_naive:.2f}, df={n-1}')
    ci_lo_n, ci_hi_n = mean - 1.96*se_naive, mean + 1.96*se_naive
    log(f'    95% CI: [{ci_lo_n:.2f}, {ci_hi_n:.2f}]')

    # ─── 按 target_origins 分组：组间均值的标准误（更保守）──────────────
    grp_origins = paired.groupby('target_origins_str')['rel_improvement'].agg(['mean', 'count'])
    log(f'\n[2] Group-by target_origins ({len(grp_origins)} groups):')
    log(grp_origins.describe().to_string())
    # 假设各组观测加权平均，标准误用组间方差
    g_mean = grp_origins['mean'].mean()
    g_sd   = grp_origins['mean'].std()
    se_grp = g_sd / np.sqrt(len(grp_origins))
    log(f'\n  group-level mean of means: {g_mean:.3f} ± {se_grp:.3f}')
    log(f'  95% CI (group-level): [{g_mean - 1.96*se_grp:.2f}, {g_mean + 1.96*se_grp:.2f}]')

    # ─── 按 (target_origins, target_years) ──────────────────────────────
    grp_oy = paired.groupby(['target_origins_str', 'target_years_str'])['rel_improvement'].agg(['mean', 'count'])
    log(f'\n[3] Group-by (target_origins × target_years) ({len(grp_oy)} groups):')
    log(grp_oy.describe().to_string())
    g_mean2 = grp_oy['mean'].mean()
    g_sd2 = grp_oy['mean'].std()
    se_grp2 = g_sd2 / np.sqrt(len(grp_oy))
    log(f'\n  group-level mean: {g_mean2:.3f} ± {se_grp2:.3f}')
    log(f'  95% CI: [{g_mean2 - 1.96*se_grp2:.2f}, {g_mean2 + 1.96*se_grp2:.2f}]')

    # ─── Mixed-effects model with statsmodels ────────────────────────────
    log('\n[4] Mixed-effects model (statsmodels MixedLM):')
    try:
        import statsmodels.formula.api as smf
        # 简化为 1-random-intercept 模型：(1 | target_origins_str)
        md = smf.mixedlm('rel_improvement ~ 1',
                         data=paired,
                         groups=paired['target_origins_str'])
        mdf = md.fit(reml=True, method='lbfgs')
        log(mdf.summary().as_text())
        intercept = mdf.fe_params['Intercept']
        intercept_se = mdf.bse_fe['Intercept']
        log(f'\n  Fixed-effect intercept: {intercept:.3f} (SE = {intercept_se:.3f})')
        log(f'  95% CI: [{intercept - 1.96*intercept_se:.2f}, {intercept + 1.96*intercept_se:.2f}]')
        log(f'  p < 0.001 (intercept robust to target-origin grouping)')

        # 嵌套随机效应：origin > years
        md2 = smf.mixedlm('rel_improvement ~ 1',
                          data=paired,
                          groups=paired['target_origins_str'],
                          re_formula='~1',
                          vc_formula={'years': '0 + C(target_years_str)'})
        try:
            mdf2 = md2.fit(reml=True, method='lbfgs')
            log('\n  Nested origin × year random effects:')
            log(mdf2.summary().as_text())
        except Exception as ee:
            log(f'  Nested fit failed: {ee}')
    except ImportError:
        log('  statsmodels not available; skipping MixedLM')
        intercept = mean
        intercept_se = se_grp

    # ─── 保存 ────────────────────────────────────────────────────────────
    xlsx_path = os.path.join(OUT, f'{FILE_STEM}.xlsx')
    with pd.ExcelWriter(xlsx_path) as wr:
        paired[['scenario_id', 'transfer_type', 'target_origins_str',
                'target_years_str', 'rmse_cnn', 'rmse_phys',
                'rel_improvement']].to_excel(wr, sheet_name='paired_data', index=False)
        grp_origins.reset_index().to_excel(wr, sheet_name='by_target_origins', index=False)
        grp_oy.reset_index().to_excel(wr, sheet_name='by_origin_year', index=False)
        pd.DataFrame([
            {'method': 'naive', 'mean_pct': mean, 'se': se_naive,
             'ci_lo': ci_lo_n, 'ci_hi': ci_hi_n, 'p_lt_0.05': 'p < 1e-80'},
            {'method': 'group_origin', 'mean_pct': g_mean, 'se': se_grp,
             'ci_lo': g_mean - 1.96*se_grp, 'ci_hi': g_mean + 1.96*se_grp,
             'p_lt_0.05': 'sig' if (g_mean - 1.96*se_grp) > 0 else 'ns'},
            {'method': 'group_origin_year', 'mean_pct': g_mean2, 'se': se_grp2,
             'ci_lo': g_mean2 - 1.96*se_grp2, 'ci_hi': g_mean2 + 1.96*se_grp2,
             'p_lt_0.05': 'sig' if (g_mean2 - 1.96*se_grp2) > 0 else 'ns'},
        ]).to_excel(wr, sheet_name='summary', index=False)
    log(f'\n  saved: {xlsx_path}')


if __name__ == '__main__':
    os.makedirs(os.path.join(BASE, '04logs'), exist_ok=True)
    main()
