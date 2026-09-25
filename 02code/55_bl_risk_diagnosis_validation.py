"""
55_bl_risk_diagnosis_validation.py: 检验"BL 残差放大可作免标签迁移风险诊断"

【为什么必须做这个实验】
论文的第一贡献是"BL 加性结构失稳可作免标签的迁移风险诊断信号"。但此前的证据只到
"残差会放大"（03/28 号），**从未证明这个信号能预测迁移后的实际风险**。
"诊断"这个词意味着**可预测性**：拿到一个新产地的光谱（无任何参考标签），
算出 BL 残差放大倍数，就应当能判断"模型迁过去会不会崩"。

若该预测性不成立，本文只能把主张降级为"BL 残差可表征域间结构差异"，
不得写作"风险诊断"。这是一次**可能推翻自己核心主张**的检验。

【检验设计（实验前预注册，禁止 post-hoc 切换）】
  预测子 A：BL 残差放大倍数 = median(目标域相对重构残差) / median(源域相对重构残差)
            低秩基底 PCA-10 仅在**源域光谱**上拟合。全程不使用任何标签。
  结局 Y ：该场景的跨域 RMSE（各方法）、品质误判率、斜率/偏置校正增益
  主检验 ：Spearman 秩相关 ρ（A 与 Y 均非正态、关系可能单调非线性）
  聚类   ：唯一目标集合（602 场景仅对应 62 个唯一目标集合，场景间不独立）
  CI     ：按唯一目标集合的 cluster bootstrap（1000 次）
  外部性 ：留一目标产地交叉验证（leave-one-target-origin-out）——
           用其余产地拟合 A→风险的单调关系，预测**从未见过**的目标产地的风险排序。
           这是"诊断"一词唯一站得住的证据形式：对新产地的**外推**能力。

  判定（预先设定，不得事后调整）：
    ρ ≥ 0.5 且 95% CI 下界 > 0，且留一验证的中位 ρ > 0.3  → "风险诊断"成立
    ρ 显著为正但未达上述阈值                              → 只能写"与迁移风险相关"
    ρ 不显著或为负                                        → 主张推翻，须删除"诊断"表述

运行方式:
    cd <项目根目录>
    python 02code/55_bl_risk_diagnosis_validation.py

输出文件:
    04outputs/55_bl_risk_diagnosis_validation.xlsx  — 逐场景预测子与结局、相关分析、留一验证
    05logs/55_bl_risk_diagnosis_validation_<时间戳>.log
"""

import os
import json
import argparse
import importlib.util
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, '..'))
_OUT = os.path.join(_BASE, '04outputs')
_LOG = os.path.join(_BASE, '05logs')
FILE_STEM = '55_bl_risk_diagnosis_validation'

_spec = importlib.util.spec_from_file_location(
    'export_utils', os.path.join(_CODE, '01_export_utils.py'))
_eu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eu)

N_COMPONENTS = 10          # 与 03 号一致：PCA-10 作为 BL 可分解性的低秩代理
FORMAL_SEEDS = [20060515, 20041210, 19810915, 2023, 2024]

# 预注册判定阈值
RHO_STRONG = 0.5
RHO_WEAK = 0.3


def make_logger():
    os.makedirs(_LOG, exist_ok=True)
    path = os.path.join(_LOG, f'{FILE_STEM}_{datetime.now():%y-%m-%d_%H%M%S}.log')

    def log(msg=''):
        print(msg, flush=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(str(msg) + '\n')
    return log


def bl_amplification(X_src, X_tgt):
    """
    BL 残差放大倍数。**必须与 03 号 compute_bl_residual 逐行同口径**，否则本脚本的
    预测子与论文图1/§2.1 报告的放大倍数不是同一个量，相关分析将失去意义。

        残差_i = ‖X_i − X̂_i‖₂ / ‖X_i‖₂
        X̂    = PCA⁻¹(PCA(X − μ_src)) + μ_src

    关键：中心化用 `StandardScaler(with_std=False)`——**只去均值，不做逐波段缩放**。
    这是 03 号的原始选择，且是对的：光谱各波段的方差结构本身携带信息，
    若逐波段标准化会把低方差波段（多为噪声）过度加权。
    （曾误用默认的 StandardScaler（含缩放）实现本函数，导致与 03 号系统性不一致——
      12 个 1→1 场景中 8 个对不上。已纠正。）

    全程只用光谱，不碰任何标签——这是"免标签诊断"能成立的前提。
    """
    scaler = StandardScaler(with_std=False).fit(X_src)
    Zs = scaler.transform(X_src)
    pca = PCA(n_components=min(N_COMPONENTS, X_src.shape[0] - 1,
                               X_src.shape[1])).fit(Zs)

    def resid(X):
        Z = scaler.transform(X)
        Xr = pca.inverse_transform(pca.transform(Z)) + scaler.mean_
        num = np.linalg.norm(X - Xr, axis=1)
        den = np.maximum(np.linalg.norm(X, axis=1), 1e-10)
        return num / den

    r_src, r_tgt = resid(X_src), resid(X_tgt)
    return float(np.median(r_tgt) / max(np.median(r_src), 1e-12)), \
           float(np.median(r_src)), float(np.median(r_tgt))


def cluster_spearman_ci(x, y, clusters, n_boot=1000, seed=11):
    """按聚类单位（唯一目标集合）重采样的 Spearman ρ 的 95% CI。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(clusters == c)[0] for c in pick])
        if len(np.unique(x[idx])) < 3:
            continue
        rhos.append(stats.spearmanr(x[idx], y[idx]).statistic)
    rhos = np.array([r for r in rhos if np.isfinite(r)])
    if len(rhos) < 50:
        return np.nan, np.nan
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_boot', type=int, default=1000)
    args = ap.parse_args()
    log = make_logger()

    log('=' * 78)
    log('55 - BL 残差放大能否**预测**迁移风险（"免标签诊断"主张的检验）')
    log('=' * 78)

    # ── 1. 对全部场景计算免标签预测子 A ───────────────────────────────
    data = _eu.load_year_region_map(years=_eu.AVAILABLE_YEARS, target_wavelengths='2025')
    with open(os.path.join(_CODE, '04_migration_scenarios.json'), 'r', encoding='utf-8') as f:
        scenarios = json.load(f)

    rows = []
    for sid, cfg in scenarios.items():
        try:
            src = _eu.merge_domain_payload(data, _eu.extract_domain_keys(cfg['source']))
            tgt = _eu.merge_domain_payload(data, _eu.extract_domain_keys(cfg['target']))
        except KeyError:
            continue
        Xs, Xt = src['X'], tgt['X']
        if len(Xs) < 15 or len(Xt) < 15:
            continue
        amp, r_src, r_tgt = bl_amplification(Xs, Xt)
        rows.append({
            '场景': sid, '迁移类型': cfg.get('type', '未知'),
            '目标集合': '+'.join(sorted(f'{y}_{r}' for y, r in _eu.extract_domain_keys(cfg['target']))),
            '目标产地': '+'.join(sorted({r for _, r in _eu.extract_domain_keys(cfg['target'])})),
            'BL放大倍数': amp, '源域中位残差': r_src, '目标域中位残差': r_tgt,
            'n_src': len(Xs), 'n_tgt': len(Xt),
        })
    A = pd.DataFrame(rows)
    log(f'\n免标签预测子已算出：{len(A)} 个场景 | '
        f'BL 放大倍数 {A["BL放大倍数"].min():.2f} ~ {A["BL放大倍数"].max():.2f}')
    log(f'唯一目标集合数：{A["目标集合"].nunique()} | 唯一目标产地组合：{A["目标产地"].nunique()}')

    # ── 2. 取结局（跨域风险）──────────────────────────────────────────
    src50 = os.path.join(_OUT, '50_formal_multiseed_benchmark.xlsx')
    if not os.path.exists(src50):
        out = os.path.join(_OUT, f'{FILE_STEM}.xlsx')
        with pd.ExcelWriter(out) as w:
            A.to_excel(w, sheet_name='表a：免标签预测子', index=False)
        log(f'\n⚠ 50 号结果尚未落地，仅产出预测子。已保存: {out}')
        log('   50 号 --merge 完成后重跑本脚本，即可做相关分析与留一验证。')
        return

    raw = pd.read_excel(src50, sheet_name='表h：全部原始结果')
    raw = raw[raw['种子'].isin(FORMAL_SEEDS)]
    per = (raw.groupby(['方法', '校正', '场景'], as_index=False)
              [['RMSE', '品质误判率']].mean())
    # 案例研究的聚合量一律取七个未校正方法都有结果的共同场景（补充材料 S3）：
    # 各方法在各自跑出的场景上算，难场景没跑完的方法会显得更好，跨方法也不可比。
    bare7 = ['PLSR', 'SVR', 'CNN', 'CNN+MMD', 'CNN+BL', 'Phys', 'Phys+BL']
    wide = per[per['校正'] == '裸'].pivot(index='场景', columns='方法', values='RMSE')
    common = wide[bare7].dropna().index
    per = per[per['场景'].isin(common)]
    log(f'共同场景（七个未校正方法都有结果）：{len(common)} 个')

    METHODS = ['PLSR', 'SVR', 'CNN', 'CNN+MMD', 'Phys+BL']
    log('\n' + '=' * 78)
    log('检验一：BL 放大倍数 与 跨域风险 的相关性（聚类单位=唯一目标集合）')
    log('=' * 78)
    log(f'{"结局":28s} {"n":>5s} {"Spearman ρ":>11s} {"95% CI":>18s} {"P":>10s}  判定')

    res = []
    for m in METHODS:
        sub = per[(per['方法'] == m) & (per['校正'] == '裸')][['场景', 'RMSE', '品质误判率']]
        j = A.merge(sub, on='场景', how='inner')
        if len(j) < 20:
            continue
        for ycol, yname in [('RMSE', f'{m} 跨域 RMSE'), ('品质误判率', f'{m} 品质误判率')]:
            rho = stats.spearmanr(j['BL放大倍数'], j[ycol])
            lo, hi = cluster_spearman_ci(j['BL放大倍数'], j[ycol], j['目标集合'],
                                         n_boot=args.n_boot)
            verdict = ('✅强' if (rho.statistic >= RHO_STRONG and lo > 0)
                       else ('⚠弱相关' if (rho.pvalue < 0.05 and rho.statistic > 0) else '🔴不成立'))
            log(f'{yname:28s} {len(j):5d} {rho.statistic:11.3f} '
                f'[{lo:6.3f},{hi:6.3f}] {rho.pvalue:10.2e}  {verdict}')
            res.append({'结局': yname, 'n': len(j), 'Spearman_rho': rho.statistic,
                        'CI_low': lo, 'CI_high': hi, 'P': rho.pvalue, '判定': verdict})

    # 校正增益（用同一份 20% 标签校正后的收益，也是一种"风险"的度量）
    for m in METHODS:
        b = per[(per['方法'] == m) & (per['校正'] == '裸')].set_index('场景')['RMSE']
        c = per[(per['方法'] == m) & (per['校正'] == '+SB')].set_index('场景')['RMSE']
        k = b.index.intersection(c.index)
        g = ((b.loc[k] - c.loc[k]) / b.loc[k] * 100).rename('校正增益').reset_index()
        j = A.merge(g, on='场景', how='inner')
        if len(j) < 20:
            continue
        rho = stats.spearmanr(j['BL放大倍数'], j['校正增益'])
        lo, hi = cluster_spearman_ci(j['BL放大倍数'], j['校正增益'], j['目标集合'],
                                     n_boot=args.n_boot)
        verdict = ('✅强' if (abs(rho.statistic) >= RHO_STRONG and (lo > 0 or hi < 0))
                   else ('⚠弱相关' if rho.pvalue < 0.05 else '🔴不成立'))
        log(f'{m + " 校正增益":28s} {len(j):5d} {rho.statistic:11.3f} '
            f'[{lo:6.3f},{hi:6.3f}] {rho.pvalue:10.2e}  {verdict}')
        res.append({'结局': f'{m} 校正增益', 'n': len(j), 'Spearman_rho': rho.statistic,
                    'CI_low': lo, 'CI_high': hi, 'P': rho.pvalue, '判定': verdict})
    res = pd.DataFrame(res)

    # ── 3. 留一目标产地交叉验证（"诊断"一词唯一站得住的证据）─────────
    log('\n' + '=' * 78)
    log('检验二：留一目标产地交叉验证 —— 能否预测**从未见过**的产地的风险排序')
    log('=' * 78)
    loo = []
    for m in METHODS:
        sub = per[(per['方法'] == m) & (per['校正'] == '裸')][['场景', 'RMSE']]
        j = A.merge(sub, on='场景', how='inner')
        origins = sorted(j['目标产地'].unique())
        for o in origins:
            te = j[j['目标产地'] == o]
            if len(te) < 8 or te['BL放大倍数'].nunique() < 3:
                continue
            # 留出产地内部：BL 放大倍数 与 实际 RMSE 的秩相关
            rho = stats.spearmanr(te['BL放大倍数'], te['RMSE'])
            loo.append({'方法': m, '留出目标产地': o, 'n_场景': len(te),
                        'Spearman_rho': rho.statistic, 'P': rho.pvalue})
    loo = pd.DataFrame(loo)
    if not loo.empty:
        log(f'{"方法":10s} {"留出产地数":>8s} {"中位 ρ":>9s} {"ρ>0 占比":>9s}  判定')
        loo_sum = []
        for m in METHODS:
            g = loo[loo['方法'] == m]
            if g.empty:
                continue
            med = g['Spearman_rho'].median()
            pos = (g['Spearman_rho'] > 0).mean() * 100
            v = '✅' if med > RHO_WEAK else ('⚠' if med > 0 else '🔴')
            log(f'{m:10s} {len(g):8d} {med:9.3f} {pos:8.0f}%  {v}')
            loo_sum.append({'方法': m, '留出产地数': len(g), '中位_rho': med,
                            'rho大于0占比%': pos, '判定': v})
        loo_sum = pd.DataFrame(loo_sum)
    else:
        loo_sum = pd.DataFrame()

    # ── 4. 结论 ───────────────────────────────────────────────────────
    log('\n' + '=' * 78)
    log('结论（对照预注册阈值）')
    log('=' * 78)
    main_rows = res[res['结局'].str.contains('RMSE')]
    med_rho = main_rows['Spearman_rho'].median() if not main_rows.empty else np.nan
    ci_ok = (main_rows['CI_low'] > 0).all() if not main_rows.empty else False
    loo_ok = (loo_sum['中位_rho'].median() > RHO_WEAK) if not loo_sum.empty else False

    log(f'  各方法 RMSE 的 Spearman ρ 中位数 : {med_rho:.3f}   （阈值 ≥{RHO_STRONG}）')
    log(f'  全部 95% CI 下界 > 0             : {ci_ok}')
    log(f'  留一产地验证中位 ρ > {RHO_WEAK}        : {loo_ok}')
    if med_rho >= RHO_STRONG and ci_ok and loo_ok:
        verdict = '✅ "BL 残差可作免标签迁移风险诊断" 成立，可保留该表述'
    elif not np.isnan(med_rho) and med_rho > 0 and ci_ok:
        verdict = ('⚠ 相关显著但强度不足 → 主张须降级为"BL 残差与迁移风险相关"，'
                   '不得写作"风险诊断"')
    else:
        verdict = '🔴 主张不成立 → 必须删除"风险诊断"表述，改为"可表征域间结构差异"'
    log(f'\n  {verdict}')

    os.makedirs(_OUT, exist_ok=True)
    out = os.path.join(_OUT, f'{FILE_STEM}.xlsx')
    with pd.ExcelWriter(out) as w:
        A.to_excel(w, sheet_name='表a：免标签预测子', index=False)
        res.to_excel(w, sheet_name='表b：相关分析', index=False)
        loo.to_excel(w, sheet_name='表c：留一产地明细', index=False)
        loo_sum.to_excel(w, sheet_name='表d：留一产地汇总', index=False)
        pd.DataFrame([{'判定': verdict, 'ρ中位数': med_rho,
                       'CI下界全>0': ci_ok, '留一通过': loo_ok}]).to_excel(
            w, sheet_name='表e：结论', index=False)
    log(f'\n已保存: {out}')


if __name__ == '__main__':
    main()
