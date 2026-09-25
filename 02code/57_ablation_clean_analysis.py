"""
57_ablation_clean_analysis.py：干净数据上的架构消融分析（表2 + §2.3）

背景：第二次独立审计查出，表2（CNN+BL vs CNN、Phys+BL vs Phys）与 §2.3（发散计数、
McNemar 检验）建立在**清洗前的脏数据**上（含 56.07 °Brix 折光仪误读样本）。50 号主实验
只跑了表1/表3 所需的 5 个方法，不含消融所需的 CNN+BL 与 Phys，故表2 无法从主实验刷新。

本脚本读取「50 号消融分片」（seed=42，方法 = CNN/CNN+BL/Phys/Phys+BL，经统一数据入口
→ 干净数据），重算：
  · 表2 三行：CNN+BL vs CNN、Phys+BL vs Phys、Phys+BL vs CNN
             （配对均值/中位 RMSE、场景相对改进、双侧 Wilcoxon、n）
  · §2.3 发散分析：Phys 与 Phys+BL 各自「测试 RMSE > 10 °Brix」的场景数；
             对不一致配对做精确 McNemar；再按唯一目标集合保守聚类后重做 McNemar。

口径与原 §2.3 完全一致（发散阈值 10、McNemar 零假设=不一致配对朝两方向等概率）。

运行方式:
    cd <项目根目录>
    python 02code/57_ablation_clean_analysis.py

输出文件:
    04outputs/57_ablation_clean_analysis.xlsx  — 表2 + §2.3 全部数字（可回填 content_csae.py）
    04outputs/57_ablation_clean_analysis.md    — 新旧对照，标出因清洗而变化的数字
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, '..'))
_OUT = os.path.join(_BASE, '04outputs')
_SHARD = os.path.join(_OUT, '50ablation_shards')

DIVERGENCE_RMSE = 10.0     # 与 §2.3 一致：测试 RMSE > 10 °Brix 记为「发散」

# 论文当前（脏数据）的表2 / §2.3 数字，用于对照
OLD = {
    'CNN+BL_vs_CNN':  dict(start=2.478, end=2.535, imp_mean=-4.18, imp_med=-2.51, p=4.96e-6, n=571),
    'PhysBL_vs_Phys': dict(start=2.022, end=1.943, imp_mean=1.61,  imp_med=0.02,  p=4.46e-2, n=551),
    'PhysBL_vs_CNN':  dict(start=2.563, end=1.976, imp_mean=22.56, imp_med=23.09, p=8.5e-87, n=569),
    'phys_diverge': 15, 'physbl_diverge': 0, 'mcnemar_p': 6.1e-5, 'mcnemar_clustered_p': 2.4e-4,
}


def load_ablation():
    files = sorted(glob.glob(os.path.join(_SHARD, 'seed*_shard*.csv')))
    if not files:
        raise SystemExit(f'🔴 未找到消融分片：{_SHARD}/seed*_shard*.csv\n'
                         f'   请先跑完 run_ablation_local.sh（8 分片）。')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # 只用「裸」（未校正）结果——表2/§2.3 讨论的是训练本身，不是事后校正
    df = df[df['校正'] == '裸'].copy()
    return df


def paired(df, method_a, method_b):
    """method_a vs method_b 的配对比较（在两者都成功的共同场景上）。
    改进定义与论文一致：以 b 为起点、a 为终点，(RMSE_b − RMSE_a)/RMSE_b。"""
    a = df[df['方法'] == method_a].set_index('场景')['RMSE']
    b = df[df['方法'] == method_b].set_index('场景')['RMSE']
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    imp = (b.values - a.values) / b.values * 100
    # 双侧 Wilcoxon（与论文一致）
    try:
        _, p = stats.wilcoxon(a.values, b.values)
    except ValueError:
        p = np.nan
    return dict(start=float(b.mean()), end=float(a.mean()),
                imp_mean=float(imp.mean()), imp_med=float(np.median(imp)),
                p=float(p), n=int(len(common)))


def mcnemar_exact(n01, n10):
    """精确 McNemar（二项检验，零假设 p=0.5）。返回双侧 P。"""
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    # 双侧精确二项
    p = 2 * stats.binom.cdf(k, n, 0.5)
    return float(min(p, 1.0))


def target_set_of(scenario_id):
    """从场景 id 提取「唯一目标集合」标识，用于保守聚类。
    50 号场景 id 形如 'Y2018-...→目标域' 或含目标域产地；此处取目标侧标签。"""
    s = str(scenario_id)
    # 约定：场景 id 里 '→' 之后是目标域；无则退回整串
    return s.split('→')[-1] if '→' in s else s


def main():
    df = load_ablation()
    n_scen = df['场景'].nunique()
    methods = sorted(df['方法'].unique())
    print(f'[57] 载入消融：{n_scen} 场景 | 方法 {methods}')

    report = ['# 干净数据消融分析（表2 + §2.3）\n',
              f'消融分片场景数：{n_scen}；方法：{methods}\n',
              '> 全部基于 seed=42、经统一数据入口的**干净数据**（已剔除 56.07 污染样本与逐位重复光谱）。\n']

    # ── 表2 三行 ────────────────────────────────────────────────────
    rows2 = {
        'CNN+BL_vs_CNN':  paired(df, 'CNN+BL', 'CNN'),
        'PhysBL_vs_Phys': paired(df, 'Phys+BL', 'Phys'),
        'PhysBL_vs_CNN':  paired(df, 'Phys+BL', 'CNN'),
    }
    report.append('## 表2：架构消融（新 vs 旧）\n')
    report.append('| 对比 | 起始RMSE | 终止RMSE | 改进(均值)% | 改进(中位)% | Wilcoxon P | n |')
    report.append('|---|---|---|---|---|---|---|')
    label = {'CNN+BL_vs_CNN': 'CNN+BL vs CNN', 'PhysBL_vs_Phys': 'Phys+BL vs Phys',
             'PhysBL_vs_CNN': 'Phys+BL vs CNN'}
    for k, r in rows2.items():
        o = OLD[k]
        report.append(f'| {label[k]} | {r["start"]:.3f} ({o["start"]:.3f}) | '
                      f'{r["end"]:.3f} ({o["end"]:.3f}) | '
                      f'{r["imp_mean"]:+.2f} ({o["imp_mean"]:+.2f}) | '
                      f'{r["imp_med"]:+.2f} ({o["imp_med"]:+.2f}) | '
                      f'{r["p"]:.2e} ({o["p"]:.1e}) | {r["n"]} ({o["n"]}) |')
    report.append('')

    # ── §2.3 发散分析 ────────────────────────────────────────────────
    phys = df[df['方法'] == 'Phys'].set_index('场景')['RMSE']
    physbl = df[df['方法'] == 'Phys+BL'].set_index('场景')['RMSE']
    common = phys.index.intersection(physbl.index)
    phys, physbl = phys.loc[common], physbl.loc[common]

    phys_div = phys > DIVERGENCE_RMSE
    physbl_div = physbl > DIVERGENCE_RMSE
    n_phys_div = int(phys_div.sum())
    n_physbl_div = int(physbl_div.sum())

    # McNemar：不一致配对 = 一个发散、另一个不发散
    n01 = int((phys_div & ~physbl_div).sum())   # Phys 发散、Phys+BL 不发散
    n10 = int((~phys_div & physbl_div).sum())   # 反向
    mcp = mcnemar_exact(n01, n10)

    # 按唯一目标集合保守聚类：每个目标集合只要有任一不一致配对就计 1 次
    disc = common[(phys_div & ~physbl_div) | (~phys_div & physbl_div)]
    tsets = {}
    for sc in disc:
        t = target_set_of(sc)
        # 方向：Phys 发散而 Phys+BL 不发散 → +1（利于 Phys+BL）
        d = 1 if (phys.loc[sc] > DIVERGENCE_RMSE and physbl.loc[sc] <= DIVERGENCE_RMSE) else -1
        tsets.setdefault(t, []).append(d)
    c01 = sum(1 for v in tsets.values() if np.mean(v) > 0)
    c10 = sum(1 for v in tsets.values() if np.mean(v) < 0)
    mcp_clustered = mcnemar_exact(c01, c10)

    # 其余正常场景的 Wilcoxon（全部共同场景）
    try:
        _, p_all = stats.wilcoxon(physbl.values, phys.values)
    except ValueError:
        p_all = np.nan

    report.append('## §2.3：物理架构内 BL 损失的净贡献（发散分析）\n')
    report.append(f'- Phys 发散场景数（RMSE>{DIVERGENCE_RMSE:g}）：**{n_phys_div}/{len(common)}**（旧 {OLD["phys_diverge"]}）')
    report.append(f'- Phys+BL 发散场景数：**{n_physbl_div}/{len(common)}**（旧 {OLD["physbl_diverge"]}）')
    report.append(f'- 不一致配对：Phys发散&Phys+BL不发散 = {n01}，反向 = {n10}')
    report.append(f'- 精确 McNemar 双侧 P = **{mcp:.2e}**（旧 {OLD["mcnemar_p"]:.1e}）')
    report.append(f'- 按唯一目标集合聚类（{len(tsets)} 个目标集合）后 McNemar P = **{mcp_clustered:.2e}**（旧 {OLD["mcnemar_clustered_p"]:.1e}）')
    report.append(f'- 全部 {len(common)} 场景 Phys+BL vs Phys 双侧 Wilcoxon P = **{p_all:.2e}**')
    report.append('')

    # ── 落盘 ─────────────────────────────────────────────────────────
    with pd.ExcelWriter(os.path.join(_OUT, '57_ablation_clean_analysis.xlsx')) as w:
        pd.DataFrame([{**{'对比': label[k]}, **v} for k, v in rows2.items()]).to_excel(
            w, sheet_name='表2', index=False)
        pd.DataFrame([{
            'Phys发散数': n_phys_div, 'PhysBL发散数': n_physbl_div,
            '不一致_Phys发散': n01, '不一致_反向': n10,
            'McNemar_P': mcp, 'McNemar_聚类P': mcp_clustered,
            '共同场景数': len(common), '目标集合数': len(tsets),
            '全场景Wilcoxon_P': p_all,
        }]).to_excel(w, sheet_name='§2.3发散', index=False)
        df.to_excel(w, sheet_name='原始_裸', index=False)

    with open(os.path.join(_OUT, '57_ablation_clean_analysis.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report) + '\n')
    print('\n'.join(report))
    print(f'\n已保存: 04outputs/57_ablation_clean_analysis.xlsx / .md')


if __name__ == '__main__':
    main()
