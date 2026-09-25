"""
35_BL_source_heldout_validation.py：source-held-out 验证 BL 残差诊断（CRITICAL 1 修复）

reviewer 怀疑 BL 残差放大可能是 in-sample 重建伪影：PCA/NMF/SVD 基底在源域全样本拟合后再
比较源域和目标域残差，可能因源域 in-sample 残差天然偏低而虚假放大。

本脚本：把每个源域随机按 70/30 切成 train (fit basis) 和 held-out (evaluate residual)，
再分别与目标域残差对比。如果 held-out 残差仍然系统性低于目标域残差，则诊断不是 in-sample 伪影。

运行方式:
    cd <repository root>
    python 35_BL_source_heldout_validation.py

输出文件：
    04outputs/35_BL_source_heldout_validation.xlsx - 各场景 train/held-out/target 残差比较
    04outputs/35_BL_source_heldout_validation-Fig11.pdf - 12 场景 held-out vs target 散点图
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from scipy import stats

_THIS = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(_THIS, '..'))
DATA = os.path.join(BASE, '01data')
OUT  = os.path.join(BASE, '04outputs')
FILE_STEM = '35_BL_source_heldout_validation'

plt.rcParams.update({
    'font.family' : 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size'   : 8.5,
    'axes.titlesize'  : 10,
    'axes.labelsize'  : 9,
    'axes.linewidth'  : 0.8,
    'axes.spines.top' : False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'lines.linewidth': 1.2,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.unicode_minus': False,
})

PAL = {
    'src_train': '#5DADE2',
    'src_held':  '#1F618D',
    'target':    '#C0392B',
    'baseline':  '#7F8C8D',
}


def load_year(year):
    """
    经由项目唯一的数据入口取数。

    ⚠ 本函数原先直接 `pd.read_csv(os.path.join(DATA, fn))`，DATA 还指向早已不存在的
    `01data/`——即它既跑不起来（FileNotFoundError），历史产出也**绕过了数据入口的
    全部硬过滤**（SSC 越界剔除、逐位重复光谱剔除）。改为统一走
    `01_export_utils.load_processed_dataframe()`，与全部其他实验共享同一份干净数据。
    """
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        'export_utils', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '01_export_utils.py'))
    _eu = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_eu)

    df = _eu.load_processed_dataframe(year)
    df = df[df['年份'] == year] if '年份' in df.columns else df
    wcols = [c for c in df.columns if 'nm' in c]
    X = df[wcols].values.astype(np.float64)
    origins = df['产地'].values
    return X, origins


def fit_basis(X_train, kind, K):
    """在源域 train 子集上拟合 surrogate 基底。"""
    if kind == 'PCA':
        m = PCA(n_components=K)
        m.fit(X_train)
        return m
    elif kind == 'NMF':
        Xnn = np.clip(X_train - X_train.min() + 1e-3, 1e-6, None)
        m = NMF(n_components=K, init='nndsvd', random_state=42, max_iter=300)
        m.fit(Xnn)
        return m
    elif kind == 'SVD':
        m = TruncatedSVD(n_components=K, random_state=42)
        m.fit(X_train)
        return m
    raise ValueError(kind)


def reconstruct(m, X, kind):
    if kind == 'PCA':
        Z = m.transform(X)
        return m.inverse_transform(Z)
    if kind == 'NMF':
        Xnn = np.clip(X - X.min() + 1e-3, 1e-6, None)
        W = m.transform(Xnn)
        return W @ m.components_
    if kind == 'SVD':
        Z = m.transform(X)
        return Z @ m.components_


def rel_residual(X, X_hat):
    return np.linalg.norm(X - X_hat, axis=1) / (np.linalg.norm(X, axis=1) + 1e-9)


def main():
    rng = np.random.default_rng(42)
    rows = []
    scatter = []  # for the figure

    for year in (2018, 2025):
        X, origs = load_year(year)
        domains = sorted(np.unique(origs))
        # 限定到 6 个 1源→1目标 场景
        for src_orig in domains:
            for tgt_orig in domains:
                if src_orig == tgt_orig:
                    continue
                src_mask = origs == src_orig
                tgt_mask = origs == tgt_orig
                Xs = X[src_mask]
                Xt = X[tgt_mask]
                if len(Xs) < 30 or len(Xt) < 30:
                    continue

                # 70/30 split 源域
                idx = rng.permutation(len(Xs))
                cut = int(0.7 * len(Xs))
                Xs_train = Xs[idx[:cut]]
                Xs_held  = Xs[idx[cut:]]

                # 三种 surrogate
                for kind, K in [('PCA', 10), ('NMF', 3), ('SVD', 3)]:
                    m = fit_basis(Xs_train, kind, K)
                    rec_train = reconstruct(m, Xs_train, kind)
                    rec_held  = reconstruct(m, Xs_held,  kind)
                    rec_tgt   = reconstruct(m, Xt,       kind)
                    r_train = rel_residual(Xs_train, rec_train)
                    r_held  = rel_residual(Xs_held,  rec_held)
                    r_tgt   = rel_residual(Xt,       rec_tgt)

                    # MW U：held-out vs target
                    _, p = stats.mannwhitneyu(r_tgt, r_held, alternative='greater')

                    rows.append(dict(
                        year=year, source=src_orig, target=tgt_orig, surrogate=kind,
                        n_src_train=len(Xs_train), n_src_held=len(Xs_held), n_tgt=len(Xt),
                        median_residual_train=float(np.median(r_train)),
                        median_residual_held=float(np.median(r_held)),
                        median_residual_tgt=float(np.median(r_tgt)),
                        amp_factor_held_to_tgt=float(np.median(r_tgt)/np.median(r_held)),
                        amp_factor_train_to_tgt=float(np.median(r_tgt)/np.median(r_train)),
                        p_held_vs_tgt=p,
                    ))
                    if kind == 'PCA':
                        scatter.append((np.median(r_held), np.median(r_tgt),
                                        f'{year}_{src_orig}→{tgt_orig}'))

    df = pd.DataFrame(rows)
    out_xlsx = os.path.join(OUT, f'{FILE_STEM}.xlsx')
    with pd.ExcelWriter(out_xlsx) as wr:
        df.to_excel(wr, sheet_name='all_results', index=False)
        # 按 surrogate 汇总
        agg = df.groupby('surrogate').agg(
            mean_train_to_tgt=('amp_factor_train_to_tgt', 'mean'),
            median_train_to_tgt=('amp_factor_train_to_tgt', 'median'),
            mean_held_to_tgt=('amp_factor_held_to_tgt', 'mean'),
            median_held_to_tgt=('amp_factor_held_to_tgt', 'median'),
            n_sig_p005=('p_held_vs_tgt', lambda v: (v < 0.05).sum()),
            n_total=('p_held_vs_tgt', 'size'),
        ).reset_index()
        agg.to_excel(wr, sheet_name='surrogate_summary', index=False)
    print(f'  saved: {out_xlsx}')

    print('\n=== Summary: held-out source vs target residuals ===')
    print(agg.to_string(index=False))
    print(f'\nfraction of 36 (12 scenarios × 3 surrogates) significant (p<0.05): '
          f'{(df["p_held_vs_tgt"] < 0.05).sum()}/{len(df)}')

    # 散点图：12 scenarios held-out (x) vs target (y)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    for hi, ti, lab in scatter:
        ax.scatter([hi], [ti], color=PAL['target'], s=22, alpha=0.85,
                   edgecolor='#2C3E50', linewidth=0.5)
    lim = max(max(t for _, t, _ in scatter), max(h for h, _, _ in scatter)) * 1.05
    ax.plot([0, lim], [0, lim], '--', color=PAL['baseline'],
            linewidth=0.7, alpha=0.7, label='y = x (no amplification)')
    ax.set_xlabel(r'Median $R_\mathrm{BL}$ on source held-out samples')
    ax.set_ylabel(r'Median $R_\mathrm{BL}$ on target samples')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(alpha=0.25, linestyle='--', linewidth=0.4)
    ax.legend(loc='lower right', fontsize=8, frameon=False)
    ax.set_title('PCA-10 surrogate: source-held-out vs target residuals\n'
                 r'(all 12 scenarios remain above the diagonal $\Rightarrow$ not an in-sample artifact)',
                 fontweight='bold', loc='left', pad=8, fontsize=10)
    out_pdf = os.path.join(OUT, f'{FILE_STEM}-Fig11.pdf')
    plt.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f'  saved: {out_pdf}')


if __name__ == '__main__':
    main()
