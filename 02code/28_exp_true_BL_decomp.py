"""
28_exp_true_BL_decomp.py：真三组分 Beer-Lambert 分解（NMF/约束 LS）

外审 codex（CRITICAL 1）指出：
    论文 §3.1 写"基于三组分 BL 分解"，但 03_exp_BL_validation.py 实际是 PCA(n=10)。
    这是科学诚信问题。

本脚本提供两条**真正的 K=3 加性 BL 分解**作为 §3.1 的方法学补强：
    1) NMF（非负矩阵分解）：A ≈ W·H，K=3。要求 W、H 均非负，
       近似 BL 的 "吸收-浓度乘积" 非负结构。
    2) 约束 NNLS（非负最小二乘）：先从源域用 NMF 找出 K=3 共享组分谱 H_src，
       再在目标域用 NNLS 求每个样本的混合权重 W_tgt，重构误差作为 BL 残差。

实验方案：
    - 12 个 1源→1目标场景
    - 在每个场景上对比 4 种 surrogate 的 BL 残差跨域放大倍数：
      * PCA(n=10)         （论文原方法）
      * NMF(K=3)
      * NNLS-fit-transfer(K=3)
      * SVD(K=3)
    - 报告各 surrogate 的源域/目标域中位残差、放大倍数、Mann-Whitney p
    - 跨 surrogate 的放大倍数 Pearson r 相关性
    - 关键：验证"BL 违反"信号不是 PCA(n=10) 的人为伪影

运行方式:
    cd 02code
    python 28_exp_true_BL_decomp.py --device cpu  # 不需要 GPU

输出文件：
    03outputs/28_exp_true_BL_decomp-图a.pdf  — 4 个 surrogate 的 BL 残差对比箱线图
    03outputs/28_exp_true_BL_decomp-图b.pdf  — surrogate 间放大倍数相关矩阵
    03outputs/28_exp_true_BL_decomp-图c.pdf  — NMF K=3 学到的源域共享组分谱
    03outputs/28_exp_true_BL_decomp.xlsx     — 完整结果
"""

import os
import importlib.util
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
from scipy.optimize import nnls
from scipy.stats import mannwhitneyu, pearsonr
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "export_utils", os.path.join(_CODE_DIR, "01_export_utils.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

setup_chinese_font   = _mod.setup_chinese_font
figure_output_path   = _mod.figure_output_path
write_excel_workbook = _mod.write_excel_workbook
get_logger           = _mod.get_logger
load_processed_year_data = _mod.load_processed_year_data
split_by_region      = _mod.split_by_region
AVAILABLE_YEARS      = _mod.AVAILABLE_YEARS

setup_chinese_font()
FILE_STEM = '28_exp_true_BL_decomp'
logger    = get_logger(FILE_STEM)
log       = logger.log

WAVELENGTHS = np.linspace(862, 2478, 229)


# ═══════════════════════════════════════════════════════════════════════════
# 各 surrogate 的拟合 + 重建误差
# ═══════════════════════════════════════════════════════════════════════════

def residual_relative(X, X_hat):
    """逐样本相对残差 = ||X - X_hat||₂ / ||X||₂"""
    res = np.linalg.norm(X - X_hat, axis=1)
    norm = np.linalg.norm(X, axis=1)
    norm = np.where(norm < 1e-10, 1e-10, norm)
    return res / norm


def fit_pca(X_src, K=10):
    """PCA-K surrogate（论文原方法的 K=10）"""
    scaler = StandardScaler(with_std=False)
    Xc = scaler.fit_transform(X_src)
    pca = PCA(n_components=K)
    pca.fit(Xc)
    return ('pca', pca, scaler)


def transform_pca(model_tup, X):
    _, pca, scaler = model_tup
    Xc = scaler.transform(X)
    Xc_hat = pca.inverse_transform(pca.transform(Xc))
    return Xc_hat + scaler.mean_


def fit_nmf(X_src, K=3, max_iter=500):
    """NMF K=3 surrogate（要求非负）"""
    X_pos = X_src - X_src.min() + 1e-3  # shift 使非负
    shift = X_src.min() - 1e-3
    nmf = NMF(n_components=K, init='nndsvd', max_iter=max_iter,
              random_state=42, tol=1e-4)
    nmf.fit(X_pos)
    return ('nmf', nmf, shift)


def transform_nmf(model_tup, X):
    _, nmf, shift = model_tup
    X_pos = X - shift
    X_pos = np.clip(X_pos, 1e-6, None)
    W = nmf.transform(X_pos)
    H = nmf.components_
    X_hat_pos = W @ H
    return X_hat_pos + shift


def fit_nnls(X_src, K=3, max_iter=500):
    """从源域 NMF 学共享组分谱 H，目标域用 NNLS 拟合混合权重"""
    X_pos = X_src - X_src.min() + 1e-3
    shift = X_src.min() - 1e-3
    nmf = NMF(n_components=K, init='nndsvd', max_iter=max_iter,
              random_state=42, tol=1e-4)
    nmf.fit(X_pos)
    H = nmf.components_  # 源域学到的共享组分谱
    return ('nnls', H, shift)


def transform_nnls(model_tup, X):
    _, H, shift = model_tup
    X_pos = X - shift
    X_pos = np.clip(X_pos, 1e-6, None)
    # 每个样本用 NNLS 求 W
    W = np.zeros((len(X_pos), H.shape[0]))
    for i in range(len(X_pos)):
        w, _ = nnls(H.T, X_pos[i])
        W[i] = w
    X_hat_pos = W @ H
    return X_hat_pos + shift


def fit_svd(X_src, K=3):
    """SVD K=3 surrogate（对照）"""
    scaler = StandardScaler(with_std=False)
    Xc = scaler.fit_transform(X_src)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V_k = Vt[:K]
    return ('svd', V_k, scaler)


def transform_svd(model_tup, X):
    _, V_k, scaler = model_tup
    Xc = scaler.transform(X)
    Xc_hat = Xc @ V_k.T @ V_k
    return Xc_hat + scaler.mean_


SURROGATES = {
    'PCA-10':     (fit_pca,   transform_pca,   {'K': 10}),
    'NMF-3':      (fit_nmf,   transform_nmf,   {'K': 3}),
    'NNLS-3':     (fit_nnls,  transform_nnls,  {'K': 3}),
    'SVD-3':      (fit_svd,   transform_svd,   {'K': 3}),
}


# ═══════════════════════════════════════════════════════════════════════════
# 场景配置
# ═══════════════════════════════════════════════════════════════════════════

def build_12_scenarios():
    """与 03_exp_BL_validation 一致的 12 个 1源→1目标场景。"""
    scenarios_2018 = [
        ('山东', '新疆'), ('山东', '陕西'), ('新疆', '山东'),
        ('新疆', '陕西'), ('陕西', '山东'), ('陕西', '新疆'),
    ]
    scenarios_2025 = [
        ('山东', '新疆'), ('山东', '甘肃'), ('新疆', '山东'),
        ('新疆', '甘肃'), ('甘肃', '山东'), ('甘肃', '新疆'),
    ]
    return [('2018', s, t) for s, t in scenarios_2018] + \
           [('2025', s, t) for s, t in scenarios_2025]


# ═══════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    log('='*80)
    log('28_exp_true_BL_decomp: 真三组分 BL 分解 vs PCA-10 对比')
    log('='*80)

    log('\n[1] 加载数据')
    data_by_year = {}
    for y in AVAILABLE_YEARS:
        X, lab, regs, wl = load_processed_year_data(y, target_wavelengths='2025')
        data_by_year[y] = {'X': X, 'y': lab, 'regions': regs, 'wavelengths': wl}
    log(f'    已加载 {AVAILABLE_YEARS}')

    log('\n[2] 12 个场景 × 4 个 surrogate')
    scenarios = build_12_scenarios()
    rows = []
    H_examples = {}  # 保留 NMF 学到的共享组分谱用于绘图

    for year_str, src_r, tgt_r in scenarios:
        year = int(year_str)
        log(f'\n  [{year}] {src_r} → {tgt_r}')
        data = data_by_year[year]
        X, y, regions = data['X'], data['y'], data['regions']
        X_src, y_src, X_tgt, y_tgt = split_by_region(X, y, regions, src_r, tgt_r)
        if len(X_src) < 30 or len(X_tgt) < 30:
            log(f'    样本不足，跳过')
            continue

        for sur_name, (fit_fn, trans_fn, params) in SURROGATES.items():
            try:
                model = fit_fn(X_src, **params)
                X_hat_src = trans_fn(model, X_src)
                X_hat_tgt = trans_fn(model, X_tgt)
                r_src = residual_relative(X_src, X_hat_src)
                r_tgt = residual_relative(X_tgt, X_hat_tgt)
                med_src = float(np.median(r_src))
                med_tgt = float(np.median(r_tgt))
                ratio = med_tgt / max(med_src, 1e-10)
                stat, p = mannwhitneyu(r_src, r_tgt, alternative='less')
                rows.append({
                    '年份': year,
                    '场景': f'{src_r}→{tgt_r}',
                    'surrogate': sur_name,
                    '源域中位残差': med_src,
                    '目标域中位残差': med_tgt,
                    '放大倍数': ratio,
                    'Mann-Whitney p': p,
                })
                log(f'    {sur_name}: src={med_src:.4f}, tgt={med_tgt:.4f}, ratio={ratio:.2f}, p={p:.2e}')
                # 保存 NMF 学到的组分谱（每个场景的）
                if sur_name == 'NMF-3' and year == 2018 and src_r == '山东':
                    _, nmf_model, _ = model
                    H_examples['2018-山东'] = nmf_model.components_
            except Exception as e:
                log(f'    {sur_name} 失败: {e}')

    df = pd.DataFrame(rows)
    log(f'\n[3] 共 {len(df)} 行结果')

    # ─── 跨 surrogate 的放大倍数相关性 ────────────────────────────────
    log('\n[4] surrogate 间放大倍数相关性')
    pivot = df.pivot(index=['年份', '场景'], columns='surrogate', values='放大倍数')
    log(pivot.to_string())
    corr_rows = []
    surr_names = list(pivot.columns)
    for i, a in enumerate(surr_names):
        for j, b in enumerate(surr_names):
            if i < j:
                pair = pivot[[a, b]].dropna()
                if len(pair) >= 3:
                    r, p = pearsonr(pair[a], pair[b])
                    corr_rows.append({'a': a, 'b': b, 'Pearson r': r,
                                      'p_value': p, 'n': len(pair)})
                    log(f'    {a} vs {b}: r={r:.3f}, p={p:.3f}, n={len(pair)}')
    corr_df = pd.DataFrame(corr_rows)

    # ─── 各 surrogate 的均值汇总 ──────────────────────────────────────
    log('\n[5] 各 surrogate 跨 12 场景均值')
    sur_summary = (df.groupby('surrogate')[['源域中位残差', '目标域中位残差', '放大倍数']]
                   .agg(['mean', 'median']))
    log(sur_summary.to_string())
    # 拍平多级列
    sur_summary.columns = ['_'.join(c).rstrip('_') for c in sur_summary.columns]

    # ─── 输出 ──────────────────────────────────────────────────────
    log('\n[6] 写出 xlsx')
    sheets = {
        '表a：完整结果': df,
        '表b：surrogate相关性': corr_df,
        '表c：surrogate汇总': sur_summary.reset_index(),
        '表d：放大倍数透视': pivot.reset_index(),
    }
    write_excel_workbook(FILE_STEM, sheets)

    # ─── 图 a：4 surrogate 在 12 场景上的放大倍数对比箱线 ───────────
    fig, ax = plt.subplots(figsize=(9, 5))
    sur_order = ['PCA-10', 'NMF-3', 'NNLS-3', 'SVD-3']
    data_box = [df[df['surrogate'] == s]['放大倍数'].values for s in sur_order]
    palette = ['#6c757d', '#e03131', '#2f9e44', '#fab005']
    bp = ax.boxplot(data_box, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch, c in zip(bp['boxes'], palette):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='No amplification')
    ax.set_xticklabels(sur_order)
    ax.set_ylabel('BL Residual Amplification Ratio (target/source)')
    ax.set_title('BL Violation Robustness across Surrogates (12 cross-origin scenarios)',
                 fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    plt.tight_layout()
    plt.savefig(figure_output_path(FILE_STEM, 0), dpi=300, bbox_inches='tight')
    plt.close()
    log(f'    saved: {FILE_STEM}-图a.pdf')

    # ─── 图 b：相关矩阵 ────────────────────────────────────────────
    if len(pivot.columns) >= 2:
        corr_mat = pivot.corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr_mat.values, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_mat.columns)))
        ax.set_xticklabels(corr_mat.columns, rotation=30)
        ax.set_yticks(range(len(corr_mat.index)))
        ax.set_yticklabels(corr_mat.index)
        for i in range(len(corr_mat)):
            for j in range(len(corr_mat)):
                ax.text(j, i, f'{corr_mat.values[i,j]:.2f}',
                        ha='center', va='center',
                        color='black' if abs(corr_mat.values[i,j]) < 0.5 else 'white')
        ax.set_title('Surrogate 间放大倍数 Pearson r', fontweight='bold')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(figure_output_path(FILE_STEM, 1), dpi=300, bbox_inches='tight')
        plt.close()
        log(f'    saved: {FILE_STEM}-图b.pdf')

    # ─── 图 c：NMF K=3 学到的源域共享组分谱 ──────────────────────────
    if H_examples:
        H = list(H_examples.values())[0]
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#e03131', '#1c7ed6', '#2f9e44']
        labels = ['Comp-1 (water-related?)', 'Comp-2 (sugar-related?)',
                  'Comp-3 (scattering/structural?)']
        for k in range(min(3, H.shape[0])):
            ax.plot(WAVELENGTHS, H[k] / H[k].max(),
                    color=colors[k], label=labels[k], linewidth=1.8)
        for peak, color, label in [(1450, 'blue', 'H₂O'),
                                    (1940, 'blue', 'H₂O'),
                                    (1580, 'orange', 'Sugar'),
                                    (2080, 'orange', 'Sugar')]:
            ax.axvline(peak, color=color, linestyle=':', alpha=0.5)
            ax.text(peak, 1.02, f'{label}\n{peak}nm',
                    ha='center', va='bottom', fontsize=8, color=color, alpha=0.7)
        ax.set_xlabel('Wavelength λ (nm)')
        ax.set_ylabel('Normalized component intensity')
        ax.set_title('NMF K=3: Shared component spectra (2018年 山东源域)',
                     fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(figure_output_path(FILE_STEM, 2), dpi=300, bbox_inches='tight')
        plt.close()
        log(f'    saved: {FILE_STEM}-图c.pdf')

    log('\n[完成]')


if __name__ == '__main__':
    main()
