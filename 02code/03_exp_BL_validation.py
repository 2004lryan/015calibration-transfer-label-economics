"""
03_exp_BL_validation.py: Beer-Lambert约束违反程度量化实验（2018/2025统一波段版）

运行方式:
    cd 02code
    python 03_exp_BL_validation.py --device auto

输出文件：
    03outputs/03_exp_BL_validation-图a.pdf  — 跨域 vs 域内 ΔRBL 箱线图
    03outputs/03_exp_BL_validation-图b.pdf  — 6种迁移场景BL误差热力图
    03outputs/03_exp_BL_validation.xlsx     — 完整BL误差统计结果
    04logs/03_exp_BL_validation.log         — 运行日志
"""

import json
import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse

# ─── 加载工具模块（带数字前缀，不能直接import）──────────────────────────────
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "export_utils", os.path.join(_CODE_DIR, "01_export_utils.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

setup_chinese_font = _mod.setup_chinese_font
figure_output_path = _mod.figure_output_path
write_excel_workbook = _mod.write_excel_workbook
get_logger = _mod.get_logger
get_device = _mod.get_device
DATA_DIR = _mod.DATA_DIR
load_processed_year_data = _mod.load_processed_year_data
split_by_region = _mod.split_by_region
AVAILABLE_YEARS = _mod.AVAILABLE_YEARS

# ─── 加载2025年数据加载器──────────────────────────────────────────────────
logger = get_logger('03_exp_BL_validation')
log = logger.log

FILE_STEM = '03_exp_BL_validation'

# ═══════════════════════════════════════════════════════════════════════════
# Beer-Lambert近似：PCA重建误差
# ═══════════════════════════════════════════════════════════════════════════

def fit_bl_components(X_train: np.ndarray, n_components: int = 10):
    """
    用PCA作为Beer-Lambert成分分解的数据驱动近似。
    X_train: 源域光谱矩阵，用于拟合成分基底
    返回：(pca模型, scaler模型)
    """
    scaler = StandardScaler(with_std=False)   # 只去均值，保留方差结构
    X_centered = scaler.fit_transform(X_train)
    pca = PCA(n_components=n_components)
    pca.fit(X_centered)
    return pca, scaler


def compute_bl_residual(X: np.ndarray, pca, scaler) -> np.ndarray:
    """
    计算每个样本的BL重建相对误差：
        ΔRBL_i = ||X_i - X̂_i||₂ / ||X_i||₂
    X̂ 是用PCA（Beer-Lambert近似）重建的光谱。
    """
    X_centered = scaler.transform(X)
    X_reconstructed_centered = pca.inverse_transform(pca.transform(X_centered))
    X_reconstructed = X_reconstructed_centered + scaler.mean_

    residuals = np.linalg.norm(X - X_reconstructed, axis=1)
    norms = np.linalg.norm(X, axis=1)
    norms = np.where(norms < 1e-10, 1e-10, norms)   # 防止除零
    return residuals / norms


# ═══════════════════════════════════════════════════════════════════════════
# 核心实验：按年份的有向跨产地BL违反程度验证
# ═══════════════════════════════════════════════════════════════════════════

def build_year_scenarios(year: int, regions: np.ndarray) -> list:
    """为单个年份生成所有有向跨产地场景。"""
    unique_regions = sorted(np.unique(regions).tolist())
    scenarios = []
    scenario_idx = 1
    for src_region in unique_regions:
        for tgt_region in unique_regions:
            if src_region == tgt_region:
                continue
            scenarios.append((src_region, tgt_region, f'Y{year}-S{scenario_idx}: {src_region}→{tgt_region}'))
            scenario_idx += 1
    return scenarios


def run_bl_validation(X, y, regions, scenario_pairs: list, n_components: int = 10):
    """
    对单个年份的数据运行所有场景的BL违反程度实验。
    返回 DataFrame，每行对应一个样本，包含：
        scenario, domain_type(src/tgt), region, bl_residual
    """
    records = []

    for src_region, tgt_region, scenario_name in scenario_pairs:
        # 分割源域和目标域
        X_src, y_src, X_tgt, y_tgt = split_by_region(
            X, y, regions,
            src_region=src_region,
            tgt_region=tgt_region
        )

        if X_src.shape[0] < 20 or X_tgt.shape[0] < 20:
            log(f'  [跳过] {scenario_name}：样本不足')
            continue

        # 用源域数据拟合BL成分结构
        pca, scaler = fit_bl_components(X_src, n_components=n_components)

        # 计算源域内BL残差（域内基线）
        bl_src = compute_bl_residual(X_src, pca, scaler)
        src_mask = np.isin(regions, src_region) if isinstance(src_region, str) else np.isin(regions, src_region)
        src_regions_vals = regions[src_mask]

        for i, reg in enumerate(src_regions_vals):
            records.append({
                'scenario': scenario_name,
                'domain_type': '源域（域内）',
                'region': reg,
                'bl_residual': bl_src[i],
            })

        # 计算目标域BL残差（跨域）
        bl_tgt = compute_bl_residual(X_tgt, pca, scaler)
        tgt_mask = np.isin(regions, tgt_region) if isinstance(tgt_region, str) else np.isin(regions, tgt_region)
        tgt_regions_vals = regions[tgt_mask]

        for i, reg in enumerate(tgt_regions_vals):
            records.append({
                'scenario': scenario_name,
                'domain_type': '目标域（跨域）',
                'region': reg,
                'bl_residual': bl_tgt[i],
            })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# 统计检验：域内 vs 跨域
# ═══════════════════════════════════════════════════════════════════════════

def statistical_test(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个场景，做Mann-Whitney U检验（不假设正态分布）。
    返回统计汇总DataFrame。
    """
    rows = []
    for scenario in df['scenario'].unique():
        sub = df[df['scenario'] == scenario]
        src_vals = sub[sub['domain_type'] == '源域（域内）']['bl_residual'].values
        tgt_vals = sub[sub['domain_type'] == '目标域（跨域）']['bl_residual'].values

        if len(src_vals) < 5 or len(tgt_vals) < 5:
            continue

        stat, pval = stats.mannwhitneyu(tgt_vals, src_vals, alternative='greater')
        median_src = np.median(src_vals)
        median_tgt = np.median(tgt_vals)
        fold_change = median_tgt / (median_src + 1e-10)

        rows.append({
            '场景': scenario,
            '源域中位BL残差': round(median_src, 4),
            '目标域中位BL残差': round(median_tgt, 4),
            '倍数变化（跨域/域内）': round(fold_change, 3),
            'Mann-Whitney U统计量': round(stat, 1),
            'p值': f'{pval:.2e}',
            '显著性（p<0.05）': '是' if pval < 0.05 else '否',
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════════════════

def plot_boxplot(df: pd.DataFrame):
    """Fig a: Cross-origin vs within-origin BL residual boxplot"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle('Beer-Lambert Constraint Violation: Cross-Origin vs Within-Origin',
                 fontsize=13, y=1.01)

    src_data = df[df['domain_type'] == '源域（域内）']['bl_residual'].values
    tgt_data = df[df['domain_type'] == '目标域（跨域）']['bl_residual'].values

    bp = ax.boxplot(
        [src_data, tgt_data],
        labels=['Source (within-origin)', 'Target (cross-origin)'],
        patch_artist=True,
        widths=0.5,
        showfliers=True,
        flierprops=dict(marker='o', markersize=3, alpha=0.4),
        medianprops=dict(color='black', linewidth=2),
    )
    bp['boxes'][0].set_facecolor('#AED6F1')
    bp['boxes'][1].set_facecolor('#F1948A')

    # Statistical test annotation
    stat, pval = stats.mannwhitneyu(tgt_data, src_data, alternative='greater')
    sig_str = f'p = {pval:.2e}' + (' ✓sig.' if pval < 0.05 else ' ✗n.s.')
    ax.set_title(f'{sig_str}', fontsize=11)
    ax.set_ylabel('BL Reconstruction Residual $\Delta R_{BL}$', fontsize=10)
    ax.set_xlabel('Domain Type', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Median annotation
    for i, vals in enumerate([src_data, tgt_data], 1):
        med = np.median(vals)
        ax.text(i, med + 0.002, f'{med:.3f}', ha='center', va='bottom',
                fontsize=9, color='black', fontweight='bold')

    plt.tight_layout()
    path = figure_output_path(FILE_STEM, 0)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    log(f'  [Fig a saved] {path}')


def plot_heatmap(stat_df: pd.DataFrame):
    """图b：全部场景BL倍数变化热力图"""
    # 创建矩阵，行=场景，列=固定为1（仅6个场景，无需pivot）
    fig, ax = plt.subplots(figsize=(5, 6))

    # 按场景排序
    stat_df_sorted = stat_df.sort_values('场景').reset_index(drop=True)
    values = stat_df_sorted['倍数变化（跨域/域内）'].values.reshape(-1, 1)

    im = ax.imshow(values, cmap='RdYlBu_r', aspect='auto',
                   vmin=0.8, vmax=max(3.0, values.max()))
    plt.colorbar(im, ax=ax, label='跨域/域内 BL残差倍数')

    ax.set_xticks([0])
    ax.set_xticklabels(['倍数'], fontsize=10)
    ax.set_yticks(range(len(stat_df_sorted)))
    ax.set_yticklabels(stat_df_sorted['场景'].values, fontsize=9)
    ax.set_title('各场景Beer-Lambert约束违反倍数（>1表示跨域时违反加剧）', fontsize=11)

    # 在格子中标数值和显著性
    for i in range(len(stat_df_sorted)):
        val = values[i, 0]
        sig = '★' if stat_df_sorted.iloc[i]['显著性（p<0.05）'] == '是' else ''
        color = 'white' if val > 2.0 else 'black'
        ax.text(0, i, f'{val:.2f}×\n{sig}', ha='center', va='center',
                fontsize=10, color=color, fontweight='bold')

    plt.tight_layout()
    path = figure_output_path(FILE_STEM, 1)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    log(f'  [图b保存] {path}')


def plot_pca_variance(X_src, pca, scenario_name: str):
    """
    图c：PCA累积方差贡献率（验证n_components=10是否足够捕获BL成分结构）
    用第一个场景的源域数据绘制。
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax.plot(range(1, len(cumvar) + 1), cumvar, 'o-', color='#2980B9', linewidth=2)
    ax.axhline(y=95, color='red', linestyle='--', alpha=0.6, label='95%方差')
    ax.axhline(y=99, color='orange', linestyle='--', alpha=0.6, label='99%方差')
    ax.set_xlabel('主成分数', fontsize=10)
    ax.set_ylabel('累积解释方差（%）', fontsize=10)
    ax.set_title(f'BL成分分解充分性验证\n{scenario_name} 源域PCA方差贡献', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0.5, len(cumvar) + 0.5)
    plt.tight_layout()
    path = figure_output_path(FILE_STEM, 2)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    log(f'  [图c保存] {path}')


# ═══════════════════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════════════════

def main():
    log('=' * 65)
    log('实验：Beer-Lambert约束违反程度量化（2018/2025统一波段版）')
    log('假设：跨域时BL重建误差显著大于域内基线误差')
    log('=' * 65)

    N_COMPONENTS = 10   # PCA成分数，对应BL模型的"主要化学成分数"
    all_results = []
    first_year_payload = None

    # 年份取自案例研究的场景定义，而不是 AVAILABLE_YEARS。后者还含一个受限年份，
    # 它从未进入 602 个迁移场景、也不在 S9~S11 的任何一张表里；跟着遍历会让本节的
    # 同年一源一目标场景由 12 个变成 18 个，与案例研究其余各节的域集合脱节。
    with open(os.path.join(_CODE_DIR, '04_migration_scenarios.json'), encoding='utf-8') as _f:
        _scen = json.load(_f)
    _case_years = set()
    for _cfg in _scen.values():
        for _side in ('source', 'target'):
            for _d in _cfg[_side].get('domains', []):
                _case_years.add(int(_d['year']))
    years = tuple(y for y in AVAILABLE_YEARS if y in _case_years)
    log(f'案例研究场景涉及的年份：{years}（AVAILABLE_YEARS={AVAILABLE_YEARS}）')

    # ─── 加载案例研究涉及的各年处理后数据 ──────────────────────────────────
    for year in years:
        log(f'\n[加载{year}年处理后数据]')
        X, y, regions, wavelengths = load_processed_year_data(year, target_wavelengths='2025')
        year_scenarios = build_year_scenarios(year, regions)

        log(f'  样本数：{X.shape[0]}，波段数：{X.shape[1]}')
        log(f'  产地分布：{dict(zip(*np.unique(regions, return_counts=True)))}')
        log(f'  糖度范围：{np.nanmin(y):.2f} ~ {np.nanmax(y):.2f} °Brix，均值 {np.nanmean(y):.2f}')
        log(f'  场景数：{len(year_scenarios)}')

        year_df = run_bl_validation(X, y, regions, year_scenarios, n_components=N_COMPONENTS)
        all_results.append(year_df)

        if first_year_payload is None:
            first_year_payload = (year, X, y, regions, year_scenarios)

    df_result = pd.concat(all_results, ignore_index=True)
    log(f'\n[运行完成] 已处理 {df_result["scenario"].nunique()} 个场景')

    # ─── 统计检验 ──────────────────────────────────────────────────────────
    log('\n[统计检验：Mann-Whitney U检验]')
    stat_df = statistical_test(df_result)

    log('\n  场景级结果摘要：')
    for _, row in stat_df.iterrows():
        log(f"  {row['场景']:<18} "
            f"域内中位={row['源域中位BL残差']:.4f}  "
            f"跨域中位={row['目标域中位BL残差']:.4f}  "
            f"倍数={row['倍数变化（跨域/域内）']:.2f}×  "
            f"p={row['p值']}  {row['显著性（p<0.05）']}")

    # ─── 总体假设判断 ───────────────────────────────────────────────────────
    significant_ratio = (stat_df['显著性（p<0.05）'] == '是').mean()
    mean_fold = stat_df['倍数变化（跨域/域内）'].mean()
    log('\n' + '=' * 65)
    log(f'假设验证结论：')
    log(f'  显著场景比例：{significant_ratio * 100:.0f}% ({(stat_df["显著性（p<0.05）"]=="是").sum()}/{len(stat_df)}个场景)')
    log(f'  跨域BL残差倍数（均值）：{mean_fold:.2f}×')

    if significant_ratio >= 0.7 and mean_fold >= 1.3:
        log('  ✅ 假设成立：跨域时Beer-Lambert约束违反程度显著增大')
        log('  → 支持Nature Food路线，PhysicsDA的BL-Loss设计有物理依据')
    elif significant_ratio >= 0.5 or mean_fold >= 1.2:
        log('  ⚠️  假设部分成立：结果倾向支持，但证据不够强')
        log('  → 建议继续推进，但论文中降低对该假设的声索强度')
    else:
        log('  ❌ 假设不成立：BL违反程度在跨域时无显著增大')
        log('  → 建议调整PhysicsDA设计，BL-Loss可能需要重新定义')

    log('=' * 65)

    # ─── 可视化 ─────────────────────────────────────────────────────────────
    log('\n[生成图表]')
    plot_boxplot(df_result)
    if len(stat_df) > 0:
        plot_heatmap(stat_df)

    # PCA方差图（用第一个场景）
    if first_year_payload is not None:
        first_year, X_first, y_first, regions_first, scenarios_first = first_year_payload
        src_region = scenarios_first[0][0]
        src_mask = (regions_first == src_region) if isinstance(src_region, str) else np.isin(regions_first, src_region)
        X_src_first = X_first[src_mask]
        if X_src_first.shape[0] > 0:
            pca_check, scaler_check = fit_bl_components(X_src_first, n_components=20)
            plot_pca_variance(X_src_first, pca_check, scenarios_first[0][2])
            log(f'  前10个主成分累积方差：'
                f'{np.sum(pca_check.explained_variance_ratio_[:10]) * 100:.1f}%')

    # ─── 写出Excel ──────────────────────────────────────────────────────────
    log('\n[写出结果表格]')

    # 汇总统计
    summary_stats = df_result.groupby('domain_type')['bl_residual'].agg(
        ['mean', 'median', 'std', 'min', 'max']).round(4).reset_index()
    summary_stats.columns = ['域类型', '均值', '中位数', '标准差', '最小值', '最大值']

    write_excel_workbook(FILE_STEM, {
        0: ('BL违反统计检验', stat_df),
        1: ('各域BL残差分布', summary_stats),
        2: ('全量样本BL残差', df_result),
    })
    log(f'  [Excel保存] {FILE_STEM}.xlsx')

    log('\n实验完成。')
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beer-Lambert约束违反程度量化实验')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='设备选择: auto(自动检测), cpu, cuda, mps')
    args = parser.parse_args()

    device = get_device(args.device)
    log(f"设备: {device}")

    main()
