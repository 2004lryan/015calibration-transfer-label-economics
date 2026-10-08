"""
71_bl_vs_generic_distance.py: BL 残差放大 vs 通用域距离 —— 免标签风险预警的**增量价值**检验

【为什么必须做这个实验】
55 号已证明 BL 残差放大与迁移风险显著相关（各方法 RMSE 的 Spearman ρ 中位 ≈0.42），
但**从未证明它比通用域距离更好**。稿件 §4 末自己把这条列为 future work：
    "whether the BL-inspired residual has incremental value over generic domain-distance
     metrics (such as principal-component reconstruction distance or maximum mean
     discrepancy) in predicting cross-domain risk has not been directly compared here"
而 MMD 与 Wasserstein 在 §2.3 的 n* 分析里**已经算过**——即这是"可做而未做"。
三份模拟审稿（06doc/03REBUTTAL_PREP.md §四）交叉综述把它列为**最高优先级** P2。

若 BL 残差并不优于（甚至不如）通用距离，则"BL 启发的物理量提供了独特信号"这一层
必须撤下，主张退回到"某个域间结构差异度量与迁移风险相关"——通用距离也能做到。
这同样是一次**可能推翻自己主张**的检验（与 55 号同性质）。

【检验设计（实验前写定，禁止事后改口径）】
  受试预测子（全部免标签，只用源域/目标域光谱，不碰任何标签）：
    BL放大倍数        本文量：源域 PCA-10 基底下 median(目标域相对残差)/median(源域相对残差)
    目标域绝对残差     同一基底下目标域中位相对残差本身（不做比值）——检验"比值"结构是否必要
    MMD2_中位启发     RBF 核 MMD²，带宽用**该场景**的中位距离启发式
    MMD2_全局带宽     RBF 核 MMD²，带宽用**全场景池化**的中位距离（防"逐场景自适应"削弱基线）
    SlicedWasserstein 2000 个随机投影方向上的一维 Wasserstein-1 均值
    CORAL距离         ‖Cov_s − Cov_t‖_F（二阶矩距离，与稿中 CORAL 方法同族）
    均值漂移          ‖μ_s − μ_t‖₂（一阶矩）
    PCA子空间夹角      源/目标各自 top-10 主子空间的 Grassmann 距离（主角度均方根）

  结局 Y：该场景的**未校正**跨域 RMSE（PLSR/SVR/CNN/CNN+MMD/Phys+BL，跨 5 固定种子取均值）
  主检验：Spearman ρ；聚类单位=唯一目标集合（602 场景仅 62 个唯一目标集合）
  增量检验（本脚本的核心，55 号没有的部分）：
    ① 配对自助 Δρ = |ρ(BL放大)| − |ρ(基线)|，**同一批聚类重采样**同时算两者 → 95% CI
       CI 跨 0 = 无法认定 BL 更好（这正是审稿人要的判别性证据）
    ② 偏相关：秩变换后 ρ(BL放大, RMSE | 最强基线) → BL 是否还有独立贡献
    ③ 产地内秩相关（**不是**留一外推）：把场景按目标产地分组，看各预测子在**单个产地内部**
       对风险的排序能力。审计指出必须这样命名——本检验并未「在其余产地上拟合、预测留出产地」，
       它衡量的是「同一产地内不同场景之间能否排对风险次序」。稿件 §3.4 的
       'leave-one-origin-out cross-validation' 措辞（源自 55 号同款实现）有同一问题，须一并更正。

  判定（预先设定；末尾 verdict 逻辑必须与本段逐字对应——外部审计 F3 曾判 FAIL，
  原因是初版把这里的「对全部基线」在代码里放宽成了「胜多于负」的计数规则）：
    对**全部** 5 方法 × 7 基线 = 35 组，Δρ 的 95% CI 下界均 > 0
                                             → BL 残差有增量价值，可保留"物理量独特"表述
    存在 CI 跨 0 的组，但无任何一组 CI 上界 < 0，且偏相关显著
                                             → 只能说"不劣于通用距离、且含部分独立信息"
    存在 Δρ 上界 < 0 的组                     → 该组上 BL **劣于**通用距离，须如实报告

运行方式:
    cd <项目根目录>
    <venv>/bin/python 02code/71_bl_vs_generic_distance.py

输出文件:
    04outputs/71_bl_vs_generic_distance.xlsx
    05logs/71_bl_vs_generic_distance_<时间戳>.log
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, '..'))
_OUT = os.path.join(_BASE, '04outputs')
_LOG = os.path.join(_BASE, '05logs')
FILE_STEM = '71_bl_vs_generic_distance'

_spec = importlib.util.spec_from_file_location(
    'export_utils', os.path.join(_CODE, '01_export_utils.py'))
assert _spec is not None
assert _spec.loader is not None
_eu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eu)

FloatArray = npt.NDArray[np.float64]

N_COMPONENTS = 10                     # 与 03/55 号一致：PCA-10 作为 BL 可分解性低秩代理
FORMAL_SEEDS = [20060515, 20041210, 19810915, 2023, 2024]
METHODS = ['PLSR', 'SVR', 'CNN', 'CNN+MMD', 'Phys+BL']
N_PROJ = 2000                         # sliced Wasserstein 的随机投影数（F4 修订：200→2000，压掉投影近似噪声）
PRED_MAIN = 'BL放大倍数'
BASELINES = ['目标域绝对残差', 'MMD2_中位启发', 'MMD2_全局带宽',
             'SlicedWasserstein', 'CORAL距离', '均值漂移', 'PCA子空间夹角']
ALL_PREDICTORS = [PRED_MAIN, *BASELINES]


def make_logger() -> Callable[..., None]:
    os.makedirs(_LOG, exist_ok=True)
    path = os.path.join(_LOG, f'{FILE_STEM}_{datetime.now():%y-%m-%d_%H%M%S}.log')

    def log(msg: object = '') -> None:
        print(msg, flush=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(str(msg) + '\n')
    return log


# ─── 免标签预测子 ────────────────────────────────────────────────────────────
def bl_amplification(X_src: FloatArray, X_tgt: FloatArray) -> tuple[float, float, float]:
    """
    BL 残差放大倍数。**与 03/55 号逐行同口径**（只去均值、不做逐波段缩放），
    否则本脚本的主预测子与论文 §3.4/图 6 报告的量不是同一个，比较将失去意义。
    """
    scaler = StandardScaler(with_std=False).fit(X_src)
    Zs = scaler.transform(X_src)
    pca = PCA(n_components=min(N_COMPONENTS, X_src.shape[0] - 1, X_src.shape[1])).fit(Zs)

    def resid(X: FloatArray) -> FloatArray:
        Z = scaler.transform(X)
        Xr = pca.inverse_transform(pca.transform(Z)) + scaler.mean_
        num = np.linalg.norm(X - Xr, axis=1)
        den = np.maximum(np.linalg.norm(X, axis=1), 1e-10)
        return np.asarray(num / den, dtype=np.float64)

    r_src, r_tgt = resid(X_src), resid(X_tgt)
    return (float(np.median(r_tgt)) / max(float(np.median(r_src)), 1e-12),
            float(np.median(r_src)), float(np.median(r_tgt)))


# ⚠ 公平性修订（2026-08-10，外部审计 F4 判 FAIL 后）：
# 初版对 MMD 设了 MMD_MAX_N=400 的每域子采样上限、带宽只用 300 个池化样本，
# 而 BL 放大倍数用全部样本。审计 panel 一致指出：
# 这是**只向基线注入近似噪声**的不对称设置，在"BL 是否更优"的对抗比较里不可接受。
# 修法选择「加强基线」而不是「削弱 BL」——全部预测子一律用**该场景的全部样本**，
# MMD 与带宽都不再子采样。O(n²) 用 ‖a‖²+‖b‖²−2a·b 展开后完全跑得动。
SUBSAMPLE_DISABLED = True


def _pairwise_sq_dists(A: FloatArray, B: FloatArray) -> FloatArray:
    """‖a−b‖² = ‖a‖² + ‖b‖² − 2a·b。不可用广播相减——那会物化 m×n×d，本数据集直接爆内存。"""
    a2 = (A ** 2).sum(1)[:, None]
    b2 = (B ** 2).sum(1)[None, :]
    d = a2 + b2 - 2.0 * (A @ B.T)
    return np.asarray(np.maximum(d, 0.0), dtype=np.float64)


def mmd2_rbf(X_src: FloatArray, X_tgt: FloatArray, gamma: float) -> float:
    """无偏 MMD²（RBF 核）。gamma = 1/(2σ²)。**用全部样本**，不子采样（见 F4 修订说明）。"""
    Kss = np.exp(-gamma * _pairwise_sq_dists(X_src, X_src))
    Ktt = np.exp(-gamma * _pairwise_sq_dists(X_tgt, X_tgt))
    Kst = np.exp(-gamma * _pairwise_sq_dists(X_src, X_tgt))
    m, n = len(X_src), len(X_tgt)
    s = (Kss.sum() - np.trace(Kss)) / (m * (m - 1))
    t = (Ktt.sum() - np.trace(Ktt)) / (n * (n - 1))
    return float(s + t - 2.0 * Kst.mean())


def median_heuristic_gamma(X: FloatArray) -> float:
    """中位距离启发式带宽：σ² = median(‖xi−xj‖²)/2 → gamma = 1/(2σ²)。用全部样本。"""
    d2 = _pairwise_sq_dists(X, X)
    med = float(np.median(d2[np.triu_indices_from(d2, k=1)]))
    return 1.0 / max(med, 1e-12)


def sliced_wasserstein(X_src: FloatArray, X_tgt: FloatArray, rng: np.random.Generator,
                       n_proj: int = N_PROJ) -> float:
    """随机投影方向上的一维 Wasserstein-1 距离均值。"""
    d = X_src.shape[1]
    V = rng.normal(size=(d, n_proj))
    V /= np.linalg.norm(V, axis=0, keepdims=True)
    Ps, Pt = X_src @ V, X_tgt @ V
    vals = [stats.wasserstein_distance(Ps[:, j], Pt[:, j]) for j in range(n_proj)]
    return float(np.mean(vals))


def coral_distance(X_src: FloatArray, X_tgt: FloatArray) -> float:
    """‖Cov_s − Cov_t‖_F —— 与稿中 CORAL 方法同族的二阶矩距离。"""
    return float(np.linalg.norm(np.cov(X_src, rowvar=False) - np.cov(X_tgt, rowvar=False), 'fro'))


def pca_subspace_distance(X_src: FloatArray, X_tgt: FloatArray, k: int = N_COMPONENTS) -> float:
    """源/目标 top-k 主子空间的 Grassmann 距离（主角度的均方根）。"""
    kk = int(min(k, X_src.shape[0] - 1, X_tgt.shape[0] - 1, X_src.shape[1]))
    Us = PCA(n_components=kk).fit(X_src).components_.T
    Ut = PCA(n_components=kk).fit(X_tgt).components_.T
    sv = np.clip(np.linalg.svd(Us.T @ Ut, compute_uv=False), -1.0, 1.0)
    return float(np.sqrt(np.sum(np.arccos(sv) ** 2)))


# ─── 统计工具 ────────────────────────────────────────────────────────────────
def cluster_boot_indices(clusters: npt.NDArray[np.str_], n_boot: int,
                         seed: int) -> list[npt.NDArray[np.int64]]:
    """
    预先生成 n_boot 组按聚类单位重采样的行索引。
    **所有预测子共用同一批索引**——这是配对 Δρ 检验能成立的前提
    （若各自独立重采样，两条 ρ 的自助分布不再配对，差值 CI 会被高估）。
    """
    uniq = np.unique(clusters)
    pos = {c: np.where(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        out.append(np.concatenate([pos[c] for c in pick]))
    return out


def spearman_ci(x: FloatArray, y: FloatArray,
                boot_idx: list[npt.NDArray[np.int64]]) -> tuple[float, float, float, FloatArray]:
    """
    点估计 ρ + 按给定重采样索引的 95% CI + **与 boot_idx 等长、跳过处填 NaN** 的自助值数组。
    等长对齐是配对 Δρ 的前提：两个预测子在第 k 次重采样上必须对应同一批行。
    """
    rho = float(stats.spearmanr(x, y).statistic)
    arr = np.full(len(boot_idx), np.nan, dtype=np.float64)
    for k, idx in enumerate(boot_idx):
        if len(np.unique(x[idx])) < 3:
            continue
        r = stats.spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(r):
            arr[k] = r
    ok = arr[np.isfinite(arr)]
    if len(ok) < 50:
        return rho, np.nan, np.nan, arr
    return rho, float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5)), arr


def partial_spearman(x: FloatArray, y: FloatArray, z: FloatArray) -> tuple[float, float]:
    """秩变换后控制 z 的偏相关：ρ(x, y | z)。"""
    rx, ry, rz = (stats.rankdata(v).astype(np.float64) for v in (x, y, z))

    def resid(a: FloatArray) -> FloatArray:
        Z = np.column_stack([np.ones_like(rz), rz])
        beta, *_ = np.linalg.lstsq(Z, a, rcond=None)
        return np.asarray(a - Z @ beta, dtype=np.float64)

    ex, ey = resid(rx), resid(ry)
    r = stats.pearsonr(ex, ey)
    return float(r.statistic), float(r.pvalue)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=11)
    args = ap.parse_args()
    log = make_logger()

    log('=' * 88)
    log('71 - BL 残差放大 vs 通用域距离：免标签风险预警的增量价值检验（返修 P2）')
    log('=' * 88)

    # ── 1. 逐场景计算全部免标签预测子 ──────────────────────────────────
    data = _eu.load_year_region_map(years=_eu.AVAILABLE_YEARS, target_wavelengths='2025')
    with open(os.path.join(_CODE, '04_migration_scenarios.json'), encoding='utf-8') as f:
        scenarios: dict[str, Any] = json.load(f)

    # 注：F4 修订取消子采样后，本函数内不再需要顶层 rng——
    # 唯一还用随机性的是 sliced Wasserstein 的投影方向，它在循环内按场景确定性地取。

    # 全局带宽：先池化所有域的光谱估一次中位距离，全场景共用
    pooled = np.vstack([np.asarray(v['X'], dtype=np.float64) for v in data.values()])
    gamma_global = median_heuristic_gamma(pooled)
    log(f'\n全局 RBF 带宽（池化中位启发式）gamma = {gamma_global:.6g}')

    rows: list[dict[str, Any]] = []
    for i, (sid, cfg) in enumerate(scenarios.items()):
        try:
            src = _eu.merge_domain_payload(data, _eu.extract_domain_keys(cfg['source']))
            tgt = _eu.merge_domain_payload(data, _eu.extract_domain_keys(cfg['target']))
        except KeyError:
            continue
        Xs = np.asarray(src['X'], dtype=np.float64)
        Xt = np.asarray(tgt['X'], dtype=np.float64)
        if len(Xs) < 15 or len(Xt) < 15:
            continue
        amp, r_src, r_tgt = bl_amplification(Xs, Xt)
        # 每个场景一个确定性 rng：两个 MMD 变体因此看到**同一批**子采样，
        # 差异只来自带宽选择，不来自抽样噪声。
        srng = np.random.default_rng([args.seed, i])
        g_local = median_heuristic_gamma(np.vstack([Xs, Xt]))
        rows.append({
            '场景': sid, '迁移类型': cfg.get('type', '未知'),
            '目标集合': '+'.join(sorted(f'{y}_{r}' for y, r in _eu.extract_domain_keys(cfg['target']))),
            '目标产地': '+'.join(sorted({r for _, r in _eu.extract_domain_keys(cfg['target'])})),
            PRED_MAIN: amp,
            '目标域绝对残差': r_tgt,
            'MMD2_中位启发': mmd2_rbf(Xs, Xt, g_local),
            'MMD2_全局带宽': mmd2_rbf(Xs, Xt, gamma_global),
            'SlicedWasserstein': sliced_wasserstein(Xs, Xt, srng),
            'CORAL距离': coral_distance(Xs, Xt),
            '均值漂移': float(np.linalg.norm(Xs.mean(0) - Xt.mean(0))),
            'PCA子空间夹角': pca_subspace_distance(Xs, Xt),
            '源域中位残差': r_src, 'n_src': len(Xs), 'n_tgt': len(Xt),
        })
    A = pd.DataFrame(rows)
    log(f'预测子已算出：{len(A)} 个场景 | 唯一目标集合 {A["目标集合"].nunique()} 个')
    log('\n各预测子的取值范围：')
    for p in ALL_PREDICTORS:
        log(f'  {p:20s} {A[p].min():12.4g} ~ {A[p].max():12.4g}')

    # 预测子之间的秩相关（看它们是不是同一个东西的不同写法）
    log('\n预测子两两 Spearman 相关（|ρ|>0.9 说明高度冗余）：')
    corr = A[ALL_PREDICTORS].corr(method='spearman')
    log(corr.round(3).to_string())

    # ── 2. 结局：各方法未校正跨域 RMSE ──────────────────────────────────
    raw = pd.read_excel(os.path.join(_OUT, '50_formal_multiseed_benchmark.xlsx'),
                        sheet_name='表h：全部原始结果')
    raw = raw[raw['种子'].isin(FORMAL_SEEDS)]
    per = (raw[raw['校正'] == '裸'].groupby(['方法', '场景'], as_index=False)['RMSE'].mean())
    # 案例研究的聚合量一律取七个未校正方法都有结果的共同场景（补充材料 S3）：
    # 各方法在各自跑出的场景上算，难场景没跑完的方法会显得更好，跨方法也不可比。
    bare7 = ['PLSR', 'SVR', 'CNN', 'CNN+MMD', 'CNN+BL', 'Phys', 'Phys+BL']
    wide = per.pivot(index='场景', columns='方法', values='RMSE')
    common = wide[bare7].dropna().index
    per = per[per['场景'].isin(common)]
    log(f'共同场景（七个未校正方法都有结果）：{len(common)} 个')

    # ── 3. 主分析：逐方法逐预测子的 ρ + 配对 Δρ ────────────────────────
    log('\n' + '=' * 88)
    log('检验一：各预测子对未校正跨域 RMSE 的预测力（聚类单位=唯一目标集合）')
    log('=' * 88)

    res_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    partial_rows: list[dict[str, Any]] = []

    for m in METHODS:
        sub = per[per['方法'] == m][['场景', 'RMSE']]
        j = A.merge(sub, on='场景', how='inner')
        if len(j) < 20:
            continue
        boot_idx = cluster_boot_indices(j['目标集合'].to_numpy().astype(str),
                                        args.n_boot, args.seed)
        y = j['RMSE'].to_numpy(dtype=np.float64)

        boots: dict[str, FloatArray] = {}
        log(f'\n【结局 = {m} 未校正 RMSE】  n = {len(j)}')
        log(f'{"预测子":20s} {"Spearman ρ":>11s} {"95% CI":>18s} {"P":>10s}')
        for p in ALL_PREDICTORS:
            x = j[p].to_numpy(dtype=np.float64)
            rho, lo, hi, arr = spearman_ci(x, y, boot_idx)
            pv = float(stats.spearmanr(x, y).pvalue)
            boots[p] = arr
            mark = ' ←本文量' if p == PRED_MAIN else ''
            log(f'{p:20s} {rho:11.3f} [{lo:6.3f},{hi:6.3f}] {pv:10.2e}{mark}')
            res_rows.append({'方法': m, '预测子': p, 'n': len(j), 'Spearman_rho': rho,
                             'CI_low': lo, 'CI_high': hi, 'P': pv})

        # 配对 Δ|ρ|（同一批重采样，逐次配对后再取分位）
        log(f'{"配对 Δ|ρ| = |BL放大| − |基线|":34s} {"Δ点估计":>9s} {"95% CI":>18s}  判定')
        r_main = abs(float(stats.spearmanr(j[PRED_MAIN].to_numpy(dtype=np.float64), y).statistic))
        for p in BASELINES:
            paired = np.abs(boots[PRED_MAIN]) - np.abs(boots[p])
            d = paired[np.isfinite(paired)]
            r_base = abs(float(stats.spearmanr(j[p].to_numpy(dtype=np.float64), y).statistic))
            lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            verdict = ('✅BL更强' if lo > 0 else ('🔴BL更弱' if hi < 0 else '⚪无法区分'))
            log(f'  vs {p:28s} {r_main - r_base:9.3f} [{lo:6.3f},{hi:6.3f}]  {verdict}')
            delta_rows.append({'方法': m, '基线': p, 'delta_点估计': r_main - r_base,
                               'CI_low': lo, 'CI_high': hi, '判定': verdict})

        # 偏相关：控制住最强基线后 BL 还剩多少
        best_base = max(BASELINES,
                        key=lambda p: abs(float(stats.spearmanr(j[p].to_numpy(dtype=np.float64), y).statistic)))
        pr, pp = partial_spearman(j[PRED_MAIN].to_numpy(dtype=np.float64), y,
                                  j[best_base].to_numpy(dtype=np.float64))
        log(f'  偏相关 ρ(BL放大, RMSE | {best_base}) = {pr:.3f}  (P={pp:.2e})')
        partial_rows.append({'方法': m, '最强基线': best_base, '偏相关_rho': pr, 'P': pp})

    res = pd.DataFrame(res_rows)
    delta = pd.DataFrame(delta_rows)
    partial = pd.DataFrame(partial_rows)

    # ── 4. 产地内秩相关（不是留一外推，见文件头 ③）────────────────────
    log('\n' + '=' * 88)
    log('检验二：产地内秩相关 —— 把场景按目标产地分组，各预测子在单个产地内部的风险排序能力')
    log('（注意：这**不是**留一外推——没有在其余产地上拟合再预测留出产地。稿件 §3.4 的'
        "'leave-one-origin-out' 措辞同源同问题，须一并更正）")
    log('=' * 88)
    loo_rows: list[dict[str, Any]] = []
    for m in METHODS:
        sub = per[per['方法'] == m][['场景', 'RMSE']]
        j = A.merge(sub, on='场景', how='inner')
        for p in ALL_PREDICTORS:
            vals = []
            for o in sorted(j['目标产地'].unique()):
                te = j[j['目标产地'] == o]
                if len(te) < 8 or te[p].nunique() < 3:
                    continue
                r = stats.spearmanr(te[p], te['RMSE']).statistic
                if np.isfinite(r):
                    vals.append(float(r))
            if vals:
                loo_rows.append({'方法': m, '预测子': p, '产地数': len(vals),
                                 '中位_rho': float(np.median(vals)),
                                 'rho大于0占比%': float(np.mean(np.array(vals) > 0) * 100)})
    loo = pd.DataFrame(loo_rows)
    if not loo.empty:
        piv = loo.pivot(index='预测子', columns='方法', values='中位_rho').reindex(ALL_PREDICTORS)
        piv['跨方法中位'] = piv.median(axis=1)
        log(piv.round(3).to_string())

    # ── 5. 结论 ────────────────────────────────────────────────────────
    log('\n' + '=' * 88)
    log('结论（对照预先设定的判定标准）')
    log('=' * 88)
    n_win = int((delta['判定'] == '✅BL更强').sum())
    n_lose = int((delta['判定'] == '🔴BL更弱').sum())
    n_tie = int((delta['判定'] == '⚪无法区分').sum())
    log(f'  配对 Δ|ρ| 共 {len(delta)} 组（5 方法 × {len(BASELINES)} 基线）：'
        f'BL 更强 {n_win} / 无法区分 {n_tie} / BL 更弱 {n_lose}')
    med_partial = float(partial['偏相关_rho'].median()) if not partial.empty else float('nan')
    log(f'  控制最强基线后的偏相关中位数：{med_partial:.3f}')

    # 判定逻辑必须与文件头的预注册逐字对应（外部审计 F3 判 FAIL 后重写）：
    # 三档的触发条件分别是「全部 35 组 CI 下界 >0」「无任何一组 CI 上界 <0 且偏相关显著」
    # 「存在 CI 上界 <0 的组」。禁止再用"胜多于负"这类事后计数规则。
    n_partial_sig = int((partial['P'] < 0.05).sum()) if not partial.empty else 0
    if n_win == len(delta):
        verdict = '✅ 全部 35 组 CI 下界均 >0 → BL 有增量价值，"物理启发量独特"可保留'
    elif n_lose > 0:
        verdict = (f'🔴 有 {n_lose} 组 Δρ 的 CI 上界 <0（BL 在该组劣于通用距离）→ '
                   '须在稿中如实报告这些组，不得只报有利的组')
    elif n_partial_sig >= 1:
        verdict = (f'⚠ 无任何一组显示 BL 更弱（更强 {n_win} / 无法区分 {n_tie} / 更弱 0），'
                   f'且 {n_partial_sig}/{len(partial)} 个方法上控制最强基线后偏相关仍显著 → '
                   '可写"BL 残差不劣于通用域距离，并含部分独立信息"；'
                   '**不得**写成"对全部基线更优"')
    else:
        verdict = ('⚠ 无任何一组显示 BL 更弱，但偏相关均不显著 → '
                   '只能写"与通用域距离相当"，不得暗示物理量更优')
    log(f'\n  {verdict}')

    os.makedirs(_OUT, exist_ok=True)
    out = os.path.join(_OUT, f'{FILE_STEM}.xlsx')
    with pd.ExcelWriter(out) as w:
        A.to_excel(w, sheet_name='表a：逐场景免标签预测子', index=False)
        corr.to_excel(w, sheet_name='表b：预测子间秩相关')
        res.to_excel(w, sheet_name='表c：各预测子预测力', index=False)
        delta.to_excel(w, sheet_name='表d：配对Δrho', index=False)
        partial.to_excel(w, sheet_name='表e：偏相关', index=False)
        loo.to_excel(w, sheet_name='表f：留一产地', index=False)
        pd.DataFrame([{'判定': verdict, 'BL更强组数': n_win, '无法区分': n_tie,
                       'BL更弱组数': n_lose, '偏相关中位': med_partial}]).to_excel(
            w, sheet_name='表g：结论', index=False)
    log(f'\n已保存: {out}')


if __name__ == '__main__':
    main()
