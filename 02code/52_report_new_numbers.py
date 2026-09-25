"""
52_report_new_numbers.py: 从 50 号 Formal 基准结果生成论文所需的全部新数字

把 04outputs/50_formal_multiseed_benchmark.xlsx 里的结果，翻译成论文
（02code/content_csae.py）直接可用的形式：

  · 新的 TBL1 / TBL2 / TBL3（可直接粘贴的 Python 字面量）
  · 正文 RES_22 / RES_23 / RES_24 / 摘要 / 结论 里每一个数字的新值
  · 新旧对照，并标出**结论方向是否发生改变**（这是最关键的一项：
    论文原结论"校正后 Phys+BL 显著劣于 PLSR"建立在单种子 100 场景上，
    换 5 种子 602 场景后可能不再成立，必须显式检查而不是默认沿用）

运行方式:
    cd <项目根目录>
    python 02code/52_report_new_numbers.py

输出文件:
    04outputs/52_report_new_numbers.md   — 新旧数字对照 + 可粘贴的表格字面量
"""

import os
import pandas as pd
import numpy as np

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, '..'))
_OUT = os.path.join(_BASE, '04outputs')
_XLSX = os.path.join(_OUT, '50_formal_multiseed_benchmark.xlsx')

# 论文当前（旧）数字，用于对照
OLD = {
    'PLSR': 3.987, 'SVR': 3.065, 'CNN': 2.577, 'CNN+MMD': 2.642, 'Phys+BL': 1.976,
    'improve_CNN': 23.4, 'paired_mean': 22.56, 'paired_median': 23.09,
    'cnnbl_delta': -4.18, 'sb_gain_PLSR': 70.4, 'sb_gain_CNN': 34.3, 'sb_gain_Phys': 6.7,
}


def fmt_p(p):
    """P 值排版：论文用上标形式。"""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return 'n/a'
    if p >= 0.001:
        return f'{p:.3f}'
    e = int(np.floor(np.log10(p)))
    m = p / 10 ** e
    sup = str(e).replace('-', '⁻')
    for d, s in zip('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'):
        sup = sup.replace(d, s)
    return f'{m:.1f}×10{sup}'


def main():
    if not os.path.exists(_XLSX):
        raise SystemExit(f'未找到 {_XLSX}——请先跑完 50 号并 --merge')

    def sheet(name, index=None):
        """空 sheet 要容错：锚点可能与本脚本并行跑，此时表d 为空表（无列名）。"""
        try:
            df = pd.read_excel(_XLSX, sheet_name=name)
        except Exception:
            return pd.DataFrame()
        if df.empty or (index and index not in df.columns):
            return pd.DataFrame()
        return df.set_index(index) if index else df

    A = sheet('表a：按方法汇总', '方法')
    A2 = sheet('表a2：共同场景子集', '方法')
    C = sheet('表c：预注册配对检验', '假设')
    D = sheet('表d：seed42复现锚点', '方法')
    E = sheet('表e：算力账')
    G = sheet('表g：每场景跨种子均值')
    if A.empty or C.empty:
        raise SystemExit('🔴 表a 或表c 为空——50 号 merge 未正常产出，请先检查分片')

    L = []
    P = L.append
    P('# 52_论文新数字对照表（源自 50 号 Formal 基准）\n')
    P(f'数据源：`04outputs/50_formal_multiseed_benchmark.xlsx`\n')

    # ── 0. 复现锚点：该计划已作废，此处只留退役说明 ─────────────────────
    #   原计划是「seed=42 必须与旧的 18_exp 逐场景一致，否则新数字不得入稿」。它不成立：
    #   旧 18_exp 算在脏数据上（SSC=56.07 的越界样本 + 31 行光谱-标签错配），而 50 号跑在
    #   清洗后的数据上；要求干净数据去复现被污染的旧结果，等于把污染当成正确性基准。
    #   56 号 step1 已于 2026-09-11（commit b64e82f）把锚点改成「与 git 里上一版合并表逐格
    #   比差异分布」。这份报告此前仍印着旧的禁令，与稿件现状矛盾，故改为写明退役。
    P('## 0. seed=42 复现锚点（已作废，保留说明）\n')
    if D.empty:
        P('> **该锚点已于 2026-09-11 作废（commit `b64e82f`；理由见 `02code/56_post_experiment_pipeline.py` '
          '模块串 step1）**：42 不在正式种子集 `[20060515, 20041210, 19810915, 2023, 2024]` 内，'
          '而旧的 18_exp 算在脏数据上（含 SSC=56.07 的越界样本与 31 行光谱-标签错配的重复行）——'
          '要求清洗后的数据去复现被污染的旧结果，等于把污染当成正确性基准。'
          '故本表为空不再构成「新数字不得写入论文」的阻塞。'
          '现行的等价性核验是 56 号 step1：把本版合并表与 git 里上一版逐格比、报告差异分布，'
          '并核对差异是否符合数据清洗的预期效应。\n')
    else:
        P('| 方法 | 新(seed42) | 旧 18_exp | 偏差 | 判定 |')
        P('|---|---|---|---|---|')
        for m, r in D.iterrows():
            old = r.get('旧18exp_RMSE')
            old_s = f'{old:.3f}' if pd.notna(old) else '—'
            dev = f'{r["偏差"]:+.3f}' if pd.notna(r.get('偏差')) else '—'
            P(f'| {m} | {r["RMSE_mean"]:.3f} | {old_s} | {dev} | {r["复现"]} |')
        bad = D[(D['复现'] == '⚠️不一致')]
        if len(bad):
            P(f'\n> 🔴 **{len(bad)} 个方法未复现旧结果**——两套代码不等价，须排查后再用新数字。\n')
        else:
            P('\n> ✅ 全部复现，新代码与旧 18_exp 等价，新数字可信。\n')

    # ── 1. 表1 ──────────────────────────────────────────────────────────
    n_common = int(A2['n_scenarios'].iloc[0]) if len(A2) else 0
    P(f'\n## 1. 新 TBL1（跨产地直接迁移，5 方法）\n')
    P(f'口径：**共同场景子集 n={n_common}**（所有方法 × 全部种子均跑通，消除幸存者偏差）；'
      f'RMSE 为 mean±std（跨场景），CI 为 cluster bootstrap 95%。\n')
    P('```python')
    P('TBL1 = [')
    for m in ['PLSR', 'SVR', 'CNN', 'CNN+MMD', 'Phys+BL']:
        if m not in A2.index:
            continue
        r = A2.loc[m]
        P(f"    ['{m}', '{r['RMSE_mean']:.3f}±{r['RMSE_std']:.3f}', "
          f"'[{r['RMSE_CI95_low']:.3f}, {r['RMSE_CI95_high']:.3f}]', "
          f"'{r['R2_mean']:.3f}', '{r['RPD_mean']:.3f}', '{r['RPIQ_mean']:.3f}', "
          f"'{r['Bias_mean']:+.2f}', '{int(r['R2大于0场景数'])}/{int(r['n_scenarios'])}'],")
    P(']')
    P('```\n')
    P('新旧对照（RMSE 均值）：\n')
    P('| 方法 | 新（5种子, 共同场景） | 旧（单种子, 各自场景） | 变化 |')
    P('|---|---|---|---|')
    for m in ['PLSR', 'SVR', 'CNN', 'CNN+MMD', 'Phys+BL']:
        if m in A2.index:
            new = A2.loc[m, 'RMSE_mean']
            P(f'| {m} | {new:.3f} | {OLD[m]:.3f} | {new - OLD[m]:+.3f} |')

    # ── 2. 表2：架构-约束协同设计消融（H6/H7）───────────────────────────
    P('\n## 2. 新 TBL2（架构–约束协同设计消融）\n')
    P('```python')
    P('TBL2 = [')
    for hid, label in [('H6', 'CNN+BL vs CNN'), ('H7', 'Phys+BL vs Phys')]:
        if hid not in C.index:
            continue
        r = C.loc[hid]
        P(f"    ['{label}', '{r['A_RMSE_mean']:.3f}', '{r['B_RMSE_mean']:.3f}', "
          f"'{r['相对改进_%']:+.2f}', '{r['Cliffs_delta']:+.3f}', "
          f"'{fmt_p(r['Wilcoxon_p_holm'])}', '{int(r['n_场景'])}'],")
    P(']')
    P('```\n')
    P('> 列：对比 / A的RMSE / B的RMSE / 相对改进% / Cliff\'s δ / P(Holm) / n\n')
    P('**方向核验（这是表2 的全部意义）**：\n')
    for hid, expect in [('H6', 'BL 加在通用 CNN 上应当**有害**（相对改进为负）'),
                        ('H7', 'BL 加在物理架构上应当**有益**（相对改进为正）')]:
        if hid not in C.index:
            continue
        r = C.loc[hid]
        imp, p = r['相对改进_%'], r['Wilcoxon_p_holm']
        sig = '显著' if p < 0.05 else '**不显著**'
        P(f'- {hid}（{r["A"]} vs {r["B"]}）：预期 {expect}；'
          f'实测相对改进 {imp:+.2f}%，P(Holm)={fmt_p(p)}（{sig}），'
          f'Cliff\'s δ={r["Cliffs_delta"]:+.3f}（{r["效应量级"]}）')
    P('')

    # ── 3. 表3：SB 校正（论文的关键反转点）──────────────────────────────
    P('\n## 3. 新 TBL3（slope/bias 校正对照）——**论文结论的胜负手**\n')
    P('```python')
    P('TBL3 = [')
    for m in ['PLSR', 'SVR', 'CNN', 'Phys+BL']:
        if m not in A.index or f'{m} + SB' not in A.index:
            continue
        bare, sb = A.loc[m], A.loc[f'{m} + SB']
        gain = (bare['RMSE_mean'] - sb['RMSE_mean']) / bare['RMSE_mean'] * 100
        # 列序必须与 content_csae.py 表3 的表头一致：
        #   方法 / 未校正RMSE / 校正后RMSE / 校正增益% / 校正后RPD / 校正后R² / 校正后误判率%
        P(f"    ['{m}', '{bare['RMSE_mean']:.3f}', '{sb['RMSE_mean']:.3f}', "
          f"'{gain:+.1f}', '{sb['RPD_mean']:.3f}', '{sb['R2_mean']:+.3f}', "
          f"'{sb['品质误判率_mean']*100:.1f}'],")
    P(']')
    P('```\n')
    P('> 列：方法 / 未校正RMSE / 校正后RMSE / 校正增益% / 校正后RPD / 校正后R² / 校正后误判率%\n')
    P('> （列序已与 content_csae.py 表3 表头逐列对齐，可直接粘贴）\n')

    P('\n### 3.1 结论方向核验（**最关键**）\n')
    P('论文原结论建立在单种子 100 场景上：「校正后 Phys+BL 显著劣于 PLSR 与 CNN，物理先验的收益'
      '可被一次经典校正替代并超越」。现在用 5 种子 602 场景重新检验：\n')
    P('| 假设 | 对比 | 相对改进% | P(Holm) | Cliff\'s δ | 结论 |')
    P('|---|---|---|---|---|---|')
    verdicts = {}
    for hid in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7']:
        if hid not in C.index:
            continue
        r = C.loc[hid]
        p = r['Wilcoxon_p_holm']
        imp = r['相对改进_%']
        if p >= 0.05:
            v = '无显著差异'
        elif imp > 0:
            v = f'{r["A"]} 显著更优'
        else:
            v = f'**{r["B"]} 显著更优**'
        verdicts[hid] = (imp, p, v)
        P(f'| {hid} | {r["A"]} vs {r["B"]} | {imp:+.2f} | {fmt_p(p)} | '
          f'{r["Cliffs_delta"]:+.3f} | {v} |')

    P('\n**论文结论是否需要改写？**\n')
    if 'H2' in verdicts:
        imp, p, _ = verdicts['H2']
        if p < 0.05 and imp < 0:
            P('- ✅ **H2 复现原结论**：校正后 Phys+BL 仍显著劣于 PLSR+SB。'
              '论文的"反转"论点在 5 种子 602 场景上依然成立，可沿用（数字需更新）。')
        elif p >= 0.05:
            P('- 🔴 **H2 推翻原结论**：校正后 Phys+BL 与 PLSR+SB **无显著差异**。'
              '原文"Phys+BL 反显著劣于 PLSR"的表述**必须删除**，改为'
              '"校正后各方法收敛至统计上无法区分的同一水平"——'
              '这仍支持"物理先验的收益可被经典校正替代"，但**不支持"并超越"**。')
        else:
            P('- 🔴 **H2 反向**：校正后 Phys+BL 反而显著**优于** PLSR+SB。'
              '原文的核心论点被推翻，必须彻底重写第 2.4 节、摘要与结论。')
    if 'H1' in verdicts:
        imp, p, _ = verdicts['H1']
        if p < 0.05 and imp > 0:
            P(f'- ✅ **H1 复现**：零标签下 Phys+BL 仍显著优于 CNN（改进 {imp:.2f}%）。')
        else:
            P(f'- 🔴 **H1 未复现**：零标签下 Phys+BL 相对 CNN 的优势不再显著'
              f'（改进 {imp:+.2f}%, P={fmt_p(p)}）。**这会动摇论文的立论基础**，须重写。')

    # ── 4. 正文数字替换清单 ─────────────────────────────────────────────
    P('\n## 4. 正文数字逐项替换清单\n')
    P('| 位置 | 旧值 | 新值 |')
    P('|---|---|---|')
    if 'Phys+BL' in A2.index and 'CNN' in A2.index:
        pb, cn = A2.loc['Phys+BL', 'RMSE_mean'], A2.loc['CNN', 'RMSE_mean']
        P(f'| Phys+BL 均值 RMSE | 1.976 | {pb:.3f} |')
        P(f'| CNN 均值 RMSE | 2.577 | {cn:.3f} |')
    if 'H1' in C.index:
        r = C.loc['H1']
        P(f'| 相对 CNN 改进 | 23.4% / 配对 22.56% | {r["相对改进_%"]:.2f}% |')
        P(f'| 该改进的 P 值 | 8.5×10⁻⁸⁷ | {fmt_p(r["Wilcoxon_p_holm"])} |')
        P(f'| 配对场景数 | 569 | {int(r["n_场景"])} |')
    if 'H6' in C.index:
        r = C.loc['H6']
        P(f'| CNN+BL 相对 CNN | −4.18% (n=571) | {r["相对改进_%"]:+.2f}% (n={int(r["n_场景"])}) |')
    if 'H7' in C.index:
        r = C.loc['H7']
        P(f'| Phys+BL 相对 Phys | +1.61% (n=551) | {r["相对改进_%"]:+.2f}% (n={int(r["n_场景"])}) |')
    for m, old_g in [('PLSR', 70.4), ('CNN', 34.3), ('Phys+BL', 6.7)]:
        if m in A.index and f'{m} + SB' in A.index:
            g = (A.loc[m, 'RMSE_mean'] - A.loc[f'{m} + SB', 'RMSE_mean']) / A.loc[m, 'RMSE_mean'] * 100
            P(f'| {m} 的校正增益 | {old_g}% | {g:+.1f}% |')
    sb_rows = [A.loc[f'{m} + SB'] for m in ['PLSR', 'SVR', 'CNN', 'Phys+BL'] if f'{m} + SB' in A.index]
    if sb_rows:
        sb = pd.DataFrame(sb_rows)
        P(f'| 校正后 RMSE 区间 | 1.62~1.65 | {sb["RMSE_mean"].min():.3f}~{sb["RMSE_mean"].max():.3f} |')
        P(f'| 校正后 RPD 区间 | 0.965~0.986 | {sb["RPD_mean"].min():.3f}~{sb["RPD_mean"].max():.3f} |')
        P(f'| 校正后误判率 | 35.2%~36.5% | {sb["品质误判率_mean"].min()*100:.1f}%~'
          f'{sb["品质误判率_mean"].max()*100:.1f}% |')

    # ── 5. 方法学口径（写进 Methods，审稿人会查）───────────────────────
    n_seed = G['场景'].nunique() if len(G) else 0
    P('\n## 5. 新的方法学口径（须写入 Methods，替换"单种子/100场景"的旧表述）\n')
    P(f'- 场景：全量 **602** 个（不再是 100 个分层采样）')
    P('- 种子：**5 粒** `[20060515, 20041210, 19810915, 2023, 2024]`；'
      'seed=42 的复现锚点已作废，见本报告第 0 节')
    P(f'- 聚类单位：**场景**。同场景的 5 个种子是重复测量，非独立样本；'
      f'先按场景对种子取均值，n_primary = 场景数（**不得**写成 602×5=3010）')
    P(f'- 检验：Wilcoxon signed-rank（主）+ paired t（辅），Holm 族内校正（7 条预注册假设）')
    P(f'- 效应量：Cliff\'s δ（主）+ Cohen\'s d（辅）')
    P(f'- CI：cluster bootstrap（按场景重采样 1000 次）')
    if len(E):
        e = E.iloc[0]
        P(f'- 算力：{e.get("GPU型号", "")} ×1，wall-clock {e.get("总_wall_clock_h", "?")} h，'
          f'累计 {e.get("累计_GPU_hours", "?")} GPU·h，峰值显存 {e.get("峰值显存_GB", "?")} GB')

    md = '\n'.join(L)
    out = os.path.join(_OUT, '52_report_new_numbers.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(md)
    print(md)
    print(f'\n\n已保存: {out}')


if __name__ == '__main__':
    main()
