"""
49_plot_csae_figures.py：《农业工程学报》投稿图（图1～图3）

═══ 图表契约（先定契约，再写代码；nature-figure 规程）═══

图1  主张：跨产地比尔–朗伯加性可分解性系统性失稳，且该失稳不依赖低秩基底的选择。
     a 哑铃图 —— 12 个场景各一行，源域中位残差 ○ → 目标域中位残差 ●，连线示放大。
        选哑铃而非并排箱线：本现象的本质是**同一场景内的配对放大**，
        并排箱线只呈现两个边缘分布，会把配对关系丢掉。
     b 一致性散点 —— PCA-10 基底的放大倍数 vs 其余基底，1:1 参考线 + 边际须。

图2  主张：未校正时物理架构在典型场景（中位数）下误差最低，但少数运行发散，均值被拉高。
     a Raincloud（半小提琴 + 箱线 + 抖动散点 + 均值标记）—— 5 方法在全部场景上的 RMSE 分布；
        箱线给中位数与四分位距，标记给均值，二者的差距就是发散运行的影响。纵轴裁在 10 °Brix，
        轴外场景数标在顶部；小提琴的核密度只用轴内的值估计，否则几个上千 °Brix 的离群值
        会把带宽撑大，整条小提琴变成一块矩形。
     b Forest plot（中位数 ± 95% bootstrap CI）—— 按 4 类迁移类型分层。
        01CLAUDE.md §7.5 钦定：正文主图用 forest/dot plot 含 95% CI。用中位数而非均值：
        发散运行把个别类型的均值推到几十 °Brix，均值图只剩一条横线，读不出典型场景的排序。

图3  主张：用同一批目标标签做一次斜率/偏置校正后，各方法收敛到一条窄带，
     物理架构不再领先，经典基线反超。
     a 哑铃图 —— 各方法中位 RMSE 从"未校正"到"校正后"的移动（95% bootstrap CI）；
        标注为逐场景校正增益的中位数，与表 S3 同口径。
     b 校正增益 vs 未校正 RMSE（逐场景散点 + 方法级中位点）：未校正误差越大，校正回收越多。

═══ 为什么不用 3D ═══
本刊硬约束为「图中字 7 磅 + 黑白可辨」。3D 的透视会压变形轴标签，7 磅字在透视下不可读；
其深度信息靠阴影/颜色编码，黑白打印下崩溃。且本数据本质是二维的（方法 × 场景 × 指标），
强行 3D 只会引入遮挡。故采用当代顶刊的高信息密度 2D 手法。

═══ 期刊硬规范（投稿须知 + 写作模版-2024）═══
  · 图中字号 7 磅；中文宋体，英文 Times New Roman
  · 图题、图注、坐标名、图例名 均须中英文对照
  · 主要线条不小于 0.5 磅
  · 图表总数 ≤6 幅 → 本文 3 图 + 3 表
  · 配色 color-blind safe，禁红绿组合，黑白打印可辨（01CLAUDE.md §8.2）

═══ 双语双通路（zh / en）═══
本脚本一次运行出两套图：
  · LANG='zh' —— 中英文同框版（《农业工程学报》硬性要求图题/轴名/图例中英对照），
                 文件名不带后缀，中文稿 Elsevier_zh.tex 依赖之，**输出必须保持稳定**；
  · LANG='en' —— 纯英文版（英文稿投稿用），文件名带 `-en` 后缀，图内不得出现任何汉字。
显示文本一律经 T(zh, en) / T_origin() 过滤；**数据键（sheet 名、列名、取值）绝不参与翻译**——
它们是 read_excel/筛选/groupby 的键，改了就读不到数据。判据：进 set_xlabel/set_title/
ax.text/legend/label= 的中文才翻，进 df[...]/.str.startswith/== 的中文不动。

运行方式:
    cd <项目根目录>
    python 02code/49_plot_csae_figures.py

输出文件:
    06doc/01manuscript/figs/Fig1.png / .pdf   — BL 失稳诊断
    06doc/01manuscript/figs/Fig2.png / .pdf   — 5 方法零标签迁移表现
    06doc/01manuscript/figs/Fig3.png / .pdf   — 斜率/偏置校正对照（核心发现）
    ……以及 FigSpec / FigArch / FigHeat，及各自的 `-en` 纯英文版
"""

import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from scipy import stats

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, '..'))
_OUT = os.path.join(_BASE, '04outputs')
# 稿件（Elsevier_zh/en.tex、Supplementary_zh/en.tex）引的是 figs/，不是 figs-csae/。
# 写进 figs-csae/ 的图不会进 PDF——2026-09-11 查到补充材料的 Fig1/2/3 停在 07-21 的
# 旧数据版，就是因为这里的目录名与稿件对不上（00LATEX_GAP.md B11）。
_FIG = os.path.join(_BASE, '06doc', '01manuscript', 'figs')
os.makedirs(_FIG, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# 语言开关：'zh' = 中英同框版（现状，文件名无后缀）；'en' = 纯英文版（`-en` 后缀）
# 由 __main__ 依次置为 'zh' → 'en'，每次都重跑一遍完整绘图流程。
# ═══════════════════════════════════════════════════════════════════════
LANG = 'zh'

# 产地名：**仅用于显示**的中→英对照。数据侧的 '新疆'/'山东'/… 是筛选键，原样保留。
ORIGIN_EN = {
    '新疆': 'Xinjiang',
    '山东': 'Shandong',
    '陕西': 'Shaanxi',
    '甘肃': 'Gansu',
}


# ── 字体栈：**每个语言分支各自显式设定，不继承上一轮状态** ──────────────
# 坑（姊妹脚本 66 实测）：export_utils.setup_chinese_font() 会把
# rcParams['font.sans-serif'] 常驻改成 SimHei；若英文版继承中文字体，
# 拉丁字母会被中文字体渲成方块/全角宽度——pdftotext 查汉字仍是 0，但排版不能投英文刊。
# 本脚本 font.family 用的是**显式字体名列表**（不是 'sans-serif' 别名），
# 故 sans-serif 那条常驻污染不落到本脚本；但仍每轮显式复位，杜绝渲染顺序串扰。
_FONT_STACK = {
    # zh：Latin 命中 Times New Roman，汉字在 TNR 中无字形→逐字符回退到宋体。
    'zh': ['Times New Roman', 'SimSun', 'Songti SC'],
    # en：**纯拉丁**栈，链上不含任何中文字体，物理上杜绝中文字形渲染拉丁字母。
    #     已核 Times New Roman 覆盖英文版全部非 ASCII 字形（→ ≈ ° × – λ），不会掉到兜底。
    #     仍用衬线 TNR 而非 Helvetica/Arial：期刊规定「英文 Times New Roman」，
    #     且与中文版正文字形一致，换无衬线反而是排版倒退。
    'en': ['Times New Roman', 'DejaVu Serif'],
}


def _apply_font() -> None:
    """按当前 LANG 显式设定字体栈（每轮绘图入口调用一次）。"""
    plt.rcParams.update({
        'font.family':        list(_FONT_STACK[LANG]),
        'mathtext.fontset':   'stix',
        'axes.unicode_minus': False,
        'font.size':          FS,
    })


def T(zh: str, en: str) -> str:  # noqa: N802  单字母大写是刻意的：调用点极密集，短名才不撑行
    """双语显示文本：zh 模式返回中英同框串（现状），en 模式返回纯英文串。"""
    return en if LANG == 'en' else zh


def T_origin(s: str) -> str:  # noqa: N802  与 T() 同族，命名保持一致
    """翻译**显示用**的产地串：'山东→新疆' → 'Shandong→Xinjiang'（en 模式）。

    只作用于即将进入 tick label / 图例的字符串副本，绝不回写任何 DataFrame 的键。
    """
    if LANG != 'en':
        return s
    for zh, en in ORIGIN_EN.items():
        s = s.replace(zh, en)
    return s

# ═══════════════════════════════════════════════════════════════════════
# 统一视觉主题（三张图共用；改这里即可全局换肤）
# ═══════════════════════════════════════════════════════════════════════

FS = 7                    # 图中字号 7 磅（期刊硬规定）
CM = 1 / 2.54
# 成图宽度必须**严格等于**版面插入宽度（W_FULL = 18 cm）。
# 早先用 bbox_inches='tight' + 经验补偿量，裁剪量随布局变化而漂移（实测 15.5~18.1 cm 都出现过），
# 7 磅字随之被缩放成 7.0~8.1 磅。改为固定画布 + constrained_layout、**不做 tight 裁剪**，
# 画布宽度即成图宽度，与版面 1:1，字号不再漂移。
W_FIG = 18.0

plt.rcParams.update({
    'font.size':        FS,
    'axes.labelsize':   FS,
    'axes.titlesize':   FS,
    'xtick.labelsize':  FS,
    'ytick.labelsize':  FS,
    'legend.fontsize':  FS,
    # font.family 必须是**列表**才触发 matplotlib(≥3.6) 的逐字符回退：
    # 英文/数字命中 Times New Roman，中文在 TNR 中无字形、自动落到宋体。
    # 宋体用 SimSun（weight=400）而非 Songti SC（macOS 只注册了 weight=900 的粗宋）。
    'font.family':      ['Times New Roman', 'SimSun', 'Songti SC'],
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'axes.linewidth':   0.5,          # ≥0.5 磅
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2.0,
    'ytick.major.size': 2.0,
    'lines.linewidth':  0.6,
    'legend.frameon':   False,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'pdf.fonttype':     42,           # 可编辑文本
    'ps.fonttype':      42,
})

# ═══════════════════════════════════════════════════════════════════════
# 统一主题色板（Wong 色盲安全；与 66_paper_figures.py 中的 THEME 逐字节一致）
# 语义锚点（跨全部 9 张图一致）：
#   · signal   朱红 —— "物理知情/更先进的一方"（49 Phys+BL / 66 深度·物理），全项目唯一强调色
#   · positive 绿   —— "校正后 / 正向改善 / 目标自建"
#   · deep_*   蓝   —— 深度模型族
#   · base_*   中性灰 —— 经典/简单基线族（退到背景）
#   · CAT4 蓝/橙/绿/紫 —— 分类四元（产地 / 漂移类型 / 基准方法），同一 quartet 复用
# ═══════════════════════════════════════════════════════════════════════
THEME = {
    'signal':   '#D55E00',   # 朱红 vermillion
    'positive': '#009E73',   # 绿 bluish-green
    'deep_1':   '#56B4E9',   # 天蓝 sky-blue
    'deep_2':   '#0072B2',   # 深蓝 blue
    'base_1':   '#BBBBBB',   # 浅灰
    'base_2':   '#888888',   # 深灰
    'amber':    '#E69F00',   # 琥珀 orange
    'purple':   '#CC79A7',   # 紫 reddish-purple
    'ref':      '#4D4D4D',   # 参考线 / 注记
    'grid':     '#C9CDD2',   # 网格
}
CAT4 = [THEME['deep_2'], THEME['amber'], THEME['positive'], THEME['purple']]

# ── 语义常量：全部指向 THEME，改 THEME 即全项目换肤（语义固定，三图一致）──────
C_BASE1 = THEME['base_1']    # PLSR      基线族·浅灰
C_BASE2 = THEME['base_2']    # SVR       基线族·深灰
C_DEEP1 = THEME['deep_1']    # CNN       深度族·天蓝
C_DEEP2 = THEME['deep_2']    # CNN+MMD   深度族·深蓝
C_PHYS  = THEME['signal']    # Phys+BL   物理族·朱红（信号色，全文唯一强调色）
C_CORR  = THEME['positive']  # 校正后     方向色·绿
C_SRC   = THEME['deep_2']    # 源域
C_TGT   = THEME['signal']    # 目标域
C_GREY  = THEME['ref']       # 参考线/注记

# 颜色 + 形状**双编码**：黑白打印下靠形状仍可区分（期刊要求黑白可辨）
METHOD_STYLE = {
    'PLSR':    dict(c=C_BASE1, m='o', z=2),
    'SVR':     dict(c=C_BASE2, m='^', z=2),
    'CNN':     dict(c=C_DEEP1, m='D', z=3),
    'CNN+MMD': dict(c=C_DEEP2, m='v', z=3),
    'Phys+BL': dict(c=C_PHYS,  m='s', z=5),   # 信号色 + 最高层级
}
METHODS5 = ['PLSR', 'SVR', 'CNN', 'CNN+MMD', 'Phys+BL']
FORMAL_SEEDS = [20060515, 20041210, 19810915, 2023, 2024]

_SRC50 = os.path.join(_OUT, '50_formal_multiseed_benchmark.xlsx')


def _common_scenarios():
    """全部方法都跑通的场景。补充材料的表 S1/S3 算在这个子集上，图必须同口径，
    否则同一个方法在图上和表里是两个数。"""
    g = pd.read_excel(_SRC50, sheet_name='表g：每场景跨种子均值')
    piv = g.pivot(index='场景', columns='方法_校正', values='RMSE')
    return set(piv.dropna().index)


def _need50(fn):
    """50 号结果未落地时**响亮失败**——绝不用旧数据或测试数据凑一张图（Fig9 教训）。"""
    if not os.path.exists(_SRC50):
        raise SystemExit(
            f'🔴 {fn} 需要 {os.path.basename(_SRC50)}，但该文件不存在。\n'
            f'   请先跑完 50 号实验并 --merge。结果落地前不得绘制此图。')


def save(fig, stem):
    # 不传 bbox_inches='tight'：保持画布宽度 = 成图宽度 = 版面宽度（18 cm），
    # 否则裁剪量会随布局漂移，把 7 磅字缩放成别的字号。
    # 纯英文版另存 `-en`，中文版文件名保持不变（中文稿的 \includegraphics 依赖之）。
    stem = stem if LANG == 'zh' else f'{stem}-{LANG}'
    for ext, kw in (('png', dict(dpi=600)), ('pdf', {})):
        fig.savefig(os.path.join(_FIG, f'{stem}.{ext}'), **kw)
    plt.close(fig)
    print(f'  ✅ {stem}.png / .pdf')


def half_violin(ax, data, pos, color, width=0.32, side='left', ymax=None):
    """半小提琴（raincloud 的云）。手写 KDE 以精确控制方向与裁剪。
    给了 ymax 时只用轴内的值估计密度：轴外的离群值不画，也不许它们撑大带宽。"""
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    if ymax is not None:
        data = data[data <= ymax]
    if len(data) < 5:
        return
    kde = stats.gaussian_kde(data)
    lo, hi = data.min(), data.max()
    ys = np.linspace(lo, hi, 200)
    dens = kde(ys)
    dens = dens / dens.max() * width
    xs = pos - dens if side == 'left' else pos + dens
    ax.fill_betweenx(ys, pos, xs, facecolor=color, alpha=0.35,
                     edgecolor=color, linewidth=0.5, zorder=2)


def cluster_ci(v, n_boot=1000, seed=7, stat=np.mean):
    """按场景重采样的 95% CI（聚类单位 = 场景）；stat 取均值或中位数。"""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(v)
    boots = np.array([stat(v[rng.integers(0, n, n)]) for _ in range(n_boot)])
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


# ═══════════════════════════════════════════════════════════════════════
# 图1：BL 加性可分解性的跨产地失稳
# ═══════════════════════════════════════════════════════════════════════
def fig1():
    d = pd.read_excel(os.path.join(_OUT, '03_exp_BL_validation.xlsx'),
                      sheet_name='表c：全量样本BL残差')
    t = pd.read_excel(os.path.join(_OUT, '28_exp_true_BL_decomp.xlsx'),
                      sheet_name='表d：放大倍数透视')

    fig = plt.figure(figsize=(W_FIG * CM, 7.4 * CM), layout='constrained')
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.06)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0])

    # ── a 哑铃图：源域 → 目标域的配对放大 ───────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    scen = sorted(d['scenario'].unique())
    rows = []
    for s in scen:
        sub = d[d.scenario == s]
        src = sub[sub.domain_type.str.startswith('源')]['bl_residual'].median()
        tgt = sub[sub.domain_type.str.startswith('目标')]['bl_residual'].median()
        # 场景名形如 'Y2018-S1: 山东→新疆'，取年份与产地对
        year = s.split('-')[0].replace('Y', '')
        pair = s.split(': ')[-1]
        rows.append((f'{pair}', year, src, tgt, tgt / src))
    rows.sort(key=lambda r: r[4])                       # 按放大倍数升序，读者顺着看
    ypos = np.arange(len(rows))

    for i, (pair, year, src, tgt, amp) in enumerate(rows):
        ax.plot([src, tgt], [i, i], color=C_GREY, lw=0.7, alpha=0.55,
                zorder=1, solid_capstyle='round')
        ax.scatter(src, i, s=17, marker='o', facecolor='white',
                   edgecolor=C_SRC, linewidth=0.9, zorder=3)
        ax.scatter(tgt, i, s=20, marker='o', facecolor=C_TGT,
                   edgecolor='k', linewidth=0.35, zorder=4)
        ax.text(tgt * 1.16, i, f'×{amp:.1f}', va='center', ha='left',
                fontsize=FS - 0.5, color=C_TGT)

    ax.set_yticks(ypos)
    # 年份并入标签本身。早先把年份画成独立的 text 挂在轴外，坐标算在 axes 分数系里，
    # 结果直接压在 tick label 上（"20陕西→山东"）。并入标签最稳。
    # 产地名只在**显示**时译英；rows 里的 pair 源自 scenario 列，不回写。
    ax.set_yticklabels([f'{y}  {T_origin(p)}' for p, y, *_ in rows])

    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(right=ax.get_xlim()[1] * 2.6)           # 给 ×N 注记留位
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel(T('相对重构残差（中位数，对数坐标）\n'
                    'Relative reconstruction residual (median, log scale)',
                    'Relative reconstruction residual (median, log scale)'))
    ax.invert_yaxis()
    ax.grid(axis='x', ls=':', lw=0.4, alpha=0.45, zorder=0)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)

    # en 标签刻意不写成 'Source domain (within-domain)'：那样图例过宽，会压住末行的
    # ×N 注记（实测 ×4.5 被图例吞掉）。括号里已含 domain，去掉重复词既更短也更地道。
    handles = [Line2D([], [], marker='o', ls='none', mfc='white', mec=C_SRC,
                      mew=0.9, ms=3.6,
                      label=T('源域（域内）Source', 'Source (within-domain)')),
               Line2D([], [], marker='o', ls='none', mfc=C_TGT, mec='k',
                      mew=0.35, ms=4.0,
                      label=T('目标域（跨域）Target', 'Target (cross-domain)'))]
    ax.legend(handles=handles, loc='lower right', handletextpad=0.4,
              borderaxespad=0.4, labelspacing=0.3)
    ax.text(-0.20, 1.02, 'a', transform=ax.transAxes, fontweight='bold',
            fontsize=FS + 2, va='bottom')

    # ── b 低秩基底一致性 ───────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    base = t['PCA-10'].values
    # 只画 3 种代理。NNLS-3 与 NMF-3 共用同一组非负基、残差逐位相同（r=1.000000），
    # 并非独立代理，画进来等于凭空多一个自我复制的点集。
    # 沿用图1 的主色系（蓝=基准/源侧，橙红=对照/目标侧），全图语义一致。
    # 不用绿色——绿在本文主题里专指"校正后"，挪作他用会污染语义。
    series = [('NMF-3', C_SRC, 'o'), ('SVD-3', C_TGT, '^')]
    for name, col, mk in series:
        y = t[name].values
        r, pv = stats.pearsonr(base, y)
        ax.scatter(base, y, s=24, marker=mk, facecolor=col, edgecolor='k',
                   linewidth=0.35, alpha=0.92, zorder=3,
                   label=f'{name}   r = {r:.2f}')

    hi = float(t[['PCA-10', 'NMF-3', 'SVD-3']].values.max()) * 1.10
    ax.plot([0, hi], [0, hi], ls='--', c=C_GREY, lw=0.6, zorder=2,
            label=T('1:1 参考线  1:1 reference', '1:1 reference'))
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(T('PCA-10 基底的残差放大倍数\n'
                    'Residual amplification ratio (PCA-10 basis)',
                    'Residual amplification ratio (PCA-10 basis)'))
    ax.set_ylabel(T('其他基底的残差放大倍数\n'
                    'Residual amplification ratio (other bases)',
                    'Residual amplification ratio (other bases)'))
    # 数据集中在左上/对角线附近，图例放右下的空白区，不压数据
    ax.legend(loc='lower right', handletextpad=0.4, borderaxespad=0.5,
              labelspacing=0.32)
    ax.grid(ls=':', lw=0.4, alpha=0.45, zorder=0)
    ax.text(-0.24, 1.02, 'b', transform=ax.transAxes, fontweight='bold',
            fontsize=FS + 2, va='bottom')

    save(fig, 'Fig1')


# ═══════════════════════════════════════════════════════════════════════
# 图2：5 种方法的零标签跨产地迁移表现
# ═══════════════════════════════════════════════════════════════════════
def fig2():
    _need50('fig2')
    raw = pd.read_excel(_SRC50, sheet_name='表h：全部原始结果')
    raw = raw[(raw['校正'] == '裸') & (raw['种子'].isin(FORMAL_SEEDS))
              & (raw['场景'].isin(_common_scenarios()))]
    # 每场景先对种子取均值：同场景的多个种子是重复测量，聚类单位 = 场景。
    # 直接把 场景×种子 全部铺进分布会把 n 虚增数倍。
    per = raw.groupby(['方法', '场景', '迁移类型'], as_index=False)['RMSE'].mean()

    fig = plt.figure(figsize=(W_FIG * CM, 7.6 * CM), layout='constrained')
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.06)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.22])

    # ── a Raincloud：全部场景上的 RMSE 分布 ────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    rng = np.random.default_rng(3)
    n_by_m = {}
    ymax = 10.0     # 本文称 RMSE>10 °Brix 为发散；轴外的场景在顶部计数，不隐藏
    for i, m in enumerate(METHODS5):
        st = METHOD_STYLE[m]
        v = per[per['方法'] == m]['RMSE'].values
        v = v[np.isfinite(v)]
        half_violin(ax, v, i, st['c'], width=0.34, side='left', ymax=ymax)
        n_out = int((v > ymax).sum())
        if n_out:
            ax.text(i + 0.17, ymax * 0.985, f'+{n_out} > {ymax:g}', ha='center', va='top',
                    fontsize=FS - 1.4, color=st['c'])
        # 抖动散点（雨）
        jitter = rng.uniform(0.06, 0.28, len(v))
        ax.scatter(i + jitter, v, s=1.6, color=st['c'], alpha=0.32,
                   linewidths=0, zorder=2, rasterized=True)
        # 箱线（细）
        bp = ax.boxplot([v], positions=[i], widths=0.14, showfliers=False,
                        patch_artist=True, zorder=4,
                        medianprops=dict(color='k', lw=0.8),
                        boxprops=dict(facecolor='white', edgecolor='k', lw=0.5),
                        whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
        # 均值标记（信号色方块 / 各自形状）：与箱线的中位线对照，差距即发散运行的影响
        ax.scatter(i, v.mean(), s=15, marker=st['m'], facecolor=st['c'],
                   edgecolor='k', linewidth=0.4, zorder=6)
        n_by_m[m] = len(v)

    ax.set_xticks(range(len(METHODS5)))
    ax.set_xticklabels([f'{m}\nn={n_by_m[m]}' for m in METHODS5],
                       rotation=0, ha='center', fontsize=FS - 0.5)
    ax.set_ylabel(T('均方根误差  RMSE/(°Brix)', 'RMSE/(°Brix)'))
    ax.set_xlabel(T('方法  Method', 'Method'))
    ax.set_ylim(0, ymax)
    ax.grid(axis='y', ls=':', lw=0.4, alpha=0.45, zorder=0)
    ax.text(-0.20, 1.02, 'a', transform=ax.transAxes, fontweight='bold',
            fontsize=FS + 2, va='bottom')

    # ── b Forest plot：按迁移类型分层的中位数 ± 95% CI ───────────────
    ax = fig.add_subplot(gs[0, 1])
    # types 同时是**筛选键**（对齐 '迁移类型' 列取值）与显示文本，故原值必须保留；
    # 显示时才按语言取 f'{tp}\n{tp_en}'（zh）或 tp_en（en）。
    types = ['1源→1目标', '1源→多目标', '多源→1目标', '多源→多目标']
    types_en = ['1→1', '1→multi', 'multi→1', 'multi→multi']
    gap, row = 0.0, []
    ylab, ytick = [], []
    for ti, (tp, tp_en) in enumerate(zip(types, types_en)):
        for mi, m in enumerate(METHODS5):
            st = METHOD_STYLE[m]
            v = per[(per['方法'] == m) & (per['迁移类型'] == tp)]['RMSE'].values
            v = v[np.isfinite(v)]
            if len(v) < 3:
                gap += 1
                continue
            y = gap
            lo, hi = cluster_ci(v, stat=np.median)
            ax.plot([lo, hi], [y, y], color=st['c'], lw=1.1, solid_capstyle='round',
                    zorder=st['z'])
            ax.scatter(np.median(v), y, s=16, marker=st['m'], facecolor=st['c'],
                       edgecolor='k', linewidth=0.35, zorder=st['z'] + 3)
            row.append((y, m))
            gap += 1
        # 组标签
        ax.text(-0.015, gap - len(METHODS5) / 2 - 0.5,
                T(f'{tp}\n{tp_en}', tp_en), transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=FS - 0.5)
        if ti < len(types) - 1:
            ax.axhline(gap + 0.0, color='#DDDDDD', lw=0.5, zorder=0)
        gap += 1.2

    ax.set_yticks([])
    ax.set_ylim(-1.0, gap - 0.6)
    ax.invert_yaxis()
    ax.set_xlabel(T('均方根误差（中位数与 95% 置信区间）\n'
                    'RMSE (median with 95% CI)/(°Brix)',
                    'RMSE (median with 95% CI)/(°Brix)'))
    ax.grid(axis='x', ls=':', lw=0.4, alpha=0.45, zorder=0)
    ax.spines['left'].set_visible(False)
    handles = [Line2D([], [], marker=METHOD_STYLE[m]['m'], ls='-',
                      color=METHOD_STYLE[m]['c'], mec='k', mew=0.35, ms=3.6,
                      lw=1.1, label=m) for m in METHODS5]
    ax.legend(handles=handles, loc='lower right', ncol=1, handletextpad=0.4,
              borderaxespad=0.4, labelspacing=0.3)
    ax.text(-0.30, 1.02, 'b', transform=ax.transAxes, fontweight='bold',
            fontsize=FS + 2, va='bottom')

    save(fig, 'Fig2')


# ═══════════════════════════════════════════════════════════════════════
# 图3：斜率/偏置校正对照（本文核心发现）
# ═══════════════════════════════════════════════════════════════════════
def fig3():
    _need50('fig3')
    A = pd.read_excel(_SRC50, sheet_name='表a2：共同场景子集').set_index('方法')
    raw = pd.read_excel(_SRC50, sheet_name='表h：全部原始结果')
    raw = raw[raw['种子'].isin(FORMAL_SEEDS) & raw['场景'].isin(_common_scenarios())]
    per = raw.groupby(['方法', '校正', '场景'], as_index=False)['RMSE'].mean()

    shown = [m for m in ['PLSR', 'SVR', 'CNN', 'Phys+BL']
             if m in A.index and f'{m} + SB' in A.index]
    if not shown:
        print('  ⚠ 50 号结果缺 +SB 行，跳过 Fig3')
        return

    fig = plt.figure(figsize=(W_FIG * CM, 7.2 * CM), layout='constrained')
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.06)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.06, 1.0])

    # ── a 哑铃图：校正把所有方法拉到同一条窄带上 ────────────────────
    ax = fig.add_subplot(gs[0, 0])
    med_corr = {}
    for i, m in enumerate(shown):
        st = METHOD_STYLE[m]
        b = per[(per['方法'] == m) & (per['校正'] == '裸')].set_index('场景')['RMSE']
        c = per[(per['方法'] == m) & (per['校正'] == '+SB')].set_index('场景')['RMSE']
        k = b.index.intersection(c.index)
        b, c = b.loc[k].to_numpy(), c.loc[k].to_numpy()
        x0, x1 = float(np.median(b)), float(np.median(c))
        med_corr[m] = x1
        b_lo, b_hi = cluster_ci(b, stat=np.median)
        c_lo, c_hi = cluster_ci(c, stat=np.median)
        # 连线 = 校正的作用；箭头指向终点
        ax.annotate('', xy=(x1, i), xytext=(x0, i),
                    arrowprops=dict(arrowstyle='-|>', color=st['c'], lw=1.0,
                                    shrinkA=2.5, shrinkB=2.5,
                                    mutation_scale=6), zorder=2)
        # 未校正：空心（各方法自身形状）
        ax.errorbar(x0, i, xerr=[[x0 - b_lo], [b_hi - x0]],
                    fmt='none', ecolor=st['c'], elinewidth=0.6, capsize=1.2,
                    capthick=0.5, zorder=3)
        ax.scatter(x0, i, s=24, marker=st['m'], facecolor='white',
                   edgecolor=st['c'], linewidth=0.9, zorder=4)
        # 校正后：实心绿（方向色）
        ax.errorbar(x1, i, xerr=[[x1 - c_lo], [c_hi - x1]],
                    fmt='none', ecolor=C_CORR, elinewidth=0.6, capsize=1.2,
                    capthick=0.5, zorder=3)
        ax.scatter(x1, i, s=24, marker=st['m'], facecolor=C_CORR,
                   edgecolor='k', linewidth=0.35, zorder=5)
        # 标注 = 逐场景校正增益（RMSE 相对下降）的中位数，与表 S3"校正增益"同口径、同为正号；
        # 哑铃的向左箭头已表达"误差下降"的方向。
        gain = float(np.median((b - c) / b * 100))
        ax.text(x0 * 1.02, i - 0.30, f'{gain:.0f}%', va='center', ha='left',
                fontsize=FS - 0.8, color=st['c'])

    # 收敛带：校正后各方法的极差
    lo = min(med_corr.values())
    hi = max(med_corr.values())
    ax.axvspan(lo, hi, color=C_CORR, alpha=0.12, zorder=0)
    # 注记居中于收敛带（靠近左轴），en 必须折行——单行 'Converged band after correction'
    # 会顶出左侧轴线之外。
    ax.text((lo + hi) / 2, -0.85, T('校正后收敛带\nConverged band',
                                    'Converged band\nafter correction'),
            ha='center', va='center', fontsize=FS - 0.8, color=C_CORR)

    ax.set_yticks(range(len(shown)))
    ax.set_yticklabels(shown)
    ax.set_ylim(-1.3, len(shown) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel(T('均方根误差（中位数与 95% 置信区间）\n'
                    'RMSE (median with 95% CI)/(°Brix)',
                    'RMSE (median with 95% CI)/(°Brix)'))
    ax.grid(axis='x', ls=':', lw=0.4, alpha=0.45, zorder=0)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    handles = [Line2D([], [], marker='o', ls='none', mfc='white', mec=C_GREY,
                      mew=0.9, ms=3.8, label=T('未校正  Uncorrected', 'Uncorrected')),
               Line2D([], [], marker='o', ls='none', mfc=C_CORR, mec='k',
                      mew=0.35, ms=3.8,
                      label=T('斜率/偏置校正后  Corrected', 'Slope/bias corrected'))]
    ax.legend(handles=handles, loc='lower right', handletextpad=0.4,
              borderaxespad=0.4, labelspacing=0.3)
    ax.text(-0.24, 1.02, 'a', transform=ax.transAxes, fontweight='bold',
            fontsize=FS + 2, va='bottom')

    # ── b 校正增益 vs 未校正误差 ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    n_below = {}
    for m in shown:
        st = METHOD_STYLE[m]
        b = per[(per['方法'] == m) & (per['校正'] == '裸')].set_index('场景')['RMSE']
        c = per[(per['方法'] == m) & (per['校正'] == '+SB')].set_index('场景')['RMSE']
        k = b.index.intersection(c.index)
        bx, cy = b.loc[k].values, c.loc[k].values
        ok = np.isfinite(bx) & np.isfinite(cy) & (bx > 0)
        g = (bx[ok] - cy[ok]) / bx[ok] * 100
        n_below[m] = int((g < -100).sum())
        ax.scatter(bx[ok], g, s=3.2, marker=st['m'], facecolor=st['c'],
                   alpha=0.30, linewidths=0, zorder=st['z'], rasterized=True)
        # 方法级汇总点（大、描边），让四个方法的位置一目了然
        ax.scatter(np.median(bx[ok]), np.median(g), s=34, marker=st['m'],
                   facecolor=st['c'], edgecolor='k', linewidth=0.5,
                   zorder=st['z'] + 6, label=m)

    ax.axhline(0, color=C_GREY, lw=0.6, ls='--', zorder=1)
    # 机制（定性）：未校正误差越大 → 校正回收越多；因校正把各方法拉入同一窄带，
    # 故未校正越差者增益越大。散点自证该趋势，不再叠加基于 4 个方法中位数的弱 ρ。
    ax.set_xlabel(T('未校正均方根误差\nUncorrected RMSE/(°Brix)',
                    'Uncorrected RMSE/(°Brix)'))
    ax.set_ylabel(T('校正后 RMSE 相对下降\nRMSE reduction/%',
                    'RMSE reduction after correction/%'))
    # 绝大多数场景的增益落在 [−100, 100]；校正后反而变差一倍以上的场景跌出轴外。
    # 裁轴以显主趋势，轴外点数逐方法现算、如实标注，不隐藏。
    ax.set_ylim(-100, 100)
    below = [(m, k) for m, k in n_below.items() if k]
    if below:
        ax.text(0.97, 0.03,
                '\n'.join(T(f'{m}：{k} 个场景增益 < −100%（轴外）',
                             f'{m}: {k} scenario{"s" if k > 1 else ""} below −100% (off axis)')
                           for m, k in below),
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=FS - 1.2, color=METHOD_STYLE[below[0][0]]['c'])
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc='upper left', handletextpad=0.35, borderaxespad=0.3,
              labelspacing=0.3, markerscale=0.8)
    ax.grid(ls=':', lw=0.4, alpha=0.45, zorder=0)
    ax.text(-0.22, 1.02, 'b', transform=ax.transAxes, fontweight='bold',
            fontsize=FS + 2, va='bottom')

    save(fig, 'Fig3')


# ═══════════════════════════════════════════════════════════════════════
# 图（架构）：物理知情可微分解架构 Phys+BL 的前向流与训练目标（纯示意，无数据依赖）
# ═══════════════════════════════════════════════════════════════════════
def fig_arch():
    """Phys+BL 的两种实例化画在同一张图上：骨架共享，差异按变体标记逐框写出。逐项对着代码画。

    骨架：输入 → 编码器 → 逐样本非负因子 →（回归头 → L_reg）与（加性重构 → 重构损失）。
    ◆ 苹果案例研究：13_model_physics_informed.py（V2）经 50 号调用——全连接编码器、三头
        （分量谱 a_ik(λ)、光程 b_i、浓度 c_ik(λ)，三者都由编码逐样本输出，a 与 c 还逐波长）、
        回归头吃的是 a/c 沿波长的均值与标准差加 b（4K+1 = 13 维），不吃重构谱；
        重构损失在对数域、源域与目标域训练子集（无标签）对称求和。
    ● 统一基准：64 号 PhysBL——与 CNN/DANN/Deep CORAL 同一卷积编码器，K=8 个非负丰度，
        分量谱 a_k(λ) 是全体样本共享的可学参数（秩 ≤ K；画成不接编码器的参数框），线性回归头，
        重构项为线性空间 MSE、权重 0.3；源域训练只用源域光谱，微调只用那 n 条有标签的目标光谱。

    中文版一段之内不混写汉字与 $…$：mathtext 会把同一串里的汉字交给缺字的数学字体，印成方框。
    故混排的行拆成若干段，逐段量出宽度后按同一基线自左向右接排（run()）。
    """
    import matplotlib._mathtext as _mt
    from matplotlib.mathtext import MathTextParser
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    S, M, B, N = FS - 1.0, FS - 0.6, 'bold', 'normal'
    # 上下标缩放：mathtext 默认 0.7，6 磅行的下标只剩 4.2 磅；取 5.05/S，使最小字形不低于 5 磅。
    # 只在本图内生效，出图后复原并清空解析缓存，不影响其余各图。
    shrink0 = _mt.SHRINK_FACTOR
    _mt.SHRINK_FACTOR = 5.05 / S
    MathTextParser._parse_cached.cache_clear()

    # 画布宽 = 正文 \textwidth（CAS 双栏实测 494.50888 pt = 17.38 cm），以 width=\textwidth 排入时 1:1，
    # 图中字号即成品字号；按 18 cm 作图会被缩到 96.6%，5.05 磅的下标印出来只剩 4.88 磅。
    W = 494.50888 / 72.27 * 2.54
    H = 40.6                                   # 纵向坐标单位数；横纵同比例，1 单位 = W/100 cm
    # 按 600 dpi 量字宽：默认 100 dpi 下字形前进量取整，长串累积误差会让接排的下一段压字
    fig = plt.figure(figsize=(W * CM, H * W / 100 * CM), dpi=600)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, H)
    ax.axis('off')
    UNIT_PT = W / 100 / 2.54 * 72              # 1 个坐标单位对应的磅数
    rdr = fig.canvas.get_renderer()

    FC_IN, FC_ENC, FC_PHYS, FC_LOSS = '#EEEEEE', '#DCEEFB', '#FBE3D2', '#D6F0E6'
    EC_ENC, EC_PHYS, EC_LOSS = C_DEEP2, C_PHYS, C_CORR
    # 两种实例化：颜色 + 形状双编码，黑白打印靠 ◆ / ● 区分
    VAR = {'cs': ('#882255', 'D', 2.7), 'ub': ('#332288', 'o', 3.0)}
    IND = 1.5                                  # 变体行：标记之后的缩进

    def box(x, y, w, h, fc, ec, lw=0.9, ls='-'):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0,rounding_size=1.1',
            facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=3))

    def arr(x1, y1, x2, y2, c=C_GREY, lw=1.0, ls='-'):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=8,
            color=c, lw=lw, ls=ls, zorder=2, shrinkA=0.5, shrinkB=0.5))

    def mark(key, x, y):
        c, m, ms = VAR[key]
        ax.plot([x], [y], marker=m, ms=ms, color=c, mec=c, ls='none', zorder=5)

    def cjk(s):
        return any('\u3000' <= ch <= '\u9fff' or '\uff00' <= ch <= '\uffef' for ch in s)

    def run(x, yb, parts, fs, w=N, c='k'):
        """若干段（文字段、$…$ 段，或以坐标单位计的空隙）按基线 yb 自左向右接排，返回右端 x。
        相邻两段合起来不致汉字与 $…$ 同串时先合并，少一处接缝。"""
        segs = []
        for s in parts:
            if segs and isinstance(s, str) and isinstance(segs[-1], str) \
                    and not (cjk(segs[-1] + s) and '$' in segs[-1] + s):
                segs[-1] += s
            else:
                segs.append(s)
        for s in segs:
            if isinstance(s, (int, float)):
                x += s
                continue
            o = ax.text(x, yb, s, fontsize=fs, fontweight=w, color=c,
                        ha='left', va='baseline', zorder=4)
            x = ax.transData.inverted().transform((o.get_window_extent(rdr).x1, 0))[0]
        return x

    def tall(line):
        return any(isinstance(s, str) and ('\\Sigma' in s or '\\hat' in s or '^{' in s) for s in line)

    def pitch(fs, line):
        return fs * (1.62 if tall(line) else 1.3) / UNIT_PT

    def block(x, yc, items):
        """items：('h', 行) 粗体标题；('cs'|'ub', [行, …]) 一种实例化的若干行；('curve', 高)。
        以 yc 为中心自上而下排；变体行前画标记、随后缩进，字色同标记。返回 (顶, 底, 最右)。"""
        GAP = 0.45
        hs = []
        for kind, body in items:
            if kind == 'curve':
                hs.append(body)
            elif kind == 'h':
                hs.append(pitch(M, body))
            else:
                hs.append(GAP + sum(pitch(S, ln) for ln in body))
        top = yc + sum(hs) / 2
        y, xr = top, x
        for (kind, body), h in zip(items, hs):
            if kind == 'curve':
                xx = np.linspace(x + 0.4, x + 10.6, 120)
                u = (xx - xx[0]) / (xx[-1] - xx[0]) * 10 + 3
                yy = y - h / 2 + 1.1 * np.exp(-((u - 6) ** 2) / 5) \
                    + 0.8 * np.exp(-((u - 10.5) ** 2) / 2.5) - 0.5 * np.sin((u - 3) / 2.0)
                ax.plot(xx, yy, color=C_DEEP2, lw=0.7, zorder=4)
            elif kind == 'h':
                xr = max(xr, run(x, y - 0.95 * M / UNIT_PT, body, M, w=B))
            else:
                c = VAR[kind][0]
                yy = y - GAP
                for j, ln in enumerate(body):
                    p = pitch(S, ln)
                    yb = yy - p + 0.3 * S / UNIT_PT
                    if j == 0:
                        mark(kind, x + 0.55, yb + 0.33 * S / UNIT_PT)
                    xr = max(xr, run(x + IND, yb, ln, S, c=c))
                    yy -= p
            y -= h
        return top, y, xr

    def fits(name, b, x, y, w, h):
        """框内文字的上、下、右三边都须离框线至少 0.5 个单位。"""
        top, bot, right = b
        assert top <= y + h - 0.5 and bot >= y + 0.5 and right <= x + w - 0.5, \
            (name, round(top, 2), round(bot, 2), round(right, 2), y, y + h, x + w)

    # ── 版面（坐标单位）：顶部两行图例，中间两行框，底部两条数据路线 ──
    X_IN, X_ENC, X_FAC, X_HEAD, X_LOSS = 1.0, 17.5, 32.9, 58.4, 81.5
    W_IN, W_ENC, W_FAC, W_HEAD, W_LOSS = 14.6, 13.4, 22.1, 20.8, 17.5
    TOP, BOT = (23.1, 11.0), (8.6, 11.0)       # (y, 高)
    YC_T, YC_B = TOP[0] + TOP[1] / 2, BOT[0] + BOT[1] / 2
    Y_COL = (BOT[0], TOP[0] + TOP[1] - BOT[0])  # 输入与编码器：两行通高

    # ── 图例：两种实例化与各自的总损失 ──
    XF = 41.0
    for key, yb, name, loss in (
            ('cs', 38.1, T('苹果案例研究（K = 3）', 'Apple case study (K = 3)'),
             r'$L=L_{\mathrm{reg}}+\lambda_{\mathrm{BL}}\,[\,L_{\mathrm{BL}}(X_s)+L_{\mathrm{BL}}(X_t)\,],'
             r'\ \ \lambda_{\mathrm{BL}}=0.5$'),
            ('ub', 35.6, T('统一基准，全部 5 个基准（K = 8）', 'Unified benchmark, all five benchmarks (K = 8)'),
             r'$L=L_{\mathrm{reg}}+0.3\,L_{\mathrm{rec}}$')):
        c = VAR[key][0]
        mark(key, 1.6, yb + 0.33 * FS / UNIT_PT)
        run(2.8, yb, [name], FS, w=B, c=c)
        run(XF, yb, [T('总损失', 'total loss'), 1.2, loss], FS, c=c)

    # ── 输入 ──
    box(X_IN, Y_COL[0], W_IN, Y_COL[1], FC_IN, C_GREY)
    b_in = block(X_IN + 0.9, (Y_COL[0] + TOP[0] + TOP[1]) / 2, [
        ('h', [T('输入光谱', 'Input spectrum')]),
        ('h', [r'$x_i(\lambda)$']),
        ('curve', 3.4),
        ('cs', [[T('229 通道，', '229 channels,')], [T('逐波段标准化', 'band-wise standardised')]]),
        ('ub', [[T('SNV 标准化', 'SNV-standardised')]]),
    ])
    fits('in', b_in, X_IN, Y_COL[0], W_IN, Y_COL[1])

    # ── 编码器 ──
    box(X_ENC, Y_COL[0], W_ENC, Y_COL[1], FC_ENC, EC_ENC)
    fits('enc', block(X_ENC + 0.9, (Y_COL[0] + TOP[0] + TOP[1]) / 2, [
        ('h', [T('编码器', 'Encoder')]),
        ('cs', [[T('2 层全连接（64）', '2 FC layers (64),')], ['ReLU + BN']]),
        ('ub', [[T('一维卷积网络，', '1D-CNN, shared')],
                [T('与 CNN、DANN、', 'with CNN, DANN')],
                [T('Deep CORAL 共用', 'and Deep CORAL')]]),
    ]), X_ENC, Y_COL[0], W_ENC, Y_COL[1])

    # ── 逐样本非负因子（编码器输出）──
    box(X_FAC, TOP[0], W_FAC, TOP[1], FC_PHYS, EC_PHYS)
    fits('fac', block(X_FAC + 0.9, YC_T, [
        ('h', [T('逐样本非负因子', 'Non-negative factors, per sample')]),
        ('cs', [[T('三个头，指数变换：', 'three heads, exponential:')],
                [r'$a_{ik}(\lambda),\ b_i,\ c_{ik}(\lambda)>0$']]),
        ('ub', [[T('丰度头，softplus：', 'abundance head, softplus:')],
                [r'$c_{ik}\geq 0$', T('，光程并入丰度', ', path length absorbed')]]),
    ]), X_FAC, TOP[0], W_FAC, TOP[1])

    # ── 共享分量谱（参数，不接编码器）──
    box(X_FAC, BOT[0], W_FAC, BOT[1], FC_PHYS, EC_PHYS, ls=(0, (3, 1.5)))
    fits('par', block(X_FAC + 0.9, YC_B, [
        ('h', [T('分量谱', 'Component spectra')]),
        ('cs', [[T('逐样本预测（见上）：', 'predicted per sample (above):')],
                [r'$a_{ik}(\lambda)$']]),
        ('ub', [[r'$a_{k}(\lambda)\geq 0,\ k=1,\ldots,8$', T('：可学习参数，', ': learned')],
                [T('全体样本共用一组', 'parameters, one set for all samples')]]),
    ]), X_FAC, BOT[0], W_FAC, BOT[1])

    # ── 回归头与回归损失 ──
    box(X_HEAD, TOP[0], W_HEAD, TOP[1], FC_ENC, EC_ENC)
    fits('reg', block(X_HEAD + 0.9, YC_T, [
        ('h', [T('回归头', 'Regression head')]),
        ('cs', [[T('MLP，输入 ', 'MLP on '), r'$4K+1=13$', T(' 维摘要：', ' summaries:')],
                [r'$a_{ik}$', T('、', ', '), r'$c_{ik}$', T(' 沿波长的均值与标准差，', ' mean and s.d. over')],
                [T('及 ', 'wavelength, and '), r'$b_i$']]),
        ('ub', [[T('线性：', 'linear: '), r'$\hat{y}_i=\mathbf{w}^{\top}\mathbf{c}_i+w_0$']]),
    ]), X_HEAD, TOP[0], W_HEAD, TOP[1])
    box(X_LOSS, TOP[0], W_LOSS, TOP[1], FC_ENC, EC_ENC)
    fits('lreg', block(X_LOSS + 0.9, YC_T, [
        ('h', [T('回归损失 ', 'Regression loss '), r'$L_{\mathrm{reg}}$']),
        ('cs', [[T('源域 SSC 标签', 'source SSC labels')]]),
        ('ub', [[T('源域标签；', 'source labels;')],
                [T('微调：', 'fine-tuning: '), r'$n$', T(' 个目标标签', ' target labels')]]),
    ]), X_LOSS, TOP[0], W_LOSS, TOP[1])

    # ── 加性重构与重构损失 ──
    box(X_HEAD, BOT[0], W_HEAD, BOT[1], FC_PHYS, EC_PHYS)
    fits('rec', block(X_HEAD + 0.9, YC_B, [
        ('h', [T('加性重构', 'Additive reconstruction')]),
        ('cs', [[r'$\hat{x}_i(\lambda)=\Sigma_{k=1}^{3}\,a_{ik}(\lambda)\,b_i\,c_{ik}(\lambda)$']]),
        ('ub', [[r'$\hat{x}_i(\lambda)=\Sigma_{k=1}^{8}\,c_{ik}\,a_{k}(\lambda)$', T('，秩 ≤ 8', ', rank ≤ 8')]]),
    ]), X_HEAD, BOT[0], W_HEAD, BOT[1])
    box(X_LOSS, BOT[0], W_LOSS, BOT[1], FC_LOSS, EC_LOSS)
    fits('lrec', block(X_LOSS + 0.9, YC_B, [
        ('h', [T('重构损失', 'Reconstruction loss')]),
        ('cs', [[r'$L_{\mathrm{BL}}$', T('：对数幅值', ': log magnitude')],
                [r'$\mathrm{MSE}\,(\ln(\hat{x}+\varepsilon),\,\ln(|x|+\varepsilon))$']]),
        ('ub', [[r'$L_{\mathrm{rec}}$', T('：线性，', ': linear, '), r'$\mathrm{MSE}\,(\hat{x},\,x)$']]),
    ]), X_LOSS, BOT[0], W_LOSS, BOT[1])

    # ── 前向箭头 ──
    arr(X_IN + W_IN, YC_T, X_ENC, YC_T)
    arr(X_ENC + W_ENC, YC_T, X_FAC, YC_T)
    arr(X_FAC + W_FAC, YC_T, X_HEAD, YC_T, c=EC_ENC)
    arr(X_FAC + W_FAC, TOP[0] + 1.5, X_HEAD, BOT[0] + BOT[1] - 2.5, c=EC_PHYS)
    arr(X_FAC + W_FAC, YC_B - 2.0, X_HEAD, YC_B - 2.0, c=EC_PHYS, ls=(0, (3, 1.5)))
    arr(X_HEAD + W_HEAD, YC_T, X_LOSS, YC_T, c=EC_ENC)
    arr(X_HEAD + W_HEAD, YC_B, X_LOSS, YC_B, c=EC_LOSS)

    # ── 数据路线：哪些光谱进入重构比对（两种实例化各一条，互不相交）──
    y_rt = BOT[0]
    for key, xa, xb, y_low, parts in (
            ('cs', X_IN + 8.5, X_LOSS + 6.0, 5.9,
             [T('进入重构比对的光谱：源域光谱 + 无标签目标域光谱',
                'spectra in the reconstruction comparison: source spectra + unlabelled target spectra')]),
            ('ub', X_IN + 3.0, X_LOSS + 12.5, 2.3,
             [T('进入重构比对的光谱：源域训练用源域光谱，微调用那 ',
                'spectra in the reconstruction comparison: source spectra in source training, the '),
              r'$n$',
              T(' 条有标签目标光谱；不用无标签目标光谱',
                ' labelled target spectra in fine-tuning; no unlabelled target spectra')])):
        c = VAR[key][0]
        ax.plot([xa, xa, xb], [y_rt, y_low, y_low], color=c, lw=0.8, ls=(0, (4, 2)), zorder=1)
        arr(xb, y_low, xb, y_rt, c=c, lw=0.8)
        yb = y_low + 0.75
        mark(key, 18.6, yb + 0.33 * (FS - 1.2) / UNIT_PT)
        run(19.8, yb, parts, FS - 1.2, c=c)

    save(fig, 'FigArch')
    _mt.SHRINK_FACTOR = shrink0
    MathTextParser._parse_cached.cache_clear()


# ═══════════════════════════════════════════════════════════════════════
# 图（热力图）：1源→1目标跨产地场景的难度结构（源域×目标域中位 RMSE）
# ═══════════════════════════════════════════════════════════════════════
def fig_heat():
    _need50('fig_heat')
    raw = pd.read_excel(_SRC50, sheet_name='表h：全部原始结果')

    domains = ['2018_山东', '2018_新疆', '2018_陕西', '2025_新疆', '2025_山东', '2025_甘肃']
    idx = {dm: i for i, dm in enumerate(domains)}

    def heat_matrix(method):
        d = raw[(raw['方法'] == method) & (raw['校正'] == '裸')].copy()
        sp = d['场景'].astype(str).str.split('→', expand=True)
        d['src'], d['tgt'] = sp[0], sp[1]
        d = d[~d['src'].str.contains(r'\+') & ~d['tgt'].str.contains(r'\+')]
        # 跨种子取中位数 = 该 1→1 场景值：一粒种子发散（RMSE 上千 °Brix）就会把均值推离其余四粒
        g = d.groupby(['src', 'tgt'])['RMSE'].median()
        M = np.full((6, 6), np.nan)
        for (s, tg), v in g.items():
            if s in idx and tg in idx:
                M[idx[s], idx[tg]] = v
        return M

    M = heat_matrix('Phys+BL')
    fig = plt.figure(figsize=(W_FIG * 0.66 * CM, 8.2 * CM), layout='constrained')
    ax = fig.add_subplot(111)
    vmax = np.nanpercentile(M, 96)
    im = ax.imshow(M, cmap='cividis', aspect='equal', vmin=np.nanmin(M), vmax=vmax)
    thr = np.nanmean([np.nanmin(M), vmax])
    for i in range(6):
        for j in range(6):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f'{M[i, j]:.1f}', ha='center', va='center',
                        fontsize=FS - 1.3, color='white' if M[i, j] < thr else 'k')
    # domains 是**筛选键**（与 '场景' 列拆出的 src/tgt 逐字匹配），原样保留；
    # 只在生成 tick label 时译英。
    labs = [T_origin(dm).replace('_', '\n') for dm in domains]
    ax.set_xticks(range(6))
    ax.set_xticklabels(labs, fontsize=FS - 1.2)
    ax.set_yticks(range(6))
    ax.set_yticklabels(labs, fontsize=FS - 1.2)
    ax.set_xlabel(T('目标域 Target domain', 'Target domain'))
    ax.set_ylabel(T('源域 Source domain', 'Source domain'))
    # 年份分块分隔线（2018 | 2025）
    for p in (2.5,):
        ax.axhline(p, color='white', lw=1.4)
        ax.axvline(p, color='white', lw=1.4)
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
    ax.grid(which='minor', color='white', lw=0.6)
    ax.tick_params(which='minor', length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.03)
    cb.set_label(T('未校正中位 RMSE/(°Brix)\nUncorrected median RMSE',
                   'Uncorrected median RMSE/(°Brix)'), fontsize=FS - 0.6)
    cb.ax.tick_params(labelsize=FS - 1)
    save(fig, 'FigHeat')


# ═══════════════════════════════════════════════════════════════════════
# 图（数据/预处理）：预处理前后光谱对比
#   主张：产地间存在系统性的散射基线/尺度偏移（本文所研究的对象），
#         而主试验刻意不施加散射校正类预处理（SNV/MSC，见 §1.2），
#         去噪+平滑+标准化后该跨产地结构差异依然保留——为物理层建模张本。
#   数据链：复刻 02_data_processing 拿到 原始/预处理后 两端（逐元素差 <1e-12 已验证），
#          样本集对齐 01_export_utils.load_processed_dataframe 的两条 L1 硬过滤。
#   为何不用柱/单折线：光谱本质是多产地曲线族 + 不确定度带，属谱学标准图型，
#          非 §8.2 所禁的"单系列折线"；颜色 + 线型双编码，黑白可辨。
# ═══════════════════════════════════════════════════════════════════════
# 产地配色 = CAT4 分类四元（蓝/橙/绿/紫），与 66 号漂移类型四元同一 quartet；
# 线型双编码（黑白可辨）。色值经 THEME 表达，全项目一致。
ORIGIN_STYLE = {
    '新疆': dict(c=THEME['deep_2'],   ls='-',  en='Xinjiang'),
    '山东': dict(c=THEME['amber'],    ls='--', en='Shandong'),
    '陕西': dict(c=THEME['positive'], ls='-.', en='Shaanxi'),
    '甘肃': dict(c=THEME['purple'],   ls=':',  en='Gansu'),
}
ORIGIN_ORDER = {2018: ['新疆', '山东', '陕西'], 2025: ['新疆', '山东', '甘肃']}
SSC_PHYS_MAX = 20.0   # 与 01_export_utils.SSC_PHYSICAL_MAX 一致（越界读数硬剔除）


def _spectra_two_stage(dp, year):
    """返回 (原始对齐光谱, 预处理后光谱, 产地, 波长)，样本集与全部实验一致。

    预处理前 = 加载对齐→(2025)五面聚合→Mahalanobis 剔异常 之后、去噪之前的原始反射光谱；
    预处理后 = 再经 db4 小波去噪 + Savitzky--Golay 平滑(11,3) + 逐波段标准化；
    样本集   = 再套 SSC>20 越界剔除 + 逐位重复光谱整组剔除（论文分析集）。
    """
    from sklearn.preprocessing import StandardScaler
    # 原始光谱的正典源是共享库 05data（01CLAUDE.md §4.5.1：读 05data → 写 03data/processed）。
    # 这里早先写的是 dp.DATA_DIR（=03data），而 2018/2025 的 03data 原始副本已按数据卫生
    # #36 核对 SHA-256 后删除，于是 fig_spectra 会 FileNotFoundError。改指 dp._MULTIYEAR_RAW。
    dat = dp._MULTIYEAR_RAW
    if year == 2018:
        X, y, reg, wl = dp.load_excel_data(
            os.path.join(dat, '01新疆-2山东-3陕西苹果光谱数据(含糖度）2018.xlsx'))
    else:
        X, y, reg, wl = dp.load_csv_data_2025(
            os.path.join(dat, '新疆-山东-甘肃苹果光谱数据2025.csv'),
            os.path.join(dat, '新疆-山东-甘肃苹果糖度数据2025.csv'))
        X = dp.aggregate_5faces(X, method='robust_mean')
        yy, rr = [], []
        for i in range(0, (len(y) // 5) * 5, 5):
            yy.append(np.nanmean(y[i:i + 5]))
            rr.append(reg[i])
        y, reg = np.array(yy), np.array(rr)
    is_out, _ = dp.detect_outliers_mahalanobis(X, threshold_percentile=10)
    X, y, reg = X[~is_out], y[~is_out], reg[~is_out]
    X_raw = X.copy()
    X_pre = StandardScaler().fit_transform(
        dp.savitzky_golay_smooth(dp.wavelet_denoise(X, 'db4', 3), 11, 3))
    keep = y <= SSC_PHYS_MAX
    key = np.array(['|'.join(f'{v:.10f}' for v in row) for row in np.round(X_pre, 10)])
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    keep &= cnt[inv] == 1
    return X_raw[keep], X_pre[keep], reg[keep], np.asarray(wl, dtype=float)


def _spectra_panel(ax, X, reg, wl, year, ylabel, title):
    for org in ORIGIN_ORDER[year]:
        m = reg == org
        if m.sum() < 2:
            continue
        st = ORIGIN_STYLE[org]
        mu, sd = X[m].mean(0), X[m].std(0)
        ax.fill_between(wl, mu - sd, mu + sd, color=st['c'], alpha=0.14,
                        linewidth=0, zorder=2)
        # org 是**筛选键**（reg == org），只在图例文本里译英。
        ax.plot(wl, mu, color=st['c'], ls=st['ls'], lw=1.0, zorder=3,
                label=T(f"{org} {st['en']}", st['en']))
    ax.set_xlim(590, 1001)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left', fontsize=FS)
    ax.margins(x=0)


def fig_spectra():
    import importlib.util
    sp = importlib.util.spec_from_file_location(
        'dp', os.path.join(_CODE, '02_data_processing.py'))
    dp = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(dp)
    # 导入 02_data_processing 会调用其 setup_chinese_font()，覆盖本脚本字体，此处按当前
    # 语言显式复位（en 版必须落到纯拉丁栈，否则拉丁字母会被中文字体渲染）。
    _apply_font()

    fig, axes = plt.subplots(2, 2, figsize=(W_FIG * CM, 12.6 * CM),
                             layout='constrained')
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.05, hspace=0.06)
    rows = [(2018, T('反射强度 Reflectance intensity/(a.u.)',
                     'Reflectance intensity/(a.u.)')),
            (2025, T('相对反射率 Relative reflectance', 'Relative reflectance'))]
    tags = [['a', 'b'], ['c', 'd']]
    # zh 用全角空格分隔面板号；en 用半角双空格，避免英文图里出现全角标点。
    sep = T('　', '  ')
    for r, (year, raw_ylab) in enumerate(rows):
        X_raw, X_pre, reg, wl = _spectra_two_stage(dp, year)
        # en 去掉 (raw)/(preprocessed) 括注：中文版里它们是「预处理前/后」的英文对照，
        # 到了纯英文版就与前半句同义重复了。
        _spectra_panel(axes[r, 0], X_raw, reg, wl, year, raw_ylab,
                       T(f"{tags[r][0]}　{year} 年 · 预处理前 (raw)",
                         f"{tags[r][0]}{sep}{year} · before preprocessing"))
        _spectra_panel(axes[r, 1], X_pre, reg, wl, year,
                       T('标准化幅值 Standardised amplitude',
                         'Standardised amplitude'),
                       T(f"{tags[r][1]}　{year} 年 · 预处理后 (preprocessed)",
                         f"{tags[r][1]}{sep}{year} · after preprocessing"))
        axes[r, 0].legend(loc='upper right', handlelength=1.6,
                          borderaxespad=0.3, labelspacing=0.25)
        for c in (0, 1):
            axes[r, c].axvline(970, color=C_GREY, ls=(0, (1, 2)), lw=0.5, zorder=1)
    # ~970 nm O–H（水/糖）合频/倍频：在 2025 原始面板的水吸收下陷处轻标一次
    axes[1, 0].text(0.60, 0.12, T('≈970 nm\nO–H（水/糖）', '≈970 nm\nO–H (water/sugar)'),
                    transform=axes[1, 0].transAxes,
                    fontsize=FS - 1.4, color=C_GREY, ha='left', va='bottom')
    for c in (0, 1):
        axes[1, c].set_xlabel(T('波长 Wavelength/nm', 'Wavelength/nm'))
    save(fig, 'FigSpec')


def draw_all():
    """按当前 LANG 走一遍**完整**绘图流程。

    每个语言分支都从 _apply_font() 重新起手、整套重画，不复用上一轮的 figure 或
    rcParams 残留状态——matplotlib 的 rcParams 是全局的，靠"默认状态"会串扰。
    """
    _apply_font()
    fig_spectra()      # 会 exec 02_data_processing（内含 setup_chinese_font），其内部再复位一次
    fig_arch()
    fig1()
    fig2()
    fig3()
    fig_heat()


if __name__ == '__main__':
    print('按《农业工程学报》规范绘图（7 磅字 / 中英双语 / 线宽≥0.5磅 / 色盲安全 + 形状双编码）')
    if '--only-arch' in sys.argv:
        # 架构示意图不读任何结果文件，单独重画它时不必连带重出其余五张数据图。
        for _lang in ('zh', 'en'):
            LANG = _lang
            _apply_font()
            fig_arch()
        sys.exit(0)
    try:
        # zh 先、en 后：zh 是既有产物，顺序不变以保证其输出稳定。
        for _lang in ('zh', 'en'):
            LANG = _lang
            print(f'── LANG={LANG}（{"中英同框版" if LANG == "zh" else "纯英文版 -en"}）')
            draw_all()
    except SystemExit as e:
        # ⚠ 此处原先只 print 后正常结束，脚本以**退出码 0** 收场——
        #   于是 56 号流水线会把「图没画出来」当成「绘图成功」，47 号照样出一份缺图的 docx。
        #   `_need50` 的守卫被这个 except 整个架空了。
        #   现在：Fig1 仍然照出（它不依赖 50 号），但**必须以非 0 退出**，让调用方停下来。
        print(e)
        print('  → 图1 已出（不依赖 50 号）。图2、图3 需 50 号实验落地后重跑本脚本。')
        print(f'\n输出目录: {_FIG}')
        sys.exit(1)
    print(f'\n输出目录: {_FIG}')
