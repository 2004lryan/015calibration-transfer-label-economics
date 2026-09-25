"""
66_paper_figures.py: 生成多基准清算论文的三张主图(清算曲线/决策相图/负律散点),中英双版。

数据源:
    04outputs/65_paper_analysis.xlsx (reckoning/phase)
    04outputs/63_crossover_analysis.xlsx (nstar_per_task/scaling_fit/lobo)
输出文件:
    04outputs/66_paper_figures-图a.pdf / -fig-a.pdf  — 清算曲线(简单 vs 经典昂贵 vs 深度/物理,四漂移类型)
    04outputs/66_paper_figures-图b.pdf / -fig-b.pdf  — 决策相图(漂移类型×标签预算→获胜方法)
    04outputs/66_paper_figures-图c.pdf / -fig-c.pdf  — 负律散点(交叉预算 n* 对域漂移量不可预测)
    06doc/01manuscript/figs/{FigReckon,FigPhase,FigLaw}.pdf      — 中版拷贝(供中文稿引用)
    06doc/01manuscript/figs/{FigReckon,FigPhase,FigLaw}-en.pdf   — 英版拷贝(供英文稿引用)
    05logs/66_paper_figures_*.log
运行方式:
    cd <项目根目录绝对路径>
    python 02code/66_paper_figures.py

═══ 视觉规程(与 49_plot_csae_figures.py 共用同一套房内风格,nature-figure / §8)═══
  · 统一主题色板 THEME(Wong 色盲安全):两脚本逐字节一致,改此处即全项目换肤。
  · 统一字形:英文/公式 Times New Roman,中文宋体 SimSun(与 49 号衬线族一致);
             英文版字体栈**不含任何中文字体**,物理上杜绝中文字形渲染拉丁字母。
  · 主图禁裸 matplotlib 默认样式;线宽≥0.5磅;去顶/右框;面板小标签 a/b/c/d;
    颜色 + 形状/线型双编码(黑白打印可辨);误差带注明为 95% CI。
  · axes.unicode_minus=False + 对数轴改用 FuncFormatter:根治此前对数刻度负指数
    (U+2212)在拉丁字体缺字形而渲成乱码(FigLaw x 轴曾出现 "10¤¹")的问题。
"""
import importlib.util
import os
import shutil
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, ".."))
_OUT = os.path.join(_BASE, "04outputs")
_FIGDIR = os.path.join(_BASE, "06doc", "01manuscript", "figs")
_es = importlib.util.spec_from_file_location("export_utils", os.path.join(_CODE, "01_export_utils.py"))
assert _es is not None  # 对已知存在的 .py 文件 spec 恒非 None
assert _es.loader is not None  # 同理 loader 恒非 None
_eu = importlib.util.module_from_spec(_es)
_es.loader.exec_module(_eu)

# ═══════════════════════════════════════════════════════════════════════
# 统一主题色板(Wong 色盲安全;与 49_plot_csae_figures.py 中的 THEME 逐字节一致)
# 语义锚点(跨全部 9 张图一致):
#   · signal   朱红 —— "物理知情/更先进的一方"(66 深度·物理 / 49 Phys+BL),全项目唯一强调色
#   · positive 绿   —— "校正后 / 正向改善 / 目标自建"
#   · deep_*   蓝   —— 深度模型族
#   · base_*   中性灰 —— 经典/简单基线族(退到背景)
#   · CAT4 蓝/橙/绿/紫 —— 分类四元(产地 / 漂移类型 / 基准方法),同一 quartet 复用
# ═══════════════════════════════════════════════════════════════════════
THEME = {
    "signal":   "#D55E00",   # 朱红 vermillion
    "positive": "#009E73",   # 绿 bluish-green
    "deep_1":   "#56B4E9",   # 天蓝 sky-blue
    "deep_2":   "#0072B2",   # 深蓝 blue
    "base_1":   "#BBBBBB",   # 浅灰
    "base_2":   "#888888",   # 深灰
    "amber":    "#E69F00",   # 琥珀 orange
    "purple":   "#CC79A7",   # 紫 reddish-purple
    "ref":      "#4D4D4D",   # 参考线 / 注记
    "grid":     "#C9CDD2",   # 网格
}
CAT4 = [THEME["deep_2"], THEME["amber"], THEME["positive"], THEME["purple"]]

# ── 统一字形:英文/公式 TNR,中文宋体;英文版栈无中文字体 ────────────────
# 坑(实测):export_utils.setup_chinese_font() 会把 font.sans-serif 常驻改成 SimHei,
# 若英文版继承中文字体,拉丁字母会被中文字体渲成方块/全角宽度(pdftotext 查汉字仍是 0,
# 但排版不能投英文刊)。故本脚本改为**每语言分支各自显式设定字体栈**,不再依赖
# setup_chinese_font(),并与 49 号采用同一套衬线族(全项目字形一致)。
_FONT_STACK = {
    "zh": ["Times New Roman", "SimSun", "Songti SC"],  # Latin→TNR,汉字逐字符回退宋体
    "en": ["Times New Roman", "DejaVu Serif"],          # 纯拉丁栈,链上无中文字体
}
FS = 7  # 图中字号(全项目统一 7 磅)


def _apply_font(lang: str) -> None:
    """按语言显式设定字体栈(每轮绘图入口调用一次,杜绝渲染顺序串扰)。"""
    plt.rcParams["font.family"] = list(_FONT_STACK[lang])


# 语言无关的房内风格(与 49 号同一套):去顶右框、细线、可编辑 PDF 文本
plt.rcParams.update({
    "font.size":         FS,
    "axes.labelsize":    FS,
    "axes.titlesize":    FS,
    "xtick.labelsize":   FS,
    "ytick.labelsize":   FS,
    "legend.fontsize":   FS,
    "font.family":       list(_FONT_STACK["zh"]),
    "mathtext.fontset":  "stix",
    "axes.unicode_minus": False,          # 用 ASCII 连字符,根治 U+2212 缺字形乱码
    "axes.linewidth":    0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size":  2.0,
    "ytick.major.size":  2.0,
    "lines.linewidth":   0.9,
    "legend.frameon":    False,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

# 漂移类型顺序与双语名(SHIFTS 为**数据键**,原样保留;仅显示名双语)
SHIFTS = ["instrument", "season", "origin_year_instrument", "lab"]
SHIFT_ZH = {"instrument": "仪器漂移", "season": "季节漂移",
            "origin_year_instrument": "跨产地/年漂移", "lab": "跨实验室漂移(MIR)"}
SHIFT_EN = {"instrument": "Instrument", "season": "Season",
            "origin_year_instrument": "Origin/Year", "lab": "Lab (MIR)"}
NCAL = [0, 5, 10, 20, 40]

# 清算曲线三方法族:(数据键, zh 名, en 名, 颜色, marker, 线型)。
# 数据键 '简单经典'/'经典昂贵'/'深度物理' 是 group 列取值,绝不改。
# 深度·物理 → 朱红 signal(与 49 号 Phys+BL 同色,坐实"更先进一方=朱红");
# 绿色因此专留给"校正/正向",不再像旧版那样一色两义。颜色+marker+线型三重编码,黑白可辨。
RECK_GROUPS = [
    ("简单经典", "简单经典", "Simple",              THEME["deep_2"], "o", "-"),
    ("经典昂贵", "经典昂贵", "Classical-expensive", THEME["amber"],  "s", (0, (5, 2))),
    ("深度物理", "深度/物理", "Deep/Physics",        THEME["signal"], "D", "-"),
]

# 决策相图方法双语与配色(win_* 键去前缀后的方法名为**数据键**,原样保留)
METH_ZH = {"zero_shot": "零迁移", "coral": "免标签对齐", "sbc": "斜率/偏置",
           "pds": "分段直接", "target_only": "目标自建", "model_update": "模型更新"}
METH_EN = {"zero_shot": "Zero-shot", "coral": "CORAL", "sbc": "Slope/Bias",
           "pds": "PDS", "target_only": "Target-only", "model_update": "Model-update"}
METH_COLOR = {"zero_shot": THEME["base_2"], "coral": THEME["deep_2"], "sbc": THEME["amber"],
              "pds": THEME["purple"], "target_only": THEME["positive"], "model_update": THEME["signal"]}
# 负律散点各漂移类型的形状(与 CAT4 颜色配对,双编码)
SHIFT_MARKER = {"instrument": "o", "season": "s", "origin_year_instrument": "D", "lab": "^"}
SHIFT_COLOR = dict(zip(SHIFTS, CAT4, strict=True))


def _parse_ci(s: object) -> tuple[float, float]:
    try:
        lo, hi = str(s).strip("[]").split(",")
        return float(lo), float(hi)
    except Exception:
        return np.nan, np.nan


def _style(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2, width=0.5)


def _panel_tag(ax: Axes, tag: str, x: float = -0.15, y: float = 1.03) -> None:
    ax.text(x, y, tag, transform=ax.transAxes, fontweight="bold",
            fontsize=FS + 2, va="bottom", ha="left")


def make_reckoning(reck: pd.DataFrame) -> Callable[[str], Figure]:
    def plot(lang: str) -> Figure:
        _apply_font(lang)
        sz = SHIFT_ZH if lang == "zh" else SHIFT_EN
        ylab = "目标域均方根误差 RMSEP" if lang == "zh" else "Target-domain RMSEP"
        xlab = "目标标签预算 n" if lang == "zh" else "Target-label budget n"
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True)
        for ax_i, (ax, st) in enumerate(zip(axes.ravel(), SHIFTS, strict=False)):
            sub = reck[reck["shift_type"] == st]
            for key, zh, en, col, mk, ls in RECK_GROUPS:
                lbl = zh if lang == "zh" else en
                d = sub[sub["group"] == key].set_index("n_cal").reindex(NCAL)
                y = d["median_rmsep"].values
                ci = d["ci95"].apply(_parse_ci)
                lo = np.array([c[0] for c in ci])
                hi = np.array([c[1] for c in ci])
                m = np.isfinite(y)
                xs = np.array(NCAL)[m]
                ax.fill_between(xs, lo[m], hi[m], color=col, alpha=0.13, lw=0, zorder=2)
                ax.plot(xs, y[m], ls=ls, color=col, lw=1.2, marker=mk, ms=3.6,
                        mec="white", mew=0.5, label=lbl, zorder=4)
            ax.set_title(sz[st], fontsize=FS + 1, pad=3)
            ax.set_xticks(NCAL)
            ax.grid(axis="y", ls=":", lw=0.4, color=THEME["grid"], alpha=0.7, zorder=0)
            ax.margins(x=0.03)
            _style(ax)
            _panel_tag(ax, "abcd"[ax_i])
        axes[1, 0].set_xlabel(xlab)
        axes[1, 1].set_xlabel(xlab)
        axes[0, 0].set_ylabel(ylab)
        axes[1, 0].set_ylabel(ylab)
        # 图例:三方法族 + 95% CI 带说明(误差带类型显式标注,§8.2)
        handles: list[Line2D | Patch] = [
            Line2D([], [], color=col, lw=1.2, ls=ls, marker=mk, ms=3.6,
                   mec="white", mew=0.5, label=(zh if lang == "zh" else en))
            for _, zh, en, col, mk, ls in RECK_GROUPS]
        handles.append(Patch(facecolor=THEME["ref"], alpha=0.18, edgecolor="none",
                             label=("阴影为 95% CI" if lang == "zh" else "Shaded: 95% CI")))
        axes[0, 0].legend(handles=handles, loc="upper left", fontsize=FS - 0.5,
                          handlelength=1.7, handletextpad=0.5, labelspacing=0.35,
                          borderaxespad=0.3)
        fig.tight_layout(pad=0.6, w_pad=1.0, h_pad=0.8)
        return fig
    return plot


def make_phase(phase: pd.DataFrame) -> Callable[[str], Figure]:
    # 每(shift,n_cal)取获胜方法(win_* 最大者);数据逻辑不变
    win = {}
    for _, r in phase.iterrows():
        fr = {k.replace("win_", ""): r[k] for k in phase.columns if k.startswith("win_")}
        top = max(fr, key=lambda k: fr[k])  # 等价 key=fr.get,改用下标索引以满足严格类型
        win[(r["shift_type"], int(r["n_cal"]))] = (top, fr[top])

    def plot(lang: str) -> Figure:
        _apply_font(lang)
        sz = SHIFT_ZH if lang == "zh" else SHIFT_EN
        mz = METH_ZH if lang == "zh" else METH_EN
        xlab = "目标标签预算 n" if lang == "zh" else "Target-label budget n"
        title = ("样本内频率图：各漂移类型 × 标签预算下最常获胜的方法" if lang == "zh"
                 else "In-sample frequency map: most frequent winner by shift type × label budget")
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        used: list[str] = []
        pad = 0.06  # 单元间留白(圆角砖块效果)
        for i, st in enumerate(SHIFTS):
            for j, nc in enumerate(NCAL):
                meth, frac = win.get((st, nc), (None, 0))
                if meth is None:
                    continue
                if meth not in used:
                    used.append(meth)
                col = METH_COLOR.get(meth, "#cccccc")
                ax.add_patch(FancyBboxPatch(
                    (j + pad, i + pad), 1 - 2 * pad, 1 - 2 * pad,
                    boxstyle="round,pad=0,rounding_size=0.10",
                    facecolor=col, edgecolor="white", lw=0.8, alpha=0.92, zorder=3))
                ax.text(j + 0.5, i + 0.60, mz.get(meth, meth), ha="center", va="center",
                        fontsize=FS - 0.5, color="white", weight="bold", zorder=4)
                ax.text(j + 0.5, i + 0.32, f"{frac:.0%}", ha="center", va="center",
                        fontsize=FS - 1.0, color="white", alpha=0.92, zorder=4)
        ax.set_xlim(0, len(NCAL))
        ax.set_ylim(0, len(SHIFTS))
        ax.set_xticks(np.arange(len(NCAL)) + 0.5)
        ax.set_xticklabels([str(n) for n in NCAL])
        ax.set_yticks(np.arange(len(SHIFTS)) + 0.5)
        ax.set_yticklabels([sz[s] for s in SHIFTS])
        ax.set_xlabel(xlab)
        ax.set_title(title, fontsize=FS + 1, pad=6)
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="box")
        # 返修 R1.5:n=0 列只有免标签方法可用(直接迁移、CORAL),在图上直接标出
        ax.set_ylim(len(SHIFTS), -0.6)
        ax.text(0.5, -0.12, ("仅免标签方法" if lang == "zh" else "label-free\nmethods only"),
                ha="center", va="bottom", fontsize=FS - 1, color=THEME["ref"], style="italic")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        # 图例:颜色→方法(仅列出图中出现的方法),置于绘图区右侧
        handles = [Patch(facecolor=METH_COLOR[m], edgecolor="white", lw=0.6,
                         label=mz.get(m, m)) for m in used]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  fontsize=FS - 0.5, handlelength=1.2, handletextpad=0.5,
                  labelspacing=0.5, borderaxespad=0.0,
                  title=("最常获胜" if lang == "zh" else "Most frequent winner"),
                  title_fontsize=FS - 0.5)
        fig.tight_layout(pad=0.5)
        return fig
    return plot


def make_law(nst: pd.DataFrame) -> tuple[Callable[[str], Figure], float]:
    d = nst.dropna(subset=["n_simple_suffices", "mmd"]).copy()
    x = np.log10(d["mmd"].values + 1e-9)
    y = np.log10(d["n_simple_suffices"].values + 1.0)
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)

    def plot(lang: str) -> Figure:
        _apply_font(lang)
        sz = SHIFT_ZH if lang == "zh" else SHIFT_EN
        xlab = ("域漂移量（最大均值差异 MMD，对数坐标）" if lang == "zh"
                else "Domain-shift magnitude (MMD, log scale)")
        ylab = ("交叉预算 n*（简单方法追平所需标签数）" if lang == "zh"
                else "Crossover budget n* (labels for simple to catch up)")
        # 标题的 R² 用 unicode 上标(非 $..$ mathtext):CJK 与 mathtext 混排时,非数学段会
        # 走 STIX 字体集渲染而 STIX 不含汉字→整串汉字变豆腐块。改纯文本即逐字符回退到宋体。
        title = (f"交叉预算不可由域漂移量预测（对数-对数 R²={r2:.3f}）" if lang == "zh"
                 else f"n* is not predictable from domain-shift magnitude (log-log R²={r2:.3f})")

        # 主散点 + 边际分布(高信息密度 2D):顶部 MMD 密度、右侧 n* 分布
        fig = plt.figure(figsize=(6.6, 4.8))
        gs = fig.add_gridspec(2, 2, width_ratios=[4.2, 1.0], height_ratios=[1.0, 4.2],
                              wspace=0.04, hspace=0.04)
        ax = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

        mmd_all = d["mmd"].values.astype(float)
        for st in SHIFTS:
            s = d[d["shift_type"] == st]
            if len(s) == 0:
                continue
            ax.scatter(s["mmd"], s["n_simple_suffices"], s=26, alpha=0.8,
                       marker=SHIFT_MARKER[st], facecolor=SHIFT_COLOR[st],
                       edgecolor="white", linewidth=0.4, zorder=4, label=sz[st])
        # 近水平拟合线(负律的直观呈现)
        xs = np.linspace(mmd_all.min(), mmd_all.max(), 60)
        ax.plot(xs, 10 ** (a + b * np.log10(xs + 1e-9)) - 1.0, "--", color=THEME["ref"],
                lw=1.1, zorder=3, label=("拟合（近水平）" if lang == "zh" else "Fit (near-flat)"))
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(ls=":", lw=0.4, color=THEME["grid"], alpha=0.7, zorder=0)
        ax.legend(fontsize=FS - 0.5, frameon=False, loc="upper right",
                  handletextpad=0.5, labelspacing=0.35, borderaxespad=0.3)
        _style(ax)

        # 顶部边际:MMD(对数轴)密度直方(灰,退到背景)
        logm = np.log10(mmd_all + 1e-9)
        cnt, edges = np.histogram(logm, bins=14)
        widths = 10 ** edges[1:] - 10 ** edges[:-1]
        ax_top.bar(10 ** edges[:-1], cnt, width=widths, align="edge",
                   color=THEME["base_1"], edgecolor="white", linewidth=0.3, zorder=2)
        ax_top.set_xscale("log")
        ax_top.axis("off")

        # 右侧边际:n*(离散取值)按取值计数横条,固定条高(取值间距≥5,不会重叠)
        vals, counts = np.unique(d["n_simple_suffices"].values.astype(float), return_counts=True)
        ax_right.barh(vals, counts, height=1.8, color=THEME["base_1"],
                      edgecolor="white", linewidth=0.3, zorder=2)
        ax_right.axis("off")

        # 不调 tight_layout:边际 gridspec + 共享轴与之不兼容(会告警);
        # save_figure_bilingual 存图时统一 bbox_inches='tight' 裁掉白边,标题不会被截。
        ax_top.set_title(title, fontsize=FS + 1, pad=4, loc="center")
        return fig
    return plot, r2


def main() -> None:
    log = _eu.get_logger("66_paper_figures")
    reck = pd.read_excel(os.path.join(_OUT, "65_paper_analysis.xlsx"), sheet_name="reckoning")
    phase = pd.read_excel(os.path.join(_OUT, "63_crossover_analysis.xlsx"), sheet_name="phase_diagram")
    nst = pd.read_excel(os.path.join(_OUT, "63_crossover_analysis.xlsx"), sheet_name="nstar_per_task")

    p_reck = _eu.save_figure_bilingual(make_reckoning(reck), "66_paper_figures", 0)
    p_phase = _eu.save_figure_bilingual(make_phase(phase), "66_paper_figures", 1)
    law_fn, r2 = make_law(nst)
    p_law = _eu.save_figure_bilingual(law_fn, "66_paper_figures", 2)
    log.log(f"清算曲线 {p_reck}")
    log.log(f"决策相图 {p_phase}")
    log.log(f"负律散点 {p_law}  (n_simple_suffices~MMD log-log R2={r2:.4f})")

    # 拷贝到 figs:zh→无后缀(中文稿引用),en→-en 后缀(英文稿引用)。两版都重生成。
    for paths, stem in [(p_reck, "FigReckon"), (p_phase, "FigPhase"), (p_law, "FigLaw")]:
        for lang, suf in (("zh", ""), ("en", "-en")):
            dst = f"{stem}{suf}.pdf"
            shutil.copyfile(paths[lang], os.path.join(_FIGDIR, dst))
            log.log(f"拷贝 {os.path.basename(paths[lang])} -> figs/{dst}")
    print("三张主图完成 -> 04outputs/66_paper_figures-图{a,b,c}.pdf")
    print("           + figs/{FigReckon,FigPhase,FigLaw}[.pdf / -en.pdf]")
    print(f"负律 log-log R2 = {r2:.4f}")


if __name__ == "__main__":
    main()
