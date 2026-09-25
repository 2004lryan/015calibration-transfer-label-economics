"""
76_revision_figures.py: 返修新增主图 FigStruct（漂移"结构而非幅度"），脚本 74 结果的可视化，中英双版。
【为什么必须做这个实验】
    审稿人 1 的唯一 major：机理洞见要在已引入的公开基准上演示。74 号已算出五基准 98 域对的
    低阶份额 f_mean / f_coral、高阶残余与低秩失稳指标 A，及其与 CORAL 增益、零标签 NRMSEP 的相关
    （域对聚类 bootstrap 95% CI）。本脚本只画图、不算新数。返修二轮（导师意见）把原独立的
    标度律否定图并入本图第 (b) 格，使"幅度不行、结构可以"这一对照落在同一张图上：
    仪器漂移几乎全是低阶变换（a）、交叉预算不能由漂移幅度预测（b）、低阶份额追踪 CORAL 增益（c）、
    A 预警直接迁移的失败（d）。
数据源:
    04outputs/74_shift_structure_diagnosis.xlsx（表a：by_benchmark / correlations / per_task）
    04outputs/63_crossover_analysis.xlsx（sheet: nstar_per_task，供 (b) 格标度律否定）
输出文件:
    04outputs/76_revision_figures-图a.pdf / -fig-a.pdf
    06doc/01manuscript/figs/FigStruct.pdf（中文稿引用）/ FigStruct-en.pdf（英文稿引用）
    05logs/76_revision_figures_*.log
运行方式:
    cd <项目根目录绝对路径>
    python 02code/76_revision_figures.py
视觉规程:与 66 号逐字节一致(THEME Wong 色板 / 字体栈 / rcParams / 面板小标签),见 66 号文件头。
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
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, ".."))
_OUT = os.path.join(_BASE, "04outputs")
_FIGDIR = os.path.join(_BASE, "06doc", "01manuscript", "figs")
_es = importlib.util.spec_from_file_location("export_utils", os.path.join(_CODE, "01_export_utils.py"))
assert _es is not None
assert _es.loader is not None
_eu = importlib.util.module_from_spec(_es)
_es.loader.exec_module(_eu)

THEME = {
    "signal": "#D55E00",
    "positive": "#009E73",
    "deep_1": "#56B4E9",
    "deep_2": "#0072B2",
    "base_1": "#BBBBBB",
    "base_2": "#888888",
    "amber": "#E69F00",
    "purple": "#CC79A7",
    "ref": "#4D4D4D",
    "grid": "#C9CDD2",
}
_FONT_STACK = {
    "zh": ["Times New Roman", "SimSun", "Songti SC"],
    "en": ["Times New Roman", "DejaVu Serif"],
}
FS = 7


def _apply_font(lang: str) -> None:
    plt.rcParams["font.family"] = list(_FONT_STACK[lang])


plt.rcParams.update(
    {
        "font.size": FS,
        "axes.labelsize": FS,
        "axes.titlesize": FS,
        "xtick.labelsize": FS,
        "ytick.labelsize": FS,
        "legend.fontsize": FS,
        "font.family": list(_FONT_STACK["zh"]),
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "lines.linewidth": 0.9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# 基准顺序按低阶份额从高到低(与正文表 struct 一致);颜色+形状双编码
BENCH = ["corn", "tablet", "ossl_mir", "mango", "apple"]
BENCH_EN = {
    "corn": "Corn (instrument)",
    "tablet": "Tablet (instrument)",
    "ossl_mir": "Soil (laboratory)",
    "mango": "Mango (season)",
    "apple": "Apple (origin/year)",
}
BENCH_ZH = {
    "corn": "玉米（仪器）",
    "tablet": "药片（仪器）",
    "ossl_mir": "土壤（实验室）",
    "mango": "芒果（季节）",
    "apple": "苹果（产地/年份）",
}
BENCH_COLOR = {
    "corn": THEME["deep_2"],
    "tablet": THEME["deep_1"],
    "ossl_mir": THEME["purple"],
    "mango": THEME["positive"],
    "apple": THEME["signal"],
}
BENCH_MARKER = {"corn": "o", "tablet": "s", "ossl_mir": "^", "mango": "D", "apple": "v"}


def _style(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2, width=0.5)


def _panel_tag(ax: Axes, tag: str, x: float = -0.18, y: float = 1.03) -> None:
    ax.text(x, y, tag, transform=ax.transAxes, fontweight="bold", fontsize=FS + 2, va="bottom", ha="left")


def _rho(corr: pd.DataFrame, pred: str, tgt: str) -> tuple[float, float, float]:
    r = corr[(corr["predictor"] == pred) & (corr["target"] == tgt)].iloc[0]
    return float(r["spearman_rho"]), float(r["ci95_lo"]), float(r["ci95_hi"])


def make_struct(bb: pd.DataFrame, pt: pd.DataFrame, corr: pd.DataFrame, nst: pd.DataFrame) -> Callable[[str], Figure]:
    bb = bb.set_index("benchmark")
    rho_b = _rho(corr, "f_coral_sw1", "coral_gain")
    rho_c = _rho(corr, "A_lowrank", "nrmsep_zero_shot0")
    law = nst.dropna(subset=["n_simple_suffices", "mmd"]).copy()
    lx = np.log10(law["mmd"].to_numpy(dtype=float) + 1e-9)
    ly = np.log10(law["n_simple_suffices"].to_numpy(dtype=float) + 1.0)
    slope, intercept = np.polyfit(lx, ly, 1)
    r2_law = 1.0 - float(np.sum((ly - (intercept + slope * lx)) ** 2) / np.sum((ly - ly.mean()) ** 2))

    def plot(lang: str) -> Figure:
        _apply_font(lang)
        zh = lang == "zh"
        names = BENCH_ZH if zh else BENCH_EN
        fig, axes2 = plt.subplots(2, 2, figsize=(7.2, 5.0))
        axes = [axes2[0, 0], axes2[1, 0], axes2[1, 1]]  # a=分解, c=份额, d=放大比（b 单列在下）
        ax_law = axes2[0, 1]
        # (a) 距离分解:均值可解释 / 二阶矩再解释 / 高阶残余(SW1,基准中位)
        ax = axes[0]
        y = np.arange(len(BENCH))[::-1]
        f_mean = bb.loc[BENCH, "f_mean_sw1"].to_numpy(dtype=float)
        f_coral = bb.loc[BENCH, "f_coral_sw1"].to_numpy(dtype=float)
        resid = 1.0 - f_coral
        parts = [
            (f_mean, THEME["base_2"], "均值匹配可消除" if zh else "removed by mean matching"),
            (f_coral - f_mean, THEME["deep_2"], "二阶矩（CORAL）再消除" if zh else "further removed by CORAL"),
            (resid, THEME["signal"], "高阶残余" if zh else "higher-order residual"),
        ]
        left = np.zeros(len(BENCH))
        handles_a = []
        for vals, col, lab in parts:
            ax.barh(y, vals, left=left, color=col, edgecolor="white", lw=0.4, height=0.62)
            left = left + vals
            handles_a.append(Patch(facecolor=col, edgecolor="white", lw=0.4, label=lab))
        for yi, r in zip(y, resid, strict=True):
            ax.text(1.005, yi, f"{r:.0%}", va="center", ha="left", fontsize=FS - 1, color=THEME["signal"])
        ax.set_yticks(y)
        ax.set_yticklabels([names[b] for b in BENCH])
        ax.set_xlim(0, 1.13)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("源–目标距离份额（sliced-W1）" if zh else "Share of source–target distance (sliced-W1)")
        ax.legend(
            handles=handles_a,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=1,
            fontsize=FS - 1,
            handlelength=1.0,
            handletextpad=0.4,
            labelspacing=0.25,
            borderaxespad=0.0,
        )
        _style(ax)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        _panel_tag(ax, "a", x=-0.42)
        # (b) 交叉预算不能由漂移幅度预测:任务散点 + 各基准分别拟合(斜率不一致) + 总体近水平
        ax = ax_law
        for b in BENCH:
            g = law[law["benchmark"] == b]
            if len(g) == 0:
                continue
            rng = np.random.default_rng(20260908 + len(g))
            jit = 1.0 + 0.085 * rng.standard_normal(len(g))
            ax.scatter(
                g["mmd"],
                g["n_simple_suffices"].to_numpy(dtype=float) * jit,
                s=11,
                marker=BENCH_MARKER[b],
                color=BENCH_COLOR[b],
                alpha=0.8,
                linewidths=0.3,
                edgecolors="white",
                zorder=3,
            )
            if len(g) >= 6:
                gx = np.log10(g["mmd"].to_numpy(dtype=float) + 1e-9)
                gy = np.log10(g["n_simple_suffices"].to_numpy(dtype=float) + 1.0)
                sb, ib = np.polyfit(gx, gy, 1)
                xs_b = np.linspace(gx.min(), gx.max(), 30)
                ax.plot(10**xs_b, 10 ** (ib + sb * xs_b) - 1.0, color=BENCH_COLOR[b], lw=0.8, alpha=0.75, zorder=2)
        xs = np.linspace(lx.min(), lx.max(), 60)
        ax.plot(
            10**xs,
            10 ** (intercept + slope * xs) - 1.0,
            ls="--",
            color=THEME["ref"],
            lw=1.1,
            zorder=4,
            label=("总体拟合（近水平）" if zh else "Pooled fit (near-flat)"),
        )
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([5, 10, 20, 40]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(3.6, 70)
        ax.set_xlabel("漂移幅度（最大均值差异 MMD，对数）" if zh else "Shift magnitude (MMD, log)")
        ax.set_ylabel("交叉预算 n*（所需标签数）" if zh else "Crossover budget $n^{*}$ (labels)")
        ax.text(
            0.03,
            0.97,
            (
                f"对数-对数 R² = {r2_law:.3f}\n细线：各基准分别拟合，斜率互不一致"
                if zh
                else f"log-log R² = {r2_law:.3f}\nthin lines: per-benchmark fits, slopes disagree"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=FS - 1,
            color=THEME["ref"],
        )
        ax.legend(fontsize=FS - 1, frameon=False, loc="lower left", handlelength=1.6, borderaxespad=0.3)
        _style(ax)
        _panel_tag(ax, "b", x=-0.24)
        # (b) 低阶份额 vs CORAL 增益(任务点)
        ax = axes[1]
        for b in BENCH:
            g = pt[pt["benchmark"] == b]
            ax.scatter(
                g["f_coral_sw1"],
                g["coral_gain"],
                s=11,
                marker=BENCH_MARKER[b],
                color=BENCH_COLOR[b],
                alpha=0.8,
                linewidths=0.3,
                edgecolors="white",
                label=names[b],
                zorder=3,
            )
        ax.set_xlabel("低阶份额（前两阶矩可消除）" if zh else "Low-order share $f_{\\mathrm{CORAL}}$")
        ax.set_ylabel("CORAL 增益（相对误差下降）" if zh else "CORAL gain (relative error reduction)")
        ax.axhline(0, color=THEME["grid"], lw=0.5, zorder=1)
        ax.text(
            0.03,
            0.97,
            f"Spearman ρ = {rho_b[0]:.2f} [{rho_b[1]:.2f}, {rho_b[2]:.2f}]",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=FS - 1,
            color=THEME["ref"],
        )
        _style(ax)
        _panel_tag(ax, "c", x=-0.24)
        # (c) 低秩残差放大比 A vs 零标签 NRMSEP(对数横轴)
        ax = axes[2]
        for b in BENCH:
            g = pt[pt["benchmark"] == b]
            ax.scatter(
                g["A_lowrank"],
                g["nrmsep_zero_shot0"],
                s=11,
                marker=BENCH_MARKER[b],
                color=BENCH_COLOR[b],
                alpha=0.8,
                linewidths=0.3,
                edgecolors="white",
                zorder=3,
            )
        ax.set_xscale("log")
        ticks = [1, 2, 5, 10, 20, 50]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("低秩残差放大比 A（对数）" if zh else "Low-rank residual amplification $A$ (log)")
        ax.set_ylabel("零标签 NRMSEP（直接迁移）" if zh else "Zero-label NRMSEP (direct transfer)")
        ax.axhline(1, color=THEME["grid"], lw=0.5, zorder=1)
        ax.text(
            0.03,
            0.97,
            f"Spearman ρ = {rho_c[0]:.2f} [{rho_c[1]:.2f}, {rho_c[2]:.2f}]",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=FS - 1,
            color=THEME["ref"],
        )
        _style(ax)
        _panel_tag(ax, "d", x=-0.24)
        # 共享图例(基准,颜色+形状)置于整图下方
        handles = [
            Line2D([], [], marker=BENCH_MARKER[b], color=BENCH_COLOR[b], linestyle="none", markersize=4, label=names[b])
            for b in BENCH
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=5,
            bbox_to_anchor=(0.5, -0.018),
            fontsize=FS - 0.5,
            handletextpad=0.3,
            columnspacing=1.2,
        )
        fig.tight_layout(pad=0.5, w_pad=1.6, h_pad=1.4, rect=(0, 0.055, 1, 1))
        return fig

    return plot


def main() -> None:
    log = _eu.get_logger("76_revision_figures")
    src = os.path.join(_OUT, "74_shift_structure_diagnosis.xlsx")
    xl = pd.ExcelFile(src)
    bb = xl.parse(xl.sheet_names[0])  # 表a：by_benchmark
    pt = xl.parse("per_task")
    corr = xl.parse("correlations")
    nst = pd.read_excel(os.path.join(_OUT, "63_crossover_analysis.xlsx"), sheet_name="nstar_per_task")
    paths = _eu.save_figure_bilingual(make_struct(bb, pt, corr, nst), "76_revision_figures", 0)
    for lang, suf in (("zh", ""), ("en", "-en")):
        dst = os.path.join(_FIGDIR, f"FigStruct{suf}.pdf")
        shutil.copyfile(paths[lang], dst)
        log.log(f"拷贝 {os.path.basename(paths[lang])} -> figs/FigStruct{suf}.pdf")
    print("FigStruct 完成 -> 04outputs/76_revision_figures-图a.pdf + figs/FigStruct[.pdf / -en.pdf]")


if __name__ == "__main__":
    main()
