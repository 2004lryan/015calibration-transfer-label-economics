"""
66_paper_figures.py: 生成多基准清算论文的三张主图(清算曲线/决策相图/负律散点),中英双版。

数据源:
    04outputs/65_paper_analysis.xlsx (reckoning/phase)
    04outputs/63_crossover_analysis.xlsx (nstar_per_task/scaling_fit/lobo)
输出文件:
    04outputs/66_paper_figures-图a.pdf / -fig-a.pdf  — 清算曲线(简单 vs 经典昂贵 vs 深度/物理,四漂移类型)
    04outputs/66_paper_figures-图b.pdf / -fig-b.pdf  — 决策相图(漂移类型×标签预算→获胜方法)
    04outputs/66_paper_figures-图c.pdf / -fig-c.pdf  — 负律散点(交叉预算 n* 对域漂移量不可预测)
    06doc/01manuscript/figs-csae/{FigReckon,FigPhase,FigLaw}.pdf  — 供中文稿引用的 zh 版拷贝
    05logs/66_paper_figures_*.log
运行方式:
    cd <项目根目录绝对路径>
    python 02code/66_paper_figures.py
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
from matplotlib.patches import Rectangle

plt.rcParams["axes.unicode_minus"] = False

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, ".."))
_OUT = os.path.join(_BASE, "04outputs")
_FIGDIR = os.path.join(_BASE, "06doc", "01manuscript", "figs-csae")
_es = importlib.util.spec_from_file_location("export_utils", os.path.join(_CODE, "01_export_utils.py"))
assert _es is not None  # 对已知存在的 .py 文件 spec 恒非 None
assert _es.loader is not None  # 同理 loader 恒非 None
_eu = importlib.util.module_from_spec(_es)
_es.loader.exec_module(_eu)

# Wong 色盲安全配色
C_SIMPLE, C_CLASSIC, C_DEEP = "#0072B2", "#E69F00", "#009E73"
# 漂移类型顺序与双语名
SHIFTS = ["instrument", "season", "origin_year_instrument", "lab"]
SHIFT_ZH = {"instrument": "仪器漂移", "season": "季节漂移",
            "origin_year_instrument": "跨产地/年漂移", "lab": "跨实验室漂移(MIR)"}
SHIFT_EN = {"instrument": "Instrument", "season": "Season",
            "origin_year_instrument": "Origin/Year", "lab": "Lab (MIR)"}
NCAL = [0, 5, 10, 20, 40]
# 方法双语与配色(决策相图)
METH_ZH = {"zero_shot": "零迁移", "coral": "免标签对齐", "sbc": "斜率/偏置",
           "pds": "分段直接", "target_only": "目标自建", "model_update": "模型更新"}
METH_EN = {"zero_shot": "Zero-shot", "coral": "CORAL", "sbc": "Slope/Bias",
           "pds": "PDS", "target_only": "Target-only", "model_update": "Model-update"}
METH_COLOR = {"zero_shot": "#999999", "coral": "#0072B2", "sbc": "#E69F00",
              "pds": "#CC79A7", "target_only": "#009E73", "model_update": "#D55E00"}


def _parse_ci(s: object) -> tuple[float, float]:
    try:
        lo, hi = str(s).strip("[]").split(",")
        return float(lo), float(hi)
    except Exception:
        return np.nan, np.nan


def _style(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3)


def make_reckoning(reck: pd.DataFrame) -> Callable[[str], Figure]:
    def plot(lang: str) -> Figure:
        if lang == "zh":
            _eu.setup_chinese_font()
        sz = SHIFT_ZH if lang == "zh" else SHIFT_EN
        g_simple = "简单经典" if lang == "zh" else "Simple"
        g_classic = "经典昂贵" if lang == "zh" else "Classical-expensive"
        g_deep = "深度/物理" if lang == "zh" else "Deep/Physics"
        ylab = "目标域均方根误差 RMSEP" if lang == "zh" else "Target-domain RMSEP"
        xlab = "目标标签预算 n" if lang == "zh" else "Target-label budget n"
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
        for ax, st in zip(axes.ravel(), SHIFTS, strict=False):
            sub = reck[reck["shift_type"] == st]
            for grp, lbl, col in [("简单经典", g_simple, C_SIMPLE),
                                  ("经典昂贵", g_classic, C_CLASSIC),
                                  ("深度物理", g_deep, C_DEEP)]:
                d = sub[sub["group"] == grp].set_index("n_cal").reindex(NCAL)
                y = d["median_rmsep"].values
                ci = d["ci95"].apply(_parse_ci)
                lo = np.array([c[0] for c in ci])
                hi = np.array([c[1] for c in ci])
                m = np.isfinite(y)
                ax.plot(np.array(NCAL)[m], y[m], "-o", color=col, lw=1.6, ms=4, label=lbl)
                ax.fill_between(np.array(NCAL)[m], lo[m], hi[m], color=col, alpha=0.15, lw=0)
            ax.set_title(sz[st], fontsize=10)
            ax.set_xticks(NCAL)
            _style(ax)
        axes[1, 0].set_xlabel(xlab)
        axes[1, 1].set_xlabel(xlab)
        axes[0, 0].set_ylabel(ylab)
        axes[1, 0].set_ylabel(ylab)
        axes[0, 0].legend(fontsize=8, frameon=False, loc="best")
        fig.tight_layout()
        return fig
    return plot


def make_phase(phase: pd.DataFrame) -> Callable[[str], Figure]:
    # 每(shift,n_cal)取获胜方法
    win = {}
    for _, r in phase.iterrows():
        fr = {k.replace("win_", ""): r[k] for k in phase.columns if k.startswith("win_")}
        top = max(fr, key=lambda k: fr[k])  # 等价 key=fr.get，改用下标索引以满足严格类型
        win[(r["shift_type"], int(r["n_cal"]))] = (top, fr[top])

    def plot(lang: str) -> Figure:
        if lang == "zh":
            _eu.setup_chinese_font()
        sz = SHIFT_ZH if lang == "zh" else SHIFT_EN
        mz = METH_ZH if lang == "zh" else METH_EN
        xlab = "目标标签预算 n" if lang == "zh" else "Target-label budget n"
        title = "决策相图：各漂移类型×标签预算下的获胜方法" if lang == "zh" \
            else "Decision phase diagram: winning method by shift type × label budget"
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        for i, st in enumerate(SHIFTS):
            for j, nc in enumerate(NCAL):
                meth, frac = win.get((st, nc), (None, 0))
                if meth is None:
                    continue
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=METH_COLOR.get(meth, "#ccc"),
                                       edgecolor="white", lw=1.5, alpha=0.85))
                ax.text(j + 0.5, i + 0.62, mz.get(meth, meth), ha="center", va="center",
                        fontsize=7.5, color="white", weight="bold")
                ax.text(j + 0.5, i + 0.30, f"{frac:.0%}", ha="center", va="center",
                        fontsize=7, color="white")
        ax.set_xlim(0, len(NCAL))
        ax.set_ylim(0, len(SHIFTS))
        ax.set_xticks(np.arange(len(NCAL)) + 0.5)
        ax.set_xticklabels(NCAL)  # type: ignore[arg-type]  # matplotlib 运行时接受 int 刻度标签，stub 仅声明 str
        ax.set_yticks(np.arange(len(SHIFTS)) + 0.5)
        ax.set_yticklabels([sz[s] for s in SHIFTS])
        ax.set_xlabel(xlab)
        ax.set_title(title, fontsize=10)
        ax.invert_yaxis()
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        fig.tight_layout()
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
        if lang == "zh":
            _eu.setup_chinese_font()
        sz = SHIFT_ZH if lang == "zh" else SHIFT_EN
        xlab = "域漂移量（最大均值差异 MMD，对数轴）" if lang == "zh" \
            else "Domain-shift magnitude (MMD, log axis)"
        ylab = "交叉预算 n*（简单方法追平所需标签数）" if lang == "zh" \
            else "Crossover budget n* (labels for simple to catch up)"
        title = (f"交叉预算不可由域漂移量预测（对数-对数 $R^2$={r2:.3f}）" if lang == "zh"
                 else f"n* is not predictable from domain-shift magnitude (log-log $R^2$={r2:.3f})")
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for st in SHIFTS:
            s = d[d["shift_type"] == st]
            ax.scatter(s["mmd"], s["n_simple_suffices"], s=34, alpha=0.75,
                       color=METH_COLOR["coral"] if False else None,
                       label=sz[st], edgecolor="white", lw=0.5,
                       c=[{"instrument": "#0072B2", "season": "#E69F00",
                           "origin_year_instrument": "#009E73", "lab": "#CC79A7"}[st]])
        xs = np.linspace(d["mmd"].min(), d["mmd"].max(), 50)
        ax.plot(xs, 10 ** (a + b * np.log10(xs + 1e-9)) - 1.0, "--", color="#555555", lw=1.3,
                label=("拟合（近水平）" if lang == "zh" else "Fit (near-flat)"))
        ax.set_xscale("log")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc="best")
        _style(ax)
        fig.tight_layout()
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

    # 拷贝 zh 版到 figs-csae 供中文稿引用
    for src, dst in [(p_reck["zh"], "FigReckon.pdf"), (p_phase["zh"], "FigPhase.pdf"),
                     (p_law["zh"], "FigLaw.pdf")]:
        shutil.copyfile(src, os.path.join(_FIGDIR, dst))
        log.log(f"拷贝 {os.path.basename(src)} -> figs-csae/{dst}")
    print("三张主图完成 -> 04outputs/66_paper_figures-图{a,b,c}.pdf + figs-csae/{FigReckon,FigPhase,FigLaw}.pdf")
    print(f"负律 log-log R2 = {r2:.4f}")


if __name__ == "__main__":
    main()
