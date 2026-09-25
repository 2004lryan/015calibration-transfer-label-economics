"""
78_map_and_regret_figure.py: 合并图 FigMap（返修图 3），中英双版，一列两面板。
【为什么必须做这个实验】
    返修意见（导师，2026-09-08）：原图 3（样本内频率图）单独成图信息量偏低，图 3 与图 4
    宜合并为一列两面板并缩小。同时审稿人 1 第 5 条要求点明零标签下只有免标签方法可用，
    审稿人 2 第 1 条要求回答"漂移未知时怎么办"。把"样本内谁最常赢"（描述）与"当成处方
    用在未见基准上的代价"（留一基准遗憾；前瞻检验尚待进行，见正文局限第五条）画在同一张图的上下两格，正是这两条意见的图示：
    上格给规则，下格给该规则外推时的代价，读者一眼看到"图是描述、处方才能推广"。
    本脚本不重训任何模型，只读已落地结果。
输入文件:
    04outputs/73_heuristic_map_lobo.xlsx（sheet: per_task 给 (a)，summary 给 (b)）
输出文件:
    04outputs/78_map_and_regret_figure-图a.pdf / -fig-a.pdf
    06doc/01manuscript/figs/FigMap.pdf（中文稿）/ FigMap-en.pdf（英文稿）
    05logs/78_map_and_regret_figure_*.log
运行方式:
    cd <项目根目录绝对路径>
    python 02code/78_map_and_regret_figure.py
视觉规程:与 66/76/77 号一致（THEME Wong 色板 / 字体栈 / 7 磅字号 / 误差带类型显式标注）。
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
from matplotlib.patches import FancyBboxPatch

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
NCAL = [0, 5, 10, 20, 40]
SHIFTS = ["instrument", "season", "lab", "origin_year_instrument"]
SHIFT_ZH = {"instrument": "仪器", "season": "季节", "lab": "实验室", "origin_year_instrument": "产地/年份"}
SHIFT_EN = {
    "instrument": "Instrument",
    "season": "Season",
    "lab": "Laboratory",
    "origin_year_instrument": "Origin/year",
}
METH_ZH = {
    "coral": "CORAL",
    "sbc": "斜率/偏置",
    "pds": "PDS",
    "model_update": "模型更新",
    "target_only": "目标自建",
    "zero_shot": "直接迁移",
    "cnn_zeroshot": "CNN 零样本",
    "cnn_finetune": "CNN 微调",
    "dann": "DANN",
    "deepcoral": "Deep CORAL",
    "physbl_zeroshot": "Phys+BL 零样本",
    "physbl_ft": "Phys+BL 微调",
}
METH_EN = {
    "coral": "CORAL",
    "sbc": "Slope/bias",
    "pds": "PDS",
    "model_update": "Model update",
    "target_only": "Target-only",
    "zero_shot": "Direct transfer",
    "cnn_zeroshot": "Zero-shot CNN",
    "cnn_finetune": "Fine-tuned CNN",
    "dann": "DANN",
    "deepcoral": "DeepCORAL",
    "physbl_zeroshot": "Zero-shot Phys+BL",
    "physbl_ft": "Fine-tuned Phys+BL",
}
# 深度族统一走一条暖色阶，读者一眼能看出哪一列整列翻给了深度族。
METH_COLOR = {
    "coral": THEME["amber"],
    "sbc": THEME["positive"],
    "pds": THEME["purple"],
    "model_update": THEME["deep_2"],
    "target_only": THEME["base_2"],
    "zero_shot": THEME["base_1"],
    "deepcoral": "#F0A57C",
    "cnn_zeroshot": "#E8845B",
    "cnn_finetune": THEME["signal"],
    "physbl_ft": "#B24502",
    "dann": "#A34A00",
    "physbl_zeroshot": "#7A3B12",
}
# 留一基准遗憾:处方 -> (中文, 英文, 颜色, 线型, 标记)
RULES: list[tuple[str, str, str, str, str, str]] = [
    (
        "always_simple",
        "最简可用校正（0 标签 CORAL，否则斜率/偏置）",
        "Simplest applicable correction",
        THEME["amber"],
        "-",
        "o",
    ),
    ("always_model_update", "始终模型更新", "Always model update", THEME["deep_2"], "-", "s"),
    ("always_target_only", "始终目标自建", "Always target-only", THEME["base_2"], ":", "^"),
    (
        "B_shift_unknown_bal",
        "频率图当规则（类型未知）",
        "Frequency map as a rule (type unknown)",
        THEME["purple"],
        "--",
        "D",
    ),
    (
        "A_shift_known_bal",
        "频率图当规则（类型已知）",
        "Frequency map as a rule (type known)",
        THEME["signal"],
        "--",
        "v",
    ),
    ("default_deep", "默认深度模型", "Default deep model", THEME["deep_1"], "-", "P"),
]
plt.rcParams.update(
    {
        "font.size": FS,
        "font.family": list(_FONT_STACK["zh"]),
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _apply_font(lang: str) -> None:
    plt.rcParams["font.family"] = list(_FONT_STACK[lang])


def _style(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2, width=0.5)


def _panel_tag(ax: Axes, tag: str, x: float = -0.085, y: float = 1.02) -> None:
    ax.text(x, y, tag, transform=ax.transAxes, fontweight="bold", fontsize=FS + 2, va="bottom", ha="left")


def _winners(per_task: pd.DataFrame) -> dict[tuple[str, int], list[tuple[str, float]]]:
    """每个(漂移类型, 预算)单元里，全部 12 个候选方法的获胜频率组成，降序。

    逐任务的 oracle 就是该任务该预算下误差最小的方法，与 73 号 insample_map 同一口径。
    这里不能用 63 号的 phase_diagram：那张表只给 6 个经典方法开了 win_ 列，胜者是深度
    方法时该单元会退成"最常获胜的经典方法"，而份额仍按全部任务算，于是土壤列被画成
    「CORAL 25%」——真正的胜者是 DeepCORAL 42%。
    """
    d = per_task[per_task["universe"] == "fig4"].drop_duplicates(["task_id", "n_cal"])
    out: dict[tuple[str, int], list[tuple[str, float]]] = {}
    for (st, nc), g in d.groupby(["shift_type", "n_cal"]):
        vc = g["oracle"].value_counts(normalize=True)
        out[(str(st), int(nc))] = [(str(m), float(v)) for m, v in vc.items() if v > 0]
    return out


def make_map_and_regret(per_task: pd.DataFrame, summ: pd.DataFrame) -> Callable[[str], Figure]:
    win = _winners(per_task)
    reg = summ[(summ.universe == "fig4") & (summ.benchmark == "ALL")].copy()
    reg["n_cal"] = reg["n_cal"].astype(str)

    def plot(lang: str) -> Figure:
        _apply_font(lang)
        sz = SHIFT_ZH if lang == "zh" else SHIFT_EN
        mz = METH_ZH if lang == "zh" else METH_EN
        # 一行两列，两格各近正方：(a) 频率图转置为 预算(行) x 漂移类型(列)；(b) 遗憾曲线
        fig = plt.figure(figsize=(7.2, 4.15))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.13], width_ratios=[1.0, 1.06], wspace=0.30, hspace=0.06)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_l = fig.add_subplot(gs[1, :])
        ax_l.axis("off")
        # ── (a) 频率图：行=标签预算，列=漂移类型 ──
        pad = 0.05
        for j, st in enumerate(SHIFTS):
            for i, nc in enumerate(NCAL):
                comp = win.get((st, nc))
                if not comp:
                    continue
                meth, frac = comp[0]
                ax_a.add_patch(
                    FancyBboxPatch(
                        (j + pad, i + pad),
                        1 - 2 * pad,
                        1 - 2 * pad,
                        boxstyle="round,pad=0,rounding_size=0.07",
                        facecolor=METH_COLOR.get(meth, "#cccccc"),
                        edgecolor="white",
                        lw=0.7,
                        alpha=0.93,
                        zorder=3,
                    )
                )
                lbl = mz.get(meth, meth).replace(" ", "\n").replace("/", "/\n")
                ax_a.text(
                    j + 0.5,
                    i + 0.40,
                    lbl,
                    ha="center",
                    va="center",
                    fontsize=FS - 1.6,
                    color="white",
                    weight="bold",
                    zorder=5,
                    linespacing=1.0,
                )
                ax_a.text(
                    j + 0.5,
                    i + 0.72,
                    f"{frac:.0%}",
                    ha="center",
                    va="center",
                    fontsize=FS - 1.8,
                    color="white",
                    alpha=0.95,
                    zorder=5,
                )
                x0, wtot, ybar, hbar = j + 0.17, 0.66, i + 0.855, 0.065
                acc = 0.0
                for k, (m2, v2) in enumerate(comp):
                    w = wtot * v2
                    ax_a.add_patch(
                        FancyBboxPatch(
                            (x0 + acc, ybar),
                            w,
                            hbar,
                            boxstyle="square,pad=0",
                            facecolor="white" if k == 0 else METH_COLOR.get(m2, "#cccccc"),
                            edgecolor="white",
                            lw=0.2,
                            alpha=0.95 if k == 0 else 0.85,
                            zorder=5,
                        )
                    )
                    acc += w
        ax_a.set_xlim(0, len(SHIFTS))
        ax_a.set_ylim(len(NCAL), 0)
        ax_a.set_aspect("equal", adjustable="box")
        ax_a.set_xticks(np.arange(len(SHIFTS)) + 0.5)
        ax_a.set_xticklabels([sz[s2].replace("/", "/\n").replace(" ", "\n") for s2 in SHIFTS], fontsize=FS - 1)
        ax_a.set_yticks(np.arange(len(NCAL)) + 0.5)
        ax_a.set_yticklabels([str(n) for n in NCAL])
        ax_a.set_ylabel("目标标签预算 n" if lang == "zh" else "Target-label budget n")
        ax_a.set_title(
            "样本内频率图：最常获胜的方法" if lang == "zh" else "In-sample frequency map: most frequent winner",
            fontsize=FS + 0.3,
            pad=4,
        )
        ax_a.text(
            len(SHIFTS) + 0.10,
            0.5,
            "仅免标签方法" if lang == "zh" else "label-free only",
            rotation=-90,
            ha="left",
            va="center",
            fontsize=FS - 1.8,
            color=THEME["ref"],
            style="italic",
        )
        for sp in ax_a.spines.values():
            sp.set_visible(False)
        ax_a.tick_params(length=0)
        _panel_tag(ax_a, "a", x=-0.26, y=1.01)
        # ── (b) 留一基准遗憾 ──
        for key, zh, en, col, ls, mk in RULES:
            d = reg[reg.rule == key].set_index("n_cal")
            ys = [100 * float(d.loc[str(n), "median_regret"]) if str(n) in d.index else np.nan for n in NCAL]
            xs = np.array(NCAL, dtype=float)
            m = np.isfinite(ys)
            ax_b.plot(
                xs[m],
                np.array(ys)[m],
                ls=ls,
                color=col,
                lw=1.1,
                marker=mk,
                ms=3.0,
                mec="white",
                mew=0.4,
                zorder=4,
                label=(zh if lang == "zh" else en),
            )
            if "pooled" in d.index:
                lo, hi = 100 * float(d.loc["pooled", "ci95_lo"]), 100 * float(d.loc["pooled", "ci95_hi"])
                ax_b.plot([44.5, 44.5], [lo, hi], color=col, lw=1.5, solid_capstyle="butt", zorder=4, alpha=0.85)
                ax_b.plot(
                    [44.5],
                    [100 * float(d.loc["pooled", "median_regret"])],
                    marker=mk,
                    ms=3.0,
                    color=col,
                    mec="white",
                    mew=0.4,
                    zorder=5,
                )
        ax_b.axhline(0, color=THEME["ref"], lw=0.6, zorder=2)
        # 标注文字按当期数据现算：写死的字符串在重跑之后点会动、字不会动。
        for key in ("always_simple", "default_deep"):
            d2 = reg[(reg.rule == key) & (reg.n_cal == "pooled")]
            if len(d2):
                ax_b.annotate(
                    f"{100 * float(d2.median_regret.iloc[0]):.1f}%",
                    xy=(44.5, 100 * float(d2.median_regret.iloc[0])),
                    xytext=(5, 0),
                    textcoords="offset points",
                    fontsize=FS - 1.6,
                    color={r[0]: r[3] for r in RULES}[key],
                    va="center",
                    weight="bold",
                )
        ax_b.set_yscale("symlog", linthresh=10, linscale=0.8)
        ax_b.set_ylim(-2.0, 420)
        ax_b.set_yticks([0, 5, 10, 30, 100, 300])
        ax_b.set_yticklabels(["0", "5", "10", "30", "100", "300"])
        ax_b.set_xticks([*NCAL, 44.5])
        ax_b.set_xticklabels([*[str(n) for n in NCAL], "汇总" if lang == "zh" else "pooled"], fontsize=FS - 1)
        ax_b.set_xlim(-3.0, 52)
        ax_b.set_xlabel("目标标签预算 n" if lang == "zh" else "Target-label budget n", labelpad=1.5)
        ax_b.set_ylabel("留一基准遗憾（%，对称对数刻度）" if lang == "zh" else "Leave-one-benchmark-out regret (%, symmetric log)")
        ax_b.set_title(
            "留一基准遗憾" if lang == "zh" else "Leave-one-benchmark-out regret",
            fontsize=FS + 0.3,
            pad=4,
        )
        ax_b.grid(axis="y", ls=":", lw=0.4, color=THEME["grid"], alpha=0.7, zorder=0)
        _style(ax_b)
        _panel_tag(ax_b, "b", x=-0.24, y=1.01)
        # ── 底部图例：只列 (b) 的处方线；(a) 的瓦片自带方法名，无需色例 ──
        hs: list[Line2D] = [
            Line2D([], [], color=r[3], ls=r[4], marker=r[5], ms=3.0, lw=1.1, label=(r[1] if lang == "zh" else r[2]))
            for r in RULES
        ]
        ax_l.legend(
            handles=hs,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.50),
            ncol=3,
            fontsize=FS - 1.4,
            handlelength=1.6,
            handletextpad=0.4,
            labelspacing=0.35,
            columnspacing=1.6,
            borderaxespad=0.0,
            frameon=False,
        )
        fig.subplots_adjust(left=0.10, right=0.975, top=0.94, bottom=0.045)
        return fig

    return plot


def main() -> None:
    log = _eu.get_logger("78_map_and_regret_figure")
    per_task = pd.read_excel(os.path.join(_OUT, "73_heuristic_map_lobo.xlsx"), sheet_name="per_task")
    summ = pd.read_excel(os.path.join(_OUT, "73_heuristic_map_lobo.xlsx"), sheet_name="summary")
    log.log(f"逐任务表 {len(per_task)} 行；遗憾表 {len(summ)} 行")
    paths = _eu.save_figure_bilingual(make_map_and_regret(per_task, summ), "78_map_and_regret_figure", 0)
    for lang, suf in (("zh", ""), ("en", "-en")):
        dst = os.path.join(_FIGDIR, f"FigMap{suf}.pdf")
        shutil.copyfile(paths[lang], dst)
        log.log(f"拷贝 {os.path.basename(paths[lang])} -> figs/FigMap{suf}.pdf")
    print("FigMap 完成 -> figs/FigMap[.pdf / -en.pdf]")


if __name__ == "__main__":
    main()
