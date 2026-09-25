"""
77_framework_figure.py: 研究框架图 FigFramework（返修新增图 1），中英双版。
【为什么必须做这个实验】
    导师返修意见（2026-09-08）：稿件缺一张总览框架图。审稿人 2 亦嫌全文过长、主线不清；
    一张"五基准 → 统一协议 → 三族方法 × 标签预算 → 四条分析线（+ SI 苹果案例）"的框架图
    让读者在进入方法节之前就看到主线与各节的对应关系。本脚本只画示意图，不含任何数据计算。
    视觉风格参照 CILS 已刊文章的总览示意图（彩色填充框 + 小图标 + 内嵌缩略示意曲线），
    四个阶段各占一条带、带内用卡片，阶段之间用宽箭头连接。
输出文件:
    04outputs/77_framework_figure-图a.pdf / -fig-a.pdf
    06doc/01manuscript/figs/FigFramework.pdf（中文稿）/ FigFramework-en.pdf（英文稿）
    05logs/77_framework_figure_*.log
运行方式:
    cd <项目根目录绝对路径>
    python 02code/77_framework_figure.py
视觉规程:与 66/76 号一致(THEME Wong 色板 / 字体栈 / 7 磅字号);示意曲线为手绘形状,不代表任何数据。
"""

import importlib.util
import os
import shutil
from collections.abc import Callable
from typing import Any

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, ".."))
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
W, H = 144.0, 100.0  # 画布单位(等比例,1 单位 = 0.05 英寸)
Glyph = Callable[[Axes, float, float, float, float, str], None]

# ---- 文案(双语) ----
TXT: dict[str, dict[str, str]] = {
    "en": {
        "s1": "Inputs: five calibration-transfer benchmarks, four shift types, two modalities",
        "s2": "One transfer protocol for all 126 ordered source → target tasks",
        "s3": "Three method families of increasing cost, each at every label budget",
        "s4": "Four analyses on the same runs (Sections 3.1–3.4) and one case study",
        "corn": "Corn|3 instruments · NIR|instrument shift|24 tasks",
        "tablet": "Tablet (IDRC)|2 instruments · NIR|instrument shift|6 tasks",
        "mango": "Mango|4 seasons · NIR|seasonal shift|12 tasks",
        "soil": "Soil (OSSL)|3 laboratories · MIR|laboratory shift|12 tasks",
        "apple": "Apple (in-house)|3 origins/season, 4 total · NIR|origin/year shift|72 tasks",
        "p1": "Source domain|trains the model|(PLS or network)",
        "p2": "Target domain|calibration pool: n ∈ {0, 5, 10, 20, 40} labels|held-out test set, fixed across budgets",
        "p3": "Evaluation|RMSEP and NRMSEP = RMSEP/σ|5 fixed seeds · bootstrap clustered on task",
        "m1": "Simple corrections|CORAL (label-free, transductive)|slope/bias correction (2 parameters)",
        "m2": "Classical model update|PDS · model update|target-only PLS",
        "m3": "Deep / physics representations|1D-CNN · DANN · Deep CORAL|Phys+BL (Beer–Lambert prior)",
        "a1": "3.1 Label-budget reckoning|best simple vs best deep;|selection-free re-analysis",
        "a2": "3.2 Label-free alignment|frequency map →|leave-one-benchmark-out regret",
        "a3": "3.3 Scaling-law test|n* vs shift magnitude|(MMD, W1, dimensionality)",
        "a4": "3.4 Shift structure|low-order share f and low-rank|amplification A, from spectra only",
        "case": (
            "Apple case study (3.5; S1–S12)|Phys+BL vs classical transfer,|"
            "602 scenarios; BL instability A;|matched-budget slope/bias control"
        ),
        "out": "Outputs",
        "o1": "when the simplest correction suffices",
        "o2": "a prescription that needs no shift diagnosis",
        "o3": "no scaling law from shift magnitude",
        "o4": "structure, not magnitude, tracks where simple corrections win",
        "lbl_shift": "source / target spectra",
        "lbl_budget": "label budget n",
    },
    "zh": {
        "s1": "输入：五个标定转移基准，四类漂移，两种模态",
        "s2": "126 个有序「源 → 目标」迁移任务，同一套协议",
        "s3": "三族方法，代价递增，每族都跑遍全部标签预算",
        "s4": "同一批运行上的四条分析线（3.1–3.4 节）与一个案例",
        "corn": "玉米|3 台仪器 · 近红外|仪器漂移|24 个任务",
        "tablet": "药片（IDRC）|2 台仪器 · 近红外|仪器漂移|6 个任务",
        "mango": "芒果|4 个季节 · 近红外|季节漂移|12 个任务",
        "soil": "土壤（OSSL）|3 个实验室 · 中红外|实验室漂移|12 个任务",
        "apple": "苹果（自采）|每季 3 产地 · 共 4 产地 · 近红外|产地/年份漂移|72 个任务",
        "p1": "源域|训练模型|（PLS 或网络）",
        "p2": "目标域|标定池：n ∈ {0, 5, 10, 20, 40} 个标签|留出测试集，各预算下固定",
        "p3": "评估|RMSEP 与 NRMSEP = RMSEP/σ|5 个固定种子 · 任务聚类 bootstrap",
        "m1": "简单校正|CORAL（免标签、转导式）|斜率/偏置校正（2 个参数）",
        "m2": "经典模型更新|PDS · 模型更新|仅目标域 PLS",
        "m3": "深度 / 物理表示|1D-CNN · DANN · Deep CORAL|Phys+BL（Beer–Lambert 先验）",
        "a1": "3.1 标签预算清算|最佳简单法对最佳深度法；|无选择偏差复算",
        "a2": "3.2 免标签对齐|频率图 →|留一基准遗憾",
        "a3": "3.3 标度律检验|n* 对漂移幅度|（MMD、W1、维数）",
        "a4": "3.4 漂移结构|低阶份额 f 与低秩放大比 A，|只用光谱即可算出",
        "case": "苹果案例（3.5 节；S1–S12）|Phys+BL 对经典转移，|602 个场景；BL 失稳指标 A；|同预算斜率/偏置对照",
        "out": "输出",
        "o1": "最简校正何时够用",
        "o2": "无需诊断漂移类型的处方",
        "o3": "漂移幅度不构成标度律",
        "o4": "与简单法胜负相关的是漂移的结构而非幅度",
        "lbl_shift": "源域 / 目标域光谱",
        "lbl_budget": "标签预算 n",
    },
}
_SHADOW: list[pe.AbstractPathEffect] = [
    pe.withSimplePatchShadow(offset=(0.7, -0.7), shadow_rgbFace="#000000", alpha=0.10)
]


def _rbox(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    fc: str,
    ec: str,
    lw: float = 0.6,
    r: float = 1.0,
    ls: str = "-",
    z: float = 2,
    shadow: bool = False,
) -> FancyBboxPatch:
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        lw=lw,
        linestyle=ls,
        zorder=z,
    )
    if shadow:
        p.set_path_effects(_SHADOW)
    ax.add_patch(p)
    return p


def _stage(ax: Axes, y: float, h: float, tint: str, color: str, num: str, title: str) -> None:
    """一条阶段带:淡色底 + 左上角编号圆 + 标题。"""
    _rbox(ax, 2.0, y, W - 4.0, h, fc=tint, ec=color, lw=0.5, r=1.6, z=1)
    ax.add_patch(Circle((5.6, y + h - 2.3), 1.7, facecolor=color, edgecolor="none", zorder=3))
    ax.text(5.6, y + h - 2.3, num, ha="center", va="center", fontsize=FS, color="white", weight="bold", zorder=4)
    ax.text(8.2, y + h - 2.3, title, ha="left", va="center", fontsize=FS, color=color, weight="bold", zorder=4)


def _card(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    color: str,
    text: str,
    lang: str,
    glyph: Glyph | None = None,
    gh: float = 0.0,
    dashed: bool = False,
    hdr_icon: str | None = None,
    body_fs: float = FS - 0.8,
    body_ha: str = "center",
) -> None:
    """卡片:彩色标题条(白字)+ 白底正文 + 可选底部示意小图。text 用 | 分行,首段为标题。"""
    hh = 3.3
    card = _rbox(ax, x, y, w, h, fc="white", ec=color, lw=0.7, r=1.0, ls="--" if dashed else "-", shadow=True)
    hdr = Rectangle((x, y + h - hh), w, hh, facecolor=color, edgecolor="none", zorder=2.5)
    ax.add_patch(hdr)
    hdr.set_clip_path(card)
    head, *body = text.split("|")
    ax.text(
        x + 1.4, y + h - hh / 2, head, ha="left", va="center", fontsize=FS - 0.3, color="white", weight="bold", zorder=4
    )
    if hdr_icon:
        _hdr_icon(ax, hdr_icon, x + w - 2.6, y + h - hh / 2, 1.15)
    ty = y + h - hh - 0.9
    bx = x + w / 2 if body_ha == "center" else x + 1.6
    for line in body:
        ax.text(bx, ty, line, ha=body_ha, va="top", fontsize=body_fs, color="#222222", zorder=4)
        ty -= body_fs * 0.31
    if glyph is not None and gh > 0:
        glyph(ax, x + 1.6, y + 1.0, w - 3.2, gh, color)


def _hdr_icon(ax: Axes, kind: str, cx: float, cy: float, s: float) -> None:
    """标题条右侧的白色小图标(纯装饰,示意样本类型/域)。"""
    kw: dict[str, Any] = {"facecolor": "white", "edgecolor": "none", "zorder": 4, "alpha": 0.95}
    if kind == "corn":  # 玉米穗:椭圆 + 粒点
        ax.add_patch(Ellipse((cx, cy), 1.3 * s, 2.4 * s, **kw))
        for dy in (-0.6, 0.0, 0.6):
            for dx in (-0.3, 0.3):
                ax.add_patch(
                    Circle((cx + dx * s, cy + dy * s), 0.16 * s, facecolor="#0072B2", edgecolor="none", zorder=5)
                )
    elif kind == "tablet":  # 药片:胶囊
        ax.add_patch(
            FancyBboxPatch(
                (cx - 1.2 * s, cy - 0.55 * s), 2.4 * s, 1.1 * s, boxstyle="round,pad=0,rounding_size=0.55", **kw
            )
        )
        ax.plot([cx, cx], [cy - 0.55 * s, cy + 0.55 * s], color="#56B4E9", lw=0.6, zorder=5)
    elif kind == "mango":  # 芒果:斜椭圆 + 叶
        ax.add_patch(Ellipse((cx, cy - 0.1 * s), 2.2 * s, 1.5 * s, angle=-25, **kw))
        ax.add_patch(Ellipse((cx + 0.6 * s, cy + 0.85 * s), 0.9 * s, 0.35 * s, angle=30, **kw))
    elif kind == "soil":  # 土壤:三层剖面
        for i, wdt in enumerate((2.4, 2.0, 1.6)):
            ax.add_patch(Rectangle((cx - wdt * s / 2, cy - 0.9 * s + i * 0.62 * s), wdt * s, 0.5 * s, **kw))
    elif kind == "apple":  # 苹果:圆 + 梗 + 叶
        ax.add_patch(Circle((cx, cy - 0.15 * s), 1.05 * s, **kw))
        ax.plot([cx, cx + 0.1 * s], [cy + 0.8 * s, cy + 1.3 * s], color="white", lw=0.8, zorder=5)
        ax.add_patch(Ellipse((cx + 0.55 * s, cy + 1.05 * s), 0.8 * s, 0.32 * s, angle=30, **kw))
    elif kind == "db":  # 圆柱数据库
        ax.add_patch(Rectangle((cx - 1.0 * s, cy - 0.9 * s), 2.0 * s, 1.6 * s, **kw))
        ax.add_patch(Ellipse((cx, cy + 0.7 * s), 2.0 * s, 0.7 * s, **kw))
        ax.add_patch(Ellipse((cx, cy - 0.9 * s), 2.0 * s, 0.7 * s, **kw))
    elif kind == "target":  # 靶心
        ax.add_patch(Circle((cx, cy), 1.15 * s, **kw))
        ax.add_patch(Circle((cx, cy), 0.75 * s, facecolor=THEME["ref"], edgecolor="none", zorder=5))
        ax.add_patch(Circle((cx, cy), 0.35 * s, facecolor="white", edgecolor="none", zorder=6))
    elif kind == "check":  # 勾
        ax.plot(
            [cx - 0.9 * s, cx - 0.2 * s, cx + 1.0 * s],
            [cy, cy - 0.8 * s, cy + 0.8 * s],
            color="white",
            lw=1.1,
            zorder=5,
        )


# ---- 底部示意小图(手绘形状,不代表数据) ----
def _mini_axes(ax: Axes, x: float, y: float, w: float, h: float) -> None:
    ax.plot([x, x, x + w], [y + h, y, y], color=THEME["base_2"], lw=0.45, zorder=3, solid_capstyle="round")


def _spectra_glyph(ax: Axes, x: float, y: float, w: float, h: float, color: str) -> None:
    """源域/目标域两条光谱示意:同形状、目标域带偏移与斜率(漂移)。"""
    _mini_axes(ax, x, y, w, h)
    t = np.linspace(0, 1, 120)
    base = 0.25 + 0.22 * np.exp(-(((t - 0.3) / 0.12) ** 2)) + 0.38 * np.exp(-(((t - 0.72) / 0.10) ** 2)) + 0.12 * t
    for k, (c, off, sl) in enumerate(((THEME["base_2"], 0.0, 0.0), (color, 0.16, 0.14))):
        ax.plot(x + 0.6 + t * (w - 1.2), y + 0.5 + (base + off + sl * t) * (h - 1.1), color=c, lw=0.8, zorder=3 + k)


def _coral_glyph(ax: Axes, x: float, y: float, w: float, h: float, color: str) -> None:
    """二阶矩对齐:灰色源分布 + 彩色目标分布 → 对齐后重叠。"""
    cy = y + h / 2
    ax.add_patch(
        Ellipse(
            (x + w * 0.2, cy + 0.3),
            w * 0.26,
            h * 0.55,
            angle=25,
            facecolor=THEME["base_1"],
            edgecolor="none",
            alpha=0.9,
            zorder=3,
        )
    )
    ax.add_patch(
        Ellipse(
            (x + w * 0.31, cy - 0.5),
            w * 0.26,
            h * 0.42,
            angle=-30,
            facecolor=color,
            edgecolor="none",
            alpha=0.75,
            zorder=3,
        )
    )
    _small_arrow(ax, (x + w * 0.5, cy), (x + w * 0.62, cy))
    ax.add_patch(
        Ellipse(
            (x + w * 0.8, cy),
            w * 0.26,
            h * 0.55,
            angle=25,
            facecolor=THEME["base_1"],
            edgecolor="none",
            alpha=0.9,
            zorder=3,
        )
    )
    ax.add_patch(
        Ellipse((x + w * 0.8, cy), w * 0.26, h * 0.55, angle=25, facecolor=color, edgecolor="none", alpha=0.6, zorder=4)
    )


def _pds_glyph(ax: Axes, x: float, y: float, w: float, h: float, color: str) -> None:
    """分段直接校正:上下两条谱,一个窗口映射到一个点。"""
    t = np.linspace(0, 1, 80)
    s = 0.3 + 0.35 * np.exp(-(((t - 0.5) / 0.18) ** 2))
    ax.plot(x + t * w, y + h * 0.78 + s * h * 0.2, color=THEME["base_2"], lw=0.7, zorder=3)
    ax.plot(x + t * w, y + h * 0.1 + s * h * 0.2, color=color, lw=0.7, zorder=3)
    ax.add_patch(
        Rectangle(
            (x + w * 0.38, y + h * 0.66), w * 0.24, h * 0.34, facecolor=color, edgecolor="none", alpha=0.18, zorder=2
        )
    )
    _small_arrow(ax, (x + w * 0.5, y + h * 0.64), (x + w * 0.5, y + h * 0.36))
    ax.add_patch(Circle((x + w * 0.5, y + h * 0.3), 0.35, facecolor=color, edgecolor="none", zorder=4))


def _cnn_glyph(ax: Axes, x: float, y: float, w: float, h: float, color: str) -> None:
    """卷积层堆叠:由宽到窄的三层 + 输出点。"""
    widths = (0.62, 0.46, 0.30)
    for i, fw in enumerate(widths):
        ax.add_patch(
            FancyBboxPatch(
                (x + w * (0.05 + i * 0.22), y + h * (0.5 - fw / 2)),
                w * 0.13,
                h * fw,
                boxstyle="round,pad=0,rounding_size=0.3",
                facecolor=color,
                edgecolor="none",
                alpha=0.35 + 0.25 * i,
                zorder=3,
            )
        )
    _small_arrow(ax, (x + w * 0.66, y + h / 2), (x + w * 0.8, y + h / 2))
    ax.add_patch(Circle((x + w * 0.88, y + h / 2), 0.55, facecolor=color, edgecolor="none", zorder=4))


def _budget_glyph(ax: Axes, x: float, y: float, w: float, h: float, color: str) -> None:
    """3.1:误差随标签预算下降的两条曲线,简单法(彩)与深度法(灰)在网格内追平。"""
    _mini_axes(ax, x, y, w, h)
    t = np.linspace(0, 1, 60)
    ax.plot(
        x + 0.5 + t * (w - 1.0),
        y + 0.4 + (0.25 + 0.55 * np.exp(-3.2 * t)) * (h - 0.8),
        color=THEME["base_2"],
        lw=0.8,
        zorder=3,
    )
    ax.plot(
        x + 0.5 + t * (w - 1.0), y + 0.4 + (0.22 + 0.35 * np.exp(-6.0 * t)) * (h - 0.8), color=color, lw=0.8, zorder=4
    )
    for k in np.linspace(0, 1, 5):
        ax.plot(x + 0.5 + k * (w - 1.0), y + 0.05, marker="|", color=THEME["base_2"], ms=2.2, mew=0.5, zorder=3)


def _freq_glyph(ax: Axes, x: float, y: float, w: float, h: float, color: str) -> None:
    """3.2:漂移类型 × 预算的频率图瓦片。"""
    nc, nr = 5, 3
    cw, ch = w / nc, h / nr
    pat = [[1, 1, 1, 1, 0], [1, 1, 0, 0, 0], [2, 1, 1, 0, 0]]
    cols = {0: THEME["base_1"], 1: color, 2: THEME["deep_1"]}
    for r in range(nr):
        for c in range(nc):
            ax.add_patch(
                Rectangle(
                    (x + c * cw + 0.12, y + r * ch + 0.12),
                    cw - 0.24,
                    ch - 0.24,
                    facecolor=cols[pat[r][c]],
                    edgecolor="none",
                    alpha=0.85,
                    zorder=3,
                )
            )


def _scaling_glyph(ax: Axes, x: float, y: float, w: float, h: float, color: str) -> None:
    """3.3:n* 对漂移幅度的散点无趋势,一条平坦虚线。"""
    _mini_axes(ax, x, y, w, h)
    rng = np.random.default_rng(7)
    px = rng.uniform(0.1, 0.95, 11)
    py = rng.uniform(0.15, 0.9, 11)
    ax.scatter(x + 0.5 + px * (w - 1.0), y + 0.4 + py * (h - 0.8), s=3.5, color=color, edgecolor="none", zorder=4)
    ax.plot([x + 0.5, x + w - 0.5], [y + 0.4 + 0.5 * (h - 0.8)] * 2, color=THEME["base_2"], lw=0.7, ls="--", zorder=3)


def _structure_glyph(ax: Axes, x: float, y: float, w: float, h: float, color: str) -> None:
    """3.4:各基准的距离分解堆叠条(低阶份额 彩 / 高阶残余 灰)。"""
    _mini_axes(ax, x, y, w, h)
    shares = (0.92, 0.91, 0.78, 0.75, 0.77)
    bw = (w - 1.0) / 5 * 0.7
    for i, s in enumerate(shares):
        bx = x + 0.5 + i * (w - 1.0) / 5 + 0.2
        ax.add_patch(Rectangle((bx, y + 0.2), bw, s * (h - 0.6), facecolor=color, edgecolor="none", zorder=3))
        ax.add_patch(
            Rectangle(
                (bx, y + 0.2 + s * (h - 0.6)),
                bw,
                (1 - s) * (h - 0.6),
                facecolor=THEME["base_1"],
                edgecolor="none",
                zorder=3,
            )
        )


def _apple_glyph(ax: Axes, x: float, y: float, w: float, h: float, color: str) -> None:
    """案例:苹果 + 三条产地/年份箭头(示意跨产地跨年迁移)。"""
    cx, cy = x + w * 0.22, y + h * 0.5
    ax.add_patch(Circle((cx, cy - 0.2), h * 0.36, facecolor=color, edgecolor="none", alpha=0.85, zorder=3))
    ax.plot([cx, cx + 0.15], [cy + h * 0.14, cy + h * 0.36], color=THEME["ref"], lw=0.7, zorder=4)
    ax.add_patch(
        Ellipse((cx + 0.55, cy + h * 0.3), 1.0, 0.4, angle=30, facecolor=THEME["positive"], edgecolor="none", zorder=4)
    )
    for k, dy in enumerate((0.72, 0.5, 0.28)):
        _small_arrow(
            ax,
            (x + w * 0.42, y + h * dy),
            (x + w * 0.92, y + h * dy),
            color=(color, THEME["base_2"], THEME["base_2"])[k],
        )


def _small_arrow(
    ax: Axes, p0: tuple[float, float], p1: tuple[float, float], color: str = THEME["ref"], ls: str = "-"
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=5,
            lw=0.6,
            color=color,
            linestyle=ls,
            shrinkA=0,
            shrinkB=0,
            zorder=3,
        )
    )


def _flow_arrow(
    ax: Axes, p0: tuple[float, float], p1: tuple[float, float], color: str = THEME["base_2"], ls: str = "-"
) -> None:
    """阶段之间的宽箭头(填充三角头)。"""
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="simple,head_length=0.9,head_width=1.6,tail_width=0.55",
            facecolor=color,
            edgecolor="none",
            linestyle=ls,
            mutation_scale=2.6,
            shrinkA=0,
            shrinkB=0,
            zorder=1.5,
            alpha=0.9,
        )
    )


def _chevron_down(ax: Axes, cx: float, ytop: float, ybot: float, color: str) -> None:
    """阶段之间居中的宽箭头:矩形杆 + 三角头。"""
    shaft_w, head_w = 2.2, 4.6
    head_h = min(2.0, (ytop - ybot) * 0.55)
    ax.add_patch(
        Rectangle(
            (cx - shaft_w / 2, ybot + head_h),
            shaft_w,
            ytop - ybot - head_h,
            facecolor=color,
            edgecolor="none",
            zorder=1.5,
        )
    )
    ax.add_patch(
        Polygon(
            [[cx - head_w / 2, ybot + head_h], [cx + head_w / 2, ybot + head_h], [cx, ybot]],
            closed=True,
            facecolor=color,
            edgecolor="none",
            zorder=1.5,
        )
    )


def make_framework() -> Callable[[str], Figure]:
    def plot(lang: str) -> Figure:
        plt.rcParams["font.family"] = list(_FONT_STACK[lang])
        t = TXT[lang]
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.axis("off")
        tint = {"s1": "#F3F7FB", "s2": "#F6F6F6", "s3": "#FFF9EE", "s4": "#F1F9F5"}
        # ── 阶段 1:五基准 ──
        y1, h1 = 78.0, 21.0
        _stage(ax, y1, h1, tint["s1"], THEME["deep_2"], "1", t["s1"])
        cw, ch, cy = 26.4, 16.0, y1 + 1.2
        xs = [4.0 + i * (cw + 1.6) for i in range(5)]
        cols = [THEME["deep_2"], THEME["deep_1"], THEME["positive"], THEME["purple"], THEME["signal"]]
        keys = ["corn", "tablet", "mango", "soil", "apple"]
        for x, key, c in zip(xs, keys, cols, strict=True):
            _card(ax, x, cy, cw, ch, c, t[key], lang, glyph=_spectra_glyph, gh=4.6, hdr_icon=key)
            _flow_arrow(ax, (x + cw / 2, cy - 0.1), (x + cw / 2, cy - 3.0), color=c)
        ax.text(
            W - 4.0,
            y1 + h1 - 2.3,
            t["lbl_shift"],
            ha="right",
            va="center",
            fontsize=FS - 1.2,
            color=THEME["base_2"],
            style="italic",
            zorder=4,
        )
        # ── 阶段 2:统一协议 ──
        y2, h2 = 57.7, 17.0
        _stage(ax, y2, h2, tint["s2"], THEME["ref"], "2", t["s2"])
        pw, ph, py = (27.0, 52.0, 45.0), 11.8, y2 + 1.2
        px = [4.0, 4.0 + 27.0 + 6.0, 4.0 + 27.0 + 6.0 + 52.0 + 6.0]
        for x, wdt, key, icon in zip(px, pw, ("p1", "p2", "p3"), ("db", "target", "check"), strict=True):
            _card(ax, x, py, wdt, ph, THEME["ref"], t[key], lang, hdr_icon=icon)
        for i in range(2):
            _flow_arrow(ax, (px[i] + pw[i] + 0.4, py + ph / 2), (px[i + 1] - 0.4, py + ph / 2))
        # 标签预算刻度(五个逐渐变大的点)
        bx0, bx1, by = px[1] + 6.0, px[1] + pw[1] - 18.0, py + 2.4
        ax.plot([bx0, bx1], [by, by], color=THEME["base_2"], lw=0.5, zorder=3)
        for k, n in enumerate((0, 5, 10, 20, 40)):
            xx = bx0 + k * (bx1 - bx0) / 4
            ax.add_patch(Circle((xx, by), 0.32 + 0.14 * k, facecolor=THEME["amber"], edgecolor="none", zorder=4))
            ax.text(xx, by - 0.85, str(n), ha="center", va="top", fontsize=FS - 1.8, color=THEME["ref"], zorder=4)
        ax.text(
            bx1 + 2.0,
            by,
            t["lbl_budget"],
            ha="left",
            va="center",
            fontsize=FS - 1.4,
            color=THEME["base_2"],
            style="italic",
            zorder=4,
        )
        # ── 阶段 3:三族方法 ──
        y3, h3 = 39.9, 14.5
        _stage(ax, y3, h3, tint["s3"], THEME["amber"], "3", t["s3"])
        mw, mh, my = 44.0, 9.6, y3 + 1.2
        fam = [
            ("m1", THEME["amber"], _coral_glyph),
            ("m2", THEME["base_2"], _pds_glyph),
            ("m3", THEME["deep_2"], _cnn_glyph),
        ]
        for i, (key, c, g) in enumerate(fam):
            x = 4.0 + i * (mw + 2.0)
            _card(ax, x, my, mw, mh, c, t[key], lang, body_ha="left")
            g(ax, x + mw - 12.0, my + 0.8, 10.5, 4.6, c)
        _chevron_down(ax, W / 2, y2 - 0.2, y3 + h3 + 0.2, THEME["base_1"])
        # ── 阶段 4:四条分析线 + 苹果案例 ──
        y4, h4 = 16.1, 20.5
        _stage(ax, y4, h4, tint["s4"], THEME["positive"], "4", t["s4"])
        aw, ah, ay = 25.6, 14.6, y4 + 1.2
        ana = [("a1", _budget_glyph), ("a2", _freq_glyph), ("a3", _scaling_glyph), ("a4", _structure_glyph)]
        for i, (key, g) in enumerate(ana):
            x = 4.0 + i * (aw + 1.6)
            _card(ax, x, ay, aw, ah, THEME["positive"], t[key], lang, glyph=g, gh=4.4)
        xc = 4.0 + 4 * (aw + 1.6)
        wc = W - 4.0 - xc
        _card(
            ax,
            xc,
            ay,
            wc,
            ah,
            THEME["signal"],
            t["case"],
            lang,
            glyph=_apple_glyph,
            gh=4.0,
            dashed=True,
            body_fs=FS - 1.1,
        )
        # 苹果基准 → 案例:沿右缘的虚线,说明案例来自同一自采数据
        ax.annotate(
            "",
            xy=(xc + wc - 2.0, ay + ah + 0.2),
            xytext=(xs[4] + cw - 2.0, cy - 0.2),
            arrowprops={
                "arrowstyle": "-|>",
                "color": THEME["signal"],
                "lw": 0.7,
                "ls": (0, (2.5, 1.5)),
                "connectionstyle": "arc3,rad=-0.28",
                "mutation_scale": 6,
            },
            zorder=1.2,
        )
        _chevron_down(ax, W / 2, y3 - 0.2, y4 + h4 + 0.2, THEME["base_1"])
        # ── 输出横幅 ──
        yo, ho = 2.8, 10.0
        _rbox(ax, 2.0, yo, W - 4.0, ho, fc=THEME["deep_2"], ec="none", r=1.6, z=2, shadow=True)
        ax.text(
            5.0,
            yo + ho / 2,
            t["out"],
            ha="left",
            va="center",
            fontsize=FS + 1,
            color="white",
            weight="bold",
            zorder=4,
            rotation=90,
        )
        items = [t["o1"], t["o2"], t["o3"], t["o4"]]
        for i, item in enumerate(items):
            col, row = i % 2, i // 2
            ix = 12.0 + col * 66.0
            iy = yo + ho - 2.7 - row * 4.4
            _hdr_icon(ax, "check", ix, iy, 0.75)
            ax.text(ix + 2.6, iy, item, ha="left", va="center", fontsize=FS - 0.5, color="white", zorder=4)
        _chevron_down(ax, W / 2, y4 - 0.2, yo + ho + 0.2, THEME["base_1"])
        fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
        return fig

    return plot


def main() -> None:
    log = _eu.get_logger("77_framework_figure")
    paths = _eu.save_figure_bilingual(make_framework(), "77_framework_figure", 0)
    for lang, suf in (("zh", ""), ("en", "-en")):
        dst = os.path.join(_FIGDIR, f"FigFramework{suf}.pdf")
        shutil.copyfile(paths[lang], dst)
        log.log(f"拷贝 {os.path.basename(paths[lang])} -> figs/FigFramework{suf}.pdf")
    print("FigFramework 完成 -> figs/FigFramework[.pdf / -en.pdf]")


if __name__ == "__main__":
    main()
