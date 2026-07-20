"""
62_ablation_multiseed_analysis.py: 多种子架构消融分析（表2 多种子升级，回应 Round B MAJOR 4）

背景：表2（CNN+BL vs CNN、Phys+BL vs Phys）此前为单种子（seed=42，见 57 号），审稿人
Round B MAJOR 4 要求多随机种子复现，并补 paired 置信区间与效应量。50 号主实验已含
CNN、Phys+BL 的 5 固定种子结果（表g），缺 CNN+BL、Phys 两条消融臂；后者由 50 号以
`--methods CNN+BL,Phys --shard_dir 04outputs/50_ablation_shards` 在 5 种子上补跑（61 号启动器）。

本脚本合并两处数据 → 逐场景取跨种子均值 → 计算表2 三行，每行报告：
  · 起止 RMSE、场景相对改进（均值/中位）、双侧 Wilcoxon P
  · 以场景为聚类单位的 cluster bootstrap 95% CI（配对差）
  · 效应量 Cliff's δ
  · 有效收敛场景数 n（多种子口径：某方法在该场景至少一个种子收敛）
并给出 §2.3 发散分析（Phys vs Phys+BL 跨种子均值 RMSE > 10 的场景数 + 精确 McNemar）。

数据缺失（消融分片未跑完）时优雅降级：只报现有臂并提示。

运行方式:
    cd /Volumes/Lin Ryan/01_科研竞赛/02论文竞赛/015苹果SSC迁移
    python 02code/62_ablation_multiseed_analysis.py

输出文件:
    04outputs/62_ablation_multiseed_analysis.xlsx  — 多种子表2 + §2.3 全部数字
    05logs/62_ablation_multiseed_analysis_YY-MM-DD_HHMMSS.log  — 运行日志
"""
import glob
import importlib.util
import os
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, ".."))
_OUT = os.path.join(_BASE, "04outputs")
_ABL_DIR = os.path.join(_OUT, "50_ablation_shards")
_MAIN_XLSX = os.path.join(_OUT, "50_formal_multiseed_benchmark.xlsx")

_spec = importlib.util.spec_from_file_location(
    "export_utils", os.path.join(_CODE, "01_export_utils.py"))
assert _spec is not None  # 对已知存在的 .py 文件 spec 恒非 None
assert _spec.loader is not None  # 同理 loader 恒非 None
_eu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eu)
get_logger = _eu.get_logger
log_experiment_header = _eu.log_experiment_header
write_script_workbook = _eu.write_script_workbook

BOOT_SEED = 20060515
N_BOOT = 3000
DIVERGENCE_RMSE = 10.0
FORMAL_SEEDS = [20060515, 20041210, 19810915, 2023, 2024]


def cross_seed_mean(df_long: pd.DataFrame) -> pd.Series:
    """长表(含 场景/种子/RMSE) → 每场景跨种子均值 RMSE（index=场景）。"""
    return df_long.groupby("场景")["RMSE"].mean()


def load_main_arms() -> dict[str, pd.Series]:
    """从 50 号表g（每场景跨种子均值）取裸 CNN、Phys+BL。"""
    g = pd.read_excel(_MAIN_XLSX, sheet_name="表g：每场景跨种子均值")
    out: dict[str, pd.Series] = {}
    for m in ("CNN", "Phys+BL"):
        sub = g[g["方法_校正"] == m]
        out[m] = sub.set_index("场景")["RMSE"]
    return out


def load_ablation_arms() -> tuple[dict[str, pd.Series], int, list[Any]]:
    """从 50_ablation_shards 取裸 CNN+BL、Phys 的跨种子均值。缺失返回空。"""
    files = sorted(glob.glob(os.path.join(_ABL_DIR, "seed*_shard*.csv")))
    if not files:
        return {}, 0, []
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["校正"] == "裸"].copy()
    seeds = sorted(df["种子"].unique().tolist())
    out: dict[str, pd.Series] = {}
    for m in ("CNN+BL", "Phys"):
        sub = df[df["方法"] == m]
        if len(sub):
            out[m] = cross_seed_mean(sub)
    return out, len(files), seeds


def cliffs_delta(a: npt.NDArray[Any], b: npt.NDArray[Any]) -> float:
    """Cliff's δ（a 相对 b）。配对差的非参效应量用符号占比近似：这里用配对差 d=a-b。"""
    d = a - b
    return float((np.sum(d < 0) - np.sum(d > 0)) / len(d))   # a<b(更优)占比 − a>b 占比


def cluster_bootstrap_ci(
    diff: pd.Series, seed: int = BOOT_SEED, n_boot: int = N_BOOT
) -> tuple[float, float, float]:
    """以场景为聚类单位（每场景一个配对差）的 percentile bootstrap 95%CI。"""
    vals = diff.values.astype(float)
    rng = np.random.default_rng(seed)
    n = len(vals)
    means = np.array([vals[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def paired_row(a: pd.Series, b: pd.Series, label: str) -> dict[str, Any]:
    """a vs b 配对（共同场景）。改进 = (RMSE_b − RMSE_a)/RMSE_b，b 为起点。"""
    common = a.index.intersection(b.index)
    aa, bb = a.loc[common].values, b.loc[common].values
    imp = (bb - aa) / bb * 100
    try:
        _, p = stats.wilcoxon(aa, bb)
    except ValueError:
        p = np.nan
    diff = pd.Series(aa - bb, index=common)          # a − b（负=a更优）
    md, lo, hi = cluster_bootstrap_ci(diff)
    return {
        "对比": label, "起始RMSE": round(float(bb.mean()), 3),
        "终止RMSE": round(float(aa.mean()), 3),
        "改进均值%": round(float(imp.mean()), 2), "改进中位%": round(float(np.median(imp)), 2),
        "配对差均值": round(md, 3), "CI下界": round(lo, 3), "CI上界": round(hi, 3),
        "CliffsDelta": round(cliffs_delta(aa, bb), 3),
        "WilcoxonP": float(p), "n": len(common),
        "CI不跨0": "是" if (lo < 0 and hi < 0) or (lo > 0 and hi > 0) else "否",
    }


def mcnemar_exact(n01: int, n10: int) -> float:
    n = n01 + n10
    if n == 0:
        return 1.0
    return float(min(2 * stats.binom.cdf(min(n01, n10), n, 0.5), 1.0))


def main() -> None:
    logger = get_logger("62_ablation_multiseed_analysis")
    main_arms = load_main_arms()
    abl_arms, n_files, seeds = load_ablation_arms()
    arms = {**main_arms, **abl_arms}
    log_experiment_header(logger, {
        "任务": "多种子架构消融（表2 升级，Round B MAJOR 4）",
        "现有臂(表g)": list(main_arms.keys()),
        "消融臂(分片)": list(abl_arms.keys()) or "（尚无：50_ablation_shards 为空）",
        "消融分片数": n_files, "消融种子": seeds or "—",
    })

    have = set(arms)
    pairs = [("CNN+BL", "CNN", "CNN+BL vs CNN（H6：BL 损失加于通用 CNN）"),
             ("Phys+BL", "Phys", "Phys+BL vs Phys（H7：BL 损失加于物理架构）"),
             ("Phys+BL", "CNN", "Phys+BL vs CNN（架构收益，参照）")]
    rows = []
    for a_m, b_m, label in pairs:
        if a_m in have and b_m in have:
            rows.append(paired_row(arms[a_m], arms[b_m], label))
            r = rows[-1]
            logger.log(f"{label}: 起{r['起始RMSE']}→终{r['终止RMSE']} 改进{r['改进均值%']:+.2f}% "
                       f"配对差CI[{r['CI下界']},{r['CI上界']}] δ={r['CliffsDelta']} n={r['n']} 不跨0={r['CI不跨0']}")
        else:
            miss = [m for m in (a_m, b_m) if m not in have]
            logger.log(f"{label}: ⏳ 缺臂 {miss}，待消融分片跑完")
    tbl2 = pd.DataFrame(rows) if rows else pd.DataFrame([{"提示": "消融分片未就绪，表2 待补"}])

    # ── §2.3 发散分析（Phys vs Phys+BL，跨种子均值 RMSE > 10）──
    div_rows = []
    if "Phys" in have and "Phys+BL" in have:
        phys, physbl = arms["Phys"], arms["Phys+BL"]
        common = phys.index.intersection(physbl.index)
        pv, pbv = phys.loc[common] > DIVERGENCE_RMSE, physbl.loc[common] > DIVERGENCE_RMSE
        n01 = int((pv & ~pbv).sum())
        n10 = int((~pv & pbv).sum())
        div_rows.append({"Phys发散场景": int(pv.sum()), "Phys+BL发散场景": int(pbv.sum()),
                         "n01(Phys发散/BL不发散)": n01, "n10(反向)": n10,
                         "McNemar精确P": mcnemar_exact(n01, n10), "共同场景": len(common)})
        logger.log(f"§2.3 发散：Phys {int(pv.sum())} vs Phys+BL {int(pbv.sum())}（阈值{DIVERGENCE_RMSE}）"
                   f" McNemar P={mcnemar_exact(n01, n10):.3g}")
    div_df = pd.DataFrame(div_rows) if div_rows else pd.DataFrame([{"提示": "Phys 臂未就绪"}])

    out = write_script_workbook(__file__, {0: ("表2多种子", tbl2), "§2.3发散": div_df})
    logger.log(f"已写出: {out}")
    print("表2 可算行数:", len(rows), "| 消融臂:", list(abl_arms.keys()) or "无（待分片）")


if __name__ == "__main__":
    main()
