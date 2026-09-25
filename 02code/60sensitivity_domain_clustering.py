"""
60sensitivity_domain_clustering.py: 跨域配对差对"聚类单位选择"的稳健性敏感性分析

回应 Round B 审稿 CRITICAL 1：602 个跨产地场景并非 602 个独立实验，
而是由 6 个原子产地-年域（2018/2025 × 山东/新疆/陕西/甘肃派生的 6 个组合）
反复组合而成；每个场景对应一个唯一"源集→目标集"域对（602=602，域对层级不粗化）。
本脚本对同一批 50 号跨种子均值，在三种聚类单位下重做 cluster bootstrap 95% CI——
场景(602) / 目标集(62) / 源集(62)——检验关键配对差的方向与显著性是否稳健于聚类选择，
从而说明结论不依赖"把组合场景当独立实验"这一过强假设，并据此弱化极小名义 P 值的呈现。

表 b 对同一批配对差改取**中位数**做同样的三档 cluster bootstrap。物理架构有少数运行发散
（RMSE 上百乃至上万 °Brix），一粒种子发散就把该场景的跨种子均值拉到极端值，配对差的均值
于是由这几个场景决定，回答的是「期望误差」；中位数回答「典型场景」。两者方向可以相反，
所以各给一张表，稿件分别引用。

运行方式:
    cd <repository root>
    python 02code/60sensitivity_domain_clustering.py

输出文件:
    04outputs/60sensitivity_domain_clustering.xlsx  — 三档聚类下各配对差的均值（表a）与中位数（表b）及 95%CI
    05logs/60sensitivity_domain_clustering_YY-MM-DD_HHMMSS.log  — 运行日志
"""
import importlib.util
import os
import collections

import numpy as np
import pandas as pd

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, ".."))

_spec = importlib.util.spec_from_file_location(
    "export_utils", os.path.join(_CODE, "01_export_utils.py"))
_eu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eu)

get_logger = _eu.get_logger
log_experiment_header = _eu.log_experiment_header
write_script_workbook = _eu.write_script_workbook

SRC_XLSX = os.path.join(_BASE, "04outputs", "50_formal_multiseed_benchmark.xlsx")
SHEET = "表g：每场景跨种子均值"
BOOT_SEED = 20060515        # 固定 bootstrap 种子（Formal 首种子），保证 CI 可复现
N_BOOT = 3000               # ≥1000（第七章 7.2）

# 关注的配对差（mA − mB）：负值表示 mA 的 RMSE 更小（更优）。
# 第四个字段表示是否进总判定：前三条是稿件主张所依托的比较；SVR+SB 一条说明「经典基线」里
# SVR 的稳健程度，逐条报告、不进总判定
COMPARISONS = [
    ("未校正 Phys+BL − CNN", "Phys+BL", "CNN", True),
    ("校正后 PLSR+SB − Phys+BL+SB", "PLSR + SB", "Phys+BL + SB", True),
    ("校正后 CNN+SB − Phys+BL+SB", "CNN + SB", "Phys+BL + SB", True),
    ("校正后 SVR+SB − Phys+BL+SB", "SVR + SB", "Phys+BL + SB", False),
]


def src_set(scene: str) -> frozenset:
    """场景 'A+B→C+D' 的源集 {A,B}。"""
    return frozenset(scene.split("→")[0].split("+"))


def tgt_set(scene: str) -> frozenset:
    """场景 'A+B→C+D' 的目标集 {C,D}。"""
    return frozenset(scene.split("→")[1].split("+"))


CLUSTERINGS = [
    ("场景(602)", lambda s: s),
    ("目标集(62)", tgt_set),
    ("源集(62)", src_set),
]


def cluster_bootstrap_ci(diff: pd.Series, key_fn, seed: int, n_boot: int, stat=np.mean):
    """对配对差 Series（index=场景）按 key_fn 聚类后做 cluster bootstrap 95%CI。

    重采样单位为聚类（整簇同时进出），簇内保留全部场景值——避免把相关场景当独立样本。
    stat 为统计量（均值或中位数），点估计与每次重采样都用它。
    """
    clusters = collections.defaultdict(list)
    for scene, val in diff.items():
        clusters[key_fn(scene)].append(val)
    cluster_vals = [np.asarray(v, dtype=float) for v in clusters.values()]
    rng = np.random.default_rng(seed)
    n_cl = len(cluster_vals)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_cl, n_cl)
        boots[b] = stat(np.concatenate([cluster_vals[i] for i in idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(stat(diff.to_numpy())), float(lo), float(hi), n_cl


def paired_diff(per: pd.DataFrame, m_a: str, m_b: str) -> pd.Series:
    """两方法在共同场景上的配对 RMSE 差（m_a − m_b），index=场景。"""
    a = per[per["方法_校正"] == m_a].set_index("场景")["RMSE"]
    b = per[per["方法_校正"] == m_b].set_index("场景")["RMSE"]
    k = a.index.intersection(b.index)
    return (a.loc[k] - b.loc[k]).dropna()


def main() -> None:
    logger = get_logger("60sensitivity_domain_clustering")
    log_experiment_header(logger, {
        "任务": "跨域配对差对聚类单位选择的稳健性（Round B CRITICAL 1）",
        "数据源": f"{os.path.basename(SRC_XLSX)} / {SHEET}",
        "bootstrap": f"cluster bootstrap × {N_BOOT}，种子 {BOOT_SEED}",
        "聚类单位": " / ".join(name for name, _ in CLUSTERINGS),
    })

    per = pd.read_excel(SRC_XLSX, sheet_name=SHEET)

    # 域结构说明表：坐实"6 原子域、602=602 域对"这一关键事实
    scenes = per["场景"].unique()
    atoms = sorted({d for s in scenes for d in s.replace("→", "+").split("+")})
    domain_pairs = {(src_set(s), tgt_set(s)) for s in scenes}
    meta_df = pd.DataFrame([
        ("唯一场景数", len(scenes)),
        ("唯一(源集,目标集)域对数", len(domain_pairs)),
        ("唯一源集数", len({src_set(s) for s in scenes})),
        ("唯一目标集数", len({tgt_set(s) for s in scenes})),
        ("原子产地-年域数", len(atoms)),
        ("原子产地-年域清单", "，".join(atoms)),
    ], columns=["项", "值"])
    logger.log(f"域结构：{len(scenes)} 场景 / {len(domain_pairs)} 域对 / "
               f"{len(atoms)} 原子域 {atoms}")

    tables = {}
    for col, stat in (("配对差均值", np.mean), ("配对差中位", np.median)):
        rows = []
        for label, m_a, m_b, _ in COMPARISONS:
            diff = paired_diff(per, m_a, m_b)
            for cname, cfn in CLUSTERINGS:
                est, lo, hi, n_cl = cluster_bootstrap_ci(diff, cfn, BOOT_SEED, N_BOOT, stat)
                excl0 = (lo < 0 and hi < 0) or (lo > 0 and hi > 0)
                rows.append({
                    "比较": label, "聚类单位": cname,
                    col: round(est, 6),
                    "CI下界": round(lo, 6), "CI上界": round(hi, 6),
                    "聚类数": n_cl, "共同场景数": len(diff),
                    "CI不跨0": "是" if excl0 else "否",
                })
                logger.log(f"{col} {label:<28s} | {cname:<10s} {est:+.3f}  "
                           f"95%CI [{lo:+.3f}, {hi:+.3f}]  簇{n_cl}  不跨0={'是' if excl0 else '否'}")
        tables[col] = pd.DataFrame(rows)

    out = write_script_workbook(__file__, {
        0: ("三档聚类稳健性", tables["配对差均值"]),
        1: ("中位配对差三档聚类", tables["配对差中位"]),
        "域结构说明": meta_df,
    })
    logger.log(f"已写出: {out}")

    # 稳健性判定：每个比较是否三档聚类全部同向且 CI 全不跨 0（均值与中位数各判一次）
    for col, df in tables.items():
        all_robust = True
        for label, _, _, key in COMPARISONS:
            sub = df[df["比较"] == label]
            signs = {np.sign(v) for v in sub[col]}
            robust = len(signs) == 1 and bool((sub["CI不跨0"] == "是").all())
            if key:
                all_robust &= robust
            logger.log(f"【稳健性·{col}】{label:<28s} 三档同向且CI不跨0 = {robust}"
                       + ("" if key else "（逐条报告，不进总判定）"))
        logger.log(f"总判定（{col}）：全部关键配对差对聚类单位选择稳健 = {all_robust}")
        print(f"稳健性总判定（{col}，三档聚类全部同向且 CI 不跨 0）:", all_robust)


if __name__ == "__main__":
    main()
