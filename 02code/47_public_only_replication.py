"""
47_public_only_replication.py: 把统一基准的三个头条量在「只用四个公共基准」上重做一遍

背景:
    讨论段（§4.4）写着「restricting attention to the 54 tasks of the four public
    benchmarks leaves the reading unchanged」，但整包里没有任何数字支撑这句话，它是
    稿件中唯一一句实质性却不可核验的断言。而同一节又自己披露：苹果上整个深度族在自己
    的源域上都只到常数预测器水平（S14），所以「苹果 72 / 126 个任务」既是最大的一块
    证据，也是最容易被质疑为「对手没学会」的一块。

    本脚本不重训任何模型，只用已落盘的逐任务结果，把三件事各做一份公共基准版本：
      表a 免选择族均值对比（68 号口径：族内取可运行成员的逐任务均值 NRMSEP）；
      表b 基准级配对差与基准等权 bootstrap（68 号口径，公共版只剩 4 个基准）；
      表c 留一基准遗憾（73 号 fig4 宇宙，公共版在 4 个基准间留一）；
      表d 漂移结构诊断的相关（74 号口径，配对聚类 bootstrap）；
      表e 边际那两条相关的留一基准稳定性——74 号的 lobo 只覆盖 CORAL 回收与直接迁移
          误差两个被解释量，承重的「简单减深度边际」没有留一版本，回复信却已按留一
          口径陈述，这里把它补齐。
    每张表都把「全部五基准 / 仅四个公共基准」并排，方向变没变一眼可见。

    去掉苹果同时去掉了整个 origin_year_instrument 漂移类（苹果是该类唯一的基准），
    所以表a 的该类行在公共版里不存在——这不是缺失，是该口径下这一类没有证据，
    正文引用时必须照此说明。

运行方式:
    cd <项目根目录绝对路径>
    python 02code/47_public_only_replication.py

输出文件:
    04outputs/47_public_only_replication.xlsx
    05logs/47_public_only_replication_YY-MM-DD_HHMMSS.log
"""
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


def _load(name: str, path: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, os.path.join(_CODE, path))
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_eu = _load("export_utils", "01_export_utils.py")
_m73 = _load("m73", "73_heuristic_map_lobo.py")
_m74 = _load("m74", "74_shift_structure_diagnosis.py")

SIMPLE = ["coral", "sbc"]
DEEP = ["cnn_zeroshot", "cnn_finetune", "dann", "deepcoral", "physbl_zeroshot", "physbl_ft"]
NCAL = [0, 5, 10, 20, 40]
SHIFTS = ["instrument", "season", "origin_year_instrument", "lab"]
PUBLIC = ("corn", "tablet", "mango", "ossl_mir")   # 四个公共基准；苹果是本课题自采
SEED = 20060515
NBOOT_BENCH = 5000

PREDICTORS = ["f_coral_mmd2", "f_coral_sw1", "residual_mmd2", "A_lowrank", "A_lowrank_oos", "mmd2_raw"]
TARGETS = ["coral_gain", "nrmsep_zero_shot0", "nrmsep_coral0",
           "margin0_deep_minus_simple", "margin20_deep_minus_simple"]

F64 = npt.NDArray[np.float64]


class _Shim:
    """73 号的 _run_universe 只用 logger.log(...)，这里把它接到本脚本的日志上。"""

    def __init__(self, logger: Any) -> None:
        self._logger = logger

    def log(self, *args: Any, **kwargs: Any) -> None:
        self._logger.log(*args, **kwargs)


def _cliff(diff: F64) -> float:
    """配对差的 Cliff δ：>0 表示简单族误差更小。"""
    d = diff[np.isfinite(diff)]
    return float((d > 0).mean() - (d < 0).mean()) if len(d) else float("nan")


def _family_rows(tm: pd.DataFrame, tag: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for st in SHIFTS:
        for nc in NCAL:
            sub = tm[(tm.shift_type == st) & (tm.n_cal == nc)]
            piv = sub.pivot_table(index="task_id", columns="method", values="nrmsep")
            sc = [m for m in SIMPLE if m in piv.columns]
            dc = [m for m in DEEP if m in piv.columns]
            if not sc or not dc:
                continue
            sm, dm = piv[sc].mean(axis=1).to_numpy(), piv[dc].mean(axis=1).to_numpy()
            ok = np.isfinite(sm) & np.isfinite(dm)
            sm, dm = sm[ok], dm[ok]
            if len(sm) < 3:
                continue
            diff = dm - sm
            try:
                p = float(stats.wilcoxon(diff).pvalue)
            except ValueError:
                p = float("nan")
            rows.append({"universe": tag, "shift_type": st, "n_cal": nc, "n_task": len(sm),
                         "simple_nrmsep_med": round(float(np.median(sm)), 3),
                         "deep_nrmsep_med": round(float(np.median(dm)), 3),
                         "cliffs_delta": round(_cliff(diff), 3),
                         "simple_win_frac": round(float((diff > 0).mean()), 3),
                         "wilcoxon_p": p})
    return rows


def _benchmark_level(tm: pd.DataFrame, tag: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per: list[dict[str, object]] = []
    summ: list[dict[str, object]] = []
    for nc in (0, 20):
        rows: list[dict[str, object]] = []
        for bench in sorted(tm["benchmark"].unique()):
            sub = tm[(tm.benchmark == bench) & (tm.n_cal == nc)]
            piv = sub.pivot_table(index="task_id", columns="method", values="nrmsep")
            sc = [m for m in SIMPLE if m in piv.columns]
            dc = [m for m in DEEP if m in piv.columns]
            if not sc or not dc:
                continue
            diff = (piv[dc].mean(axis=1) - piv[sc].mean(axis=1)).dropna()
            if len(diff) == 0:
                continue
            rows.append({"universe": tag, "n_cal": nc, "benchmark": bench, "n_task": len(diff),
                         "mean_diff_deep_minus_simple": round(float(diff.mean()), 3),
                         "simple_better": bool(diff.mean() > 0)})
        per += rows
        vals = np.array([float(str(r["mean_diff_deep_minus_simple"])) for r in rows])
        rng = np.random.default_rng(SEED)          # 每档重新播种，两个宇宙可逐位复现
        boots = [float(np.mean(rng.choice(vals, len(vals), replace=True))) for _ in range(NBOOT_BENCH)]
        summ.append({"universe": tag, "n_cal": nc, "n_benchmarks": len(rows),
                     "benchmark_balanced_mean_diff": round(float(vals.mean()), 3),
                     "boot95_lo": round(float(np.percentile(boots, 2.5)), 3),
                     "boot95_hi": round(float(np.percentile(boots, 97.5)), 3),
                     "n_benchmarks_simple_better": int(sum(bool(r["simple_better"]) for r in rows))})
    return per, summ


def _regret(tm: pd.DataFrame, tag: str, logger: Any) -> pd.DataFrame:
    pt, _rules, _ins, _choices = _m73._run_universe(tm, tag, _Shim(logger))
    pooled = _m73._summarise(pt, ["rule"]).assign(universe=tag, n_cal="pooled")
    by_n = _m73._summarise(pt, ["rule", "n_cal"]).assign(universe=tag)
    return pd.concat([pooled, by_n], ignore_index=True)


def _correlations(per_task: pd.DataFrame, tag: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cl = per_task["pair"].astype(str).to_numpy()
    for pr in PREDICTORS:
        for tg in TARGETS:
            rho, lo, hi, p, n = _m74._spearman_ci(
                per_task[pr].to_numpy(dtype=float), per_task[tg].to_numpy(dtype=float), cl)
            rows.append({"universe": tag, "predictor": pr, "target": tg, "n": n,
                         "spearman_rho": round(rho, 3), "ci95_lo": round(lo, 3),
                         "ci95_hi": round(hi, 3), "p_nominal": p,
                         "ci_excludes_0": bool(np.isfinite(lo) and np.isfinite(hi) and lo * hi > 0)})
    return rows


def _margin_lobo(per_task: pd.DataFrame) -> list[dict[str, object]]:
    """对两个 margin 被解释量做留一基准相关：每次剔掉一个基准，在其余四个上重算。"""
    rows: list[dict[str, object]] = []
    benches = sorted(per_task["benchmark"].unique())
    for pr in ("f_coral_mmd2", "A_lowrank"):
        for tg in ("margin0_deep_minus_simple", "margin20_deep_minus_simple"):
            for held in benches:
                rest = per_task[per_task["benchmark"] != held]
                rho, lo, hi, _p, n = _m74._spearman_ci(
                    rest[pr].to_numpy(dtype=float), rest[tg].to_numpy(dtype=float),
                    rest["pair"].astype(str).to_numpy())
                rows.append({"predictor": pr, "target": tg, "held_out": held, "n_rest": n,
                             "rho_rest": round(rho, 3), "ci95_lo": round(lo, 3),
                             "ci95_hi": round(hi, 3),
                             "ci_excludes_0": bool(np.isfinite(lo) and np.isfinite(hi) and lo * hi > 0)})
    return rows


def main() -> int:
    logger = _eu.get_logger(os.path.basename(__file__).replace(".py", ""))
    tm_all = _m73._load_task_medians()
    tm_pub = tm_all[tm_all["benchmark"].isin(PUBLIC)].copy()
    logger.log("全部基准 %d 任务；四个公共基准 %d 任务（%s）",
               tm_all["task_id"].nunique(), tm_pub["task_id"].nunique(),
               "、".join(sorted(tm_pub["benchmark"].unique())))

    fam = pd.DataFrame(_family_rows(tm_all, "all5") + _family_rows(tm_pub, "public4"))
    per_a, sum_a = _benchmark_level(tm_all, "all5")
    per_p, sum_p = _benchmark_level(tm_pub, "public4")
    bench_per = pd.DataFrame(per_a + per_p)
    bench_sum = pd.DataFrame(sum_a + sum_p)
    reg = pd.concat([_regret(tm_all, "all5", logger), _regret(tm_pub, "public4", logger)],
                    ignore_index=True)

    pt74 = pd.read_excel(os.path.join(_OUT, "74_shift_structure_diagnosis.xlsx"), sheet_name="per_task")
    cor = pd.DataFrame(_correlations(pt74, "all5")
                       + _correlations(pt74[pt74["benchmark"].isin(PUBLIC)].copy(), "public4"))
    lobo = pd.DataFrame(_margin_lobo(pt74))

    dst = os.path.join(_OUT, "47_public_only_replication.xlsx")
    with pd.ExcelWriter(dst) as xw:
        fam.to_excel(xw, sheet_name="表a：免选择族均值", index=False)
        bench_per.to_excel(xw, sheet_name="表b：基准级配对差", index=False)
        bench_sum.to_excel(xw, sheet_name="表b2：基准等权bootstrap", index=False)
        reg.to_excel(xw, sheet_name="表c：留一基准遗憾", index=False)
        cor.to_excel(xw, sheet_name="表d：结构诊断相关", index=False)
        lobo.to_excel(xw, sheet_name="表e：边际相关留一基准", index=False)

    logger.log("表a 免选择族均值:\n%s", fam.to_string(index=False))
    logger.log("表b2 基准等权 bootstrap:\n%s", bench_sum.to_string(index=False))
    key = reg[(reg["n_cal"].astype(str) == "pooled")][["universe", "rule", "n_task", "median_regret",
                                                       "ci95_lo", "ci95_hi", "within10_frac"]]
    logger.log("表c 池化遗憾:\n%s", key.to_string(index=False))
    logger.log("表e 边际相关的留一基准稳定性:\n%s", lobo.to_string(index=False))
    logger.log("表d 承重相关（f_coral_mmd2 与两档 margin）:\n%s",
               cor[(cor["predictor"] == "f_coral_mmd2")
                   & (cor["target"].str.startswith("margin"))].to_string(index=False))
    print(f"→ {os.path.relpath(dst, _BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
