"""
73_heuristic_map_lobo.py: 频率图「漂移类型 × 标签预算 → 最频繁最优方法」的留一基准外推验证（返修 CILS R1.5/R2.1/R2.2）

【为什么必须做这个实验】
    投出稿的图 4 是 in-sample 频率图：在同一批 126 任务上数胜者，正文自认"需前瞻验证才能操作使用"。
    审稿人 2 追问：评估建立在漂移前后数据都已知之上，漂移未知、不可控时这套结果怎么用？并认为
    工作"只是评估"。审稿人 1 提醒零标签下多数方法根本不适用，CORAL 是专为此设计的。
    本脚本不重训任何模型，只用 62（经典）+ 64（深度）已落地曲线，把频率图变成可外推的选法规则并
    如实报告它离 oracle 有多远：
      规则 A（漂移类型已知）：留一基准（LOBO）——只用其余基准的任务数胜者，再用于被留出的基准；
                              该漂移类型在其余基准里不存在时规则 A 不可用（记 NA），退到规则 B。
      规则 B（漂移类型未知）：只看标签预算，用其余基准全部任务的最频繁胜者——这正是"漂移未知"场景。
      固定基线：always-simple（n=0 用 CORAL，n>0 用 SBC，即正文的实用建议）、always-model_update、
                always-target_only、默认深度（n=0 cnn_zeroshot，n>0 cnn_finetune）、oracle（每任务最优）。
    遗憾 regret = RMSEP(所选)/RMSEP(oracle) − 1（同任务同测试集，量纲抵消，可跨基准池化）；
    另报 NRMSEP（=RMSEP/σ_tgt）与「落在 oracle 5%/10% 以内」的任务占比；任务聚类 bootstrap 95% CI。
    方法宇宙两档：fig4（与 65 号图 4 完全一致：各预算只含该预算下有曲线的方法）；
                  extended（免标签方法的 n=0 结果顺延到各预算——同一测试集，其误差不随标签变——
                  对应"有标签时仍可选免标签法"的现实选择集）。主结果取 fig4 档，extended 作敏感性。
    不引入新主张：验证的是既有 C2a（频率图的可用性边界）。

运行方式:
    cd <项目根目录绝对路径>
    python 02code/73_heuristic_map_lobo.py

输出文件:
    04outputs/73_heuristic_map_lobo.xlsx — per_task / summary / rule_tables / insample_map / paper_table
    05logs/73_heuristic_map_lobo_YY-MM-DD_HHMMSS.log
"""

import importlib.util
import os

import numpy as np
import numpy.typing as npt
import pandas as pd

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, ".."))
_OUT = os.path.join(_BASE, "04outputs")
_es = importlib.util.spec_from_file_location("export_utils", os.path.join(_CODE, "01_export_utils.py"))
assert _es is not None  # 对已知存在的 .py 文件 spec 恒非 None
assert _es.loader is not None  # 同理 loader 恒非 None
_eu = importlib.util.module_from_spec(_es)
_es.loader.exec_module(_eu)

SIMPLE = ["coral", "sbc"]
CLASSIC_EXP = ["model_update", "target_only"]
DEEP = ["cnn_zeroshot", "cnn_finetune", "dann", "deepcoral", "physbl_zeroshot", "physbl_ft"]
UNIVERSE = SIMPLE + CLASSIC_EXP + DEEP + ["zero_shot", "pds"]  # 与 65 号 phase 完全一致
LABEL_FREE = ["zero_shot", "coral", "cnn_zeroshot", "dann", "deepcoral", "physbl_zeroshot"]
NCAL = [0, 5, 10, 20, 40]
RNG = np.random.default_rng(20060515)
RNG_TASK = np.random.default_rng(20060515)   # 跨预算池化的按任务聚类 bootstrap 专用，不扰动 RNG 的既有随机流
NBOOT = 2000


def _load_task_medians() -> pd.DataFrame:
    """每 (task, method, n_cal) 跨种子×重复取中位 RMSEP，并接上 σ_tgt 得 NRMSEP。"""
    cl = pd.read_excel(os.path.join(_OUT, "62_crossover_engine.xlsx"), sheet_name="curves")
    dp = pd.read_excel(os.path.join(_OUT, "64_deep_transfer_server.xlsx"), sheet_name="curves")
    allr = pd.concat([cl, dp], ignore_index=True)
    allr = allr[allr["method"].isin(UNIVERSE) & allr["n_cal"].isin(NCAL)]
    tm = allr.groupby(["benchmark", "shift_type", "task_id", "method", "n_cal"], as_index=False)["rmsep"].median()
    ns = pd.read_excel(os.path.join(_OUT, "63_crossover_analysis.xlsx"), sheet_name="nstar_per_task")
    sig = ns[["task_id", "y_std_tgt"]].drop_duplicates("task_id")
    tm = tm.merge(sig, on="task_id", how="left")
    tm["nrmsep"] = tm["rmsep"] / tm["y_std_tgt"]
    return tm


def _extend_label_free(tm: pd.DataFrame) -> pd.DataFrame:
    """extended 宇宙：免标签方法的 n=0 行复制到每个预算（同测试集，误差不随标签变）。"""
    base = tm[(tm["n_cal"] == 0) & tm["method"].isin(LABEL_FREE)]
    ext = [tm]
    for nc in NCAL:
        if nc == 0:
            continue
        have = set(map(tuple, tm.loc[tm["n_cal"] == nc, ["task_id", "method"]].values))
        add = base[[(t, m) not in have for t, m in zip(base["task_id"], base["method"], strict=True)]].copy()
        add["n_cal"] = nc
        ext.append(add)
    return pd.concat(ext, ignore_index=True)


def _winners(tm: pd.DataFrame) -> pd.DataFrame:
    """每 (task, n_cal) 的 oracle 最优方法及其 RMSEP。"""
    idx = tm.groupby(["task_id", "n_cal"])["rmsep"].idxmin()
    w = tm.loc[idx, ["benchmark", "shift_type", "task_id", "n_cal", "method", "rmsep", "nrmsep"]]
    return w.rename(columns={"method": "oracle", "rmsep": "rmsep_oracle", "nrmsep": "nrmsep_oracle"}).reset_index(
        drop=True
    )


def _rank_by_frequency(w: pd.DataFrame, balanced: bool = False) -> list[str]:
    """按胜者频次排序；balanced=True 时每个训练基准等权（任务权重 1/该基准任务数），
    防止任务数最多的基准（苹果 72/126）单独决定规则。"""
    if not balanced:
        return list(w["oracle"].value_counts().index)
    wt = 1.0 / w.groupby("benchmark")["task_id"].transform("count")
    return list(wt.groupby(w["oracle"]).sum().sort_values(ascending=False).index)


def _pick_available(ranked: list[str], avail: set[str]) -> str | None:
    for m in ranked:
        if m in avail:
            return m
    return None


def _lookup(tm: pd.DataFrame) -> dict[tuple[str, int], dict[str, tuple[float, float]]]:
    d: dict[tuple[str, int], dict[str, tuple[float, float]]] = {}
    for r in tm.itertuples(index=False):
        d.setdefault((r.task_id, int(r.n_cal)), {})[r.method] = (float(r.rmsep), float(r.nrmsep))
    return d


def _boot_ci_median(x: npt.NDArray[np.float64]) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan"), float("nan")
    meds = [float(np.median(RNG.choice(x, len(x), replace=True))) for _ in range(NBOOT)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def _boot_ci_median_by_task(g: pd.DataFrame) -> tuple[float, float]:
    """按任务聚类的 bootstrap：每次有放回抽任务，被抽中任务的全部 (预算) 单元格一起进入。

    跨预算池化时同一任务有多个单元格，彼此相关；逐格独立重采样会把区间算窄。
    """
    blocks = [grp["regret"].values.astype(float) for _, grp in g.groupby("task_id", sort=True)]
    if sum(len(b) for b in blocks) < 3:
        return float("nan"), float("nan")
    k = len(blocks)
    meds = []
    for _ in range(NBOOT):
        idx = RNG_TASK.integers(0, k, size=k)
        meds.append(float(np.median(np.concatenate([blocks[i] for i in idx]))))
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def _summarise(pt: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """分组汇总遗憾。分组含 n_cal 时每个任务只有一格，逐行 bootstrap 即任务 bootstrap；
    不含 n_cal（跨预算池化）时改用按任务聚类的 bootstrap（独立的随机流 RNG_TASK）。"""
    rows = []   # 存 6 位小数：稿件按百分数一位小数显示，存 4 位再显示会双重舍入（如 0.208500→20.8 应为 20.9）
    for k, g in pt.groupby(keys, dropna=False):
        g = g.dropna(subset=["regret"])
        if len(g) == 0:
            continue
        reg = g["regret"].values.astype(float)
        if "n_cal" in keys:
            lo, hi = _boot_ci_median(reg)
        else:
            _boot_ci_median(reg)   # 照旧推进 RNG，使其后各分组（按预算）的区间与存档逐位一致；结果不用
            lo, hi = _boot_ci_median_by_task(g)
        row = dict(zip(keys, k if isinstance(k, tuple) else (k,), strict=True))
        row.update(
            {
                "n_task": len(g),
                "median_regret": round(float(np.median(reg)), 6),
                "ci95_lo": round(lo, 6),
                "ci95_hi": round(hi, 6),
                "mean_regret": round(float(np.mean(reg)), 6),
                "within5_frac": round(float(np.mean(reg <= 0.05)), 6),
                "within10_frac": round(float(np.mean(reg <= 0.10)), 6),
                "median_nrmsep_chosen": round(float(np.nanmedian(g["nrmsep_chosen"])), 3),
                "median_nrmsep_oracle": round(float(np.nanmedian(g["nrmsep_oracle"])), 3),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


# 固定处方族：不需要任何漂移诊断的规则，(n=0 用什么, n>0 用什么)。
# 正文的实用建议 always_simple 是其中一员；C 规则在留出基准之外先把整族比一遍再选，
# 因此它的遗憾不含「看过全部五个基准才挑出这条处方」的循环成分。
FIXED_PRESCRIPTIONS: dict[str, tuple[str | None, str | None]] = {
    "always_simple": ("coral", "sbc"),
    "always_model_update": (None, "model_update"),
    "always_target_only": (None, "target_only"),
    "default_deep": ("cnn_zeroshot", "cnn_finetune"),
}


def _prescription_method(name: str, nc: int) -> str | None:
    zero, pos = FIXED_PRESCRIPTIONS[name]
    return zero if nc == 0 else pos


def _median_regret_on(w_sub: pd.DataFrame, lk: dict[tuple[str, int], dict[str, tuple[float, float]]],
                      name: str) -> float:
    """一条固定处方在给定任务集合上的中位遗憾；该处方在某单元不可运行时该单元不计入。"""
    vals: list[float] = []
    for r in w_sub.itertuples(index=False):
        m = _prescription_method(name, int(r.n_cal))
        avail = lk.get((r.task_id, int(r.n_cal)), {})
        if m is None or m not in avail:
            continue
        vals.append(avail[m][0] / r.rmsep_oracle - 1.0)
    return float(np.median(vals)) if vals else float("inf")


def _run_universe(tm: pd.DataFrame, universe: str,
                  logger: object) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    w = _winners(tm)
    lk = _lookup(tm)
    benches = sorted(w["benchmark"].unique())

    # in-sample 频率图（应与 65 号 phase 一致）
    ins = []
    for (st, nc), g in w.groupby(["shift_type", "n_cal"]):
        vc = g["oracle"].value_counts()
        ins.append(
            {
                "universe": universe,
                "shift_type": st,
                "n_cal": nc,
                "top_method": vc.index[0],
                "top_frac": round(float(vc.iloc[0] / len(g)), 3),
                "n_task": len(g),
            }
        )
    insample = pd.DataFrame(ins)

    per_rows, rule_rows, choice_rows = [], [], []
    for held in benches:
        train = w[w["benchmark"] != held]
        test = w[w["benchmark"] == held]
        # 处方族本身也走留一：只看其余四个基准，选中位遗憾最小的那条固定处方
        train_scores = {k: _median_regret_on(train, lk, k) for k in FIXED_PRESCRIPTIONS}
        best_fixed = min(train_scores, key=lambda k: train_scores[k])
        choice_rows.append({"universe": universe, "held_out": held, "selected": best_fixed,
                            **{f"train_median_regret[{k}]": round(v, 4) for k, v in train_scores.items()}})
        for nc in NCAL:
            tr = train[train["n_cal"] == nc]
            te = test[test["n_cal"] == nc]
            if len(te) == 0 or len(tr) == 0:
                continue
            rank_b = _rank_by_frequency(tr)
            rank_b_bal = _rank_by_frequency(tr, balanced=True)
            for st in sorted(te["shift_type"].unique()):
                tr_a = tr[tr["shift_type"] == st]
                rank_a = _rank_by_frequency(tr_a) if len(tr_a) else []
                rank_a_bal = _rank_by_frequency(tr_a, balanced=True) if len(tr_a) else []
                rule_rows.append(
                    {
                        "universe": universe,
                        "held_out": held,
                        "shift_type": st,
                        "n_cal": nc,
                        "ruleA_available": bool(len(tr_a)),
                        "ruleA_top": rank_a[0] if rank_a else None,
                        "ruleA_bal_top": rank_a_bal[0] if rank_a_bal else None,
                        "ruleA_n_train": len(tr_a),
                        "ruleB_top": rank_b[0],
                        "ruleB_bal_top": rank_b_bal[0],
                        "ruleB_n_train": len(tr),
                    }
                )
                sub = te[te["shift_type"] == st]
                for r in sub.itertuples(index=False):
                    avail = lk[(r.task_id, nc)]
                    choices = {
                        "A_shift_known": _pick_available(rank_a, set(avail)) if rank_a else None,
                        "B_shift_unknown": _pick_available(rank_b, set(avail)),
                        "A_shift_known_bal": _pick_available(rank_a_bal, set(avail)) if rank_a_bal else None,
                        "B_shift_unknown_bal": _pick_available(rank_b_bal, set(avail)),
                        "always_simple": "coral" if nc == 0 else "sbc",
                        "always_model_update": None if nc == 0 else "model_update",
                        "always_target_only": None if nc == 0 else "target_only",
                        "default_deep": "cnn_zeroshot" if nc == 0 else "cnn_finetune",
                        "C_best_fixed_prescription": _prescription_method(best_fixed, nc),
                    }
                    for rule, m in choices.items():
                        if m is None or m not in avail:
                            per_rows.append(
                                {
                                    "universe": universe,
                                    "held_out": held,
                                    "benchmark": r.benchmark,
                                    "shift_type": st,
                                    "task_id": r.task_id,
                                    "n_cal": nc,
                                    "rule": rule,
                                    "chosen": m,
                                    "oracle": r.oracle,
                                    "rmsep_chosen": np.nan,
                                    "rmsep_oracle": r.rmsep_oracle,
                                    "regret": np.nan,
                                    "nrmsep_chosen": np.nan,
                                    "nrmsep_oracle": r.nrmsep_oracle,
                                    "note": "rule unavailable" if m is None else "method not run for task",
                                }
                            )
                            continue
                        rm, nr = avail[m]
                        per_rows.append(
                            {
                                "universe": universe,
                                "held_out": held,
                                "benchmark": r.benchmark,
                                "shift_type": st,
                                "task_id": r.task_id,
                                "n_cal": nc,
                                "rule": rule,
                                "chosen": m,
                                "oracle": r.oracle,
                                "rmsep_chosen": rm,
                                "rmsep_oracle": r.rmsep_oracle,
                                "regret": rm / r.rmsep_oracle - 1.0,
                                "nrmsep_chosen": nr,
                                "nrmsep_oracle": r.nrmsep_oracle,
                                "note": "",
                            }
                        )
    per_task = pd.DataFrame(per_rows)
    rules = pd.DataFrame(rule_rows)
    choices_df = pd.DataFrame(choice_rows)
    # 本项目的 logger.log 只做拼接、不做 %-格式化，故用 f-string
    logger.log(f"[{universe}] in-sample 频率图:\n{insample.to_string(index=False)}")  # type: ignore[attr-defined]
    logger.log(f"[{universe}] 处方族留一选择:\n{choices_df.to_string(index=False)}")  # type: ignore[attr-defined]
    return per_task, rules, insample, choices_df


def main() -> None:
    logger = _eu.get_logger(os.path.basename(__file__).replace(".py", ""))
    tm0 = _load_task_medians()
    logger.log("任务数=%d  方法=%s", tm0["task_id"].nunique(), sorted(tm0["method"].unique()))
    out_pt, out_rules, out_ins, out_sum, out_choice = [], [], [], [], []
    for universe, tm in [("fig4", tm0), ("extended", _extend_label_free(tm0))]:
        pt, rules, ins, choices = _run_universe(tm, universe, logger)
        out_pt.append(pt)
        out_rules.append(rules)
        out_ins.append(ins)
        out_choice.append(choices)
        s1 = _summarise(pt, ["universe", "rule", "n_cal"]).assign(benchmark="ALL")
        s2 = _summarise(pt, ["universe", "rule", "benchmark", "n_cal"])
        s3 = _summarise(pt, ["universe", "rule"]).assign(benchmark="ALL", n_cal="pooled")
        out_sum += [s1, s2, s3]
    per_task = pd.concat(out_pt, ignore_index=True)
    rules = pd.concat(out_rules, ignore_index=True)
    insample = pd.concat(out_ins, ignore_index=True)
    summary = pd.concat(out_sum, ignore_index=True)
    prescription_choice = pd.concat(out_choice, ignore_index=True)

    # 论文用表：fig4 宇宙，按预算池化五基准，规则 × 预算
    paper = summary[(summary["universe"] == "fig4") & (summary["benchmark"] == "ALL")].copy()
    paper = paper[
        paper["rule"].isin(
            [
                "A_shift_known",
                "A_shift_known_bal",
                "B_shift_unknown",
                "B_shift_unknown_bal",
                "always_simple",
                "default_deep",
                "always_model_update",
                "always_target_only",
                "C_best_fixed_prescription",
            ]
        )
    ]
    paper = paper.sort_values(["n_cal", "rule"], key=lambda s: s.map(lambda v: str(v).zfill(6)))

    logger.log("汇总（fig4，ALL）:\n%s", paper.to_string(index=False))
    path = _eu.write_script_workbook(
        __file__,
        {
            0: ("paper_table", paper),
            "summary": summary,
            "per_task": per_task,
            "rule_tables": rules,
            "insample_map": insample,
            "prescription_choice": prescription_choice,
        },
    )
    logger.log("落盘: %s", path)
    print(paper.to_string(index=False))
    print(
        "\nrule_tables (fig4, ruleA 可用性):\n",
        rules[rules["universe"] == "fig4"].groupby(["held_out", "shift_type"])["ruleA_available"].first().to_string(),
    )
    print(
        "\n注: regret=RMSEP(所选)/RMSEP(oracle)-1，同任务同测试集；A=漂移类型已知的 LOBO 规则，"
        "B=漂移类型未知的仅按预算规则；单基准漂移类型下 A 不可用即退到 B；"
        "C=对固定处方族本身也走留一（在其余四个基准上选中位遗憾最小的那条处方，再用于留出基准），"
        "用来剥掉「看过全部五个基准才挑出 always_simple」的循环成分。"
    )


if __name__ == "__main__":
    main()
