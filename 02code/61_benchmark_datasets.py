"""
61_benchmark_datasets.py: 多基准标定转移统一数据管道 + 迁移任务注册表（idea v3 地基）

把 5(+)个经典标定转移基准统一成同一 schema，供 crossover 引擎(62号)与服务器全量实验消费：
    corn(三仪器,仪器漂移,配对) / tablet(双仪器,仪器漂移,配对) /
    mango(四季节,季节漂移,非配对) / apple(自有,产地×年×仪器漂移,非配对) / OSSL(土壤跨实验室,待接)

统一 schema (每个基准):
    dict(name, modality, shift_type, properties[list],
         domains={dom_key: dict(X=(n,p) float, Y=(n,n_prop) float, wl=(p,) float)},
         paired=bool)
迁移任务 = (benchmark, src_dom, tgt_dom, prop_idx, prop_name); 生成全部有序域对 × 成分。

原始数据只读自 05data(正典源),派生统一数组缓存到 03data/processed/benchmarks/*.npz。
读 05data → 写 03data/processed 是 §4.5.1 法定预处理流向,绝不回写源。

运行方式:
    cd <repository root>
    python 02code/61_benchmark_datasets.py           # 载入全部+打印任务统计+缓存 npz

输出文件:
    03data/processed/benchmarks/<name>.npz           — 各基准统一数组缓存
    03data/processed/benchmarks/transfer_tasks.csv   — 迁移任务注册表
    05logs/61_benchmark_datasets_YY-MM-DD_HHMMSS.log
"""
import importlib.util
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.io import loadmat

_CODE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_CODE, ".."))
_es = importlib.util.spec_from_file_location("export_utils", os.path.join(_CODE, "01_export_utils.py"))
assert _es is not None  # 对已知存在的 .py 文件 spec 恒非 None
assert _es.loader is not None  # 同理 loader 恒非 None
_eu = importlib.util.module_from_spec(_es)
_es.loader.exec_module(_eu)

DATA05 = os.path.normpath(os.path.join(_BASE, "..", "..", "05data"))
PROC = os.path.join(_BASE, "03data", "processed", "benchmarks")
APPLE_PROC = os.path.join(_BASE, "03data")   # 02_data_{year}.csv 所在


# ---------------------------------------------------------------- corn
def load_corn() -> dict[str, Any]:
    m = loadmat(os.path.join(DATA05, "036_corn_nir_3instruments", "corn.mat"),
                struct_as_record=False, squeeze_me=True)
    wl = np.linspace(1100, 2498, 700)        # 文档波长: 1100–2498nm, 2nm 步长(仅供作图)
    Y = np.asarray(m["propvals"].data, float)     # 80×4
    props = ["moisture", "oil", "protein", "starch"]
    domains = {}
    for inst in ["m5", "mp5", "mp6"]:
        domains[inst] = {"X": np.asarray(m[f"{inst}spec"].data, float), "Y": Y.copy(), "wl": wl.copy()}
    return {"name": "corn", "modality": "NIR", "shift_type": "instrument",
                "properties": props, "domains": domains, "paired": True}


# ---------------------------------------------------------------- tablet
def load_tablet() -> dict[str, Any]:
    m = loadmat(os.path.join(DATA05, "037_tablet_nir_shootout2002", "nir_shootout_2002.mat"),
                struct_as_record=False, squeeze_me=True)
    props = ["weight", "hardness", "assay"]
    wl = np.linspace(600, 1898, 650)         # 文档波长: 600–1898nm, 2nm 步长(仅供作图)
    domains = {}
    for inst in ["1", "2"]:
        X = np.vstack([np.asarray(m[f"{s}_{inst}"].data, float)
                       for s in ["calibrate", "validate", "test"]])
        Y = np.vstack([np.asarray(m[f"{s}_Y"].data, float)
                       for s in ["calibrate", "validate", "test"]])
        domains[f"inst{inst}"] = {"X": X, "Y": Y, "wl": wl.copy()}
    return {"name": "tablet", "modality": "NIR", "shift_type": "instrument",
                "properties": props, "domains": domains, "paired": True}


# ---------------------------------------------------------------- mango
def load_mango() -> dict[str, Any]:
    p = os.path.join(DATA05, "041_mango_anderson_crossseason", "NAnderson2020MendeleyMangoNIRData.csv")
    df = pd.read_csv(p)
    wl_cols = [c for c in df.columns if str(c).strip().lstrip("-").isdigit()]
    wl = np.asarray([float(c) for c in wl_cols], float)
    domains = {}
    for season in sorted(df["Season"].dropna().unique()):
        sub = df[df["Season"] == season]
        X = sub[wl_cols].to_numpy(float)
        y = sub["DM"].to_numpy(float).reshape(-1, 1)
        ok = np.isfinite(X).all(1) & np.isfinite(y).ravel() & (y.ravel() > 0)
        if ok.sum() >= 40:
            domains[f"S{int(season)}"] = {"X": X[ok], "Y": y[ok], "wl": wl.copy()}
    return {"name": "mango", "modality": "NIR", "shift_type": "season",
                "properties": ["DM"], "domains": domains, "paired": False}


# ---------------------------------------------------------------- apple (自有)
def load_apple() -> dict[str, Any]:
    """自有苹果三季。

    ⚠ 2026-09-09 起改走 01_export_utils.load_processed_dataframe——它是全项目唯一的
    苹果数据入口,统一施加两条与数据量无关的硬规则(参考 SSC 越界剔除、逐位相同光谱整组剔除)。
    此前本函数直接读 CSV、只做越界剔除,于是统一基准里留着 2018 的 31 行、2019 的 30 行
    「光谱相同而标签不同」的错配样本,而案例研究已把它们剔掉——两条管线用的不是同一份数据,
    与正文「三条清洗规则在数据读取入口统一实施」的表述不符。
    """
    domains = {}
    for year in [2018, 2019, 2025]:
        fp = os.path.join(APPLE_PROC, f"02_data_{year}.csv")
        if not os.path.exists(fp):
            continue
        df = _eu.load_processed_dataframe(year)
        bc = [c for c in df.columns if c.startswith("波段")]
        wl = np.asarray([float(c.split("_")[1].replace("nm", "")) for c in bc], float)
        X = df[bc].to_numpy(float)
        y = df["SSC含量(°Brix)"].to_numpy(float).reshape(-1, 1)
        reg = df["产地"].to_numpy(str)
        ok = np.isfinite(X).all(1) & np.isfinite(y.ravel())   # 越界与重复谱已在入口剔除
        X, y, reg = X[ok], y[ok], reg[ok]
        inst = "A" if year in (2018, 2019) else "B"
        for org in np.unique(reg):
            if org == "未知":
                continue
            mask = reg == org
            if mask.sum() >= 30:
                domains[f"{year}_{org}_{inst}"] = {"X": X[mask], "Y": y[mask], "wl": wl.copy()}
    return {"name": "apple", "modality": "NIR", "shift_type": "origin_year_instrument",
                "properties": ["SSC"], "domains": domains, "paired": False}


# ---------------------------------------------------------------- OSSL 全球土壤(双模态,跨实验室)
OSSL_DIR = os.path.join(DATA05, "038_ossl_global_soil_spectral")
OSSL_GZ = os.path.join(OSSL_DIR, "ossl_all_L1_v1.2.csv.gz")
OSSL_SUBLIBS = ["KSSL.SSL", "LUCAS.SSL", "ICRAF.ISRIC", "AFSIS1.SSL", "CAF.SSL"]  # 主要库=跨实验室域
OSSL_CAP = 700          # 每库上限(8GB 本机内存安全)
OSSL_TARGET = "c.tot"   # 总碳(≈SOC),土壤光谱头号目标


def _build_ossl_cache(log: Any = None) -> None:
    """分块读 622MB gz,按 sub-library 子采样,产 ossl_visnir.npz / ossl_mir.npz。仅需跑一次。"""
    hdr = pd.read_csv(OSSL_GZ, nrows=0, compression="gzip")
    cols = list(hdr.columns)
    vis = [c for c in cols if c.startswith("scan_visnir.") and c.endswith("_ref")]
    mir = [c for c in cols if c.startswith("scan_mir.") and c.endswith("_abs")]
    dcol = "dataset.code_ascii_txt"
    tcol = next((c for c in cols if c.startswith("c.tot") and "a622" in c), None)
    scol = next((c for c in cols if c.startswith("sand.tot")), None)
    ycols = [c for c in [tcol, scol] if c]
    usecols = [dcol, *ycols, *vis, *mir]
    wl_vis = np.array([float(c.split(".")[1].split("_")[0]) for c in vis])
    wl_mir = np.array([float(c.split(".")[1].split("_")[0]) for c in mir])
    buf: dict[str, list[pd.DataFrame]] = {sl: [] for sl in OSSL_SUBLIBS}
    got = dict.fromkeys(OSSL_SUBLIBS, 0)
    for chunk in pd.read_csv(OSSL_GZ, usecols=usecols, compression="gzip", chunksize=8000):
        for sl in OSSL_SUBLIBS:
            if got[sl] >= OSSL_CAP:
                continue
            sub = chunk[(chunk[dcol] == sl) & chunk[tcol].notna()]
            if len(sub) == 0:
                continue
            need = OSSL_CAP - got[sl]
            sub = sub.iloc[:need]
            buf[sl].append(sub)
            got[sl] += len(sub)
        if all(got[sl] >= OSSL_CAP for sl in OSSL_SUBLIBS):
            break
        if log:
            log.log(f"    OSSL 累计: {got}")
    for modality, spec_cols, wl in [("visnir", vis, wl_vis), ("mir", mir, wl_mir)]:
        doms = {}
        for sl in OSSL_SUBLIBS:
            if not buf[sl]:
                continue
            df = pd.concat(buf[sl], ignore_index=True)
            X = df[spec_cols].to_numpy(float)
            Y = df[ycols].to_numpy(float)
            ok = np.isfinite(X).all(1) & np.isfinite(Y[:, 0])
            if ok.sum() >= 40:
                doms[sl.replace(".SSL", "").replace(".ISRIC", "")] = {"X": X[ok], "Y": Y[ok], "wl": wl.copy()}
        np.savez_compressed(
            os.path.join(PROC, f"ossl_{modality}.npz"),
            **{f"X__{k}": v["X"] for k, v in doms.items()},
            **{f"Y__{k}": v["Y"] for k, v in doms.items()},
            **{f"wl__{k}": v["wl"] for k, v in doms.items()},
            props=np.array(["c.tot", "sand.tot"][:len(ycols)], dtype=object),
            domain_keys=np.array(list(doms.keys()), dtype=object),
        )
        if log:
            log.log(f"  ossl_{modality}: 域={list(doms.keys())} 形状={{k:v['X'].shape for k,v in doms.items()}}")


def load_ossl(modality: str) -> dict[str, Any]:
    """从缓存载入 OSSL 某模态(cross-lab)。modality∈{visnir,mir}。"""
    fp = os.path.join(PROC, f"ossl_{modality}.npz")
    z = np.load(fp, allow_pickle=True)
    keys = list(z["domain_keys"])
    props = [str(p) for p in z["props"]]

    def _bin(
        X: npt.NDArray[np.float64], wl: npt.NDArray[np.float64], target: int = 340
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """高维光谱分箱到 ~target 带(MIR 1701 平滑,降维提速,sel_k 不再爆)。"""
        p = X.shape[1]
        if p <= target:
            return X, wl
        f = p // target
        nb = p // f
        Xb = X[:, :nb * f].reshape(X.shape[0], nb, f).mean(2)
        wlb = wl[:nb * f].reshape(nb, f).mean(1)
        return Xb, wlb

    doms = {}
    for k in keys:
        Xb, wlb = _bin(z[f"X__{k}"], z[f"wl__{k}"])
        doms[k] = {"X": Xb, "Y": z[f"Y__{k}"], "wl": wlb}
    return {"name": f"ossl_{modality}", "modality": ("MIR" if modality == "mir" else "visNIR"),
                "shift_type": "lab", "properties": props, "domains": doms, "paired": False}


def load_cached(name: str) -> dict[str, Any]:
    """从 npz 缓存重建统一 schema(服务器无 05data 时的唯一来源;本机亦更快)。"""
    z = np.load(os.path.join(PROC, f"{name}.npz"), allow_pickle=True)
    keys = [str(k) for k in z["domain_keys"]]
    doms = {k: {"X": z[f"X__{k}"], "Y": z[f"Y__{k}"], "wl": z[f"wl__{k}"]} for k in keys}
    props = [str(p) for p in z["props"]]
    paired: bool
    if "meta" in z.files:
        nm, mod, shift, paired_str = [str(x) for x in z["meta"]]
        paired = (paired_str == "True")
    else:   # ossl 缓存无 meta
        nm, mod = name, ("MIR" if "mir" in name else "visNIR" if "visnir" in name else "NIR")
        shift, paired = "lab", False
    return {"name": nm, "modality": mod, "shift_type": shift, "properties": props, "domains": doms, "paired": paired}


def _cache_first(name: str, raw_fn: Callable[[], dict[str, Any]]) -> Callable[[], dict[str, Any]]:
    """缓存存在则读缓存,否则从 05data 原始构建。"""
    def _l() -> dict[str, Any]:
        if os.path.exists(os.path.join(PROC, f"{name}.npz")):
            return load_cached(name)
        return raw_fn()
    return _l


LOADERS = {"corn": _cache_first("corn", load_corn), "tablet": _cache_first("tablet", load_tablet),
           "mango": _cache_first("mango", load_mango), "apple": _cache_first("apple", load_apple)}
for _mod in ["visnir", "mir"]:
    _fp = os.path.join(PROC, f"ossl_{_mod}.npz")
    if os.path.exists(_fp):
        try:
            _nk = len(np.load(_fp, allow_pickle=True)["domain_keys"])
        except Exception:
            _nk = 0
        if _nk >= 2:      # ≥2 域才能做迁移
            LOADERS[f"ossl_{_mod}"] = (lambda m: (lambda: load_ossl(m)))(_mod)


def build_transfer_tasks(bench: dict[str, Any], max_apple_pairs: int | None = None) -> list[dict[str, Any]]:
    """生成一个基准的全部有序域对 × 成分 迁移任务。apple 域多,可限跨仪器/同仪器子集。"""
    keys = list(bench["domains"].keys())
    tasks = []
    for s in keys:
        for t in keys:
            if s == t:
                continue
            # apple: 标注同仪器/跨仪器,便于分层
            extra = {}
            if bench["name"] == "apple":
                extra["same_instrument"] = s.split("_")[-1] == t.split("_")[-1]
            for pi, pn in enumerate(bench["properties"]):
                tasks.append(dict(benchmark=bench["name"], src=s, tgt=t,
                                  prop_idx=pi, prop_name=pn,
                                  shift_type=bench["shift_type"], paired=bench["paired"],
                                  modality=bench["modality"], **extra))
    return tasks


def main() -> None:
    import sys
    log = _eu.get_logger("61_benchmark_datasets")
    os.makedirs(PROC, exist_ok=True)
    if "--build-ossl" in sys.argv:
        log.log("构建 OSSL 双模态缓存(分块读 622MB gz)...")
        _build_ossl_cache(log)
        log.log("OSSL 缓存构建完成。")
        return
    _eu.log_experiment_header(log, {"任务": "多基准标定转移统一管道", "基准": list(LOADERS)})
    all_tasks = []
    summary = []
    raw_builders = {"corn": load_corn, "tablet": load_tablet, "mango": load_mango, "apple": load_apple}
    for name, loader in raw_builders.items():   # main 从 05data 原始重建(消费端用缓存优先 LOADERS)
        try:
            b = loader()
        except Exception as e:
            log.log(f"⚠ {name} 载入失败: {e}")
            continue
        doms = b["domains"]
        tasks = build_transfer_tasks(b)
        all_tasks += tasks
        nsamp = {k: v["X"].shape for k, v in doms.items()}
        log.log(f"[{name}] modality={b['modality']} shift={b['shift_type']} paired={b['paired']} "
                 f"域数={len(doms)} 成分={b['properties']} 任务数={len(tasks)}")
        log.log(f"    域形状: {nsamp}")
        summary.append({"benchmark": name, "modality": b["modality"], "shift_type": b["shift_type"],
                            "n_domains": len(doms), "n_props": len(b["properties"]), "n_tasks": len(tasks)})
        # 缓存 npz (供服务器)
        np.savez_compressed(
            os.path.join(PROC, f"{name}.npz"),
            **{f"X__{k}": v["X"] for k, v in doms.items()},
            **{f"Y__{k}": v["Y"] for k, v in doms.items()},
            **{f"wl__{k}": v["wl"] for k, v in doms.items()},
            meta=np.array([b["name"], b["modality"], b["shift_type"], str(b["paired"])], dtype=object),
            props=np.array(b["properties"], dtype=object),
            domain_keys=np.array(list(doms.keys()), dtype=object),
        )
    tdf = pd.DataFrame(all_tasks)
    tdf.to_csv(os.path.join(PROC, "transfer_tasks.csv"), index=False)
    sdf = pd.DataFrame(summary)
    log.log("\n===== 基准汇总 =====\n" + sdf.to_string(index=False))
    log.log(f"\n迁移任务总数: {len(all_tasks)}  (缓存: {PROC})")
    log.log("按 shift_type 分布:\n" + tdf["shift_type"].value_counts().to_string())
    print(sdf.to_string(index=False))
    print(f"\n迁移任务总数: {len(all_tasks)}")
    print(tdf["shift_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
