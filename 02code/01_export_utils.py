"""
01_export_utils.py: 项目统一输入输出工具模块

运行方式:
    作为模块导入使用，不单独运行

功能：
    - 路径管理 (BASE_DIR, DATA_DIR, OUTPUT_DIR, LOG_DIR)
    - 设备自动选择 (CPU/CUDA/MPS)
    - 绘图工具 (中文字体、PDF导出、规范命名)
    - 日志记录 (输出、表格、统计)
    - 光谱/域适应专用工具 (波长解析、年份-产地数据加载、域合并、按产地切分)

    ── 以下为 ARIS 模板同步补齐部分（见文件末尾"模板同步区"）──
    - 六路随机种子 (set_random_seed)、Formal 阶段 CUDA 守卫 (enforce_publication_grade_device)
    - Checkpoint + 模型卡 + 数据卡 (save_best_checkpoint / write_model_card / write_datasheet)
    - wandb 统一入口 (init_wandb)、算力跟踪 (track_compute)
    - 统计检验 (compare_methods / apply_multi_comp_correction / compare_methods_multi)
    - LLH-GBID 多模型综合评价 (compute_gbid / gbid_sensitivity_* / gbid_bootstrap)
    - 环境捕获与依赖锁定 (capture_env_info / freeze_requirements / compute_sha256)

注意：本模块文件名以数字开头，无法被 `import 01_export_utils` 直接导入。
全项目统一通过 importlib.util.spec_from_file_location("export_utils", ".../01_export_utils.py")
+ exec_module 的方式加载（见各实验脚本头部）。因此本文件的**顶层语句在每个实验脚本
启动时都会执行一次**，顶层不得放置任何可能失败或产生副作用的重操作。
"""
import datetime
import hashlib
import json
import os
import platform
import random
import re
import string
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager, suppress
from typing import Any, Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# ─── 数组类型别名（mypy --strict 要求 ndarray 带类型参数）──────────────────────
# matplotlib.pyplot 在运行时重导出了 Figure/Axes，但其 type stub 未导出，
# 因此注解必须直接引用 matplotlib.figure.Figure / matplotlib.axes.Axes。
FloatArray = npt.NDArray[np.floating[Any]]
"""浮点数组：效用值 u、权重 W/v、GBID 分解量等一切实值向量/矩阵。"""

IntArray = npt.NDArray[np.integer[Any]]
"""整数数组：维度分组索引 dim_groups、排名矩阵等。"""

AnyArray = npt.NDArray[Any]
"""dtype 不定的数组：返回字典中同时含浮点指标与整数排名时使用。"""

# ─── 种子集合常量（第五章 Item 4 双轨锚定：常量轨）────────────────────────────
# 双轨锚定的另一轨为 refine-logs/0NSTATISTICAL_PROTOCOL.md 预注册文档
RECOMMENDED_SEEDS: tuple[int, ...] = (20060515, 20041210, 19810915, 2023, 2024)
"""Formal 阶段固定 5 种子集合；所有入口脚本 argparse --seeds 默认引用此常量。

**本项目保留立项时锁定的种子集，未随 2026-07-27 的模板更新改动**——预注册的效力
来自「事前锁定且此后不变」，不是「与模板一致」。已产出论文数字的实验跑的就是这一套。
上报 `n_seeds` 用 `n_distinct_seeds()`；本集合无重复，与 len() 相等，接口统一即可。
"""

DEV_PILOT_SEED: int = 2004
"""Pilot 阶段单 dev 种子；严禁出现在 Formal 5 种子集合中。"""

# ─── Formal 阶段 stage 集合（第五章 Item 5 MPS 豁免子条款）─────────────────────
FORMAL_STAGES: frozenset[str] = frozenset(
    {
        "main_experiment",
        "baseline",
        "external_validation",
        "ablation_final",
        "negative_control",
        "sensitivity_main_figure",
    }
)
"""Formal 阶段 stage 白名单；enforce_publication_grade_device 用其判定是否强制 CUDA。"""


# ─── 路径管理 ─────────────────────────────────────────────────────────────────
# 本文件位于 02code/，上一级即项目根目录
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(_THIS_DIR, '..'))   # 项目根目录 (02code/ → root)
# 数据目录默认 03data；可用环境变量 PROJECT_DATA_DIR 重定向到任意路径（便携性：
# 在远程算力机等非项目布局的环境运行时，把已清洗数据放到中性目录并指向它即可，
# 环境变量未设时行为与原先完全一致，不影响本地任何脚本。
DATA_DIR   = os.environ.get('PROJECT_DATA_DIR') or os.path.join(BASE_DIR, '03data')
OUTPUT_DIR = os.path.join(BASE_DIR, '04outputs')
LOG_DIR    = os.path.join(BASE_DIR, '05logs')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ─── 模板同步：其余标准目录常量（第二章目录树）────────────────────────────────
# 注意：与模板不同，本文件**不在导入时调用 ensure_project_dirs()**。
# 原因：本模块被 40+ 个实验脚本以 importlib.exec_module 加载，导入即执行顶层语句；
# 若在此自动 makedirs，会在 03data/ 下凭空创建 processed/ 目录，且一旦 datalock.sh
# 把 03data/ 置为只读，makedirs 将抛 PermissionError 并连带打挂全部实验脚本。
# 需要建目录的调用方请显式调用 ensure_project_dirs()。
CODE_DIR = os.path.join(BASE_DIR, '02code')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
DOC_DIR = os.path.join(BASE_DIR, '06doc')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, '07checkpoints')
REFINE_LOGS_DIR = os.path.join(BASE_DIR, 'refine-logs')
WIKI_DIR = os.path.join(BASE_DIR, 'research-wiki')

AVAILABLE_YEARS = (2018, 2019, 2025)
REFERENCE_YEAR = 2025
PROCESSED_DATA_FILES = {
    year: f'02_data_{year}.csv'
    for year in AVAILABLE_YEARS
}

_REFERENCE_WAVELENGTHS_CACHE: dict[int, FloatArray] = {}
_METADATA_COLUMNS = ['年份', '产地', 'SSC含量', 'SSC含量(°Brix)']
_WAVELENGTH_PATTERN = re.compile(r'([0-9]+(?:\.[0-9]+)?)nm')


# ─── 设备自动选择 ─────────────────────────────────────────────────────────────
def get_device(device_str: str = 'auto') -> str:
    """自动选择计算设备：优先CUDA > MPS > CPU"""
    if device_str == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    elif device_str == 'cuda':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device_str == 'mps':
        return 'mps' if torch.backends.mps.is_available() else 'cpu'
    return 'cpu'


# ─── 中文字体与全局图表设置 ──────────────────────────────────────────────────
def setup_chinese_font() -> None:
    """优先加载 SimHei / Arial Unicode MS / Microsoft YaHei，并关闭负号乱码。"""
    candidates = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
    available = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    chosen = None
    for c in candidates:
        if c in available:
            chosen = c
            break
    if chosen:
        plt.rcParams['font.sans-serif'] = [chosen]
    else:
        # 兜底：macOS 系统中文字体
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    # 嵌入字体到PDF，避免中文在LaTeX中显示为方块
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42


# ─── 索引→字母序列 (0→a, 1→b, ..., 25→z, 26→aa, ...) ──────────────────────
def _index_to_letter(index: int) -> str:
    letters = string.ascii_lowercase
    result = ''
    n = index
    while True:
        result = letters[n % 26] + result
        n = n // 26 - 1
        if n < 0:
            break
    return result


# ─── 图表命名规范 ─────────────────────────────────────────────────────────────
def figure_output_path(file_stem: str, index: int) -> str:
    """
    生成标准图表输出路径。
    例如：figure_output_path('pilot', 0) → 'OUTPUT_DIR/pilot-图a.pdf'
    """
    letter = _index_to_letter(index)
    filename = f'{file_stem}-图{letter}.pdf'
    return os.path.join(OUTPUT_DIR, filename)


# ─── 表格命名规范 ─────────────────────────────────────────────────────────────
def excel_output_path(file_stem: str) -> str:
    """生成标准 Excel 输出路径：OUTPUT_DIR/文件名.xlsx"""
    return os.path.join(OUTPUT_DIR, f'{file_stem}.xlsx')


def excel_data_path(file_stem: str) -> str:
    """生成数据处理输出路径：DATA_DIR/文件名.xlsx (仅用于02_data_processing)"""
    return os.path.join(DATA_DIR, f'{file_stem}.xlsx')


def table_sheet_name(index: int, title: str) -> str:
    """
    生成规范化 Sheet 名：表a：标题
    自动替换 Excel 非法字符 []:*?/\\ 为 _，并截断至 31 字符。
    """
    letter = _index_to_letter(index)
    raw = f'表{letter}：{title}'
    cleaned = re.sub(r'[\[\]:*?/\\]', '_', raw)
    return cleaned[:31]


def write_excel_workbook(file_stem: str, sheets: dict[Any, Any]) -> str:
    """
    统一写出多 Sheet Excel 工作簿。
    sheets: {index_or_str: (title, DataFrame)} 或 {str: DataFrame}
    支持两种格式：
      {0: ('标题', df), 1: ('标题2', df2)}
      {'自定义sheetname': df}
    返回输出路径。
    """
    path = excel_output_path(file_stem)
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for key, val in sheets.items():
            if isinstance(val, tuple):
                idx, title, df = key, val[0], val[1]
                sheet_name = table_sheet_name(idx, title)
            else:
                sheet_name = str(key)[:31]
                df = val
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return path


# ─── 处理后数据统一加载 ─────────────────────────────────────────────────────
def get_label_column(df: pd.DataFrame) -> str:
    """返回糖度标签列名，兼容 SSC含量 与 SSC含量(°Brix)。"""
    for col in ['SSC含量(°Brix)', 'SSC含量']:
        if col in df.columns:
            return col
    raise KeyError('未找到糖度标签列：SSC含量 或 SSC含量(°Brix)')


def get_spectrum_columns(df: pd.DataFrame) -> list[str]:
    """返回光谱列名列表。"""
    return [c for c in df.columns if c not in _METADATA_COLUMNS]


def parse_wavelength_from_column(col_name: str) -> float:
    """从列名中解析波长，例如 波段001_591.19nm → 591.19。"""
    if not isinstance(col_name, str):
        return np.nan
    match = _WAVELENGTH_PATTERN.search(col_name)
    if match is None:
        return np.nan
    return float(match.group(1))


def get_spectrum_wavelengths(df: pd.DataFrame, spectrum_cols: list[str] | None = None) -> FloatArray:
    """返回给定DataFrame光谱列对应的波长数组。"""
    spectrum_cols = spectrum_cols or get_spectrum_columns(df)
    return np.array([parse_wavelength_from_column(col) for col in spectrum_cols], dtype=np.float32)


# 参考 SSC 的物理上限。成熟富士苹果的可溶性固形物约 10~18 °Brix，
# 生理上限约 20；超过即为折光仪读数错误，不是真实样本。
SSC_PHYSICAL_MAX = 20.0


def load_processed_dataframe(year: int) -> pd.DataFrame:
    """
    读取 03data/02_data_YEAR.csv，并剔除参考 SSC 越界的读数异常样本。

    ⚠ 2026-07-14 修复的严重数据缺陷：
      2018 年含 1 个 SSC = 56.07 °Brix（陕西）、2025 年含 1 个 SSC = 37.12 °Brix（甘肃）。
      苹果的 SSC 生理上限约 20 °Brix，这两个值只可能是折光仪读数错误。
      而**全项目此前没有任何地方剔除它们**——论文却写着"已剔除 2 个 SSC>20 的异常样本"。

      实测破坏力：602 个迁移场景中，**357 个（59.3%）的目标域含该异常样本**，
      其中 **69 个（11.5%）落进测试集**。单点误差 |56.07 − 13| ≈ 43 °Brix，
      在 n≈40 的测试集上把 RMSE 从约 2.0 直接抬到约 7.1（×3.5）。
      它同时污染训练集拟合与斜率/偏置校正的拟合。

      本过滤属项目约定 §4.5 的 L1 白名单（"硬规则剔除：NaN/越界/质量码"）——
      它是一条与数据量无关的硬规则，不构成数据泄漏。

    在此处过滤是因为本函数是全项目唯一的数据入口，所有实验脚本都经由它取数。
    """
    if year not in PROCESSED_DATA_FILES:
        supported = ', '.join(str(y) for y in AVAILABLE_YEARS)
        raise ValueError(f'year={year} 不在当前可用年份集合中，可用年份: {supported}')
    path = os.path.join(DATA_DIR, PROCESSED_DATA_FILES[year])
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)

    # ── 硬规则 1：剔除参考 SSC 越界的读数异常样本 ──────────────────────
    label_col = get_label_column(df)
    n0 = len(df)
    df = df[df[label_col] <= SSC_PHYSICAL_MAX].reset_index(drop=True)
    if len(df) < n0:
        print(f'[01_export_utils] {year} 年剔除 SSC > {SSC_PHYSICAL_MAX:g} °Brix 的读数异常样本 '
              f'{n0 - len(df)} 个（{n0} → {len(df)}）')

    # ── 硬规则 2：剔除逐位完全相同的光谱（整组剔除）────────────────────
    #
    # ⚠ 2026-07-14 查出的第二个数据缺陷：
    #   2018 年**原始采集数据**中存在 18 组逐位完全相同的光谱（涉及 37 行），
    #   而它们的参考 SSC 标签**各不相同**（例如三条相同光谱对应 12.13 / 14.17 / 14.37 °Brix）。
    #   重复全部出现在**连续行号**上，2025 年数据则一组也没有——
    #   典型的采集/导出缺陷（光谱未在两次测量间刷新，果实 N+1 拿到了果实 N 的光谱）。
    #
    #   后果：这些行的「光谱–标签」是**错配**的。同一条光谱不可能同时属于两个糖度不同的果实，
    #   故每组中至少有一条是错的，且无法判定是哪一条。把已知错配的样本留在监督基准里不可辩护。
    #
    #   处置：**整组剔除**（而非保留首条）。保留任一条都需要假设采集缺陷的方向，
    #   而该方向无从证实；整组剔除是唯一不需要额外假设的选择，代价是损失约 2.7% 样本。
    #
    #   本过滤同属项目约定 §4.5 的 L1 白名单（"硬规则剔除：NaN/越界/质量码"），
    #   且为与数据量无关的硬规则，不构成数据泄漏。§7.6 亦要求做近邻重复检查。
    spec_cols = get_spectrum_columns(df)
    if spec_cols:
        key = df[spec_cols].round(10).astype(str).agg('|'.join, axis=1)
        dup_mask = key.duplicated(keep=False)          # keep=False → 整组标记
        n_dup = int(dup_mask.sum())
        if n_dup:
            df = df[~dup_mask].reset_index(drop=True)
            print(f'[01_export_utils] {year} 年剔除逐位重复的光谱 {n_dup} 行'
                  f'（光谱相同但标签不同 → 光谱-标签错配，整组剔除）')
    return df


def get_reference_wavelengths(reference_year: int = REFERENCE_YEAR) -> FloatArray:
    """
    获取参考波长网格。
    默认使用 2025 年处理后数据的 229 波段，供跨年份实验统一输入维度。
    """
    if reference_year not in _REFERENCE_WAVELENGTHS_CACHE:
        df = load_processed_dataframe(reference_year)
        _REFERENCE_WAVELENGTHS_CACHE[reference_year] = get_spectrum_wavelengths(df)
    return _REFERENCE_WAVELENGTHS_CACHE[reference_year].copy()


def align_spectrum_columns(
    df: pd.DataFrame, target_wavelengths: FloatArray | None = None
) -> tuple[list[str], FloatArray]:
    """
    按目标波长网格选择最接近的光谱列。
    target_wavelengths=None 时返回原始处理后光谱列。
    """
    spectrum_cols = get_spectrum_columns(df)
    source_wavelengths = get_spectrum_wavelengths(df, spectrum_cols)

    if target_wavelengths is None:
        return spectrum_cols, source_wavelengths

    chosen_cols = []
    chosen_wavelengths = []
    used_indices = set()

    for target in target_wavelengths:
        order = np.argsort(np.abs(source_wavelengths - target))
        selected_idx = None
        for idx in order:
            idx = int(idx)
            if idx not in used_indices:
                selected_idx = idx
                used_indices.add(idx)
                break
        if selected_idx is None:
            raise ValueError(f'无法为目标波长 {target:.2f}nm 匹配唯一列')
        chosen_cols.append(spectrum_cols[selected_idx])
        chosen_wavelengths.append(source_wavelengths[selected_idx])

    return chosen_cols, np.array(chosen_wavelengths, dtype=np.float32)


def load_processed_year_data(
    year: int,
    region: str | list[str] | tuple[str, ...] | None = None,
    target_wavelengths: FloatArray | str | None = None,
    dtype: npt.DTypeLike = np.float32,
) -> tuple[FloatArray, FloatArray, AnyArray, FloatArray]:
    """
    统一读取处理后数据。

    参数：
        year: 当前可用年份之一（默认项目为 2018 / 2025）
        region: 指定产地或产地列表；None 表示读取该年份全部产地
        target_wavelengths:
            - None: 使用该年份原始处理后波段
            - '2025': 对齐到 2025 年 229 波段
            - ndarray: 对齐到指定波长网格

    返回：
        X, y, regions, wavelengths
    """
    df = load_processed_dataframe(year)

    if region is not None:
        if isinstance(region, (list, tuple, set)):
            df = df[df['产地'].isin(region)].reset_index(drop=True)
        else:
            df = df[df['产地'] == region].reset_index(drop=True)

    if isinstance(target_wavelengths, str):
        if target_wavelengths != '2025':
            raise ValueError("target_wavelengths 仅支持 None、'2025' 或 ndarray")
        target_wavelengths = get_reference_wavelengths(REFERENCE_YEAR)

    spectrum_cols, wavelengths = align_spectrum_columns(df, target_wavelengths)
    label_col = get_label_column(df)

    X = df[spectrum_cols].values.astype(dtype)
    y = df[label_col].values.astype(dtype)
    regions = df['产地'].astype(str).values

    return X, y, regions, wavelengths


def load_year_region_map(
    years: tuple[int, ...] = AVAILABLE_YEARS,
    target_wavelengths: FloatArray | str | None = '2025',
    dtype: npt.DTypeLike = np.float32,
) -> dict[tuple[int, str], dict[str, Any]]:
    """返回 {(year, region): {'X','y','regions','wavelengths'}} 的统一数据字典。"""
    data_dict: dict[tuple[int, str], dict[str, Any]] = {}
    for year in years:
        df = load_processed_dataframe(year)
        for region in df['产地'].astype(str).unique():
            X, y, regions, wavelengths = load_processed_year_data(
                year=year,
                region=region,
                target_wavelengths=target_wavelengths,
                dtype=dtype,
            )
            data_dict[(year, region)] = {
                'X': X,
                'y': y,
                'regions': regions,
                'wavelengths': wavelengths,
            }
    return data_dict


def normalize_domain_key(domain_item: object) -> tuple[int, str]:
    """将域描述统一转换为 (year, region) 元组。"""
    if isinstance(domain_item, dict):
        return int(domain_item['year']), str(domain_item['region'])
    if isinstance(domain_item, (list, tuple)) and len(domain_item) == 2:
        return int(domain_item[0]), str(domain_item[1])
    if isinstance(domain_item, str):
        year_str, region = domain_item.split('_', 1)
        return int(year_str), region
    raise TypeError(f'无法解析域描述: {domain_item!r}')


def sort_domain_keys(domain_keys: Iterable[object]) -> tuple[tuple[int, str], ...]:
    """按 (year, region) 排序并去重。"""
    normalized = [normalize_domain_key(item) for item in domain_keys]
    return tuple(sorted(set(normalized), key=lambda item: (item[0], item[1])))


def domain_key_to_label(domain_key: tuple[int, str]) -> str:
    year, region = normalize_domain_key(domain_key)
    return f'{year}_{region}'


def domain_keys_to_label(domain_keys: Iterable[object]) -> str:
    """将域集合格式化为稳定字符串标签。"""
    return '+'.join(domain_key_to_label(key) for key in sort_domain_keys(domain_keys))


def extract_domain_keys(side_config: dict[str, Any]) -> tuple[tuple[int, str], ...]:
    """
    从场景配置的 source/target 字段提取域集合。
    兼容：
      - {'year', 'region'} 单域旧格式
      - {'domains': [...]} 多域新格式
      - {'labels': [...]} 仅标签格式
    """
    if 'domains' in side_config:
        raw_items = side_config['domains']
    elif 'labels' in side_config:
        raw_items = side_config['labels']
    elif 'year' in side_config and 'region' in side_config:
        raw_items = [(side_config['year'], side_config['region'])]
    else:
        raise KeyError(f'未找到可解析的域集合字段: {side_config}')
    return sort_domain_keys(raw_items)


def merge_domain_payload(
    year_region_data: dict[tuple[int, str], dict[str, Any]], domain_keys: Iterable[object]
) -> dict[str, Any]:
    """将多个原子域合并为一个源域/目标域负载。"""
    domain_keys = sort_domain_keys(domain_keys)
    payloads = []
    for key in domain_keys:
        if key not in year_region_data:
            raise KeyError(f'缺少域数据: {key}')
        payloads.append((key, year_region_data[key]))

    X_list, y_list, regions_list, years_list, domain_labels = [], [], [], [], []
    wavelengths = None
    for key, payload in payloads:
        X_list.append(payload['X'])
        y_list.append(payload['y'])
        regions_list.append(payload['regions'])
        years_list.append(np.full(len(payload['y']), key[0], dtype=np.int32))
        domain_labels.append(np.full(len(payload['y']), domain_key_to_label(key), dtype=object))
        if wavelengths is None:
            wavelengths = payload.get('wavelengths')

    return {
        'X': np.concatenate(X_list, axis=0),
        'y': np.concatenate(y_list, axis=0),
        'regions': np.concatenate(regions_list, axis=0),
        'years': np.concatenate(years_list, axis=0),
        'domain_labels': np.concatenate(domain_labels, axis=0),
        'wavelengths': wavelengths,
        'domain_keys': domain_keys,
        'label': domain_keys_to_label(domain_keys),
        'count': len(domain_keys),
    }


def infer_scenario_pattern_type(
    src_domain_keys: Iterable[object], tgt_domain_keys: Iterable[object]
) -> str:
    """根据源/目标域集合大小返回模式类型。"""
    src_n = len(sort_domain_keys(src_domain_keys))
    tgt_n = len(sort_domain_keys(tgt_domain_keys))
    src_label = '1源' if src_n == 1 else '多源'
    tgt_label = '1目标' if tgt_n == 1 else '多目标'
    return f'{src_label}→{tgt_label}'


def split_by_region(
    X: AnyArray,
    y: AnyArray,
    regions: AnyArray,
    src_region: str | list[str] | tuple[str, ...] | set[str],
    tgt_region: str | list[str] | tuple[str, ...] | set[str],
) -> tuple[AnyArray, AnyArray, AnyArray, AnyArray]:
    """按产地名切分源域和目标域，支持单个产地或产地列表。"""
    if isinstance(src_region, (list, tuple, set)):
        src_mask = np.isin(regions, list(src_region))
    else:
        src_mask = regions == src_region

    if isinstance(tgt_region, (list, tuple, set)):
        tgt_mask = np.isin(regions, list(tgt_region))
    else:
        tgt_mask = regions == tgt_region

    return X[src_mask], y[src_mask], X[tgt_mask], y[tgt_mask]


# ─── 日志工具 ─────────────────────────────────────────────────────────────────
class Logger:
    """同时向终端和 .log 文件写出日志，支持 print() 直接重定向。"""

    def __init__(self, file_stem: str) -> None:
        log_path = os.path.join(LOG_DIR, f'{file_stem}.log')
        # SIM115 抑制理由：Logger 生命周期内持有该句柄（供 log() 持续写入），由 close()/__del__ 显式关闭，
        # 不能用 with 上下文管理器（会在 __init__ 返回时立即关闭文件）。
        self._log_file = open(log_path, 'w', encoding='utf-8', buffering=1)  # noqa: SIM115
        self._stdout = sys.stdout
        self.log_path = log_path
        # 写头部
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._write(f'# 实验日志：{file_stem}  启动时间：{ts}\n')

    def _write(self, msg: str) -> None:
        self._log_file.write(msg)
        self._log_file.flush()

    def log(self, *args: object, sep: str = ' ', end: str = '\n') -> None:
        """写一行日志（同时输出到终端）。"""
        msg = sep.join(str(a) for a in args) + end
        self._stdout.write(msg)
        self._stdout.flush()
        self._write(msg)

    def close(self) -> None:
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._write(f'\n# 实验结束时间：{ts}\n')
        self._log_file.close()

    def __del__(self) -> None:
        # 解释器退出/回收阶段 close() 可能因资源已释放而抛错，静默吞掉即可（行为等价原 try/except pass）。
        with suppress(Exception):
            self.close()


def get_logger(file_stem: str) -> 'Logger':
    """获取一个绑定到 OUTPUT_DIR/{file_stem}.log 的日志实例。"""
    return Logger(file_stem)


# ═══════════════════════════════════════════════════════════════════════════════
# 模板同步区 —— 以下全部函数从 ARIS 模板 02code/export_utils.py 同步补齐
# ═══════════════════════════════════════════════════════════════════════════════
#
# 本区以上为 015 项目原有实现（26 个函数 + Logger 类），一行未改，其中 17 个
# 光谱/域适应自研函数为本项目独有，模板中不存在。
#
# 与模板的三处受控偏差（均为"保 015 不破"的必要让步，review 时请重点看）：
#   1. figure_output_paths / save_figure_bilingual 走私有 _figure_output_path_lang，
#      不复用 015 的 figure_output_path —— 后者无 lang 形参且已被 30+ 脚本调用，
#      加参数会改动原函数签名，故另起炉灶，中文路径产物与原函数完全一致。
#   2. write_datasheet 内部惰性 makedirs(PROCESSED_DATA_DIR)，因为本文件不在
#      导入时调用 ensure_project_dirs()（理由见文件头部路径常量处注释）。
#   3. 模板顶层的 ensure_project_dirs() 自动调用被移除，仅保留函数定义。
#
# 命名冲突处置（模板与 015 同名但实现不同者，一律保留 015 版本，未覆盖）：
#   get_device / setup_chinese_font / _index_to_letter / figure_output_path /
#   excel_output_path / excel_data_path / table_sheet_name / write_excel_workbook /
#   Logger / get_logger
# ═══════════════════════════════════════════════════════════════════════════════


# ─── 标准目录创建（第二章目录树）──────────────────────────────────────────────
def ensure_project_dirs() -> None:
    """自动创建项目标准目录结构。详见项目约定第二章。

    `wandb/` 与 `runs/` 由对应工具自动管理，本函数不创建。

    与模板的差异：本模块**不在导入时自动调用**本函数（模板会）。原因见文件头部
    路径常量处注释——本模块被 40+ 实验脚本 exec_module 加载，导入即建目录会污染
    03data/ 并在 datalock 只读锁定后打挂全部脚本。需要建目录请显式调用。
    """
    for d in (
        PROCESSED_DATA_DIR,
        OUTPUT_DIR,
        LOG_DIR,
        CHECKPOINTS_DIR,
        REFINE_LOGS_DIR,
        WIKI_DIR,
    ):
        os.makedirs(d, exist_ok=True)


# ─── 文件名清洗 ──────────────────────────────────────────────────────────────
_FORBIDDEN_CHARS = re.compile(r'[\[\]:*?/\\<>|"]')


def sanitize_filename(name: str) -> str:
    """清洗文件名：替换非法字符为 `_`，截断到 200 字符。"""
    cleaned = _FORBIDDEN_CHARS.sub("_", name)
    return cleaned[:200]


# ─── 路径生成函数（模板新增部分；excel_output_path / excel_data_path 沿用 015 版）──
def csv_output_path(file_stem: str, suffix: str | None = None) -> str:
    """04outputs/<file_stem>[-<suffix>].csv"""
    name = sanitize_filename(file_stem)
    if suffix:
        name = f"{name}-{sanitize_filename(suffix)}"
    return os.path.join(OUTPUT_DIR, f"{name}.csv")


def processed_data_path(file_stem: str, suffix: str | None = None) -> str:
    """03data/processed/<file_stem>[-<suffix>]（无扩展名，调用方拼 .npz / .parquet 等）"""
    name = sanitize_filename(file_stem)
    if suffix:
        name = f"{name}-{sanitize_filename(suffix)}"
    return os.path.join(PROCESSED_DATA_DIR, name)


def checkpoint_dir(file_stem: str, seed: int, timestamp: str | None = None) -> str:
    """07checkpoints/<file_stem>_<YY-MM-DD_HHMMSS>_seed<X>/

    顺序：脚本 → 时间 → 种子，与项目约定第二章目录树及第五章 Item 6 wandb run name 一致。
    timestamp 缺省时取当前时间。返回目录路径，自动创建。
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%y-%m-%d_%H%M%S")
    name = f"{sanitize_filename(file_stem)}_{timestamp}_seed{seed}"
    path = os.path.join(CHECKPOINTS_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def log_output_path(file_stem: str, timestamp: str | None = None) -> str:
    """05logs/<file_stem>_<YY-MM-DD_HHMMSS>.log"""
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%y-%m-%d_%H%M%S")
    return os.path.join(LOG_DIR, f"{sanitize_filename(file_stem)}_{timestamp}.log")


# ─── 哈希校验 ─────────────────────────────────────────────────────────────────
def _rank_ascending(values: FloatArray) -> IntArray:
    """并列感知的升序排名：GBID 越小名次越靠前，并列一律同取最小名次。

    §15.5 / §15.8 的「第一名频率」「平均名次」「排名区间」「成对胜率」全部建立在
    本函数上，因此它的并列语义直接决定印进论文表格的数字。

    **为什么不能用 `argsort(argsort(x)) + 1`**：该写法在出现并列时按数组下标强行
    分出先后——两个 GBID 完全相同的模型会被报成「第一名频率 100% / 0%」，系统性
    偏袒索引靠前者。消融变体与完整方法在某些指标上取值一致、或权重扰动恰好扫到
    等值点，都会触发这一情形，且不抛任何异常、不留任何痕迹。

    **为什么取 `method='min'` 而非 `'average'`**：并列第一必须两边都记为第 1 名，
    下游的 `(rankings == 1)` 才能同时命中；`'average'` 会把并列第一记成 1.5，
    使「第一名频率」塌成 0% / 0%，与 §15.8 表格的语义不符。代价是并列时平均名次
    偏乐观（记作 1, 1, 3 而非 1.5, 1.5, 3），报告平均名次时须知情。
    """
    from scipy.stats import rankdata

    return np.asarray(rankdata(np.asarray(values, dtype=float), method="min"), dtype=int)


def compute_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    """计算文件 SHA-256，返回 64 字符 hex 字符串。详见第五章 Item 5。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── 六路随机种子 ────────────────────────────────────────────────────────────
def set_random_seed(seed: int, deterministic: bool = True, device: str | None = None) -> None:
    """统一封装六路种子（详见第五章 Item 5）：
    random / numpy / torch CPU & CUDA / PYTHONHASHSEED / CUBLAS_WORKSPACE_CONFIG。

    deterministic=True 时同步开启 PyTorch 确定性算子：
        cudnn.deterministic=True, cudnn.benchmark=False, use_deterministic_algorithms=True。

    device 用于 MPS 豁免分支（第五章 Item 5 MPS 子条款）：
        - device='mps' 时跳过 cudnn.* 设置并向 stderr 写入 WARN（cudnn 在 MPS 环境无效）；
          torch.use_deterministic_algorithms 仅 best-effort（多数 MPS 算子非确定性）。
        - device=None 时按 deterministic 标志全开（保留旧行为兼容性）。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        if device == "mps":
            sys.stderr.write(
                "[WARN] set_random_seed: device='mps' detected — 跳过 cudnn.* 设置"
                "（MPS 环境下 cudnn 设置无效），torch.use_deterministic_algorithms 仅 best-effort。"
                " 详见项目约定第五章 Item 5 MPS 豁免子条款。\n"
            )
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        with suppress(AttributeError, RuntimeError):
            torch.use_deterministic_algorithms(True, warn_only=True)


# ─── Formal 阶段硬件守卫（第五章 Item 5 MPS 豁免子条款）─────────────────────────
def enforce_publication_grade_device(
    device: str,
    stage: str,
    logger: Logger | None = None,
) -> None:
    """Formal 阶段硬件守卫：stage ∈ FORMAL_STAGES 且 device != 'cuda' 时 raise SystemExit。

    详见项目约定第五章 Item 5 MPS 豁免子条款"阻塞型 guard"。
    所有 Formal 阶段入口脚本必须在 set_random_seed 之后、主体逻辑之前立即调用本函数。
    禁止任何形式绕过（环境变量、try/except 吞错、注释跳过等）。

    参数：
        device: 实际使用的设备字符串（通常来自 get_device 返回）。
        stage: 当前 stage 标签；Formal 集合见 FORMAL_STAGES 常量。
        logger: 可选 Logger 实例；非空时把守卫结果写入日志。

    若 stage 不在 FORMAL_STAGES（如 'pilot' / 'preprocessing' / 'sensitivity_exploration'）
    直接放行——开发与探索阶段不受 CUDA 强制约束。
    """
    if stage not in FORMAL_STAGES:
        if logger is not None:
            logger.log(
                f"[INFO] enforce_publication_grade_device: stage={stage!r} 非 Formal，跳过守卫。"
            )
        return
    if device == "cuda":
        if logger is not None:
            logger.log(
                f"[INFO] enforce_publication_grade_device: stage={stage!r} on cuda — 守卫通过。"
            )
        return
    msg = (
        f"主结果 stage 必须 CUDA，当前 stage={stage!r}, device={device!r}；"
        f"切换 --device cuda 或将 stage 改为 'pilot' 后重跑。"
        f"（FORMAL_STAGES={sorted(FORMAL_STAGES)}；详见项目约定第五章 Item 5 MPS 豁免子条款）"
    )
    if logger is not None:
        logger.log(f"[FATAL] {msg}")
    raise SystemExit(msg)


# ─── 中英双语图表（详见第四章 Item 3）─────────────────────────────────────────
def _figure_output_path_lang(file_stem: str, index: int, lang: str = "zh") -> str:
    """双语图表路径构造（等价于模板的 figure_output_path(..., lang=...)）。

    015 原有的 figure_output_path(file_stem, index) 无 lang 形参且已被 30+ 个实验
    脚本调用，为保持其签名与产物完全不变，双语路径另起本私有函数，不改动原函数。

    lang='zh' → 04outputs/<file_stem>-图a.pdf   （与 015 原 figure_output_path 输出一致）
    lang='en' → 04outputs/<file_stem>-fig-a.pdf
    """
    letter = _index_to_letter(index)
    if lang == "zh":
        filename = f"{sanitize_filename(file_stem)}-图{letter}.pdf"
    elif lang == "en":
        filename = f"{sanitize_filename(file_stem)}-fig-{letter}.pdf"
    else:
        raise ValueError(f"lang 必须为 'zh' 或 'en'，收到: {lang!r}")
    return os.path.join(OUTPUT_DIR, filename)


def figure_output_paths(file_stem: str, index: int) -> dict[str, str]:
    """同时返回中英双语图路径。返回 {'zh': '...-图a.pdf', 'en': '...-fig-a.pdf'}"""
    return {
        "zh": _figure_output_path_lang(file_stem, index, lang="zh"),
        "en": _figure_output_path_lang(file_stem, index, lang="en"),
    }


def save_figure_bilingual(
    plot_fn: Callable[[str], Figure], file_stem: str, index: int
) -> dict[str, str]:
    """统一双语图保存入口。

    plot_fn(lang) 必须返回 matplotlib.figure.Figure，按 lang ∈ {'zh','en'} 切换文本内容。
    数据 / 配色 / 字号 / 版式必须完全一致（第四章 Item 3）。
    """
    saved: dict[str, str] = {}
    for lang in ("zh", "en"):
        fig = plot_fn(lang)
        path = _figure_output_path_lang(file_stem, index, lang=lang)
        fig.savefig(path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        saved[lang] = path
    return saved


# ─── 一脚本一 xlsx 强制入口（第四章 Item 1）───────────────────────────────────
def write_script_workbook(script_path: str, sheets: dict[Any, Any]) -> str:
    """正式脚本唯一允许的 Excel 写出入口（项目约定第四章 Item 1 强约束）。

    严格执行"一脚本一 xlsx"：file_stem 由调用方所在脚本文件名（去 `.py`）派生，
    不接受任何自定义 stem。所有 dataset × task × protocol × budget × ablation ×
    negative_control × failure_cases × external_validation 等结果必须折叠为同
    一 workbook 内的多个 sheet——禁止按业务维度拆出多个 xlsx。

    用法（必须直接传 __file__，不接受字符串拼接绕过）：
        write_script_workbook(__file__, {
            0: ('main', main_df),
            'budget': budget_df,
            'ablation': ablation_df,
            'failure_cases': failure_cases_df,
            'external_validation': external_df,
        })

    项目级中枢报表（如算力总账 compute_budget）必须由项目级中枢脚本（推荐命名
    `99project_summary.py`）调用本函数生成 `99project_summary.xlsx`，并把
    compute_budget / 各脚本元信息作为 sheet 落入；禁止单训练脚本越权写出项目级
    汇总文件。

    sheets 字典格式同 write_excel_workbook；sheet 名长度受 Excel 31 字符限制，
    冲突时由 openpyxl 报错（不静默去重，防止覆盖）。
    """
    file_stem = os.path.splitext(os.path.basename(script_path))[0]
    if not file_stem:
        raise ValueError(f"无法从 script_path 派生 file_stem: {script_path!r}")
    return write_excel_workbook(file_stem, sheets)


# ─── 数据卡（第四章 Item 10）/ 模型卡（第十一章 Item 3）───────────────────────
_DATASHEET_REQUIRED = [
    "source_and_license",
    "integrity_sha",
    "granularity_unit",
    "sample_stats",
    "split_logic",
    "known_bias",
    "label_reliability",
    "preprocessing_pointer",
    "ethics_label",
]

_MODELCARD_REQUIRED = [
    "arch",
    "params",
    "flops",
    "ckpt_sha",
    "training_config",
    "training_hardware",
    "deterministic_guarantee",
    "stage",
    "eval_summary",
    "intended_use_and_limits",
    "license_ethics",
    "traceability",
]

# 字段键保持英文（代码内引用方便、跨语言稳定），但渲染进 .md 的**标题必须是中文**——
# 第一章 Item 1 强制 .md 文档纯中文输出。数据卡/模型卡是随论文与 GitHub 一起发布的正式文档。
_DATASHEET_TITLES = {
    "source_and_license": "数据来源与许可",
    "integrity_sha": "完整性校验（SHA-256）",
    "granularity_unit": "粒度与单位",
    "sample_stats": "样本统计",
    "split_logic": "划分逻辑",
    "known_bias": "已知偏差",
    "label_reliability": "标签可靠性",
    "preprocessing_pointer": "预处理流程指针",
    "ethics_label": "伦理标签",
}

_MODELCARD_TITLES = {
    "arch": "架构",
    "params": "参数量",
    "flops": "FLOPs",
    "ckpt_sha": "权重 SHA-256",
    "training_config": "训练配置",
    "training_hardware": "训练硬件",
    "deterministic_guarantee": "确定性保证",
    "stage": "阶段（pilot / formal）",
    "eval_summary": "评估摘要",
    "intended_use_and_limits": "预期用途与边界",
    "license_ethics": "许可与伦理",
    "traceability": "追溯指针",
}
"""模型卡必填字段（第十一章 Item 3）。

training_hardware：'cuda' / 'mps' / 'cpu' + GPU 型号 + 数量。
deterministic_guarantee：'full'（CUDA + 全 deterministic 开）/ 'best-effort'（MPS + 算子降级清单）/ 'none'。
stage：'pilot' / 'formal'；Formal 阶段必须 training_hardware='cuda' + deterministic_guarantee='full'
（与第五章 Item 5 MPS 豁免子条款及 enforce_publication_grade_device guard 联动）。
"""


def _render_md_section(title: str, value: Any) -> list[str]:
    lines = [f"## {title}", ""]
    if isinstance(value, str):
        lines.append(value)
    else:
        lines.append("```json")
        lines.append(json.dumps(value, ensure_ascii=False, indent=2, default=str))
        lines.append("```")
    lines.append("")
    return lines


def write_datasheet(dataset_name: str, fields: dict[str, Any]) -> str:
    """写出数据卡 03data/processed/<dataset_name>-datasheet.md（第四章 Item 10）。

    fields 必须包含: source_and_license / integrity_sha / granularity_unit / sample_stats /
    split_logic / known_bias / label_reliability / preprocessing_pointer / ethics_label.
    缺一即 ValueError。
    """
    missing = [k for k in _DATASHEET_REQUIRED if k not in fields]
    if missing:
        raise ValueError(f"数据卡缺失必填字段: {missing}")
    # 与模板差异：模板依赖导入时的 ensure_project_dirs() 建好 processed/，
    # 本模块不在导入时建目录（理由见文件头部），故此处惰性创建。
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DATA_DIR, f"{sanitize_filename(dataset_name)}-datasheet.md")
    lines = [f"# 数据集卡：{dataset_name}", ""]
    for k in _DATASHEET_REQUIRED:
        lines.extend(_render_md_section(_DATASHEET_TITLES[k], fields[k]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def write_model_card(run_name: str, fields: dict[str, Any]) -> str:
    """写出模型卡 07checkpoints/<run_name>/model_card.md（第十一章 Item 3）。

    fields 必须包含 _MODELCARD_REQUIRED 全部字段：
        arch / params / flops / ckpt_sha / training_config / training_hardware /
        deterministic_guarantee / stage / eval_summary / intended_use_and_limits /
        license_ethics / traceability.

    Formal 阶段强校验（与第五章 Item 5 MPS 豁免子条款联动）：
        stage='formal' 时强制要求 training_hardware='cuda' 且 deterministic_guarantee='full'，
        否则 raise ValueError 阻止落盘——避免 MPS 上的模型卡被误标为 Formal 投稿级证据。
    """
    missing = [k for k in _MODELCARD_REQUIRED if k not in fields]
    if missing:
        raise ValueError(f"模型卡缺失必填字段: {missing}")
    if fields["stage"] == "formal":
        hw = fields["training_hardware"]
        det = fields["deterministic_guarantee"]
        hw_str = hw if isinstance(hw, str) else str(hw)
        if not hw_str.startswith("cuda") or det != "full":
            raise ValueError(
                f"Formal 阶段模型卡 hardware/deterministic 不合规："
                f"training_hardware={hw!r}（应以 'cuda' 开头）、"
                f"deterministic_guarantee={det!r}（应为 'full'）。"
                f" 详见项目约定第五章 Item 5 + 第十一章 Item 3。"
            )
    run_dir = os.path.join(CHECKPOINTS_DIR, sanitize_filename(run_name))
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "model_card.md")
    lines = [f"# 模型卡：{run_name}", ""]
    for k in _MODELCARD_REQUIRED:
        lines.extend(_render_md_section(_MODELCARD_TITLES[k], fields[k]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def save_best_checkpoint(
    model: torch.nn.Module,
    run_name: str,
    model_card_fields: dict[str, Any],
    *,
    tag: str = "best",
    optimizer: torch.optim.Optimizer | None = None,
    extra_state: dict[str, Any] | None = None,
) -> dict[str, str]:
    """一站式落地 checkpoint（第五章 Item 6 + 第十一章 Item 3 强制入口）。

    单次调用完成：torch.save → SHA-256 → write_model_card；自动把 SHA 注入
    model_card_fields['ckpt_sha']，调用方传入的 'ckpt_sha' 会被覆盖。

    best 与 last 需分别调用两次（tag='best' / tag='last'），每次写一个 .pt 文件。
    禁止裸 torch.save 跳过本封装——会导致 SHA-256 与模型卡缺失，违反项目约定
    第五章 Item 6 "Checkpoint 联动" 强制规范。

    参数：
        model: 待保存的 nn.Module，仅 state_dict 落盘。
        run_name: 与 wandb run name / 日志前缀一致的字符串，详见项目约定第五章 Item 6。
        model_card_fields: 模型卡字段；'ckpt_sha' 会被本函数覆盖，其余必填项见
            _MODELCARD_REQUIRED。
        tag: "best" 或 "last"，落盘文件名 `<tag>.pt`。
        optimizer: 可选，传入则一并保存 optimizer.state_dict()。
        extra_state: 可选，额外字段（如 epoch / scheduler / scaler 等）写入同一 checkpoint。

    返回：{ckpt_path, sha256, run_dir, model_card_path}
    """
    if tag not in ("best", "last"):
        raise ValueError(f"tag 必须是 'best' 或 'last'，收到: {tag!r}")
    run_dir = os.path.join(CHECKPOINTS_DIR, sanitize_filename(run_name))
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, f"{tag}.pt")
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "run_name": run_name,
        "tag": tag,
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra_state is not None:
        payload["extra_state"] = extra_state
    torch.save(payload, ckpt_path)
    sha256 = compute_sha256(ckpt_path)
    fields = {**model_card_fields, "ckpt_sha": sha256}
    model_card_path = write_model_card(run_name, fields)
    return {
        "ckpt_path": ckpt_path,
        "sha256": sha256,
        "run_dir": run_dir,
        "model_card_path": model_card_path,
    }



def save_sklearn_checkpoint(
    model: Any,
    run_name: str,
    model_card_fields: dict[str, Any],
    *,
    tag: str = "best",
    extra_state: dict[str, Any] | None = None,
) -> dict[str, str]:
    """非 torch 模型（sklearn / imblearn Pipeline / statsmodels 等）的 checkpoint 落盘入口。

    与 `save_best_checkpoint` 完全同权：单次调用完成 joblib.dump → SHA-256 →
    write_model_card，落到 `07checkpoints/<run_name>/<tag>.joblib`。

    **为什么必须有这个函数**（本条源自项目 020 的实战事故，回流至模板）：
    `save_best_checkpoint` 的签名写死 `torch.nn.Module`，而光谱 / 表格类课题的主力
    模型往往是 PLSR / SVR / RandomForest 这类 sklearn 估计器。缺了本通道的后果不是
    "换个方式存"，而是**整个项目一个权重都不存**——020 跑到投稿阶段 `07checkpoints/`
    仍然全空、零 model_card，第十一章 Item 3 整章空转，且没有任何报错。
    附带事故：模型缓存被随手 pickle 到了 `03data/processed/`，污染数据区。

    落盘格式用 `joblib` 而非 `pickle`：sklearn 官方推荐，对模型内部的大 numpy 数组
    做专门优化，体积与读写速度都显著优于裸 pickle。

    模型卡字段口径（`_MODELCARD_REQUIRED` 全部必填，与 torch 通道一致）：
        - `arch`   : 估计器类名 + 关键结构（如 "PLSRegression(n_components=12)"）
        - `params` : 可学习参数量；sklearn 填有意义的规模量（如 "coef_ shape=(215,3)"）
        - `flops`  : sklearn 估计器通常无意义，填 "N/A（sklearn 估计器，非逐层前向）"
        - `stage='formal'` 时同样强制 `training_hardware` 以 'cuda' 开头且
          `deterministic_guarantee='full'`。**纯 CPU 的 sklearn 主实验应标
          `stage='pilot'`，或在 `01PROJECT_PROTOCOL.md` B2 中具名声明该 stage
          不受 CUDA 强制**（第五章 Item 3 的 CUDA 强制针对深度模型，见 FORMAL_STAGES）。

    参数：
        model: 任意可被 joblib 序列化的估计器对象（sklearn Estimator / Pipeline / 自定义类）。
        run_name: 与 wandb run name / 日志前缀一致的字符串。
        model_card_fields: 模型卡字段；'ckpt_sha' 会被本函数覆盖。
        tag: "best" 或 "last"。
        extra_state: 可选附加状态（如折号 / 超参 / 训练集统计），与模型一起打包落盘。

    返回：{ckpt_path, sha256, run_dir, model_card_path}
    """
    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "save_sklearn_checkpoint 需要 `pip install joblib`（sklearn 的官方序列化依赖）"
        ) from exc
    if tag not in ("best", "last"):
        raise ValueError(f"tag 必须是 'best' 或 'last'，收到: {tag!r}")
    run_dir = os.path.join(CHECKPOINTS_DIR, sanitize_filename(run_name))
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, f"{tag}.joblib")
    payload: dict[str, Any] = {
        "model": model,
        "run_name": run_name,
        "tag": tag,
        "estimator_class": type(model).__name__,
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if extra_state is not None:
        payload["extra_state"] = extra_state
    joblib.dump(payload, ckpt_path)
    sha256 = compute_sha256(ckpt_path)
    fields = {**model_card_fields, "ckpt_sha": sha256}
    model_card_path = write_model_card(run_name, fields)
    return {
        "ckpt_path": ckpt_path,
        "sha256": sha256,
        "run_dir": run_dir,
        "model_card_path": model_card_path,
    }


def save_checkpoint(
    model: Any,
    run_name: str,
    model_card_fields: dict[str, Any],
    *,
    tag: str = "best",
    **kwargs: Any,
) -> dict[str, str]:
    """checkpoint 落盘的**统一分发入口**——按模型类型自动选 torch / joblib 通道。

    脚本作者不需要记住"torch 走哪个、sklearn 走哪个"：一律调本函数。
    `torch.nn.Module` → `save_best_checkpoint`；其余一切 → `save_sklearn_checkpoint`。

    这层分发存在的意义是消除"选错通道就整个项目不存权重"这一失败模式（见
    `save_sklearn_checkpoint` 的事故说明）。两条下游通道各自的强校验（模型卡必填
    字段、Formal 阶段 cuda + full 校验）都不因走本入口而放松。

    `optimizer` 仅 torch 通道支持；传给 sklearn 通道会因签名不匹配直接 TypeError，
    这是有意的——它意味着调用方对模型类型的判断本身就错了。
    """
    if isinstance(model, torch.nn.Module):
        return save_best_checkpoint(model, run_name, model_card_fields, tag=tag, **kwargs)
    return save_sklearn_checkpoint(model, run_name, model_card_fields, tag=tag, **kwargs)


# ─── 结果与指标 ────────────────────────────────────────────────────────────────
def save_predictions(
    predictions_df: pd.DataFrame, file_stem: str, suffix: str = "predictions"
) -> str:
    """保存预测结果到 04outputs/<file_stem>-<suffix>.csv（UTF-8 with BOM 便于 Excel 打开）。"""
    path = csv_output_path(file_stem, suffix=suffix)
    predictions_df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def n_distinct_seeds(seeds: Sequence[int] = RECOMMENDED_SEEDS) -> int:
    """上报用的种子数：去重后的数量。

    表格/正文里的 `n_seeds=` 一律取本函数，禁止写 `len(seeds)`。集合内允许重复
    （见 RECOMMENDED_SEEDS 说明），但重复的那一次不提供独立信息。
    """
    return len(set(seeds))


def save_metrics_summary(metrics_df: pd.DataFrame, file_stem: str) -> str:
    """保存指标汇总到 04outputs/<file_stem>.xlsx，sheet 名为 'metrics_summary'。"""
    return write_excel_workbook(file_stem, {"metrics_summary": metrics_df})


# ─── 统计检验（第七章 Item 14 重复 CV 相关性感知）─────────────────────────────
def _cliffs_delta(a: FloatArray, b: FloatArray) -> float:
    """Cliff's δ effect size for non-parametric paired comparison."""
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return 0.0
    greater = int(np.sum(a[:, None] > b[None, :]))
    less = int(np.sum(a[:, None] < b[None, :]))
    return (greater - less) / (n_a * n_b)


def _nadeau_bengio_t(
    diffs: FloatArray, n_train: int, n_test: int
) -> tuple[float, float]:
    """Nadeau-Bengio corrected resampled t-test (Nadeau & Bengio, 1999/2003)。

    修正方差公式：corrected_var = var(diffs) * (1/n + n_test/n_train)
    返回 (t_stat, p_two_sided)。
    """
    from scipy import stats

    n = len(diffs)
    if n < 2:
        raise ValueError("Nadeau-Bengio 至少需要 2 个配对样本")
    mean_diff = float(np.mean(diffs))
    var_diff = float(np.var(diffs, ddof=1))
    corrected_var = var_diff * (1.0 / n + n_test / n_train)
    if corrected_var <= 0:
        return 0.0, 1.0
    t_stat = mean_diff / float(np.sqrt(corrected_var))
    p_two = 2.0 * (1.0 - float(stats.t.cdf(abs(t_stat), df=n - 1)))
    return t_stat, p_two


def compare_methods(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    method: str = "nadeau_bengio",
    n_train: int | None = None,
    n_test: int | None = None,
    cluster_unit: str = "fold",
    protocol_path: str | None = None,
    rope: float = 0.01,
    bootstrap_n: int = 1000,
    bootstrap_ci: float = 0.95,
    rng_seed: int = 42,
) -> dict[str, Any]:
    """配对方法对比的统一封装（第七章 Item 14）。

    method:
        'nadeau_bengio' — 推荐默认，需提供 n_train / n_test
        'paired_t'     — 朴素 paired t-test（仅在预注册声明可独立时使用）
        'wilcoxon'     — 非参数回退
        'bayes'        — 通过 baycomp.two_on_single 给出 P(A 优) / P(ROPE) / P(B 优)

    cluster bootstrap CI（第七章 Item 14 强制）：
        bootstrap_n > 0 时（默认 1000 次）自动对 diffs 做"按外层折"层级
        重采样，得到 mean_diff 的 (bootstrap_ci) 置信区间，写入 mean_diff_ci_low /
        mean_diff_ci_high；若需按实体 ID 聚类需在调用前预先按实体聚合 scores。
        bootstrap_n=0 跳过（仅小样本调试 / 单测可用，正式实验禁止）。

    返回字段统一包含: test / cluster_unit / protocol_path / n_paired /
    mean_diff / std_diff / mean_diff_ci_low / mean_diff_ci_high / bootstrap_n /
    cohens_d / cliffs_delta / p_raw / 测试特定项。
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"scores_a / scores_b 形状不一致: {a.shape} vs {b.shape}")
    diffs = a - b
    result: dict[str, Any] = {
        "test": method,
        "cluster_unit": cluster_unit,
        "protocol_path": protocol_path,
        "n_paired": len(diffs),
        "mean_diff": float(np.mean(diffs)),
        "std_diff": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
    }
    result["cohens_d"] = (
        result["mean_diff"] / result["std_diff"] if result["std_diff"] > 0 else 0.0
    )
    result["cliffs_delta"] = _cliffs_delta(a, b)

    if bootstrap_n > 0 and len(diffs) > 1:
        rng = np.random.default_rng(rng_seed)
        n = len(diffs)
        boot_idx = rng.integers(0, n, size=(bootstrap_n, n))
        boot_means = diffs[boot_idx].mean(axis=1)
        alpha = (1.0 - bootstrap_ci) / 2.0
        result["mean_diff_ci_low"] = float(np.percentile(boot_means, alpha * 100))
        result["mean_diff_ci_high"] = float(
            np.percentile(boot_means, (1.0 - alpha) * 100)
        )
        result["bootstrap_n"] = bootstrap_n
        result["bootstrap_ci_level"] = bootstrap_ci

    if method == "nadeau_bengio":
        if n_train is None or n_test is None:
            raise ValueError("Nadeau-Bengio 必须提供 n_train 和 n_test")
        t_stat, p_two = _nadeau_bengio_t(diffs, n_train, n_test)
        result["t_stat"] = t_stat
        result["p_raw"] = p_two
    elif method == "paired_t":
        from scipy import stats

        t_stat, p_two = stats.ttest_rel(a, b)
        result["t_stat"] = float(t_stat)
        result["p_raw"] = float(p_two)
    elif method == "wilcoxon":
        from scipy import stats

        try:
            w_stat, p_two = stats.wilcoxon(a, b, zero_method="wilcox")
            result["w_stat"] = float(w_stat)
            result["p_raw"] = float(p_two)
        except ValueError as e:
            result["w_stat"] = None
            result["p_raw"] = 1.0
            result["error"] = str(e)
    elif method == "bayes":
        try:
            import baycomp
        except ImportError as exc:
            raise ImportError("Bayesian comparison 需要 `pip install baycomp`") from exc
        posterior = baycomp.two_on_single(a, b, rope=rope)
        # baycomp 新版直接返回 tuple (p_left, p_rope, p_right)；老版返回带 .probs() 的对象
        if isinstance(posterior, tuple):
            p_left, p_rope, p_right = posterior
        else:
            p_left, p_rope, p_right = posterior.probs()
        result["p_a_better"] = float(p_left)
        result["p_rope"] = float(p_rope)
        result["p_b_better"] = float(p_right)
        result["p_raw"] = float(1.0 - p_left)
    else:
        raise ValueError(f"未知 method: {method}")
    return result


def apply_multi_comp_correction(
    p_values: Sequence[float],
    method: str = "holm",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Holm 或 Benjamini-Hochberg (BH/FDR) 多比较校正（第七章 Item 14 强制）。

    在多组 compare_methods 调用后收集所有 p_raw，统一调本函数得校正后 p_corrected
    与拒绝标记。

    参数：
        p_values: 待校正的 p 值序列。
        method: 'holm' 或 'bh'（'fdr_bh' 同义），与 statsmodels.multipletests 一致。
        alpha: 全局显著性水平，默认 0.05。

    返回：{method, alpha, p_raw, p_corrected, rejected}
    """
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError as exc:
        raise ImportError(
            "多比较校正需要 `pip install statsmodels`"
        ) from exc
    method_map = {"holm": "holm", "bh": "fdr_bh", "fdr_bh": "fdr_bh"}
    if method not in method_map:
        raise ValueError(f"未知 method: {method!r}，仅支持 {list(method_map.keys())}")
    sm_method = method_map[method]
    reject, p_corr, _, _ = multipletests(list(p_values), alpha=alpha, method=sm_method)
    return {
        "method": method,
        "alpha": alpha,
        "p_raw": [float(p) for p in p_values],
        "p_corrected": [float(p) for p in p_corr],
        "rejected": [bool(r) for r in reject],
    }


def compare_methods_multi(
    scores_by_method: dict[str, Sequence[float]],
    rope: float = 0.01,
) -> dict[str, Any]:
    """多模型 Bayesian 配对比较（第七章 Item 14 多模型场景）。

    遍历所有 C(n, 2) 方法对，对每对调 `baycomp.two_on_multiple`（外层折作为
    "datasets" 维度，给出 hierarchical posterior）；若该 API 不可用则自动回退
    `baycomp.SignedRankTest` 或 `baycomp.two_on_single`。所有方法得分必须等长
    （n_primary = K_outer）。

    返回：{test, rope, method_names, n_pairs, pairs: [{method_a, method_b,
           p_a_better, p_rope, p_b_better}, ...]}
    """
    try:
        import baycomp
    except ImportError as exc:
        raise ImportError(
            "Bayesian 多模型比较需要 `pip install baycomp`"
        ) from exc
    method_names = list(scores_by_method.keys())
    n = len(method_names)
    if n < 2:
        raise ValueError(f"至少需要 2 个方法，收到 {n}")
    lengths = {m: len(scores_by_method[m]) for m in method_names}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"所有方法的 scores 长度必须一致，收到 {lengths}")
    pairs: list[dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a_name, b_name = method_names[i], method_names[j]
            a = np.asarray(scores_by_method[a_name], dtype=float)
            b = np.asarray(scores_by_method[b_name], dtype=float)
            posterior: Any = None
            # 优先 two_on_multiple（hierarchical，需 pystan）；任何失败回退非层级 API
            try:
                posterior = baycomp.two_on_multiple(
                    a.reshape(-1, 1), b.reshape(-1, 1), rope=rope
                )
            except Exception:  # baycomp 依赖 pystan / 旧版 API 差异，必须宽松回退
                try:
                    posterior = baycomp.SignedRankTest.probs(a, b, rope=rope)
                except (AttributeError, TypeError):
                    posterior = baycomp.two_on_single(a, b, rope=rope)
            if not isinstance(posterior, tuple):
                posterior = posterior.probs()
            p_left, p_rope, p_right = posterior
            pairs.append(
                {
                    "method_a": a_name,
                    "method_b": b_name,
                    "p_a_better": float(p_left),
                    "p_rope": float(p_rope),
                    "p_b_better": float(p_right),
                }
            )
    return {
        "test": "bayesian_pairwise",
        "rope": rope,
        "method_names": method_names,
        "n_pairs": len(pairs),
        "pairs": pairs,
    }


# ─── LLH-GBID 多模型综合评价框架（第十五章 CRITICAL）─────────────────────────
# 完整方法论见 refine-logs/02LLH_GBID_FRAMEWORK.md

_GBID_Q_GRID_DEFAULT: tuple[tuple[float, float], ...] = (
    (2.0, 2.0),
    (3.0, 2.0),
    (2.0, 3.0),
    (3.0, 3.0),
)
"""短板敏感性分析默认 q 网格（第十五章 §15.5）。"""

_GBID_EPS: float = 1e-10
"""GBID 数值稳定小量，防止除零和 log(0)。"""


def compute_utility(
    x: FloatArray,
    I: float,
    L: float,
    direction: str = "benefit",
    *,
    s: float = 1.0,
    clip: bool = True,
) -> dict[str, FloatArray]:
    """单指标效用化（第十五章 §3.1—§3.3）。

    将原始指标值 x 转换为 [0,1] 正向效用值。

    参数：
        x: 原始指标值，形状 (n_models,) 或 (n_models, n_repeats)。
        I: 理想值（效益型：最大可接受；成本型：最小可接受）。
        L: 基准值（效益型：最低可接受；成本型：最高可接受）。
        direction: 指标方向——'benefit'（越大越好）或 'cost'（越小越好）。
        s: 非线性映射指数，>1 凸（接近理想才显著得分），<1 凹（快速脱离基准即获益），=1 线性。
        clip: 是否截断到 [0,1]，False 时返回截断前效用 ũ_j。

    返回：
        {'u_tilde': ũ_j (截断前), 'u': u_j (截断后, 若 clip=True)}

    异常：
        ValueError: 若 |I - L| < _GBID_EPS（退化指标）。
        ValueError: 若 direction 不是 'benefit' 或 'cost'。
        ValueError: 若 s <= 0。
    """
    if s <= 0:
        raise ValueError(f"s 必须 > 0，收到 {s}")
    if direction not in ("benefit", "cost"):
        raise ValueError(f"direction 必须为 'benefit' 或 'cost'，收到 {direction!r}")
    denom = abs(I - L)
    if denom < _GBID_EPS:
        raise ValueError(
            f"|I - L| = {denom} < {_GBID_EPS}，指标退化（分母为零）。"
            f" 请调整 I={I}, L={L} 或将该指标退出评价并在附录登记。"
        )
    x_arr = np.asarray(x, dtype=float)
    u_tilde = (x_arr - L) / denom if direction == "benefit" else (L - x_arr) / denom
    if s != 1.0:
        u_tilde = np.sign(u_tilde) * (np.abs(u_tilde) ** s)
    result: dict[str, FloatArray] = {"u_tilde": u_tilde}
    if clip:
        result["u"] = np.clip(u_tilde, 0.0, 1.0)
    return result


def compute_target_utility(
    x: FloatArray,
    T: float = 0.0,
    tau_minus: float | None = None,
    tau_plus: float | None = None,
    *,
    delta_minus: float = 0.0,
    delta_plus: float = 0.0,
    clip: bool = True,
) -> dict[str, FloatArray]:
    """目标型指标效用化（第十五章 §3.2 统一非对称带死区形式）。

    将原始指标值 x 转换为关于目标值 T 的 [0,1] 正向效用值。

    参数：
        x: 原始指标值，形状 (n_models,) 或 (n_models, n_repeats)。
        T: 目标值（默认 0，用于偏差类指标）。
        tau_minus: 低于 T 侧的零效用偏离边界。若为 None 则与 tau_plus 相同。
        tau_plus: 高于 T 侧的零效用偏离边界。若两者均为 None，默认 tau=1.0。
        delta_minus: 低于 T 侧的无惩罚容忍半径（≥0）。
        delta_plus: 高于 T 侧的无惩罚容忍半径（≥0）。
        clip: 是否截断到 [0,1]。

    返回：
        {'u_tilde': ũ_j, 'u': u_j (若 clip=True), 'e': 有效偏离量 e_j}

    异常：
        ValueError: 若 τ^± ≤ δ^±（容忍半径不能超过零效用边界）。
    """
    if tau_minus is None and tau_plus is None:
        tau_minus = tau_plus = 1.0
    elif tau_minus is None:
        tau_minus = tau_plus
    elif tau_plus is None:
        tau_plus = tau_minus
    # mypy 窄化后仍为 float
    tm: float = tau_minus  # type: ignore[assignment]
    tp: float = tau_plus  # type: ignore[assignment]
    if tm <= delta_minus:
        raise ValueError(f"tau_minus ({tm}) 必须 > delta_minus ({delta_minus})")
    if tp <= delta_plus:
        raise ValueError(f"tau_plus ({tp}) 必须 > delta_plus ({delta_plus})")
    x_arr = np.asarray(x, dtype=float)
    e = np.zeros_like(x_arr)
    below = x_arr <= T
    above = ~below
    if np.any(below):
        e[below] = np.maximum(0.0, (T - x_arr[below]) - delta_minus) / (tm - delta_minus)
    if np.any(above):
        e[above] = np.maximum(0.0, (x_arr[above] - T) - delta_plus) / (tp - delta_plus)
    u_tilde = 1.0 - e
    result: dict[str, FloatArray] = {"u_tilde": u_tilde, "e": e}
    if clip:
        result["u"] = np.clip(u_tilde, 0.0, 1.0)
    return result


def compute_gbid(
    u: FloatArray,
    W: FloatArray,
    v: Sequence[FloatArray],
    dim_groups: Sequence[IntArray],
    q_B: float = 2.0,
    q_W: float = 2.0,
) -> dict[str, Any]:
    """LLH-GBID 主计算 + 均衡分解（第十五章 §6—§9）。

    参数：
        u: 截断后效用矩阵，形状 (n_models, n_metrics)。
        W: 维度权重，形状 (n_dims,)，默认全等权。
        v: 维度内指标条件权重列表，第 d 项形状 (|J_d|,)。
        dim_groups: 维度指标索引列表，第 d 项为 int 数组指向 u 的列。
        q_B: 维度间短板敏感性参数（≥1），主分析取 2。
        q_W: 维度内短板敏感性参数（≥1），主分析取 2。

    返回：
        {
            'gbid':        np.ndarray (n_models,) —— GBID 值 ∈ [0,1]
            'gbids':       np.ndarray (n_models,) —— 0–100 综合指数
            'u_bar':       np.ndarray (n_models,) —— 总体平均效用 ū
            'U_d':         np.ndarray (n_models, n_dims) —— 各模型各维度效用
            'C_B':         np.ndarray (n_models,) —— 维度间不均衡量
            'C_W':         np.ndarray (n_models,) —— 维度内不均衡量
            'decomp':      dict —— 分解项（仅 q_B=q_W=2 时精确）
            'D_d':         np.ndarray (n_models, n_dims) —— 维度内缺口距离
            'd_j':         np.ndarray (n_models, n_metrics) —— 指标缺口
            'params':      dict —— 本次计算参数
        }

    异常：
        ValueError: 若 q_B < 1 或 q_W < 1。
        ValueError: 若权重形状与 dim_groups 不一致。
    """
    if q_B < 1.0 or q_W < 1.0:
        raise ValueError(f"q_B={q_B}, q_W={q_W} 均须 ≥ 1")
    u_arr = np.asarray(u, dtype=float)
    if u_arr.ndim == 1:
        u_arr = u_arr.reshape(1, -1)
    n_models, n_metrics = u_arr.shape
    n_dims = len(dim_groups)
    W_arr = np.asarray(W, dtype=float)
    if W_arr.shape != (n_dims,):
        raise ValueError(f"W 形状 {W_arr.shape} 与 dim_groups 长度 {n_dims} 不一致")
    if not np.allclose(W_arr.sum(), 1.0):
        raise ValueError(f"W 之和必须为 1，当前 {W_arr.sum()}")
    if len(v) != n_dims:
        raise ValueError(f"v 长度 {len(v)} 与 dim_groups 长度 {n_dims} 不一致")
    for d in range(n_dims):
        if v[d].shape != (len(dim_groups[d]),):
            raise ValueError(
                f"v[{d}] 形状 {v[d].shape} 与 dim_groups[{d}] 长度 {len(dim_groups[d])} 不一致"
            )
        if not np.allclose(v[d].sum(), 1.0):
            raise ValueError(f"v[{d}] 之和必须为 1，当前 {v[d].sum()}")

    # 指标缺口 d_j = 1 - u_j
    d_j = 1.0 - u_arr  # (n_models, n_metrics)

    # 维度效用 U_d 与维度内缺口距离 D_d
    U_d = np.zeros((n_models, n_dims))
    D_d = np.zeros((n_models, n_dims))
    for d_idx in range(n_dims):
        cols = dim_groups[d_idx]
        v_d = np.asarray(v[d_idx], dtype=float)
        U_d[:, d_idx] = u_arr[:, cols] @ v_d  # (n_models,)
        # 维度内 L_{q_W} 缺口距离
        d_d = d_j[:, cols]  # (n_models, |J_d|)
        if abs(q_W - 1.0) < _GBID_EPS:
            D_d[:, d_idx] = d_d @ v_d
        elif np.isposinf(q_W):
            D_d[:, d_idx] = np.max(d_d, axis=1)
        else:
            D_d[:, d_idx] = (v_d * (d_d ** q_W)).sum(axis=1) ** (1.0 / q_W)

    # 总体平均效用
    u_bar = (U_d * W_arr).sum(axis=1)  # (n_models,)

    # 维度间 L_{q_B} 聚合 → GBID
    if abs(q_B - 1.0) < _GBID_EPS:
        gbid = D_d @ W_arr
    elif np.isposinf(q_B):
        gbid = np.max(D_d, axis=1)
    else:
        gbid = (W_arr * (D_d ** q_B)).sum(axis=1) ** (1.0 / q_B)

    # 均衡分解（q_B = q_W = 2 时精确，否则为近似分解）
    C_B = np.sqrt(np.maximum(0.0, ((U_d - u_bar[:, None]) ** 2) @ W_arr))
    C_W_sq = np.zeros(n_models)
    for d_idx in range(n_dims):
        cols = dim_groups[d_idx]
        v_d = np.asarray(v[d_idx], dtype=float)
        u_d = U_d[:, d_idx]  # (n_models,)
        sq_diffs = (u_arr[:, cols] - u_d[:, None]) ** 2
        C_W_sq += W_arr[d_idx] * (sq_diffs @ v_d)
    C_W = np.sqrt(np.maximum(0.0, C_W_sq))

    decomp: dict[str, FloatArray] = {}
    if abs(q_B - 2.0) < _GBID_EPS and abs(q_W - 2.0) < _GBID_EPS:
        decomp["avg_gap_sq"] = (1.0 - u_bar) ** 2
        decomp["C_B_sq"] = C_B ** 2
        decomp["C_W_sq"] = C_W ** 2
        decomp["gbid_sq"] = decomp["avg_gap_sq"] + decomp["C_B_sq"] + decomp["C_W_sq"]

    gbids = 100.0 * (1.0 - gbid)

    return {
        "gbid": gbid,
        "gbids": gbids,
        "u_bar": u_bar,
        "U_d": U_d,
        "C_B": C_B,
        "C_W": C_W,
        "decomp": decomp,
        "D_d": D_d,
        "d_j": d_j,
        "params": {"q_B": q_B, "q_W": q_W, "n_models": n_models, "n_metrics": n_metrics, "n_dims": n_dims},
    }


def gbid_sensitivity_q(
    u: FloatArray,
    W: FloatArray,
    v: Sequence[FloatArray],
    dim_groups: Sequence[IntArray],
    q_grid: Sequence[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """短板敏感性分析（第十五章 §11.1）。

    参数：
        u, W, v, dim_groups: 同 compute_gbid。
        q_grid: (q_B, q_W) 配置列表，默认 {(2,2),(3,2),(2,3),(3,3)}。

    返回：
        {
            'results':        list[dict] —— 每个 q 配置的 compute_gbid 返回
            'rankings':       np.ndarray (n_models, n_configs) —— 排名矩阵
            'mean_rank':      np.ndarray (n_models,) —— 平均名次
            'first_place_pct': np.ndarray (n_models,) —— 第一名频率(%)
            'kendall_tau':    np.ndarray (n_configs, n_configs) —— Kendall τ 矩阵
            'pairwise_win':   np.ndarray (n_models, n_models) —— 成对胜率
        }
    """
    if q_grid is None:
        q_grid = _GBID_Q_GRID_DEFAULT
    q_list = list(q_grid)
    n_configs = len(q_list)
    u_arr = np.asarray(u, dtype=float)
    if u_arr.ndim == 1:
        u_arr = u_arr.reshape(1, -1)
    n_models = u_arr.shape[0]

    results: list[dict[str, Any]] = []
    rankings = np.zeros((n_models, n_configs), dtype=int)
    for i, (q_B, q_W) in enumerate(q_list):
        r = compute_gbid(u_arr, W, v, dim_groups, q_B=q_B, q_W=q_W)
        results.append(r)
        # 排名：GBID 越小越好
        rankings[:, i] = np.argsort(np.argsort(r["gbid"])) + 1

    mean_rank = rankings.mean(axis=1)
    first_place_pct = (rankings == 1).mean(axis=1) * 100.0

    # Kendall τ 矩阵（排名间一致性）
    from scipy import stats as sp_stats
    kendall_tau = np.eye(n_configs)
    for i in range(n_configs):
        for j in range(i + 1, n_configs):
            tau, _ = sp_stats.kendalltau(rankings[:, i], rankings[:, j])
            kendall_tau[i, j] = kendall_tau[j, i] = tau

    # 成对胜率（基于主分析 q=(2,2)排名）
    pairwise_win = np.zeros((n_models, n_models))
    for i in range(n_configs):
        for a in range(n_models):
            for b in range(n_models):
                if rankings[a, i] < rankings[b, i]:
                    pairwise_win[a, b] += 1.0
    pairwise_win /= n_configs

    return {
        "results": results,
        "rankings": rankings,
        "mean_rank": mean_rank,
        "first_place_pct": first_place_pct,
        "kendall_tau": kendall_tau,
        "pairwise_win": pairwise_win,
        "q_grid": q_list,
    }


def gbid_sensitivity_weights(
    u: FloatArray,
    W: FloatArray,
    v: Sequence[FloatArray],
    dim_groups: Sequence[IntArray],
    *,
    q_B: float = 2.0,
    q_W: float = 2.0,
    n_samples: int = 1000,
    kappa_B: float = 10.0,
    kappa_W: float = 10.0,
    seed: int = 42,
) -> dict[str, Any]:
    """权重敏感性分析——两级 Dirichlet 随机权重（第十五章 §11.2）。

    参数：
        u, W, v, dim_groups, q_B, q_W: 同 compute_gbid。W/v 为主分析权重，作为 Dirichlet 中心。
        n_samples: 随机权重组数（推荐 ≥1000）。
        kappa_B: 维度间 Dirichlet 集中参数（越大越接近主分析权重）。
        kappa_W: 维度内 Dirichlet 集中参数。
        seed: 随机种子。

    返回：
        {
            'rankings':       np.ndarray (n_models, n_samples) —— 排名矩阵
            'mean_rank':      np.ndarray (n_models,)
            'first_place_pct': np.ndarray (n_models,)
            'rank_intervals': dict —— 各模型排名区间 {model_idx: (lo, hi)}
            'kendall_vs_main': np.ndarray (n_samples,) —— 每次与主排序的 Kendall τ
            'mean_kendall':    float —— 平均 Kendall τ
        }
    """
    u_arr = np.asarray(u, dtype=float)
    if u_arr.ndim == 1:
        u_arr = u_arr.reshape(1, -1)
    n_models = u_arr.shape[0]
    n_dims = len(dim_groups)
    W_main = np.asarray(W, dtype=float)
    rng = np.random.default_rng(seed)

    # 主分析排名（用于 Kendall τ 比较）
    main = compute_gbid(u_arr, W_main, v, dim_groups, q_B=q_B, q_W=q_W)
    main_rank = np.argsort(np.argsort(main["gbid"])) + 1

    rankings = np.zeros((n_models, n_samples), dtype=int)
    kendall_vs_main = np.zeros(n_samples)
    from scipy import stats as sp_stats

    for s in range(n_samples):
        # 两级 Dirichlet 扰动
        W_s = rng.dirichlet(kappa_B * W_main)
        v_s: list[FloatArray] = []
        for d in range(n_dims):
            v_d = np.asarray(v[d], dtype=float)
            v_s.append(rng.dirichlet(kappa_W * v_d))
        r_s = compute_gbid(u_arr, W_s, v_s, dim_groups, q_B=q_B, q_W=q_W)
        rankings[:, s] = np.argsort(np.argsort(r_s["gbid"])) + 1
        tau, _ = sp_stats.kendalltau(main_rank, rankings[:, s])
        kendall_vs_main[s] = tau

    mean_rank = rankings.mean(axis=1)
    first_place_pct = (rankings == 1).mean(axis=1) * 100.0
    rank_intervals = {}
    for m in range(n_models):
        lo = int(np.percentile(rankings[m], 2.5))
        hi = int(np.percentile(rankings[m], 97.5))
        rank_intervals[m] = (lo, hi)

    return {
        "rankings": rankings,
        "mean_rank": mean_rank,
        "first_place_pct": first_place_pct,
        "rank_intervals": rank_intervals,
        "kendall_vs_main": kendall_vs_main,
        "mean_kendall": float(kendall_vs_main.mean()),
        "params": {"n_samples": n_samples, "kappa_B": kappa_B, "kappa_W": kappa_W, "seed": seed},
    }


def gbid_bootstrap(
    u_all: FloatArray,
    W: FloatArray,
    v: Sequence[FloatArray],
    dim_groups: Sequence[IntArray],
    *,
    q_B: float = 2.0,
    q_W: float = 2.0,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """GBID/GBIDS 重采样统计推断（第十五章 §11.5）。

    参数：
        u_all: 效用矩阵，形状 (n_models, n_metrics, n_repeats)。
        W, v, dim_groups, q_B, q_W: 同 compute_gbid。
        n_bootstrap: bootstrap 重复次数（≥1000）。
        ci_level: 置信水平（默认 0.95）。
        seed: 随机种子。

    返回：
        {
            'gbid_mean':   np.ndarray (n_models,) —— 平均 GBID
            'gbid_sd':     np.ndarray (n_models,) —— GBID 标准差
            'gbids_mean':  np.ndarray (n_models,) —— 平均 GBIDS
            'gbids_sd':    np.ndarray (n_models,) —— GBIDS 标准差
            'gbid_ci_lo':  np.ndarray (n_models,) —— GBID percentile bootstrap CI 下界
            'gbid_ci_hi':  np.ndarray (n_models,) —— GBID percentile bootstrap CI 上界
            'gbids_ci_lo': np.ndarray (n_models,) —— GBIDS CI 下界
            'gbids_ci_hi': np.ndarray (n_models,) —— GBIDS CI 上界
            'pairwise_diff':  dict —— 配对差异统计
            'first_place_pct': np.ndarray (n_models,) —— 第一名频率(%)
            'mean_rank':      np.ndarray (n_models,) —— 平均名次
        }
    """
    u_arr = np.asarray(u_all, dtype=float)
    if u_arr.ndim == 2:
        u_arr = u_arr[:, :, np.newaxis]
    n_models, _n_metrics, n_repeats = u_arr.shape
    if n_repeats < 2:
        raise ValueError(f"至少需要 2 次重复实验，当前 n_repeats={n_repeats}")
    rng = np.random.default_rng(seed)

    # 各次重复的 GBID
    gbid_by_rep = np.zeros((n_models, n_repeats))
    gbids_by_rep = np.zeros((n_models, n_repeats))
    for r in range(n_repeats):
        res = compute_gbid(u_arr[:, :, r], W, v, dim_groups, q_B=q_B, q_W=q_W)
        gbid_by_rep[:, r] = res["gbid"]
        gbids_by_rep[:, r] = res["gbids"]

    gbid_mean = gbid_by_rep.mean(axis=1)
    gbid_sd = gbid_by_rep.std(ddof=1, axis=1)
    gbids_mean = gbids_by_rep.mean(axis=1)
    gbids_sd = gbids_by_rep.std(ddof=1, axis=1)

    # percentile bootstrap CI（§15.6 明确要求 percentile，非 BCa——不做偏差/加速校正）
    alpha = (1.0 - ci_level) / 2.0
    gbid_ci_lo = np.zeros(n_models)
    gbid_ci_hi = np.zeros(n_models)
    gbids_ci_lo = np.zeros(n_models)
    gbids_ci_hi = np.zeros(n_models)

    for m in range(n_models):
        boot_means = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_repeats, size=n_repeats)
            boot_means[b] = gbid_by_rep[m, idx].mean()
        gbid_ci_lo[m] = float(np.percentile(boot_means, alpha * 100))
        gbid_ci_hi[m] = float(np.percentile(boot_means, (1.0 - alpha) * 100))
        # GBIDS CI 由 GBID CI 线性变换
        gbids_ci_lo[m] = 100.0 * (1.0 - gbid_ci_hi[m])
        gbids_ci_hi[m] = 100.0 * (1.0 - gbid_ci_lo[m])

    # 配对差异统计
    pairwise_diff: dict[str, Any] = {}
    if n_models >= 2:
        diff_matrix = np.zeros((n_models, n_models, n_repeats))
        for a in range(n_models):
            for b in range(n_models):
                diff_matrix[a, b, :] = gbid_by_rep[a, :] - gbid_by_rep[b, :]
        pairwise_diff["diff_mean"] = diff_matrix.mean(axis=2)
        pairwise_diff["diff_sd"] = diff_matrix.std(ddof=1, axis=2)
        pairwise_diff["pct_a_better"] = (diff_matrix < 0).mean(axis=2) * 100.0

    # 排名统计
    rankings = np.zeros((n_models, n_repeats), dtype=int)
    for r in range(n_repeats):
        rankings[:, r] = np.argsort(np.argsort(gbid_by_rep[:, r])) + 1
    first_place_pct = (rankings == 1).mean(axis=1) * 100.0
    mean_rank = rankings.mean(axis=1)

    return {
        "gbid_mean": gbid_mean,
        "gbid_sd": gbid_sd,
        "gbids_mean": gbids_mean,
        "gbids_sd": gbids_sd,
        "gbid_ci_lo": gbid_ci_lo,
        "gbid_ci_hi": gbid_ci_hi,
        "gbids_ci_lo": gbids_ci_lo,
        "gbids_ci_hi": gbids_ci_hi,
        "pairwise_diff": pairwise_diff,
        "first_place_pct": first_place_pct,
        "mean_rank": mean_rank,
        "params": {
            "n_repeats": n_repeats,
            "n_bootstrap": n_bootstrap,
            "ci_level": ci_level,
            "seed": seed,
        },
    }


# ─── LLH-GBID 论文表述辅助 ────────────────────────────────────────────────────
def gbid_paper_paragraph(
    model_name: str,
    gbid_result: dict[str, Any],
    sensitivity_q: dict[str, Any] | None = None,
    sensitivity_w: dict[str, Any] | None = None,
    *,
    model_idx: int = 0,
    metric_names: Sequence[str] | None = None,
    top_metrics: Sequence[str] | None = None,
) -> str:
    """生成 LLH-GBID 论文中文表述段落（第十五章 §15.9）。

    参数：
        model_name: 模型名称（如"所提出模型"）。
        gbid_result: compute_gbid 返回结果。
        sensitivity_q: gbid_sensitivity_q 返回结果（可选，提供则追加稳健性句式）。
        sensitivity_w: gbid_sensitivity_weights 返回结果（可选）。
        model_idx: 目标模型在结果数组中的索引（默认 0 即第一个模型）。
        metric_names: 所有指标名列表，用于列举原始指标表现。
        top_metrics: 该模型表现较好的指标名子集，用于"在…等指标上表现出较强的预测能力"。

    返回：
        中文段落字符串，可直接用于论文 Results/Discussion。
    """
    u_bar = float(gbid_result["u_bar"][model_idx])
    gbids = float(gbid_result["gbids"][model_idx])
    C_B = float(gbid_result["C_B"][model_idx])
    C_W = float(gbid_result["C_W"][model_idx])

    lines: list[str] = []

    # 第一段：原始指标 + 综合排序
    metric_str = "、".join(top_metrics) if top_metrics else "多个"
    lines.append(
        f"{model_name}在{metric_str}等指标上表现出较强的预测能力。"
        f"在统一指标效用、维度权重及参照标准后，{model_name}获得最高的"
        f"GBIDS（{gbids:.1f}，ū = {u_bar:.3f}），"
        f"表明其在当前评价体系下具有最优的综合性能。"
    )

    # 第二段：均衡分解
    lines.append(
        f"从均衡分解来看，{model_name}的维度间不均衡量C_B = {C_B:.3f}、"
        f"维度内不均衡量C_W = {C_W:.3f}。"
        + (
            "其C_B < C_W，说明模型主要短板来源于维度内部指标不均衡；"
            if C_B < C_W
            else "其C_W < C_B，说明模型主要短板来源于维度间差异；"
            if C_W < C_B
            else "维度间与维度内不均衡程度相近，"
        )
        + "该分解有助于定位综合性能的具体改进方向。"
    )

    # 第三段：稳健性（若有）
    robust_parts: list[str] = []
    if sensitivity_q is not None:
        fp = float(sensitivity_q["first_place_pct"][model_idx])
        mr = float(sensitivity_q["mean_rank"][model_idx])
        robust_parts.append(
            f"不同短板惩罚参数(q_B, q_W)设置下仍保持较高排名"
            f"（第一名频率 {fp:.0f}%，平均名次 {mr:.1f}）"
        )
    if sensitivity_w is not None:
        fp_w = float(sensitivity_w["first_place_pct"][model_idx])
        mk = float(sensitivity_w["mean_kendall"])
        robust_parts.append(
            f"权重扰动下与主排序Kendall τ = {mk:.3f}"
            f"（第一名频率 {fp_w:.0f}%）"
        )
    if robust_parts:
        lines.append(
            "在" + "；".join(robust_parts) + "，"
            "说明综合评价结果具有较好的稳健性。"
        )

    return "\n\n".join(lines)


# ─── 基线预算（第七章 Item 15）────────────────────────────────────────────────
_BASELINE_BUDGET_FIELDS = [
    "hp_search_space_size",
    "n_trials",
    "seed_set",
    "max_epochs",
    "wall_clock_sec",
    "gpu_hours",
    "peak_gpu_mem_gb",
]


def log_baseline_budget(logger: Logger, budget: dict[str, Any]) -> None:
    """落盘基线算力预算到日志。budget 必须包含 _BASELINE_BUDGET_FIELDS 全部字段。"""
    missing = [k for k in _BASELINE_BUDGET_FIELDS if k not in budget]
    if missing:
        raise ValueError(f"baseline 预算字段缺失: {missing}")
    sep = "-" * 64
    logger.log(sep)
    logger.log("Baseline 算力预算（第七章 Item 15）")
    logger.log(sep)
    for k in _BASELINE_BUDGET_FIELDS:
        logger.log(f"  {k}: {budget[k]}")
    logger.log(sep)


# ─── 图表显著性标注（第八章 Item 2）───────────────────────────────────────────
def _p_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def annotate_significance(
    ax: Axes,
    pairs: list[tuple[int, int]],
    p_values: list[float],
    y_positions: list[float] | None = None,
    bracket_height_ratio: float = 0.02,
    line_kwargs: dict[str, Any] | None = None,
    text_kwargs: dict[str, Any] | None = None,
) -> None:
    """在指定 axes 上画显著性括号。

    pairs: [(0, 1), (1, 2), ...] x 轴位置对（按 tick 索引）
    p_values: 对应 p 值（建议传 `p_corrected`）
    y_positions: 每对括号的 y 坐标；缺省时根据当前 ylim 自动堆叠。
    """
    if len(pairs) != len(p_values):
        raise ValueError("pairs 与 p_values 长度不一致")
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    base_y = ylim[1]
    # 显式标注为 dict[str, Any]：混合 float/str 值会被推断为 dict[str, object]，
    # 展开成 **kwargs 时无法匹配 Axes.plot / Axes.text 的具名参数签名。
    lk: dict[str, Any] = {"linewidth": 1.2, "color": "black"}
    if line_kwargs:
        lk.update(line_kwargs)
    tk: dict[str, Any] = {"ha": "center", "va": "bottom", "fontsize": 10}
    if text_kwargs:
        tk.update(text_kwargs)
    for i, ((x1, x2), p) in enumerate(zip(pairs, p_values, strict=False)):
        y = (
            y_positions[i]
            if y_positions is not None
            else base_y + y_range * bracket_height_ratio * (i + 1) * 2
        )
        bracket_h = y_range * bracket_height_ratio
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + bracket_h, y + bracket_h, y],
            **lk,
        )
        ax.text((x1 + x2) / 2, y + bracket_h * 1.1, _p_to_stars(p), **tk)
    if y_positions is None:
        ax.set_ylim(
            ylim[0],
            base_y + y_range * bracket_height_ratio * (len(pairs) + 1) * 2.5,
        )


# ─── 算力跟踪（第十二章 Item 1）───────────────────────────────────────────────
@contextmanager
def track_compute(
    stage_name: str,
    logger: Logger | None = None,
    *,
    wandb_run: Any | None = None,
) -> Iterator[dict[str, Any]]:
    """上下文管理器：跟踪一段计算的 wall-clock / GPU hours / 峰值显存 / 能耗 / 碳排放。

    用法：
        with track_compute("训练-baseline", logger, wandb_run=run) as info:
            train(...)
        # info: {'stage', 'wall_clock_sec', 'gpu_hours', 'peak_gpu_mem_gb',
        #        'energy_kwh', 'carbon_kg'}

    若安装了 codecarbon，将额外记录 energy_kwh / carbon_kg；否则只记录硬件指标。

    wandb_run（可选）：传入 init_wandb 返回的 run 对象时，会在退出时自动以
    `compute/<key>` 前缀 log 到 wandb；失败不抛错，仅日志告警，避免 wandb 异常
    破坏正在退出的训练流程。

    多段调用：每次 with 块结束后保留 info dict，最终把多次结果拼成 list 传给
    compute_info_to_df(...) 即可生成 'compute' sheet 的 DataFrame。
    """
    info: dict[str, Any] = {"stage": stage_name}
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tracker: Any = None
    try:
        from codecarbon import EmissionsTracker

        tracker = EmissionsTracker(
            project_name=stage_name,
            log_level="error",
            save_to_file=False,
        )
        tracker.start()
    except ImportError:
        tracker = None
    try:
        yield info
    finally:
        info["wall_clock_sec"] = time.time() - t0
        info["gpu_hours"] = (
            info["wall_clock_sec"] / 3600.0 if torch.cuda.is_available() else 0.0
        )
        if torch.cuda.is_available():
            info["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        else:
            info["peak_gpu_mem_gb"] = 0.0
        if tracker is not None:
            try:
                emissions = tracker.stop()
                info["carbon_kg"] = float(emissions) if emissions is not None else 0.0
                final_data = getattr(tracker, "final_emissions_data", None)
                if final_data is not None:
                    info["energy_kwh"] = float(
                        getattr(final_data, "energy_consumed", 0.0)
                    )
            except (RuntimeError, AttributeError):
                pass
        if logger is not None:
            logger.log(f"[track_compute] {stage_name}: {info}")
        if wandb_run is not None:
            wandb_metrics = {
                f"compute/{k}": v for k, v in info.items() if k != "stage"
            }
            wandb_metrics["compute/stage"] = stage_name
            try:
                wandb_run.log(wandb_metrics)
            except Exception as exc:  # wandb 异常隔离，不破坏退出流程
                if logger is not None:
                    logger.log(f"[track_compute] wandb log 失败，已忽略: {exc}")


def compute_info_to_df(
    info: dict[str, Any] | Sequence[dict[str, Any]],
) -> pd.DataFrame:
    """把 track_compute 返回的 info dict（或多段 dict 列表）转为单行 / 多行 DataFrame。

    用于 write_script_workbook(__file__, {..., 'compute': compute_info_to_df(...)})——
    第十二章 Item 1 强制要求算力字段以 'compute' sheet 名汇入本脚本主 xlsx。
    """
    if isinstance(info, dict):
        return pd.DataFrame([info])
    return pd.DataFrame(list(info))


# ─── wandb 入口（第五章 Item 6 强制）──────────────────────────────────────────
def init_wandb(
    script_path: str,
    seed: int,
    *,
    project: str,
    stage: str,
    device: str,
    config: dict[str, Any] | None = None,
    timestamp: str | None = None,
    mode: Literal["online", "offline", "disabled", "shared"] = "online",
    **wandb_init_kwargs: Any,
) -> Any:
    """正式脚本唯一允许的 wandb.init 入口（项目约定第五章 Item 6 强约束）。

    强制按 `<脚本文件名>_<YY-MM-DD_HHMMSS>_seed<X>` 构造 run name，与 05logs/ 日志
    文件名前缀及 07checkpoints/<run_name>/ 目录完全对齐，确保 run ↔ 日志 ↔
    checkpoint ↔ 输出图表四方互查。

    项目名按 `<project>-<stage>` 拼接（如 `myproj-baselines` / `myproj-ablation`），
    禁止调用方直接调用 wandb.init 自拼 run name 或 project。

    强制注入 tags：`['device:<device>', 'stage:<stage>']`——用于事后筛 MPS / CUDA
    及 pilot / formal 分组（与第五章 Item 5 MPS 豁免子条款联动）。调用方传入的
    tags 会与本函数注入的 tags 拼接，不会被覆盖。

    参数：
        script_path: 通常传 __file__。
        seed: 当前随机种子。
        project: 项目代号（与 01IDEA_REPORT.md 中定义一致，第十三章 Item 2）。
        stage: 阶段标签（baselines / ablation / sensitivity / external / rebuttal 等）。
        device: 实际设备字符串（'cuda' / 'mps' / 'cpu'，通常来自 get_device 返回），
            用于注入 wandb tag 与事后筛选。
        config: 传给 wandb 的 hyperparameter dict（推荐与 log_experiment_header 的
            header_dict 保持一致或合并）。
        timestamp: 缺省取当前时间，与 05logs 日志保持一致时显式传入同一字符串。
        mode: "online" / "offline" / "disabled"，无外网时用 "offline"。

    返回 wandb.Run 对象。
    """
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("init_wandb 需要 `pip install wandb`") from exc
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%y-%m-%d_%H%M%S")
    file_stem = os.path.splitext(os.path.basename(script_path))[0]
    if not file_stem:
        raise ValueError(f"无法从 script_path 派生 file_stem: {script_path!r}")
    run_name = f"{sanitize_filename(file_stem)}_{timestamp}_seed{seed}"
    project_full = f"{project}-{stage}"
    forced_tags = [f"device:{device}", f"stage:{stage}"]
    user_tags = wandb_init_kwargs.pop("tags", None) or []
    if not isinstance(user_tags, list):
        user_tags = list(user_tags)
    tags = forced_tags + user_tags
    return wandb.init(
        project=project_full,
        name=run_name,
        config=config or {},
        mode=mode,
        tags=tags,
        **wandb_init_kwargs,
    )


# ─── 环境捕获与依赖锁定（第五章 Item 5）───────────────────────────────────────
_REQUIREMENTS_HEADER = (
    "# requirements.txt — 自动锁定（由 export_utils.freeze_requirements 在每次实验启动时刷新）\n"
    "# 详见项目约定第五章 Item 5\n"
    "# 切勿手动编辑——本文件由 uv pip freeze / pip freeze 覆写\n"
    "\n"
)


def freeze_requirements(path: str | None = None) -> dict[str, Any]:
    """自动锁定当前 venv 依赖到 requirements.txt（第五章 Item 5）。

    优先 `uv pip freeze`，回退 `<sys.executable> -m pip freeze`；两者都失败则不写文件，
    在返回字典的 'error' 字段说明原因。每次写入都会保留一个 4 行注释头部，
    package 行由 freeze 输出原样追加。

    返回字段：
        tool          : 'uv' / 'pip' / None
        path          : 锁定文件路径
        sha256_before : 写入前 SHA-256（若文件不存在则为 None）
        sha256_after  : 写入后 SHA-256
        changed       : 内容是否发生变化
        error         : 失败时的原因字符串
    """
    if path is None:
        path = os.path.join(BASE_DIR, "requirements.txt")
    sha_before = compute_sha256(path) if os.path.exists(path) else None
    result: dict[str, Any] = {
        "tool": None,
        "path": path,
        "sha256_before": sha_before,
        "sha256_after": None,
        "changed": False,
        "error": None,
    }
    output: str | None = None
    for cmd, tool_name in (
        (["uv", "pip", "freeze"], "uv"),
        ([sys.executable, "-m", "pip", "freeze"], "pip"),
    ):
        try:
            output = subprocess.check_output(
                cmd, stderr=subprocess.DEVNULL, text=True
            )
            result["tool"] = tool_name
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    if output is None:
        result["error"] = "uv / pip 均不可用，requirements.txt 未更新"
        return result
    final = _REQUIREMENTS_HEADER + output
    with open(path, "w", encoding="utf-8") as f:
        f.write(final)
    sha_after = compute_sha256(path)
    result["sha256_after"] = sha_after
    result["changed"] = sha_before != sha_after
    return result


def capture_env_info(auto_freeze: bool = True) -> dict[str, Any]:
    """自动捕获运行环境信息（第五章 Item 5）：Python / PyTorch / CUDA / GPU /
    依赖锁定 SHA-256 / git commit hash 等。

    auto_freeze=True（默认）时会调用 freeze_requirements() 自动刷新 requirements.txt，
    确保 SHA-256 反映当前 venv 真实快照。需禁用时（如单测）显式传 auto_freeze=False。
    """
    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": torch.backends.mps.is_available(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    if auto_freeze:
        freeze_result = freeze_requirements()
        info["requirements_freeze"] = freeze_result
        if freeze_result["sha256_after"] is not None:
            info["requirements_sha256"] = freeze_result["sha256_after"]
    else:
        req_path = os.path.join(BASE_DIR, "requirements.txt")
        if os.path.exists(req_path):
            info["requirements_sha256"] = compute_sha256(req_path)
    env_path = os.path.join(BASE_DIR, "environment.yml")
    if os.path.exists(env_path):
        info["environment_yml_sha256"] = compute_sha256(env_path)
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=BASE_DIR,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        info["git_commit"] = commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["git_commit"] = "not_a_git_repo_or_git_not_installed"
    return info


def log_experiment_header(logger: Logger, header: dict[str, Any]) -> None:
    """按第五章规范打印实验头部字段（强制内置 capture_env_info）。

    本函数内部会自动调用 capture_env_info() 并合并到 header 前部，确保每次正式实验
    都会刷新 requirements.txt、记录 git commit、Python / PyTorch / CUDA / GPU 信息。
    禁止通过任何参数绕过——若必须跳过（仅 pilot 探索或单测），请直接调用更底层的
    capture_env_info(auto_freeze=False) 自行组织 header，并在代码 review 阶段说明理由。

    用户传入的 header 字段优先级高于自动捕获字段（同名 key 会覆盖自动值）。

    header 推荐键（缺失项自动跳过）：
        script / dataset / source_domain / target_domain / split_strategy / model /
        hyperparams / random_seeds / transfer_type / target_label_ratio /
        early_stop_epoch / best_val_metric / final_test_metric / output_files /
        device / extra。
    """
    header = {**capture_env_info(), **header}
    sep = "=" * 64
    sub = "-" * 64
    logger.log(sep)
    logger.log("实验配置 / Experiment Configuration")
    logger.log(sub)
    for k, v in header.items():
        if v is None or v == "":
            continue
        if isinstance(v, dict):
            logger.log(f"{k}:")
            for kk, vv in v.items():
                logger.log(f"    {kk}: {vv}")
        elif isinstance(v, (list, tuple)):
            logger.log(f"{k}: {list(v)}")
        else:
            logger.log(f"{k}: {v}")
    logger.log(sep)
