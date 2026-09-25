"""
02_data_processing.py: 多年份光谱数据处理与异常检测

运行方式:
    cd <repository root>/02code
    python 02_data_processing.py --device auto --year all

输出文件：
    01data/02_data_processing.xlsx       — 统计汇总表（年份/产地/波段/糖度统计）
    01data/02_data_2018.csv             — 2018年完整处理数据（样本×产地×光谱×SSC）
    01data/02_data_2025.csv             — 2025年完整处理数据
    04logs/02_data_processing.log       — 处理日志
"""

import os
import sys
import importlib.util
import argparse
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
import pywt
import warnings

warnings.filterwarnings('ignore')

# ─── 加载工具模块 ──────────────────────────────────────────────────────
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "export_utils", os.path.join(_CODE_DIR, "01_export_utils.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

setup_chinese_font = _mod.setup_chinese_font
figure_output_path = _mod.figure_output_path
write_excel_workbook = _mod.write_excel_workbook
excel_data_path = _mod.excel_data_path
get_logger = _mod.get_logger
get_device = _mod.get_device
DATA_DIR = _mod.DATA_DIR
BASE_DIR = _mod.BASE_DIR
AVAILABLE_YEARS = _mod.AVAILABLE_YEARS

setup_chinese_font()

FILE_STEM = '02_data_processing'
logger = get_logger(FILE_STEM)
log = logger.log

# ─── 全局波段范围 (2018/2019/2025 共有) ──────────────────────────────────
COMMON_WAVELENGTH_MIN = 590.67
COMMON_WAVELENGTH_MAX = 1001.12

# 2018/2019/2025 原始数据统一从跨项目共享库 05data（正典源、只读）读取；
# 读 05data → 写 03data/processed 是 01CLAUDE.md §4.5.1 的法定预处理流向。
# （历史上 2018/2025 曾在 03data 存原始副本，已核 SHA-256 与 05data 字节一致后统一改读 05data 并删副本。）
_MULTIYEAR_RAW = os.path.normpath(os.path.join(
    BASE_DIR, '..', '..', '05data', '001_apple_hyperspectral_multiyear'))
RAW_FILE_2019 = os.path.join(_MULTIYEAR_RAW, '1新疆-2山东-3陕西苹果光谱数据(含糖度）2019.xlsx')


def aggregate_5faces(spectra_array: np.ndarray, method: str = 'robust_mean') -> np.ndarray:
    """将每个苹果的5个测量面聚合为单一谱图"""
    n_total = spectra_array.shape[0]
    n_bands = spectra_array.shape[1]

    # 截断到最近的5的倍数
    n_total_rounded = (n_total // 5) * 5
    if n_total != n_total_rounded:
        log(f"⚠ 截断 {n_total} → {n_total_rounded} 样本")
        spectra_array = spectra_array[:n_total_rounded]
        n_total = n_total_rounded

    n_apples = n_total // 5
    aggregated_spec = np.zeros((n_apples, n_bands))

    for i in range(n_apples):
        face_indices = np.arange(i*5, (i+1)*5)
        faces = spectra_array[face_indices, :]

        if method == 'robust_mean':
            sorted_faces = np.sort(faces, axis=0)
            aggregated_spec[i, :] = np.mean(sorted_faces[1:4, :], axis=0)  # 中间3个
        elif method == 'mean':
            aggregated_spec[i, :] = np.mean(faces, axis=0)
        elif method == 'median':
            aggregated_spec[i, :] = np.median(faces, axis=0)

    log(f"✓ 多面聚合: {n_total} → {n_apples} 样本 (方法={method})")
    return aggregated_spec


def detect_outliers_mahalanobis(spectra: np.ndarray, threshold_percentile: int = 10) -> tuple:
    """用Mahalanobis距离进行异常检测"""
    scaler = StandardScaler()
    spec_scaled = scaler.fit_transform(spectra)

    center = np.mean(spec_scaled, axis=0)
    distances = np.linalg.norm(spec_scaled - center, axis=1)

    threshold = np.percentile(distances, 100 - threshold_percentile)
    is_outlier = distances > threshold

    n_outliers = np.sum(is_outlier)
    if n_outliers > 0:
        log(f"✓ 异常检测: {n_outliers}/{len(spectra)} 样本异常 (删除最离群{threshold_percentile}%)")
    else:
        log(f"✓ 异常检测: 无异常样本")

    return is_outlier, distances


def wavelet_denoise(spectra: np.ndarray, wavelet: str = 'db4', level: int = 3) -> np.ndarray:
    """小波去噪"""
    n_samples, n_bands = spectra.shape
    denoised = np.zeros_like(spectra)

    for i in range(n_samples):
        spec = spectra[i, :]
        coeffs = pywt.wavedec(spec, wavelet, level=level, mode='symmetric')

        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(n_bands))

        coeffs_thresh = [coeffs[0]]
        for j in range(1, len(coeffs)):
            coeffs_thresh.append(pywt.threshold(coeffs[j], threshold, mode='soft'))

        denoised[i, :] = pywt.waverec(coeffs_thresh, wavelet, mode='symmetric')[:n_bands]

    log(f"✓ 小波去噪: {wavelet}, level={level}")
    return denoised


def savitzky_golay_smooth(spectra: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay平滑"""
    if window_length % 2 == 0:
        window_length += 1

    n_samples, n_bands = spectra.shape
    smoothed = np.zeros_like(spectra)

    for i in range(n_samples):
        smoothed[i, :] = signal.savgol_filter(spectra[i, :], window_length, polyorder)

    log(f"✓ Savitzky-Golay平滑: window={window_length}, polyorder={polyorder}")
    return smoothed


def load_excel_data(filepath: str):
    """加载2018 Excel格式"""
    raw = pd.read_excel(filepath, header=None)

    wavelengths_raw = raw.iloc[0, :].values
    col_headers = raw.iloc[1, :].values
    data = raw.iloc[2:, :].copy().reset_index(drop=True)

    y_col_idx = int(np.where(col_headers == 'Y')[0][0])
    z_col_idx = int(np.where(col_headers == 'Z')[0][0])

    spec_cols = [i for i in range(len(col_headers)) if i not in [y_col_idx, z_col_idx]]
    wavelengths = wavelengths_raw[spec_cols]

    # 只取共同波段
    valid_idx = (wavelengths >= COMMON_WAVELENGTH_MIN) & (wavelengths <= COMMON_WAVELENGTH_MAX)
    wavelengths = wavelengths[valid_idx]

    X = data.iloc[:, spec_cols].values.astype(float)
    X = X[:, valid_idx]
    y = data.iloc[:, y_col_idx].values.astype(float)
    z = data.iloc[:, z_col_idx].values.astype(int)

    region_map = {1: '新疆', 2: '山东', 3: '陕西'}
    regions = np.array([region_map.get(zval, '未知') for zval in z])

    # 移除NaN
    valid_rows = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid_rows]
    y = y[valid_rows]
    regions = regions[valid_rows]

    log(f"✓ 加载Excel: {X.shape[0]}样本, {X.shape[1]}波段")
    return X, y, regions, wavelengths


def load_csv_data_2025(spec_filepath: str, sugar_filepath: str):
    """加载 2025 CSV（每面一行、一张图多苹果、面交错）→ 按真实苹果编号聚合。

    ⚠ 2026-07-17 修复的严重聚合/对齐缺陷：
      2025 光谱是"一张图 3 个苹果、5 个面交错存"（行序 GS001顶/GS002顶/GS003顶/
      GS001底/…）。原实现（aggregate_5faces + np.repeat 糖度）按**连续 5 行**当一个
      苹果的 5 面、按文件顺序配糖度 —— 每个"样本"实为 3 个不同苹果的混合谱、标签又
      错位，致 2025 域内 RPD≈0.95（无信号）。现改为：
        · 光谱：按「实际苹果编号」(如 GS001) 分组、对该苹果各面取均值；
        · 糖度：每苹果 = 5 面糖度均值，按苹果编码对齐；
      修复后 2025 域内 RPD 回到 ~1.0–1.2（仍偏弱，属该仪器本身信噪比问题）。
    """
    spec_df = pd.read_csv(spec_filepath)
    if '实际苹果编号' not in spec_df.columns:
        raise ValueError('2025 光谱缺少「实际苹果编号」列，无法按苹果分组')
    apple_id = spec_df['实际苹果编号'].astype(str).str.extract(r'^([A-Za-z]+\d+)')[0]

    # 波长列 → 只取共同波段
    wl_cols, wavelengths = [], []
    for col in spec_df.columns:
        try:
            wl = float(str(col).replace('nm', '').strip())
        except ValueError:
            continue
        if COMMON_WAVELENGTH_MIN <= wl <= COMMON_WAVELENGTH_MAX:
            wl_cols.append(col)
            wavelengths.append(wl)
    wavelengths = np.array(wavelengths)

    # 按苹果编号聚合各面（缺面的苹果按其实有面数取均值）
    grouped = spec_df.assign(_apple=apple_id).groupby('_apple', sort=False)[wl_cols].mean()
    apple_ids = grouped.index.to_numpy()
    X_all = grouped.to_numpy(dtype=float)
    log(f"  2025 按苹果聚合: {spec_df.shape[0]} 面 → {len(apple_ids)} 苹果, {len(wavelengths)} 波段")

    # 糖度：每苹果 5 面均值，按苹果编码对齐（糖度文件多为 gbk）
    sugar_df = None
    for enc in ('gbk', 'gb18030', 'utf-8-sig', 'utf-8'):
        try:
            sugar_df = pd.read_csv(sugar_filepath, encoding=enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if sugar_df is None:
        raise ValueError(f'无法解码糖度文件: {sugar_filepath}')
    face_cols = [c for c in ['顶部', '底部', '侧面1', '侧面2', '侧面3'] if c in sugar_df.columns]
    ssc_map = (sugar_df.assign(_apple=sugar_df['苹果编码'].astype(str).str.strip(),
                               _ssc=sugar_df[face_cols].mean(axis=1))
               .set_index('_apple')['_ssc'])
    y_all = ssc_map.reindex(apple_ids).to_numpy(dtype=float)

    region_map = {'XJ': '新疆', 'SD': '山东', 'GS': '甘肃'}
    regions = np.array([region_map.get(str(a)[:2], '未知') for a in apple_ids])

    # 移除未匹配糖度 / 含 NaN 的苹果
    valid_rows = ~(np.isnan(X_all).any(axis=1) | np.isnan(y_all))
    X_all, y_all, regions = X_all[valid_rows], y_all[valid_rows], regions[valid_rows]

    log(f"✓ 加载CSV(2025,按苹果编号正确聚合+对齐): {X_all.shape[0]}苹果, {X_all.shape[1]}波段")
    return X_all, y_all, regions, wavelengths


def process_all_years(years_list=None, outlier_pct=10, season_scaler=True):
    """完整处理流程。outlier_pct: Mahalanobis 离群剔除百分位(默认10,C6敏感性分析可传0/5/15)。
    season_scaler: 是否在每个年份内按全部产地做逐波段标准化(默认是)。设 False 时输出停在
    SG 平滑之后,标准化只剩 50/61 号在源域上拟合的那一步,供年份级标准化的敏感性分析(106 号)。"""
    if years_list is None:
        years_list = list(AVAILABLE_YEARS)
    processed_data = {}
    fnames = {
        2018: '01新疆-2山东-3陕西苹果光谱数据(含糖度）2018.xlsx',
    }

    for year in years_list:
        log(f"\n【处理{year}年数据】")

        try:
            if year == 2018:
                # 2018 从 05data 正典源读取（与 2019/2025 一致，读 05data → 写 03data/processed）
                filepath = os.path.join(_MULTIYEAR_RAW, fnames[year])
                if not os.path.exists(filepath):
                    log(f"⚠ 文件不存在: {filepath}")
                    continue
                X, y, regions, wavelengths = load_excel_data(filepath)
                log(f"原始: {X.shape[0]}样本, {X.shape[1]}波段")
                # 2018 已是单样本，不需多面聚合

            elif year == 2019:
                # 2019 与 2018 同格式（每行一苹果、Y=糖度/Z=产地），从 05data 正典源读取
                if not os.path.exists(RAW_FILE_2019):
                    log(f"⚠ 2019 文件不存在: {RAW_FILE_2019}")
                    continue
                X, y, regions, wavelengths = load_excel_data(RAW_FILE_2019)
                log(f"原始: {X.shape[0]}样本, {X.shape[1]}波段")
                # 2019 已是单样本，不需多面聚合

            elif year == 2025:
                # 2025 从 05data 正典源读取（读 05data → 写 03data/processed）
                spec_path = os.path.join(_MULTIYEAR_RAW, '新疆-山东-甘肃苹果光谱数据2025.csv')
                sugar_path = os.path.join(_MULTIYEAR_RAW, '新疆-山东-甘肃苹果糖度数据2025.csv')
                if not os.path.exists(spec_path) or not os.path.exists(sugar_path):
                    log(f"⚠ 2025数据文件不完整")
                    continue
                # load_csv_data_2025 已按真实苹果编号聚合+对齐（不再调用 aggregate_5faces）
                X, y, regions, wavelengths = load_csv_data_2025(spec_path, sugar_path)
                log(f"原始: {X.shape[0]}苹果, {X.shape[1]}波段")
            else:
                continue

            # 异常检测
            is_outlier, _ = detect_outliers_mahalanobis(X, threshold_percentile=outlier_pct)
            X = X[~is_outlier]
            y = y[~is_outlier]
            regions = regions[~is_outlier]

            # 去噪
            X = wavelet_denoise(X, wavelet='db4', level=3)
            X = savitzky_golay_smooth(X, window_length=11, polyorder=3)

            # 标准化
            if season_scaler:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            processed_data[year] = (X, y, regions, wavelengths)
            log(f"✓ 完成: {X.shape[0]}样本保留, {X.shape[1]}波段")

        except Exception as e:
            log(f"❌ 错误: {str(e)}")
            import traceback
            traceback.print_exc()

    return processed_data


def save_results(processed_data, years_list):
    """保存结果到 01data/ 目录

    输出两类文件：
    1. 统计汇总表：02_data_processing.xlsx (多个Sheet)
    2. 完整处理数据：02_data_YEAR.csv (光谱+糖度+产地)
    """
    sheets = {}

    for year in years_list:
        if year not in processed_data:
            continue

        X, y, regions, wavelengths = processed_data[year]

        # 统计汇总
        summary_df = pd.DataFrame({
            '指标': ['样本数', '产地数', '波段数', '糖度均值', '糖度Std'],
            '数值': [X.shape[0], len(np.unique(regions)), X.shape[1],
                    f"{y.mean():.3f}", f"{y.std():.3f}"]
        })
        sheets[f'{year}年'] = summary_df

        # 完整数据表：索引+光谱+糖度+产地
        data_dict = {}
        data_dict['年份'] = [year] * X.shape[0]
        data_dict['产地'] = regions
        data_dict['SSC含量(°Brix)'] = y

        # 添加光谱列 (波段作为列名)
        for i, wl in enumerate(wavelengths):
            data_dict[f'波段{i+1:03d}_{wl:.2f}nm'] = X[:, i]

        full_df = pd.DataFrame(data_dict)

        # 保存完整数据到独立CSV
        full_data_path = os.path.join(DATA_DIR, f'02_data_{year}.csv')
        full_df.to_csv(full_data_path, index=False, encoding='utf-8')
        log(f"✓ 保存完整数据: {os.path.basename(full_data_path)} ({X.shape[0]}样本 × {X.shape[1]}波段)")

    # 保存汇总统计表
    output_path = excel_data_path(FILE_STEM)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    log(f"✓ 保存统计汇总: {os.path.basename(output_path)}")


# Main
def main(args):
    log("\n" + "="*70)
    log("数据处理: 多面聚合 + 异常检测 + 小波去噪 (统一波段版)")
    log("="*70 + "\n")
    log(f"共同波段: {COMMON_WAVELENGTH_MIN:.2f} - {COMMON_WAVELENGTH_MAX:.2f} nm\n")

    years_list = list(AVAILABLE_YEARS) if args.year == 'all' else [int(args.year)]
    invalid_years = [year for year in years_list if year not in AVAILABLE_YEARS]
    if invalid_years:
        supported = ', '.join(str(year) for year in AVAILABLE_YEARS)
        raise ValueError(f'不支持的年份: {invalid_years}，当前仅支持: {supported}')

    processed_data = process_all_years(years_list=years_list, outlier_pct=args.outlier_pct,
                                       season_scaler=not args.no_season_scaler)
    save_results(processed_data, years_list)

    log("\n" + "="*70)
    log("✓ 数据处理完成!")
    log("="*70 + "\n")

    return processed_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='苹果光谱数据处理')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--year', type=str, default='all')
    parser.add_argument('--outlier_pct', type=int, default=10,
                        help='Mahalanobis 离群剔除百分位(默认10;C6敏感性:0/5/15)')
    parser.add_argument('--no_season_scaler', action='store_true',
                        help='跳过年份内逐波段标准化(年份级标准化敏感性,106 号用)')
    args = parser.parse_args()

    device = get_device(args.device)
    log(f"设备: {device}")

    processed_data = main(args)
