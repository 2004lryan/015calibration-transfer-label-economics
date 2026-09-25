#!/usr/bin/env bash
# 104_c6_physfix_rerun_local.sh — 在本机补跑离群剔除 0% 与 15% 两档（C6）下的 Phys 与 Phys+BL
#
# 为什么要补跑：C6 是补充材料 S9 与正文 §2.4、§3.5 引用的离群剔除敏感性（产物
# 04outputs/c6_p0.xlsx、c6_p15.xlsx，2026-07-23 在服务器上跑）。其中 Phys 与 Phys+BL 经过
# 13_model_physics_informed.py，训练时回归项没有逐样本配对（修正见 038bfed），这两个方法
# 在两档下都要重跑；其余五个方法（PLSR、SVR、CNN、CNN+MMD、CNN+BL）不经过该模块，C6 原结果
# 照用。C6 之后 01_export_utils 的读数入口（SSC 越界与逐位重复光谱两条剔除规则）、02 号的
# 预处理与 50 号的划分都没有改动，50 号在每个场景划分前、每个方法开跑前都重置同一粒种子，
# 所以补跑与 C6 原结果的目标域划分逐样本相同。
#
# 数据：02 号从 05data 正典原始数据按 --outlier_pct 重新剔除，写到 $C6_DATA_ROOT/p<档>/，
# 再由 PROJECT_DATA_DIR 指给 50 号；03data 不动。开跑前先按 10% 重生成一份，与 03data 的
# 02_data_<年>.csv 逐字节比对，任一年不一致即退出。
#
# 作业：50 号，两档 × 5 粒 Formal 种子 × 24 分片，--methods Phys,Phys+BL
#   → 04outputs/c6_shards_physfix_p0/ 与 04outputs/c6_shards_physfix_p15/
# 已存在且非空的分片跳过，可断点续跑。
#
# 运行方式（项目根目录）:
#   bash 02code/104_c6_physfix_rerun_local.sh [并发进程数，默认 4]
# 可选环境变量 C6_DATA_ROOT 指定重生成数据的存放处（默认 ${TMPDIR:-/tmp}/ssc015_c6data）。
# 注意：02 号会改写受 git 跟踪的 05logs/02_data_processing.log，跑完用
#   git checkout -- 05logs/02_data_processing.log 恢复。
set -uo pipefail
PJOBS="${1:-4}"
EPOCHS=200
N_SHARDS=24
PCTS=(0 15)
SEEDS=(20060515 20041210 19810915 2023 2024)

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE          # macOS 上 torch 与 numpy 各带一份 libomp，不设则进程直接退出
export PYTHONUNBUFFERED=1
export PYTHON="${PYTHON:-python3}"
C6_DATA_ROOT="${C6_DATA_ROOT:-${TMPDIR:-/tmp}/ssc015_c6data}"
C6_DATA_ROOT="${C6_DATA_ROOT%/}"
mkdir -p 05logs

regen() {  # regen <档>：把该档的预处理数据写到 $C6_DATA_ROOT/p<档>/
  local pct=$1 dir="$C6_DATA_ROOT/p$1"
  if [[ -s "$dir/02_data_2018.csv" && -s "$dir/02_data_2025.csv" ]]; then
    echo "   p$pct 数据已存在：$dir"; return 0
  fi
  mkdir -p "$dir"
  PROJECT_DATA_DIR="$dir" nice -n 10 "$PYTHON" 02code/02_data_processing.py --outlier_pct "$pct" \
    > "05logs/104_regen_p${pct}.out" 2>&1 || { echo "!! p$pct 数据重生成失败，见 05logs/104_regen_p${pct}.out"; exit 1; }
  echo "   p$pct 数据已生成：$dir"
}

echo "== 104 C6 补跑 Phys / Phys+BL：数据目录 ${C6_DATA_ROOT}，开始于 $(date '+%F %T') =="
regen 10
for y in 2018 2019 2025; do
  cmp -s "03data/02_data_$y.csv" "$C6_DATA_ROOT/p10/02_data_$y.csv" \
    || { echo "!! 按 10% 重生成的 $y 年数据与 03data 不一致，重生成路径不可信，退出"; exit 1; }
done
echo "   10% 重生成与 03data 逐字节一致（2018、2019、2025）"
for p in "${PCTS[@]}"; do regen "$p"; done

JOBS=$(mktemp "${TMPDIR:-/tmp}/ssc015_c6physfix.XXXXXX")   # GNU 与 BSD 都认的模板写法
trap 'rm -f "$JOBS"' EXIT
: > "$JOBS"
for p in "${PCTS[@]}"; do
  mkdir -p "04outputs/c6_shards_physfix_p$p"
  for s in "${SEEDS[@]}"; do
    for ((i=0; i<N_SHARDS; i++)); do
      f="04outputs/c6_shards_physfix_p$p/seed${s}_shard$(printf '%02d' "$i").csv"
      [[ -s "$f" ]] || echo "$p $s $i"
    done
  done
done >> "$JOBS"

echo "== 待跑 $(wc -l < "$JOBS" | tr -d ' ') 个分片，并发 ${PJOBS} =="

export EPOCHS N_SHARDS C6_DATA_ROOT
xargs -P "$PJOBS" -L1 bash -c '
  p=$0; s=$1; i=$2
  PROJECT_DATA_DIR="$C6_DATA_ROOT/p$p" nice -n 10 "$PYTHON" 02code/50_formal_multiseed_benchmark.py \
    --device cpu --epochs "$EPOCHS" --seed "$s" --shard_id "$i" --n_shards "$N_SHARDS" \
    --methods Phys,Phys+BL --shard_dir "04outputs/c6_shards_physfix_p$p" \
    > "05logs/104_c6_p${p}_${s}_${i}.out" 2>&1 \
    && echo "   完成 p${p} ${s} ${i}  $(date +%T)" || echo "!! 失败 p${p} ${s} ${i}"
' < "$JOBS"

for p in "${PCTS[@]}"; do
  echo "== p${p}：$(ls "04outputs/c6_shards_physfix_p$p"/seed*_shard*.csv 2>/dev/null | wc -l | tr -d ' ')/120 个分片 =="
done
echo "== 104 结束于 $(date '+%F %T') =="
