#!/usr/bin/env bash
# 103_physfix_rerun_local.sh — 在本机补跑苹果案例研究里的 Phys 与 Phys+BL
#
# 为什么要补跑：13_model_physics_informed.py 的回归项原先把 (B,) 的预测与 (B,1) 的标签
# 直接相减，广播成 B×B，每个预测都去对批内全部标签，模型只学到批均值。修正后只有这两个
# 方法的结果会变；其余五个方法（PLSR、SVR、CNN、CNN+MMD、CNN+BL）的代码路径是配对的，
# 原分片照用。50 号在每个场景划分前、每个方法开跑前都重置同一粒种子，59 号同理，所以补跑
# 与原分片的目标域划分逐样本相同，配对比较成立。
#
# 作业：
#   case  50 号，5 粒 Formal 种子 × 24 分片，--methods Phys,Phys+BL → 04outputs/50_shards_physfix/
#   zero  59 号，种子 42 × 24 分片，--methods Phys+BL             → 04outputs/59_shards_physfix/
# 已存在且非空的分片跳过，可断点续跑。
#
# 运行方式（项目根目录）:
#   bash 02code/103_physfix_rerun_local.sh [并发进程数，默认 4]
# 本机为 Apple M1（8 核 / 8 GB）：单线程 BLAS、nice 10，4 个进程约占 2 GB 内存。
set -uo pipefail
PJOBS="${1:-4}"
EPOCHS=200
N_SHARDS=24
ZERO_SEED=42
SEEDS=(20060515 20041210 19810915 2023 2024)

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE          # macOS 上 torch 与 numpy 各带一份 libomp，不设则进程直接退出
export PYTHONUNBUFFERED=1
export PYTHON="${PYTHON:-python3}"
CASE_DIR=04outputs/50_shards_physfix
ZERO_DIR=04outputs/59_shards_physfix
mkdir -p "$CASE_DIR" "$ZERO_DIR" 05logs

JOBS=$(mktemp -t ssc015_physfix)
trap 'rm -f "$JOBS"' EXIT
: > "$JOBS"
for ((i=0; i<N_SHARDS; i++)); do
  f="$ZERO_DIR/seed${ZERO_SEED}_shard$(printf '%02d' "$i").csv"
  [[ -s "$f" ]] || echo "zero $i -"
done >> "$JOBS"
for s in "${SEEDS[@]}"; do
  for ((i=0; i<N_SHARDS; i++)); do
    f="$CASE_DIR/seed${s}_shard$(printf '%02d' "$i").csv"
    [[ -s "$f" ]] || echo "case $s $i"
  done
done >> "$JOBS"

echo "== 103 补跑 Phys / Phys+BL：待跑 $(wc -l < "$JOBS" | tr -d ' ') 个分片，并发 ${PJOBS}，开始于 $(date '+%F %T') =="

export EPOCHS N_SHARDS ZERO_SEED CASE_DIR ZERO_DIR
xargs -P "$PJOBS" -L1 bash -c '
  kind=$0
  if [[ "$kind" == "zero" ]]; then
    i=$1
    nice -n 10 "$PYTHON" 02code/59_reviewer_experiments.py \
      --device cpu --zero_epochs "$EPOCHS" --seed "$ZERO_SEED" --shard_id "$i" --n_shards "$N_SHARDS" \
      --methods Phys+BL --shard_dir "$ZERO_DIR" \
      > "05logs/103_zero_${i}.out" 2>&1 \
      && echo "   完成 zero ${i}  $(date +%T)" || echo "!! 失败 zero ${i}"
  else
    s=$1; i=$2
    nice -n 10 "$PYTHON" 02code/50_formal_multiseed_benchmark.py \
      --device cpu --epochs "$EPOCHS" --seed "$s" --shard_id "$i" --n_shards "$N_SHARDS" \
      --methods Phys,Phys+BL --shard_dir "$CASE_DIR" \
      > "05logs/103_case_${s}_${i}.out" 2>&1 \
      && echo "   完成 case ${s} ${i}  $(date +%T)" || echo "!! 失败 case ${s} ${i}"
  fi
' < "$JOBS"

echo "== 103 结束于 $(date '+%F %T')：case $(ls "$CASE_DIR"/seed*_shard*.csv 2>/dev/null | wc -l | tr -d ' ')/120，zero $(ls "$ZERO_DIR"/seed*_shard*.csv 2>/dev/null | wc -l | tr -d ' ')/24 =="
