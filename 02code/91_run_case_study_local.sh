#!/usr/bin/env bash
# 91_run_case_study_local.sh — 在本机（macOS，无 CUDA）调度 50 号案例研究基准的全部 (种子 × 分片)
#
# 为什么需要本机版：51 号的显存闸门依赖 nvidia-smi，本项目的 GPU 实例已过期（BRIEF §6），
# 案例研究必须在本机重跑。模型极小（CNN ~2 万参数、输入 229 维），瓶颈是 Python 与
# kernel launch 开销，因此用「单线程 BLAS × 多进程」把 8 个核吃满，而不是单进程多线程。
#
# 运行方式:
#   bash 02code/91_run_case_study_local.sh [每种子分片数] [并发进程数] [epochs]
#   例: bash 02code/91_run_case_study_local.sh 24 4 200
#
# 两种模式：
#   FORCE=0（默认）断点续跑，已存在且非空的分片 CSV 自动跳过；
#   FORCE=1        全量重跑，由 50 号就地覆盖同名分片（旧结果在 git 历史里，可 git checkout 取回）。
#                  合并前会逐个核对 120 个分片的 mtime 晚于本次启动时刻，杜绝新旧结果混在一起。
#
# 输出:
#   04outputs/50_shards/seed<S>_shard<NN>.csv    — 各分片结果
#   05logs/50_shard_<seed>_<shard>.out           — 各分片日志
set -uo pipefail

N_SHARDS="${1:-24}"
PARALLEL="${2:-4}"
EPOCHS="${3:-200}"
FORCE="${FORCE:-0}"

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE"

SEEDS=(20060515 20041210 19810915 2023 2024)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export SSC_BENCH=1

mkdir -p 04outputs/50_shards 05logs

JOBS=$(mktemp)
for s in "${SEEDS[@]}"; do
  for ((i=0; i<N_SHARDS; i++)); do
    f="04outputs/50_shards/seed${s}_shard$(printf '%02d' "$i").csv"
    if [[ "$FORCE" == "1" ]]; then echo "$s $i"; else [[ -s "$f" ]] || echo "$s $i"; fi
  done
done > "$JOBS"

TOTAL=$(wc -l < "$JOBS" | tr -d ' ')
echo "=========================================================="
echo " 案例研究基准（本机 CPU）"
echo "   种子      : ${SEEDS[*]}"
echo "   分片/种子 : ${N_SHARDS}   并发: ${PARALLEL}   epochs: ${EPOCHS}"
echo "   待跑作业  : ${TOTAL} / $(( ${#SEEDS[@]} * N_SHARDS ))（已完成的分片已跳过）"
echo "=========================================================="

T0=$(date +%s)
FAIL=0
if [[ "$TOTAL" -gt 0 ]]; then
  # macOS 的 xargs 没有 -a，用重定向喂作业清单
  xargs -P "$PARALLEL" -L1 bash -c '
    seed=$0; shard=$1
    nice -n 5 python3 02code/50_formal_multiseed_benchmark.py \
      --device cpu --epochs '"$EPOCHS"' \
      --seed "$seed" --shard_id "$shard" --n_shards '"$N_SHARDS"' \
      > "05logs/50_shard_${seed}_${shard}.out" 2>&1
    code=$?
    if [[ $code -ne 0 ]]; then
      echo "!! 分片失败 seed=${seed} shard=${shard} (exit ${code})，见 05logs/50_shard_${seed}_${shard}.out"
    else
      echo "   完成 seed=${seed} shard=${shard}"
    fi
    exit $code
  ' < "$JOBS" || FAIL=1
fi
rm -f "$JOBS"

echo "----------------------------------------------------------"
echo "用时 $(( ($(date +%s) - T0) / 60 )) 分钟"
if [[ "$FAIL" -ne 0 ]]; then
  echo "存在失败分片——不要合并，先看日志。"
  exit 1
fi
MISSING=0
for s in "${SEEDS[@]}"; do
  for ((i=0; i<N_SHARDS; i++)); do
    f="04outputs/50_shards/seed${s}_shard$(printf '%02d' "$i").csv"
    if [[ ! -s "$f" ]]; then echo "缺分片: $f"; MISSING=1
    elif [[ "$FORCE" == "1" ]] && [[ $(stat -f %m "$f") -lt "$T0" ]]; then
      echo "陈旧分片（本次未重写）: $f"; MISSING=1
    fi
  done
done
[[ "$MISSING" -eq 0 ]] || { echo "分片不全或含陈旧文件，拒绝合并。"; exit 1; }
echo "全部分片就绪。合并："
echo "  python3 02code/50_formal_multiseed_benchmark.py --merge"
