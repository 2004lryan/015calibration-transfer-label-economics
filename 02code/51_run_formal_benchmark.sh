#!/usr/bin/env bash
# 51_run_formal_benchmark.sh: 在**共享** GPU 上并行调度 50 号 Formal 基准的全部 (种子 × 分片) 作业
#
# 为什么要并行:
#   模型极小（CNN ~2 万参数, batch=16, 输入 229 维），单进程 GPU 利用率不足 10%，
#   瓶颈是 kernel launch 与 Python 开销而非算力。多进程共享同一张卡可把吞吐拉满。
#
# 共享机器纪律（这张卡上另有他人项目）:
#   · 显存  每进程硬上限 --gpu_mem_frac（默认 4% ≈ 1 GB），超限自己 OOM，不吃满整卡
#   · 启动前检查空闲显存，预算不足直接拒绝启动，绝不挤占
#   · CPU   nice 10 降优先级 + OMP/MKL 单线程，邻居的进程永远优先
#   · 可控  所有子进程带 SSC_BENCH 标记，51_kill.sh 一键清场
#
# 运行方式:
#   bash 02code/51_run_formal_benchmark.sh <每种子分片数> <并发进程数> [epochs] [显存占比]
#   例: bash 02code/51_run_formal_benchmark.sh 8 10 200 0.04
#
# 输出:
#   04outputs/50_shards/seed<S>_shard<NN>.csv    — 各分片结果
#   04outputs/50_formal_multiseed_benchmark.xlsx — 合并 + 统计检验后的唯一正式产出

set -uo pipefail

N_SHARDS="${1:-8}"
PARALLEL="${2:-10}"
EPOCHS="${3:-200}"
GPU_FRAC="${4:-0.04}"

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE"

SEEDS=(20060515 20041210 19810915 2023 2024)   # 仅 5 粒 Formal 种子
# seed=42 复现锚点不在此队列：它不占 GPU，由 53_anchor_check.py 在 CPU 上并行跑，
# 且改用"逐场景比对 18_exp"的更强判据，无需全量 602 场景。

# ── 共享机器：单线程 BLAS，避免 $PARALLEL 个进程各开 192 线程把机器打爆 ──
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SSC_BENCH=1                                 # 清场标记

mkdir -p 04outputs/50_shards 05logs

# ── 显存预算闸门：算清楚要占多少、卡上还剩多少，不够就不跑 ──────────────
TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
FREE_MIB=$(nvidia-smi --query-gpu=memory.free  --format=csv,noheader,nounits | head -1)
# 每进程上限 = GPU_FRAC × 总显存，再留 512 MiB/进程 的 CUDA context 余量
PER_PROC=$(awk -v t="$TOTAL_MIB" -v f="$GPU_FRAC" 'BEGIN{printf "%d", t*f + 512}')
NEED=$(( PER_PROC * PARALLEL ))
RESERVE=2048                                       # 给邻居至少留 2 GiB 缓冲

echo "=========================================================="
echo " Formal 基准调度（共享 GPU）"
echo "   种子     : ${SEEDS[*]}"
echo "   分片/种子 : ${N_SHARDS}    并发: ${PARALLEL}    epochs: ${EPOCHS}"
echo "   显存预算 : 每进程 ≤ ${PER_PROC} MiB × ${PARALLEL} = ${NEED} MiB"
echo "   卡上现状 : 空闲 ${FREE_MIB} / 总 ${TOTAL_MIB} MiB"
echo "   作业总数 : $(( ${#SEEDS[@]} * N_SHARDS ))"
echo "=========================================================="

if (( NEED + RESERVE > FREE_MIB )); then
  echo "!! 显存不足：需 ${NEED} MiB + 邻居缓冲 ${RESERVE} MiB > 空闲 ${FREE_MIB} MiB"
  echo "   降低并发（第2个参数）或显存占比（第4个参数）后重试。拒绝启动。"
  exit 1
fi

# ── 作业清单（已完成的分片自动跳过，支持断点续跑）────────────────────────
JOBS=$(mktemp)
for s in "${SEEDS[@]}"; do
  for ((i=0; i<N_SHARDS; i++)); do
    f="04outputs/50_shards/seed${s}_shard$(printf '%02d' "$i").csv"
    [[ -s "$f" ]] || echo "$s $i"
  done
done > "$JOBS"

TOTAL=$(wc -l < "$JOBS" | tr -d ' ')
echo "待跑作业: $TOTAL（已完成的分片已跳过）"

T0=$(date +%s)
if [[ "$TOTAL" -gt 0 ]]; then
  # nice 10：邻居的进程永远比我优先拿到 CPU
  xargs -a "$JOBS" -P "$PARALLEL" -L1 bash -c '
    seed=$0; shard=$1
    nice -n 10 python 02code/50_formal_multiseed_benchmark.py \
      --device cuda --epochs '"$EPOCHS"' --gpu_mem_frac '"$GPU_FRAC"' \
      --seed "$seed" --shard_id "$shard" --n_shards '"$N_SHARDS"' \
      > "05logs/50_shard_${seed}_${shard}.out" 2>&1
    code=$?
    if [[ $code -ne 0 ]]; then
      echo "!! 失败 seed=$seed shard=$shard (exit $code)" >&2
      tail -3 "05logs/50_shard_${seed}_${shard}.out" >&2
    else
      echo "   完成 seed=$seed shard=$shard"
    fi
  '
fi
rm -f "$JOBS"
T1=$(date +%s)
echo "----------------------------------------------------------"
echo "全部分片耗时: $(( (T1-T0)/60 )) min"

# ── 完整性校验 ──────────────────────────────────────────────────────────
MISSING=0
for s in "${SEEDS[@]}"; do
  for ((i=0; i<N_SHARDS; i++)); do
    f="04outputs/50_shards/seed${s}_shard$(printf '%02d' "$i").csv"
    [[ -s "$f" ]] || { echo "!! 缺失: $f"; MISSING=$((MISSING+1)); }
  done
done
if (( MISSING > 0 )); then
  echo "!! 有 $MISSING 个分片缺失。重跑本脚本即可断点续跑（已完成的会跳过）。"
  exit 1
fi

echo "全部分片齐全，开始合并 + 统计检验..."
nice -n 10 python 02code/50_formal_multiseed_benchmark.py --merge --n_boot 1000
