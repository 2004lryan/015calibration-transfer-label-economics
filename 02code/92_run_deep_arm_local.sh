#!/usr/bin/env bash
# 92_run_deep_arm_local.sh — 在本机调度 64 号深度臂的全量重跑（基准 × 种子 = 25 个作业）
#
# 为什么要重跑：64 号原实现的编码器把整条谱在波长轴上求平均、且全批量每 epoch 只走一次
# 更新，导致深度基线在**源域内部**就退化成常数预测器（88/89/90 号诊断）。修正后按固定
# 优化步数预算走小批量，各基准拿到同样多的参数更新。
#
# 运行方式:
#   bash 02code/92_run_deep_arm_local.sh [并发进程数] [rep]
#   例: bash 02code/92_run_deep_arm_local.sh 3 1
# rep 默认 1：正典的 64 号输出里 rep 只有 0 一个取值，每 (任务, 预算, 方法) 恰好 5 次运行。
#
# 断点续跑：已存在且非空的分片 CSV 自动跳过（FORCE=1 强制重跑，就地覆盖）。
#
# 输出:
#   04outputs/64_shards/<benchmark>_<seed>.csv   — 各分片
#   05logs/64_<benchmark>_<seed>.out             — 各分片日志
# 全部就绪后合并:
#   python3 02code/64_deep_transfer_server.py --merge
set -uo pipefail

PARALLEL="${1:-4}"
REP="${2:-1}"     # 正典口径：每 (task, seed) 一次划分，每格 5 次运行（rep=1）
FORCE="${FORCE:-0}"

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE"

BENCHES=(corn tablet mango ossl_mir apple)
SEEDS=(20060515 20041210 19810915 2023 2024)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

mkdir -p 04outputs/64_shards 05logs

JOBS=$(mktemp)
for b in "${BENCHES[@]}"; do
  for s in "${SEEDS[@]}"; do
    f="04outputs/64_shards/${b}_${s}.csv"
    if [[ "$FORCE" == "1" ]]; then echo "$b $s"; else [[ -s "$f" ]] || echo "$b $s"; fi
  done
done > "$JOBS"

TOTAL=$(wc -l < "$JOBS" | tr -d ' ')
echo "=========================================================="
echo " 深度臂重跑（本机 CPU）"
echo "   基准 : ${BENCHES[*]}"
echo "   种子 : ${SEEDS[*]}"
echo "   并发 : ${PARALLEL}   rep: ${REP}"
echo "   待跑 : ${TOTAL} / $(( ${#BENCHES[@]} * ${#SEEDS[@]} ))"
echo "=========================================================="

T0=$(date +%s)
FAIL=0
if [[ "$TOTAL" -gt 0 ]]; then
  xargs -P "$PARALLEL" -L1 bash -c '
    b=$0; s=$1
    nice -n 5 python3 02code/64_deep_transfer_server.py \
      --benchmarks "$b" --seeds "$s" --rep '"$REP"' --device cpu --tag "${b}_${s}" \
      > "05logs/64_${b}_${s}.out" 2>&1
    code=$?
    if [[ $code -ne 0 ]]; then
      echo "!! 分片失败 ${b} ${s} (exit ${code})，见 05logs/64_${b}_${s}.out"
    else
      echo "   完成 ${b} ${s}"
    fi
    exit $code
  ' < "$JOBS" || FAIL=1
fi
rm -f "$JOBS"

echo "----------------------------------------------------------"
echo "用时 $(( ($(date +%s) - T0) / 60 )) 分钟"
[[ "$FAIL" -eq 0 ]] || { echo "存在失败分片——不要合并，先看日志。"; exit 1; }
MISSING=0
for b in "${BENCHES[@]}"; do
  for s in "${SEEDS[@]}"; do
    f="04outputs/64_shards/${b}_${s}.csv"
    if [[ ! -s "$f" ]]; then echo "缺分片: $f"; MISSING=1
    elif [[ "$FORCE" == "1" ]] && [[ $(stat -f %m "$f") -lt "$T0" ]]; then
      echo "陈旧分片（本次未重写）: $f"; MISSING=1
    fi
  done
done
[[ "$MISSING" -eq 0 ]] || { echo "分片不全或含陈旧文件，拒绝合并。"; exit 1; }
echo "全部分片就绪。合并："
echo "  python3 02code/64_deep_transfer_server.py --merge"
