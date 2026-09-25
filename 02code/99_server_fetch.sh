#!/usr/bin/env bash
# 99_server_fetch.sh — 把服务器上跑出来的分片与合并结果取回本机
#
# 两个分片目录用 --delete 整体替换，不做增量合并：本机在迁到服务器之前算过一部分分片，
# 那批是用同一份代码但另一台机器跑的，混在一起会让「每格 5 次运行」的口径说不清。
# 04outputs 下的其它文件（下游分析产物）不动，它们仍由本机重算。
#
# 深度臂（25 个分片）比案例研究（120 个）先跑完，而统一基准那条下游分析链只依赖深度臂，
# 所以支持只取一部分先开工：第 4 个参数 deep / case / zero / all。
# zero 取严格零标签对照的 24 个分片（59 号）。
#
# 用法:  SSHPASS='<密码>' bash 02code/99_server_fetch.sh <用户@主机> <端口> <远端目录> [deep|case|zero|all] [--dry-run]
set -uo pipefail
HOSTSPEC="${1:?用法: 99_server_fetch.sh 用户@主机 端口 远端目录 [deep|case|zero|all] [--dry-run]}"
PORT="${2:?}"
REMOTE="${3:?}"
SCOPE="${4:-all}"
DRY="${5:-}"
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE"

RSH="sshpass -e ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -p ${PORT}"
mkdir -p 04outputs/50_shards 04outputs/59_shards 04outputs/64_shards 05logs/server

case "$SCOPE" in
  deep) DIRS="64_shards" ;;
  case) DIRS="50_shards" ;;
  zero) DIRS="59_shards" ;;
  all)  DIRS="50_shards 59_shards 64_shards" ;;
  *) echo "第 4 个参数只能是 deep / case / zero / all"; exit 1 ;;
esac

echo "== 分片（整体替换，本次取: ${DIRS}）=="
for d in $DIRS; do
  rsync -az --timeout=180 --stats --delete ${DRY:+--dry-run} -e "$RSH" \
    "${HOSTSPEC}:${REMOTE}/04outputs/${d}/" "04outputs/${d}/" || exit 1
  printf '  %-10s 本机现有 %s 个 csv\n' "$d" "$(ls 04outputs/${d}/*.csv 2>/dev/null | wc -l | tr -d ' ')"
done

echo "== 合并结果与源域阳性对照 =="
# 服务器只在全部分片就绪后才合并，所以这几份可能还不存在。逐个取、缺的跳过，
# 不让「还没生成」把整条取回判成失败。
FILES="04outputs/50_formal_multiseed_benchmark.xlsx
04outputs/59_reviewer_experiments.xlsx
04outputs/64_deep_transfer_server.xlsx
04outputs/89_source_reference_baselines.xlsx"
for b in corn tablet mango ossl_mir apple; do
  FILES="$FILES
04outputs/88_deep_indomain_sanity_${b}.xlsx"
done
for f in $FILES; do
  if rsync -az --timeout=180 ${DRY:+--dry-run} -e "$RSH" \
       "${HOSTSPEC}:${REMOTE}/$f" "$f" 2>/dev/null; then
    printf '  取回 %s\n' "${f##*/}"
  else
    printf '  跳过 %s（服务器上还没有）\n' "${f##*/}"
  fi
done

echo "== 日志 =="
rsync -az --timeout=300 ${DRY:+--dry-run} -e "$RSH" \
  "${HOSTSPEC}:${REMOTE}/05logs/" 05logs/server/ || exit 1

case "$SCOPE" in
  deep) echo "取回完成。深度臂齐了就: python3 02code/64_deep_transfer_server.py --merge && bash 02code/97_downstream_pipeline.sh unified" ;;
  case) echo "取回完成。案例分片齐了就: python3 02code/50_formal_multiseed_benchmark.py --merge && bash 02code/97_downstream_pipeline.sh case" ;;
  zero) echo "取回完成。零标签分片齐了就: python3 02code/59_reviewer_experiments.py --merge" ;;
  *)    echo "取回完成。下一步: bash 02code/97_downstream_pipeline.sh all" ;;
esac
