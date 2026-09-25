#!/usr/bin/env bash
# 96_server_deploy.sh — 把本项目跑实验所需的最小集合同步到服务器
#
# 只传三样：02code 下的 .py/.json/.sh（约 14 MB）、03data 的三份苹果清洗表（约 29 MB）、
# 03data/processed/benchmarks 的统一缓存（约 54 MB）。02code/checkpoints* 是旧权重（约 950 MB），不传。
#
# macOS 26+ 自带的是 openrsync，过滤规则语义与 GNU rsync 有出入，
# 所以这里在本机把要传的文件逐条列出，交给 --files-from，传什么完全可预期。
#
# 用法:  SSHPASS='<密码>' bash 02code/96_server_deploy.sh <用户@主机> <端口> <远端目录> [--dry-run]
set -uo pipefail
HOSTSPEC="${1:?用法: 96_server_deploy.sh 用户@主机 端口 远端目录 [--dry-run]}"
PORT="${2:?}"
REMOTE="${3:?}"
DRY="${4:-}"
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE"

LIST="$(mktemp -t ssc015_deploy_list)"
trap 'rm -f "$LIST"' EXIT
{
  find 02code -maxdepth 1 -type f \( -name '*.py' -o -name '*.json' -o -name '*.sh' \) -print
  for y in 2018 2019 2025; do
    [ -f "03data/02_data_${y}.csv" ] && echo "03data/02_data_${y}.csv"
  done
  find 03data/processed/benchmarks -maxdepth 1 -type f -print
  echo pyproject.toml
} | LC_ALL=C sort > "$LIST"

echo "== 待传 $(wc -l < "$LIST" | tr -d ' ') 个文件，合计 $(du -ch $(tr '\n' ' ' < "$LIST") 2>/dev/null | tail -1 | cut -f1) =="

RSH="sshpass -e ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -p ${PORT}"
exec rsync -az --timeout=180 --stats ${DRY:+--dry-run} --files-from="$LIST" \
  -e "$RSH" ./ "${HOSTSPEC}:${REMOTE}/"
