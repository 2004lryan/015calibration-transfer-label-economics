#!/usr/bin/env bash
# 107_server_season_scaler.sh — 在服务器上跑 106 号（年份级逐波段标准化的敏感性）
#
# 作业：106 号，种子 20060515，60 个 stride 分片，方法 PLSR、SVR、CNN、Phys+BL，
#       口径 all / no_test / no_target → 远端 04outputs/106_shards/
# 数据：本机用 02 号 --outlier_pct 10 --no_season_scaler 生成（样本、产地、SSC 与 03data 逐行相同，
#       光谱停在 SG 平滑之后），deploy 时传到远端 ed001data/p10ns/。
# 服务器与登录方式同 105 号：密钥登录、BatchMode、每条子命令一到两次连接、轮询间隔不短于 15 分钟，
# 作业用 setsid 脱离 SSH 会话。
#
# 子命令（项目根目录执行）:
#   deploy <无年份标准化的数据目录>   传 02code 的 .py/.json/.sh 与该数据目录
#   start [并发]                      setsid 拉起全部分片（默认 15，贴 cgroup 配额），进程组号写入远端 05logs/106.pgid
#   status                            已完成分片数、失败数、进程组是否还在
#   stop                              按进程组号停掉
#   fetch                             取回分片到 04outputs/106_shards/，服务器端日志到 05logs/server_106/
#   _run [解释器] [并发]              直接在当前机器上跑全部分片（start 在服务器上调用它；本机跑时
#                                     先设 PROJECT_DATA_DIR 指向数据目录）。每个场景的三种口径在同一
#                                     进程里跑，口径间的配对比较不受机器差异影响，分片可以分在不同机器上跑。
set -uo pipefail
HOST="${SSC015_HOST:-user@host}"
PORT="${SSC015_PORT:-22}"
REMOTE="${SSC015_REMOTE:-apple_ssc_transfer}"
PY="${SSC015_PY:-python}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=15 -p "$PORT" "$HOST")
RSH="ssh -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=15 -p $PORT"
SEED=20060515
N_SHARDS=60

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE"

cmd="${1:-}"
[[ $# -gt 0 ]] && shift
case "$cmd" in
deploy)
  SRC="${1:?用法: deploy <无年份标准化的数据目录>}"
  SRC="${SRC%/}"
  for y in 2018 2019 2025; do [[ -s "$SRC/02_data_$y.csv" ]] || { echo "!! $SRC 缺 $y 年数据"; exit 1; }; done
  LIST=$(mktemp "${TMPDIR:-/tmp}/ssc015_deploy106.XXXXXX")
  trap 'rm -f "$LIST"' EXIT
  find 02code -maxdepth 1 -type f \( -name '*.py' -o -name '*.json' -o -name '*.sh' \) ! -name '._*' -print \
    | LC_ALL=C sort > "$LIST"
  rsync -az --timeout=180 --files-from="$LIST" -e "$RSH" ./ "$HOST:$REMOTE/" || exit 1
  rsync -az --timeout=180 --exclude='._*' --exclude='*.xlsx' -e "$RSH" "$SRC/" "$HOST:$REMOTE/ed001data/p10ns/" || exit 1
  echo "== deploy 完成：代码 $(wc -l < "$LIST" | tr -d ' ') 个文件 + 数据 $SRC =="
  ;;
start)
  P="${1:-15}"
  "${SSH[@]}" "bash -s -- '$REMOTE' '$PY' '$P'" <<'EOF'
cd "$1" || exit 1
mkdir -p 05logs
if [[ -s 05logs/106.pgid ]] && kill -0 -- -"$(cat 05logs/106.pgid)" 2>/dev/null; then
  echo "106 已在跑（进程组 $(cat 05logs/106.pgid)）"; exit 0
fi
setsid nohup bash 02code/107_server_season_scaler.sh _run "$2" "$3" >> 05logs/106_server.log 2>&1 < /dev/null &
pid=$!
sleep 8
ps -o pgid= -p "$pid" | tr -d ' ' > 05logs/106.pgid
echo "已拉起 106：进程组 $(cat 05logs/106.pgid)，并发 $3"
tail -4 05logs/106_server.log
EOF
  ;;
_run)
  # 服务器上由 start 拉起；本机可直接调用
  export PYTHON="${1:-python}" PJOBS="${2:-15}" SEED N_SHARDS
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
  export KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1 LANG=C.UTF-8
  export PROJECT_DATA_DIR="${PROJECT_DATA_DIR:-$BASE/ed001data/p10ns}"
  mkdir -p 04outputs/106_shards
  echo "== 106 开始于 $(date '+%F %T')，数据 ${PROJECT_DATA_DIR}，并发 ${PJOBS} =="
  for ((i=0; i<N_SHARDS; i++)); do
    f="04outputs/106_shards/seed${SEED}_shard$(printf '%02d' "$i").csv"
    [[ -s "$f" ]] || echo "$i"
  done | xargs -P "$PJOBS" -n1 bash -c '
    i=$0
    nice -n 10 "$PYTHON" 02code/106_season_scaler_sensitivity.py --device cpu --epochs 200 \
      --seed "$SEED" --shard_id "$i" --n_shards "$N_SHARDS" --shard_dir 04outputs/106_shards \
      > "05logs/106_shard${i}.out" 2>&1 && echo "   完成 ${i}  $(date +%T)" || echo "!! 失败 ${i}"'
  echo "== 106 结束于 $(date '+%F %T')：$(ls 04outputs/106_shards/seed*_shard*.csv 2>/dev/null | wc -l)/$N_SHARDS 个分片 =="
  ;;
status)
  "${SSH[@]}" "bash -s -- '$REMOTE' '$N_SHARDS'" <<'EOF'
cd "$1" || exit 1
L=05logs/106_server.log
n=$(ls 04outputs/106_shards/seed*_shard*.csv 2>/dev/null | wc -l)
f=$(grep -c "^!! 失败" "$L" 2>/dev/null)
pg=$(cat 05logs/106.pgid 2>/dev/null)
alive=否; [[ -n "$pg" ]] && kill -0 -- -"$pg" 2>/dev/null && alive=是
echo "服务器 $(date '+%m-%d %H:%M')｜106 ${n}/$2｜失败 ${f:-0}｜仍在跑 ${alive}｜负载 $(cut -d' ' -f1 /proc/loadavg)"
tail -2 "$L"
EOF
  ;;
stop)
  "${SSH[@]}" "bash -s -- '$REMOTE'" <<'EOF'
cd "$1" || exit 1
pg=$(cat 05logs/106.pgid 2>/dev/null)
if [[ -n "$pg" ]] && kill -- -"$pg" 2>/dev/null; then echo "已停进程组 $pg"; else echo "没有在跑的 106"; fi
EOF
  ;;
fetch)
  mkdir -p 04outputs/106_shards 05logs/server_106
  rsync -az --timeout=180 --exclude='._*' -e "$RSH" "$HOST:$REMOTE/04outputs/106_shards/" 04outputs/106_shards/ || exit 1
  rsync -az --timeout=180 --exclude='._*' --include='106*' --exclude='*' -e "$RSH" "$HOST:$REMOTE/05logs/" 05logs/server_106/ || exit 1
  echo "本机现有 $(ls 04outputs/106_shards/seed*_shard*.csv 2>/dev/null | wc -l | tr -d ' ')/$N_SHARDS 个分片"
  ;;
*)
  echo "用法: bash 02code/107_server_season_scaler.sh {deploy <数据目录>|start [并发]|status|stop|fetch}"
  exit 1
  ;;
esac
