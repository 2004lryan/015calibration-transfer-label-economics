#!/usr/bin/env bash
# 109_server_unified_deep.sh — 在服务器上跑 108 号深度族（扣除目标域口径）的其余 4 粒种子
#
# 作业：108 号 --family deep --variant no_target，种子 20041210 / 19810915 / 2023 / 2024，
#       每个（种子，任务）一片（--n_shards 72 --tag _s<种子>）→ 远端 04outputs/108_shards/。
#       种子 20060515 的 72 片已在本机跑完；与正典 64 号同为 5 种子。
# 数据：02 号 --outlier_pct 10 --no_season_scaler 生成的目录，deploy 时传到远端 ed001data/p10ns/，
#       经 PROJECT_DATA_DIR 指给 108 号。61 号 load_apple 以 03data/02_data_{年}.csv 是否存在决定载入
#       哪些年份，所以 03data 下这三个文件也要传——缺一年，任务编号会整体错位而不报错；start 之前先跑 check。
# 登录方式同 105/107 号：密钥登录、BatchMode、每条子命令一到两次连接、轮询间隔不短于 15 分钟，
# 作业用 setsid 脱离 SSH 会话。
#
# 子命令（项目根目录执行）:
#   deploy <无年份标准化的数据目录>   传 02code 的 .py/.json/.sh、该数据目录、03data/02_data_*.csv 与本机已完成的分片
#   check                             远端与本机逐个比对 72 个任务的编号与内容
#   start [并发] [设备]               setsid 拉起（并发默认取远端可用 CPU 数；设备默认 cpu，可为 cuda），
#                                     进程组号写入远端 05logs/109.pgid
#   status                            已完成分片数、失败数、进程组是否还在
#   stop                              按进程组号停掉
#   fetch                             取回分片（本机已有的不覆盖）到 04outputs/108_shards/，日志到 05logs/server_109/
#   _run [解释器] [并发] [设备]       直接在当前机器上跑全部未完成的（种子，任务）；start 在服务器上调用它
set -uo pipefail
HOST="${SSC015_HOST:-user@host}"
PORT="${SSC015_PORT:-22}"
REMOTE="${SSC015_REMOTE:-apple_ssc_transfer}"
PY="${SSC015_PY:-python3}"
KNOWN="${SSC015_KNOWN_HOSTS:-$HOME/.ssh/known_hosts}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=15 -o UserKnownHostsFile="$KNOWN" -p "$PORT" "$HOST")
RSH="ssh -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=15 -o UserKnownHostsFile=$KNOWN -p $PORT"
SEEDS="20041210 19810915 2023 2024"
N_TASKS=72

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE"

# 打印全部苹果任务（编号、源→目标、源/目标样本数），本机与远端用同一段代码，check 时逐行比对
TASKLIST_PY='
import importlib.util, os
spec = importlib.util.spec_from_file_location("b61", os.path.join("02code", "61_benchmark_datasets.py"))
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
raw = b.load_apple()
for i, tk in enumerate(b.build_transfer_tasks(raw)):
    d = raw["domains"]
    print(i, tk["src"], tk["tgt"], tk["prop_name"], len(d[tk["src"]]["X"]), len(d[tk["tgt"]]["X"]))
'

cmd="${1:-}"
[[ $# -gt 0 ]] && shift
case "$cmd" in
deploy)
  SRC="${1:?用法: deploy <无年份标准化的数据目录>}"
  SRC="${SRC%/}"
  for y in 2018 2019 2025; do
    [[ -s "$SRC/02_data_$y.csv" ]] || { echo "!! $SRC 缺 $y 年数据"; exit 1; }
    [[ -s "03data/02_data_$y.csv" ]] || { echo "!! 03data 缺 02_data_$y.csv"; exit 1; }
  done
  LIST=$(mktemp "${TMPDIR:-/tmp}/ssc015_deploy109.XXXXXX")
  trap 'rm -f "$LIST"' EXIT
  { find 02code -maxdepth 1 -type f \( -name '*.py' -o -name '*.json' -o -name '*.sh' \) ! -name '._*' ! -name '.*' -print
    for y in 2018 2019 2025; do echo "03data/02_data_$y.csv"; done
    find 04outputs/108_shards -maxdepth 1 -type f -name 'deep_no_target_s*_shard*.csv' ! -name '._*' -print
  } | LC_ALL=C sort > "$LIST"
  rsync -az --timeout=180 --files-from="$LIST" -e "$RSH" ./ "$HOST:$REMOTE/" || exit 1
  "${SSH[@]}" "mkdir -p '$REMOTE/ed001data/p10ns'" || exit 1   # rsync 只建最末一级目录
  rsync -az --timeout=180 --exclude='._*' --exclude='*.xlsx' -e "$RSH" "$SRC/" "$HOST:$REMOTE/ed001data/p10ns/" || exit 1
  echo "== deploy 完成：$(wc -l < "$LIST" | tr -d ' ') 个文件（代码、03data 三季、本机已完成分片）+ 数据 $SRC =="
  ;;
check)
  L=$(PROJECT_DATA_DIR="${PROJECT_DATA_DIR:?本机须设 PROJECT_DATA_DIR}" python3 -c "$TASKLIST_PY" 2>/dev/null)
  R=$("${SSH[@]}" "cd '$REMOTE' && PROJECT_DATA_DIR='$REMOTE/ed001data/p10ns' $PY -c '$TASKLIST_PY' 2>/dev/null")
  nl=$(printf '%s\n' "$L" | grep -c .); nr=$(printf '%s\n' "$R" | grep -c .)
  if [[ "$L" == "$R" && "$nl" -eq "$N_TASKS" ]]; then
    echo "== 任务清单一致：本机与远端都是 $nl 个任务，逐行相同 =="
  else
    echo "!! 任务清单不一致（本机 $nl 行，远端 $nr 行）"; diff <(printf '%s\n' "$L") <(printf '%s\n' "$R") | head -20; exit 1
  fi
  ;;
start)
  P="${1:-}"; DEV="${2:-cpu}"
  "${SSH[@]}" "bash -s -- '$REMOTE' '$PY' '$P' '$DEV'" <<'EOF'
cd "$1" || exit 1
mkdir -p 05logs
if [[ -s 05logs/109.pgid ]] && kill -0 -- -"$(cat 05logs/109.pgid)" 2>/dev/null; then
  echo "109 已在跑（进程组 $(cat 05logs/109.pgid)）"; exit 0
fi
P="$3"
if [[ -z "$P" ]]; then
  # 可用 CPU 取 cgroup 配额与 nproc 的较小者
  P=$(nproc)
  if [[ -r /sys/fs/cgroup/cpu.max ]]; then
    read -r q per < /sys/fs/cgroup/cpu.max
    [[ "$q" != "max" ]] && (( q / per < P )) && P=$(( q / per ))
  fi
fi
setsid nohup bash 02code/109_server_unified_deep.sh _run "$2" "$P" "$4" >> 05logs/109_server.log 2>&1 < /dev/null &
pid=$!
sleep 8
ps -o pgid= -p "$pid" | tr -d ' ' > 05logs/109.pgid
echo "已拉起 109：进程组 $(cat 05logs/109.pgid)，并发 $P（nproc $(nproc)）"
tail -3 05logs/109_server.log
EOF
  ;;
status)
  "${SSH[@]}" "bash -s -- '$REMOTE'" <<'EOF'
cd "$1" || exit 1
L=05logs/109_server.log
n=$(ls 04outputs/108_shards/deep_no_target_s*_shard*.csv 2>/dev/null | wc -l)
f=$(grep -c "失败" "$L" 2>/dev/null)
pg=$(cat 05logs/109.pgid 2>/dev/null)
alive=否; [[ -n "$pg" ]] && kill -0 -- -"$pg" 2>/dev/null && alive=是
echo "服务器 $(date '+%m-%d %H:%M')｜108 深度族补种子 ${n}/288｜失败 ${f:-0}｜仍在跑 ${alive}｜负载 $(cut -d' ' -f1 /proc/loadavg)"
tail -2 "$L"
EOF
  ;;
stop)
  "${SSH[@]}" "bash -s -- '$REMOTE'" <<'EOF'
cd "$1" || exit 1
pg=$(cat 05logs/109.pgid 2>/dev/null)
if [[ -n "$pg" ]] && kill -- -"$pg" 2>/dev/null; then echo "已停进程组 $pg"; else echo "没有在跑的 109"; fi
EOF
  ;;
fetch)
  mkdir -p 04outputs/108_shards 05logs/server_109
  rsync -az --timeout=180 --ignore-existing --exclude='._*' --include='deep_no_target_s*_shard*.csv' --include='deep_no_target_s*_shard*.meta.json' --exclude='*' \
    -e "$RSH" "$HOST:$REMOTE/04outputs/108_shards/" 04outputs/108_shards/ || exit 1
  rsync -az --timeout=180 --exclude='._*' --include='109*' --include='108_deep_no_target_s*' --exclude='*' \
    -e "$RSH" "$HOST:$REMOTE/05logs/" 05logs/server_109/ || exit 1
  echo "本机现有 $(ls 04outputs/108_shards/deep_no_target_s*_shard*.csv 2>/dev/null | wc -l | tr -d ' ')/288 个补种子分片"
  ;;
_run)
  export PYTHON="${1:-python3}" PJOBS="${2:-8}" DEVICE="${3:-cpu}"
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
  export KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1 LANG=C.UTF-8
  export PROJECT_DATA_DIR="${PROJECT_DATA_DIR:-$BASE/ed001data/p10ns}"
  mkdir -p 04outputs/108_shards 05logs
  echo "== 109 开始于 $(date '+%F %T')，数据 ${PROJECT_DATA_DIR}，并发 ${PJOBS} =="
  for s in $SEEDS; do
    for ((i=0; i<N_TASKS; i++)); do
      f="04outputs/108_shards/deep_no_target_s${s}_shard$(printf '%02d' "$i").csv"
      [[ -s "$f" ]] || echo "$s $i"
    done
  done | xargs -P "$PJOBS" -n2 bash -c '
    s=$0; i=$1
    nice -n 10 "$PYTHON" 02code/108_apple_unified_season_scaler.py --family deep --variant no_target \
      --seeds "$s" --tag "_s$s" --shard "$i" --n_shards 72 --device "$DEVICE" >> "05logs/108_deep_no_target_s${s}.log" 2>&1 \
      && echo "   完成 ${s} ${i}  $(date +%T)" || echo "!! 失败 ${s} ${i}"'
  echo "== 109 结束于 $(date '+%F %T')：$(ls 04outputs/108_shards/deep_no_target_s*_shard*.csv 2>/dev/null | wc -l)/288 个分片 =="
  ;;
*)
  echo "用法: bash 02code/109_server_unified_deep.sh {deploy <数据目录>|check|start [并发] [设备]|status|stop|fetch}"
  exit 1
  ;;
esac
