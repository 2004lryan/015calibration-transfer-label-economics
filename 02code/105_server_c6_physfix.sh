#!/usr/bin/env bash
# 105_server_c6_physfix.sh — 在服务器上跑 104 号（离群剔除 0%/15% 两档下 Phys 与 Phys+BL 的补跑）
#
# 为什么上服务器：104 号共 240 个分片，本机 8 核还被系统后台服务分走一部分，约 12 片/小时；
# 服务器 cgroup 配额 15 核，可以 15 路并行。服务器上跑的就是 104 号本身：同一份代码、
# 同一份三档预处理数据（本机按 --outlier_pct 生成，10% 档与 03data 逐字节一致）、同一组种子，
# 与正典分片一样用 CPU（meta: device=cpu）。
#
# 子命令（项目根目录执行）:
#   deploy <C6数据目录>  传 02code 的 .py/.json/.sh、03data 三份清洗表（104 号拿它核对 10% 档）、
#                        <C6数据目录>/p0 p10 p15，以及本机已跑完的 C6 分片（104 号会跳过它们）
#   check                远端依赖探测：逐个真的 import，缺什么列什么
#   setup                在服务器上后台安装依赖（torch 取 CPU 轮子），进度见远端 05logs/105_install.log
#   smoke                p0 数据上 1 个场景、完整 200 轮的冒烟测试并计时（后台跑，status 带出结果）
#   start [并发]         setsid 拉起 104 号（默认 15，贴 cgroup 配额），进程组号写入远端 05logs/104.pgid
#   status               两档已完成分片数、失败数、进程组是否还在
#   stop                 按进程组号停掉 104 号及其全部 worker
#   fetch                取回两档分片到 04outputs/，服务器端日志到 05logs/server_c6/
#
# 登录走密钥（~/.ssh/id_ed25519 已由 ssh-copy-id 装到服务器），BatchMode 下不接受密码提示。
# 网关有 fail2ban 式封禁：每条子命令只开一到两次连接；自动轮询的间隔不短于 15 分钟。
# 网关还会掐断长连接，所以安装与补跑都用 setsid 脱离 SSH 会话，断线不影响。
# 可选环境变量：SSC015_HOST（默认 user@host）、SSC015_PORT（默认 22）、
#               SSC015_REMOTE（默认 apple_ssc_transfer）、SSC015_PY（远端解释器，默认 python）
set -uo pipefail
HOST="${SSC015_HOST:-user@host}"
PORT="${SSC015_PORT:-22}"
REMOTE="${SSC015_REMOTE:-apple_ssc_transfer}"
PY="${SSC015_PY:-python}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=15 -p "$PORT" "$HOST")
RSH="ssh -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=15 -p $PORT"

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE"

cmd="${1:-}"
[[ $# -gt 0 ]] && shift
case "$cmd" in
deploy)
  SRC="${1:?用法: deploy <C6数据目录，内含 p0 p10 p15>}"
  SRC="${SRC%/}"
  for p in 0 10 15; do
    [[ -s "$SRC/p$p/02_data_2018.csv" && -s "$SRC/p$p/02_data_2025.csv" ]] || { echo "!! $SRC/p$p 缺数据"; exit 1; }
  done
  LIST=$(mktemp "${TMPDIR:-/tmp}/ssc015_deploy.XXXXXX")
  trap 'rm -f "$LIST"' EXIT
  {   # ._* 是这块外置盘上 macOS 自动生成的附属文件，不传
    find 02code -maxdepth 1 -type f \( -name '*.py' -o -name '*.json' -o -name '*.sh' \) ! -name '._*' -print
    for y in 2018 2019 2025; do echo "03data/02_data_${y}.csv"; done
    for p in 0 15; do find "04outputs/c6_shards_physfix_p$p" -maxdepth 1 -type f ! -name '._*' -print 2>/dev/null; done
  } | LC_ALL=C sort > "$LIST"
  echo "== 代码、清洗表与本机已完成分片：$(wc -l < "$LIST" | tr -d ' ') 个文件 =="
  rsync -az --timeout=180 --files-from="$LIST" -e "$RSH" ./ "$HOST:$REMOTE/" || exit 1
  echo "== 三档预处理数据 =="
  rsync -az --timeout=180 --exclude='._*' -e "$RSH" "$SRC/p0" "$SRC/p10" "$SRC/p15" "$HOST:$REMOTE/c6data/" || exit 1
  echo "== deploy 完成 =="
  ;;
check)
  "${SSH[@]}" "bash -s -- '$REMOTE' '$PY'" <<'EOF'
cd "$1" 2>/dev/null && echo "远端目录：$1，02code 下 $(ls 02code | wc -l) 项" || echo "（远端目录 $1 还不存在）"
for py in "$2" /opt/conda/bin/python; do
  command -v "$py" >/dev/null 2>&1 || { echo "[$py] 不存在"; continue; }
  echo "[$py] $("$py" -V 2>&1)"
  for m in numpy pandas scipy sklearn openpyxl matplotlib seaborn torch; do
    v=$("$py" -c "import $m; print(getattr($m, '__version__', ''))" 2>/dev/null) && echo "   有 $m $v" || echo "   缺 $m"
  done
done
echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)｜负载 $(cut -d' ' -f1-3 /proc/loadavg)｜内存 $(free -g | awk '/Mem/{print $7"G 可用"}')"
ps -eo pcpu,etime,comm --sort=-pcpu | head -5
EOF
  ;;
setup)
  # 先把本脚本同步上去（安装逻辑在下面的 _install 分支，由服务器上的这份脚本执行）
  rsync -az --timeout=180 -e "$RSH" 02code/105_server_c6_physfix.sh "$HOST:$REMOTE/02code/" || exit 1
  "${SSH[@]}" "bash -s -- '$REMOTE' '$PY'" <<'EOF'
cd "$1" || exit 1
find . -name '._*' -type f -delete
mkdir -p 05logs
if [[ -s 05logs/105_install.pid ]] && kill -0 "$(cat 05logs/105_install.pid)" 2>/dev/null; then
  echo "安装已在进行（pid $(cat 05logs/105_install.pid)）"; tail -3 05logs/105_install.log; exit 0
fi
setsid nohup bash 02code/105_server_c6_physfix.sh _install "$2" > 05logs/105_install.log 2>&1 < /dev/null &
echo $! > 05logs/105_install.pid
sleep 30
echo "后台安装已起（pid $(cat 05logs/105_install.pid)），日志末行："
tail -4 05logs/105_install.log
EOF
  ;;
_install)
  # 只在服务器上由 setup 拉起。torch 取 CPU 轮子：本项目一律 --device cpu，GPU 版约 2.5 GB 用不上；
  # 三个源依次试，机房出网受限时不至于卡死在第一个。
  PYX="${1:-python}"
  say() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
  for idx in https://download.pytorch.org/whl/cpu - https://pypi.tuna.tsinghua.edu.cn/simple; do
    "$PYX" -c 'import torch' 2>/dev/null && break
    say "torch <- $idx"
    if [[ $idx == - ]]; then "$PYX" -m pip install --no-cache-dir --timeout 60 torch 2>&1 | tail -3
    else "$PYX" -m pip install --no-cache-dir --timeout 60 --index-url "$idx" torch 2>&1 | tail -3; fi
  done
  for idx in - https://pypi.tuna.tsinghua.edu.cn/simple; do
    "$PYX" -c 'import numpy,pandas,scipy,sklearn,openpyxl,matplotlib' 2>/dev/null && break
    say "科学栈 <- $idx"
    if [[ $idx == - ]]; then
      "$PYX" -m pip install --no-cache-dir --timeout 60 numpy pandas scipy scikit-learn openpyxl matplotlib 2>&1 | tail -3
    else
      "$PYX" -m pip install --no-cache-dir --timeout 60 -i "$idx" numpy pandas scipy scikit-learn openpyxl matplotlib 2>&1 | tail -3
    fi
  done
  "$PYX" -c "import torch,numpy,pandas,scipy,sklearn,openpyxl,matplotlib
print('INSTALL_OK torch', torch.__version__, '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| scipy', scipy.__version__, '| sklearn', sklearn.__version__)" 2>&1 | tail -1
  ;;
smoke)
  # 网关会掐断长连接（前台跑 200 轮的冒烟实测被断开），所以冒烟也用 setsid 脱离会话；
  # 结果写远端 05logs/105_smoke.log，status 会带出。
  "${SSH[@]}" "bash -s -- '$REMOTE' '$PY'" <<'EOF'
cd "$1" || exit 1
mkdir -p 05logs
if ps -eo args | grep -q "[5]0_formal_multiseed_benchmark.py.*ssc015_smoke"; then echo "冒烟已在跑"; exit 0; fi
T=$(mktemp -d /tmp/ssc015_smoke.XXXXXX)
setsid nohup bash -c '
  t0=$(date +%s)
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PROJECT_DATA_DIR="$1/c6data/p0" \
    "$2" 02code/50_formal_multiseed_benchmark.py --device cpu --epochs 200 --seed 2023 \
    --shard_id 0 --n_shards 1 --max_scenarios 1 --methods Phys,Phys+BL --shard_dir "$3" > "$3/smoke.log" 2>&1
  rc=$?
  echo "SMOKE exit=$rc 用时 $(( $(date +%s) - t0 )) 秒（1 个场景 × 2 个方法 × 200 轮）"
  tail -3 "$3/smoke.log"
  cut -c1-200 "$3"/seed*_shard*.csv 2>/dev/null
' _ "$1" "$2" "$T" > 05logs/105_smoke.log 2>&1 < /dev/null &
echo "冒烟已在后台拉起（输出目录 ${T}），结果见远端 05logs/105_smoke.log"
EOF
  ;;
start)
  P="${1:-15}"
  "${SSH[@]}" "bash -s -- '$REMOTE' '$PY' '$P'" <<'EOF'
cd "$1" || exit 1
mkdir -p 05logs
if [[ -s 05logs/104.pgid ]] && kill -0 -- -"$(cat 05logs/104.pgid)" 2>/dev/null; then
  echo "104 已在跑（进程组 $(cat 05logs/104.pgid)）"; exit 0
fi
PYTHON="$2" LANG=C.UTF-8 C6_DATA_ROOT="$1/c6data" \
  setsid nohup bash 02code/104_c6_physfix_rerun_local.sh "$3" >> 05logs/104_c6_physfix_rerun_server.log 2>&1 < /dev/null &
pid=$!
sleep 8
ps -o pgid= -p "$pid" | tr -d ' ' > 05logs/104.pgid
echo "已拉起 104：进程组 $(cat 05logs/104.pgid)，并发 $3"
tail -6 05logs/104_c6_physfix_rerun_server.log
EOF
  ;;
status)
  "${SSH[@]}" "bash -s -- '$REMOTE'" <<'EOF'
cd "$1" || exit 1
L=05logs/104_c6_physfix_rerun_server.log
p0=$(ls 04outputs/c6_shards_physfix_p0/seed*_shard*.csv 2>/dev/null | wc -l)
p15=$(ls 04outputs/c6_shards_physfix_p15/seed*_shard*.csv 2>/dev/null | wc -l)
f=$(grep -c "^!! 失败" "$L" 2>/dev/null)
pg=$(cat 05logs/104.pgid 2>/dev/null)
alive=否; [[ -n "$pg" ]] && kill -0 -- -"$pg" 2>/dev/null && alive=是
echo "服务器 $(date '+%m-%d %H:%M')｜p0 ${p0}/120｜p15 ${p15}/120｜失败 ${f:-0}｜仍在跑 ${alive}｜负载 $(cut -d' ' -f1 /proc/loadavg)"
[[ -z "$pg" && -s 05logs/105_install.log ]] && echo "安装日志末行：$(tail -1 05logs/105_install.log)"
[[ -z "$pg" && -s 05logs/105_smoke.log ]] && { echo "冒烟："; cat 05logs/105_smoke.log; }
exit 0
EOF
  ;;
stop)
  "${SSH[@]}" "bash -s -- '$REMOTE'" <<'EOF'
cd "$1" || exit 1
pg=$(cat 05logs/104.pgid 2>/dev/null)
if [[ -n "$pg" ]] && kill -- -"$pg" 2>/dev/null; then echo "已停进程组 $pg"; else echo "没有在跑的 104"; fi
EOF
  ;;
fetch)
  for p in 0 15; do
    mkdir -p "04outputs/c6_shards_physfix_p$p"
    rsync -az --timeout=180 --exclude='._*' -e "$RSH" "$HOST:$REMOTE/04outputs/c6_shards_physfix_p$p/" "04outputs/c6_shards_physfix_p$p/" || exit 1
  done
  mkdir -p 05logs/server_c6
  rsync -az --timeout=180 --exclude='._*' -e "$RSH" "$HOST:$REMOTE/05logs/" 05logs/server_c6/ || exit 1
  for p in 0 15; do
    echo "p${p}：本机现有 $(ls "04outputs/c6_shards_physfix_p${p}"/seed*_shard*.csv 2>/dev/null | wc -l | tr -d ' ')/120 个分片"
  done
  ;;
*)
  echo "用法: bash 02code/105_server_c6_physfix.sh {deploy <C6数据目录>|check|setup|smoke|start [并发]|status|stop|fetch}"
  exit 1
  ;;
esac
