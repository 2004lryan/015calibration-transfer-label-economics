#!/usr/bin/env bash
# 93_server_bootstrap.sh — 在服务器上「装依赖 → 验依赖 → 起全量重跑」，全程脱离终端
#
# 为什么单独一支：装 torch 要几分钟，而网关会掐掉长连接（实测 pip 装到一半就
# "Connection closed by remote host"），跟着 SSH 会话跑的进程会一起没。本脚本由 98 号用
# setsid nohup 拉起，断线不影响；进度全部落在 05logs/93_bootstrap.log，随时另开连接看。
#
# 用法（服务器上，一般由 98 号代劳）:  bash 02code/93_server_bootstrap.sh [并发数]
set -u
PJOBS="${1:-56}"
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE"
mkdir -p 05logs
NEED='numpy pandas scipy sklearn torch openpyxl matplotlib'

# 「是否已经在跑」的判断放在这里而不是调用方：pgrep -f 扫的是整条命令行，而调用方的
# 命令行里必然写着本脚本的名字，怎么写都会自己匹配自己。文件锁没有这个问题。
exec 9> 05logs/93.lock
if ! flock -n 9; then
  echo "[$(date '+%H:%M:%S')] 已有一个 93 在跑，本次退出。"
  exit 0
fi
# 本脚本由 setsid 拉起，因而是进程组组长：把 pid 记下来，停机时 kill -- -PID
# 就能连同 xargs 和所有 worker 一起收掉，不必用 pkill -f 去猜（那必然自匹配）。
echo $$ > 05logs/93.pid
trap 'rm -f 05logs/93.pid' EXIT

say() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

probe() {   # $1=解释器。打印 "PROBE <缺失包...>"；解释器跑不起来则什么都不打印。
  "$1" - "$NEED" <<'PY' 2>/dev/null
import sys
miss = []
for m in sys.argv[1].split():
    try:
        __import__(m)
    except Exception:
        miss.append(m)
# 齐全时必须正好输出 "PROBE"：print("PROBE", "") 会多出一个尾空格，
# 调用方的 case 分支就永远落到「缺（空）」那一支上。
print("PROBE" + ("".join(" " + m for m in miss)))
PY
}

say "=== 找解释器 ==="
CANDS="$(command -v python || true) $(command -v python3 || true)"
CANDS="$CANDS /usr/local/bin/python /usr/bin/python3 /opt/conda/bin/python /root/miniconda3/bin/python"
if command -v conda > /dev/null 2>&1; then
  for e in $(conda env list 2>/dev/null | awk '!/^#/ && NF>1 {print $NF}'); do
    CANDS="$CANDS $e/bin/python"
  done
fi
PYBIN=""; FALLBACK=""; SEEN=""
for c in $CANDS; do
  [ -x "$c" ] || continue
  case " $SEEN " in *" $c "*) continue ;; esac
  SEEN="$SEEN $c"
  out="$(probe "$c")"
  case "$out" in
    PROBE) say "  ✅ $c 依赖齐全"; PYBIN="$c"; break ;;
    "PROBE "*) say "  ✗ $c 缺: ${out#PROBE }"; [ -n "$FALLBACK" ] || FALLBACK="$c" ;;
    *) say "  ✗ $c 探测失败（解释器跑不起来）" ;;
  esac
done

if [ -z "$PYBIN" ]; then
  [ -n "$FALLBACK" ] || { say "!! 一个可用解释器都没有，停"; exit 1; }
  # 只有确实要自己装时才等别人的 pip：两个 pip 同时写 site-packages 会把环境写坏
  #（网关掐断长连接时上一次的 pip 会活下来），但这台机器上还住着别的项目，它们的
  # pip 装在自己的 conda 环境里，跟我们没有冲突——把这个等待放在脚本开头，等于
  # 依赖齐全时也白等最多一小时。
  for _ in $(seq 1 120); do
    pgrep -f 'pip install' > /dev/null || break
    say "  等一个仍在跑的 pip install…"
    sleep 30
  done
  say "=== 装依赖到 $FALLBACK ==="
  # torch 取 CPU 轮子：GPU 版约 2.5 GB，而本项目一律 --device cpu，装它纯属占根盘。
  # 三个源依次试，机房出网受限时不至于卡死在第一个。
  for idx in https://download.pytorch.org/whl/cpu - https://pypi.tuna.tsinghua.edu.cn/simple; do
    say "  torch ← ${idx}"
    if [ "$idx" = "-" ]; then
      "$FALLBACK" -m pip install --no-cache-dir torch 2>&1 | tail -4
    else
      "$FALLBACK" -m pip install --no-cache-dir --index-url "$idx" torch 2>&1 | tail -4
    fi
    "$FALLBACK" -c 'import torch' 2>/dev/null && { say "  torch 就绪"; break; }
  done
  for idx in - https://pypi.tuna.tsinghua.edu.cn/simple; do
    say "  科学栈 ← ${idx}"
    if [ "$idx" = "-" ]; then
      "$FALLBACK" -m pip install --no-cache-dir numpy pandas scipy scikit-learn \
        openpyxl matplotlib 2>&1 | tail -4
    else
      "$FALLBACK" -m pip install --no-cache-dir -i "$idx" numpy pandas scipy scikit-learn \
        openpyxl matplotlib 2>&1 | tail -4
    fi
    "$FALLBACK" -c 'import numpy,pandas,scipy,sklearn,openpyxl,matplotlib' 2>/dev/null \
      && { say "  科学栈就绪"; break; }
  done
  out="$(probe "$FALLBACK")"
  case "$out" in
    PROBE) say "  ✅ 装完齐全"; PYBIN="$FALLBACK" ;;
    *) say "!! 装完仍缺: ${out#PROBE } —— 不启动"; exit 2 ;;
  esac
fi

"$PYBIN" -c "import torch,sklearn,pandas;
print('  torch',torch.__version__,'| sklearn',sklearn.__version__,'| pandas',pandas.__version__)"
say "=== 起全量重跑（并发 ${PJOBS}）==="
PYTHON="$PYBIN" bash 02code/95_server_run_all.sh "$PJOBS"
say "=== 95 号退出，码 $? ==="
