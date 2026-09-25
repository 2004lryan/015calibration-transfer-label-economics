#!/usr/bin/env bash
# 98_server_verify_and_launch.sh — 拉起服务器端 bootstrap，或查看它的进度
#
# 网关会掐掉长连接（pip 装 torch 装到一半就 "Connection closed by remote host"），
# 所以真正干活的 93 号用 setsid nohup 脱离终端跑，本脚本只负责「点火」和「看火」。
# 又因为网关有 fail2ban，连续短连接会把自己锁出去，所以看进度别连着敲。
#
# 用法:
#   SSHPASS='<密码>' bash 02code/98_server_verify_and_launch.sh <用户@主机> <端口> <远端目录> start [并发数]
#   SSHPASS='<密码>' bash 02code/98_server_verify_and_launch.sh <用户@主机> <端口> <远端目录> status
set -uo pipefail
HOSTSPEC="${1:?用法见脚本头}"
PORT="${2:?}"
REMOTE="${3:?}"
MODE="${4:-status}"
PJOBS="${5:-56}"

if [ "$MODE" = "stop" ]; then
  # 按进程组停：93 号由 setsid 拉起，是组长，kill -- -PGID 连 xargs 与全部 worker 一起收。
  BODY='
p="$(cat 05logs/93.pid 2>/dev/null)"
if [ -z "$p" ]; then
  echo "  没有 05logs/93.pid，可能没在跑，或是记 pid 之前的旧版本起的。"
else
  kill -TERM -- "-$p" 2>/dev/null
  sleep 5
  kill -KILL -- "-$p" 2>/dev/null
  rm -f 05logs/93.pid
  echo "  已停（进程组 $p）"
fi'
elif [ "$MODE" = "start" ]; then
  # 重复点火由 93 号自己的文件锁挡住，这里只管点。
  BODY='
mkdir -p 05logs
setsid nohup bash 02code/93_server_bootstrap.sh "$PJOBS" >> 05logs/93_bootstrap.log 2>&1 < /dev/null &
echo "  已点火（pid $!），日志 05logs/93_bootstrap.log"
sleep 12
tail -15 05logs/93_bootstrap.log 2>/dev/null'
else
  BODY='
echo "--- 进程 ---"
# bootstrap 是否在跑用文件锁判，不用 pgrep：本条命令行里就写着脚本名和日志名，
# pgrep -f 一定会匹配到自己。
printf "  bootstrap %s ｜ worker %s ｜ pip %s\n" \
  "$(flock -n 05logs/93.lock true 2>/dev/null && echo 已结束 || echo 在跑)" \
  "$(pgrep -cf "[5]0_formal_multiseed|[6]4_deep_transfer_server|[8]8_deep_indomain|[8]9_source_reference")" \
  "$(pgrep -cf "[p]ip install")"
echo "--- 分片 ---"
printf "  案例研究 %s/120 ｜ 深度臂 %s/25 ｜ 根盘 %s\n" \
  "$(ls 04outputs/50_shards/*.csv 2>/dev/null | wc -l)" \
  "$(ls 04outputs/64_shards/*.csv 2>/dev/null | wc -l)" \
  "$(df -h / | awk "NR==2{print \$4\" 可用\"}")"
echo "--- 日志尾 ---"
tail -20 05logs/93_bootstrap.log 2>/dev/null'
fi

exec sshpass -e ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 \
  -p "$PORT" "$HOSTSPEC" "cd '$REMOTE' || exit 1; PJOBS='$PJOBS'; $BODY"
