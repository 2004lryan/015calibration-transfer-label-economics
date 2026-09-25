#!/usr/bin/env bash
# 100_server_budget_sensitivity.sh — 在服务器上按「统一放大的优化预算」重跑源域阳性对照
#
# 要回答的问题只有一个：S14 里深度族在苹果源域内只略低于恒定预测器，是固定 2000 步
# 优化预算造成的，还是这批数据本身的性质？把预算统一放大到 8000 步、五个基准一视同仁地
# 重跑一遍，就能把两者分开。
#
# 纪律（与 95 号同源）：
#   · 预算放大必须**对五个基准同时生效**。只给苹果加预算＝逐基准调参，会破坏正文
#     「各方法按原文通用配方、同一固定预算、不逐基准调参」的协议，比原来的问题更糟。
#   · 一律 --device cpu，单线程 BLAS；容器 cgroup 配额是 15 个 CPU，五个进程各占一个。
#   · 各基准各写各的 xlsx，互不共享状态；已存在且非空的输出自动跳过，可断点续跑。
#
# 用法:
#   SSHPASS='<密码>' bash 02code/100_server_budget_sensitivity.sh <用户@主机> <端口> <远端目录> start [步数]
#   SSHPASS='<密码>' bash 02code/100_server_budget_sensitivity.sh <用户@主机> <端口> <远端目录> status
#   SSHPASS='<密码>' bash 02code/100_server_budget_sensitivity.sh <用户@主机> <端口> <远端目录> fetch <本地目录>
set -uo pipefail
HOSTSPEC="${1:?用法见脚本头}"
PORT="${2:?}"
REMOTE="${3:?}"
MODE="${4:-status}"
STEPS="${5:-8000}"
# 基础镜像的 /usr/local/bin/python 是空壳，装了依赖的解释器在 conda 里且不在非交互 PATH 上
PY="${PYTHON:-/opt/conda/bin/python3}"
SSH="sshpass -e ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -p ${PORT}"

case "$MODE" in
  start)
    BODY="
cd '${REMOTE}' || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p 05logs/budget${STEPS}
for b in corn tablet mango ossl_mir apple; do
  out=\"04outputs/88_deep_indomain_sanity_steps${STEPS}_\${b}.xlsx\"
  if [ -s \"\$out\" ]; then echo \"  \$b 已有输出，跳过\"; continue; fi
  setsid nohup nice -n 10 ${PY} 02code/88_deep_indomain_sanity.py --device cpu \\
    --benchmarks \"\$b\" --train-steps ${STEPS} --out \"\$out\" \\
    > \"05logs/budget${STEPS}/\${b}.log\" 2>&1 &
  echo \"  已起 \$b pid \$!\"
done
sleep 5
echo \"  在跑 \$(pgrep -fc 88_deep_indomain_sanity) 个进程\"
"
    ;;
  status)
    BODY="
cd '${REMOTE}' || exit 1
echo \"  在跑 \$(pgrep -fc 88_deep_indomain_sanity || echo 0) 个进程\"
ls -l 04outputs/88_deep_indomain_sanity_steps${STEPS}_*.xlsx 2>/dev/null | awk '{print \"  \"\$5\" \"\$9}'
for f in 05logs/budget${STEPS}/*.log; do [ -e \"\$f\" ] && echo \"  == \$f\" && tail -2 \"\$f\"; done
"
    ;;
  fetch)
    DST="${6:-04outputs}"
    exec rsync -az --timeout=180 -e "$SSH" \
      "${HOSTSPEC}:${REMOTE}/04outputs/88_deep_indomain_sanity_steps${STEPS}_*.xlsx" "$DST/"
    ;;
  *) echo "未知模式 $MODE"; exit 2 ;;
esac

exec $SSH "$HOSTSPEC" "bash -s" <<< "$BODY"
