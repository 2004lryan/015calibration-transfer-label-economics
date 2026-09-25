#!/usr/bin/env bash
# 95_server_run_all.sh — 在多核服务器上一次跑完「案例研究 120 分片 + 深度臂 25 分片
#                        + 严格零标签对照 24 分片」
#
# 为什么单独写一份：91/92 是 macOS 版（`stat -f %m`、xargs 无 -a），且本机只能开 4 个进程。
# 本脚本把两类作业混进同一个进程池，按核数一次填满，跑完自动合并。
#
# 运行方式（服务器上）:
#   bash 02code/95_server_run_all.sh [并发进程数]
#   例: bash 02code/95_server_run_all.sh 56
#
# 纪律：
#   · 一律 --device cpu。模型极小、GPU 利用率不足 10%，且 56 个进程各占一个 CUDA context
#     会把 24 GB 显存撑爆；正典的 120 个分片本来也是 CPU 跑的（meta.json: device=cpu）。
#   · 单线程 BLAS + nice 10，机器上另有他人项目，不抢。
#   · 已存在且非空的分片自动跳过，可断点续跑。
#
# 输出:
#   04outputs/50_shards/seed<S>_shard<NN>.csv    — 案例研究分片（n_shards=24，与正典一致）
#   04outputs/64_shards/<基准>_<种子>.csv        — 深度臂分片
#   04outputs/59_shards/seed42_shard<NN>.csv     — 严格零标签 + 经典预处理对照分片
#   05logs/…                                     — 每个分片一份日志
set -uo pipefail

# 默认 16 而不是核数的一半：这台机器 `nproc --all` 报 128，但容器的 cgroup 配额是
# cpu.cfs_quota_us/period = 1500000/100000 = **15 个 CPU**。按 56 并发跑过一轮，每个进程
# 只拿到 27.5% 的核（56×0.275≈15.4，正好顶在配额上），总吞吐一样但没有一个分片能先跑完，
# 中途出事就全丢。并发贴着配额走，分片才会陆续落盘、可断点续跑。
PJOBS="${1:-16}"
EPOCHS=200
N_SHARDS=24
ZERO_SEED=42          # 59 号严格零标签对照自己的协议种子，与 Formal 的 5 粒无关
SEEDS=(20060515 20041210 19810915 2023 2024)
BENCHES=(corn tablet mango ossl_mir apple)

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# 解释器由调用方指定：服务器基础镜像的 /usr/local/bin/python 是空的，装了依赖的那个
# 未必叫 python，也未必在 PATH 上（见 98 号探测逻辑）。
export PYTHON="${PYTHON:-python}"
mkdir -p 04outputs/50_shards 04outputs/59_shards 04outputs/64_shards 05logs

JOBS=$(mktemp)
# 源域阳性对照排最前：它跑得最久，早开始才不会拖住收尾。与分片互不依赖（直接读
# benchmarks 缓存），所以混在同一个池子里并行跑。
# 88 号按基准拆成 5 个作业：它内部单进程单线程，整体排一个作业就只吃得到一个核，
# 五个基准串起来是全池最长的一条。五个基准之间没有共享状态，拆开不改变任何数字，
# 只是各写各的 xlsx，由 80 号合并。
: > "$JOBS"
for b in "${BENCHES[@]}"; do
  [[ -s "04outputs/88_deep_indomain_sanity_${b}.xlsx" ]] || echo "aux 88_${b} -"
done >> "$JOBS"
[[ -s "04outputs/89_source_reference_baselines.xlsx" ]] || echo "aux 89 -" >> "$JOBS"
# 深度臂随后：它只有 25 个作业且是后续全部下游分析的前置，先把它喂进池子
for b in "${BENCHES[@]}"; do
  for s in "${SEEDS[@]}"; do
    f="04outputs/64_shards/${b}_${s}.csv"
    [[ -s "$f" ]] || echo "deep $b $s"
  done
done >> "$JOBS"
# 严格零标签对照（59 号）：与 50 号同一套场景、同一份数据，但深度模型改用源域留出集早停，
# 另附四条经典预处理基线。单种子 24 分片，比案例分片短，排在它前面先落盘。
for ((i=0; i<N_SHARDS; i++)); do
  f="04outputs/59_shards/seed${ZERO_SEED}_shard$(printf '%02d' "$i").csv"
  [[ -s "$f" ]] || echo "zero $i -"
done >> "$JOBS"
for s in "${SEEDS[@]}"; do
  for ((i=0; i<N_SHARDS; i++)); do
    f="04outputs/50_shards/seed${s}_shard$(printf '%02d' "$i").csv"
    [[ -s "$f" ]] || echo "case $s $i"
  done
done >> "$JOBS"

TOTAL=$(wc -l < "$JOBS" | tr -d ' ')
echo "=========================================================="
echo " 服务器全量重跑"
# nproc 认 OMP_NUM_THREADS（上面刚把它设成 1），要看真实核数得用 --all
# 容器重启后这台机器换成了 cgroup v2：v1 的 cpu/cpu.cfs_quota_us 整个目录都不在了，
# 配额写在 cpu.max 的一行里（"<配额> <周期>"，未设时配额是字面量 max）。两种都认，
# 认不出就说未设——这一行只是显示，判断并发靠的是调用方传进来的 PJOBS。
CPUQ="未设"
if [[ -r /sys/fs/cgroup/cpu.max ]]; then
  read -r _q _p < /sys/fs/cgroup/cpu.max
  [[ "$_q" == "max" ]] || CPUQ="$(( _q / _p ))"
elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us && -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]]; then
  _q="$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)"
  _p="$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)"
  [[ "$_q" -le 0 ]] || CPUQ="$(( _q / _p ))"
fi
echo "   核数     : $(nproc --all)（cgroup 配额 ${CPUQ}）   并发: ${PJOBS}"
echo "   待跑作业 : ${TOTAL}（源域阳性对照 ≤6 + 深度臂 ≤25 + 零标签对照 ≤24 + 案例研究 ≤120，已完成的跳过）"
echo "=========================================================="
T0=$(date +%s)

xargs -a "$JOBS" -P "$PJOBS" -L1 bash -c '
  kind=$0
  if [[ "$kind" == "aux" ]]; then
    n=$1
    if [[ "$n" == 88_* ]]; then
      b="${n#88_}"
      cmd=("$PYTHON" 02code/88_deep_indomain_sanity.py --device cpu --benchmarks "$b" \
           --out "04outputs/88_deep_indomain_sanity_${b}.xlsx")
    else
      cmd=("$PYTHON" 02code/89_source_reference_baselines.py)
    fi
    nice -n 10 "${cmd[@]}" > "05logs/${n}_aux.out" 2>&1       && echo "   完成 aux ${n}" || echo "!! 失败 aux ${n}"
  elif [[ "$kind" == "zero" ]]; then
    i=$1
    nice -n 10 "$PYTHON" 02code/59_reviewer_experiments.py \
      --device cpu --zero_epochs '"$EPOCHS"' --seed '"$ZERO_SEED"' --shard_id "$i" --n_shards '"$N_SHARDS"' \
      > "05logs/59_shard_${i}.out" 2>&1 \
      && echo "   完成 zero ${i}" || echo "!! 失败 zero ${i}"
  elif [[ "$kind" == "deep" ]]; then
    b=$1; s=$2
    nice -n 10 "$PYTHON" 02code/64_deep_transfer_server.py \
      --benchmarks "$b" --seeds "$s" --rep 1 --device cpu --tag "${b}_${s}" \
      > "05logs/64_${b}_${s}.out" 2>&1 \
      && echo "   完成 deep ${b} ${s}" || echo "!! 失败 deep ${b} ${s}"
  else
    s=$1; i=$2
    nice -n 10 "$PYTHON" 02code/50_formal_multiseed_benchmark.py \
      --device cpu --epochs '"$EPOCHS"' --seed "$s" --shard_id "$i" --n_shards '"$N_SHARDS"' \
      > "05logs/50_shard_${s}_${i}.out" 2>&1 \
      && echo "   完成 case ${s} ${i}" || echo "!! 失败 case ${s} ${i}"
  fi
'
rm -f "$JOBS"
echo "----------------------------------------------------------"
echo "用时 $(( ($(date +%s) - T0) / 60 )) 分钟"

MISSING=0
for b in "${BENCHES[@]}"; do for s in "${SEEDS[@]}"; do
  [[ -s "04outputs/64_shards/${b}_${s}.csv" ]] || { echo "缺深度分片 ${b}_${s}"; MISSING=1; }
done; done
for ((i=0; i<N_SHARDS; i++)); do
  [[ -s "04outputs/59_shards/seed${ZERO_SEED}_shard$(printf '%02d' "$i").csv" ]] || { echo "缺零标签分片 ${i}"; MISSING=1; }
done
for s in "${SEEDS[@]}"; do for ((i=0; i<N_SHARDS; i++)); do
  [[ -s "04outputs/50_shards/seed${s}_shard$(printf '%02d' "$i").csv" ]] || { echo "缺案例分片 ${s}_${i}"; MISSING=1; }
done; done
if [[ "$MISSING" -ne 0 ]]; then echo "分片不全，不合并。"; exit 1; fi

echo "全部分片就绪，开始合并"
"$PYTHON" 02code/64_deep_transfer_server.py --merge          > 05logs/64_merge.out 2>&1 && echo "  深度臂已合并"
"$PYTHON" 02code/50_formal_multiseed_benchmark.py --merge    > 05logs/50_merge.out 2>&1 && echo "  案例研究已合并"
"$PYTHON" 02code/59_reviewer_experiments.py --merge          > 05logs/59_merge.out 2>&1 && echo "  零标签对照已合并"
for aux in 04outputs/88_deep_indomain_sanity_*.xlsx 04outputs/89_source_reference_baselines.xlsx; do
  [[ -s "$aux" ]] && echo "  ${aux##*/} 就绪" || echo "  ！${aux##*/} 缺失，见 05logs/*_aux.out"
done
echo "完成。把 04outputs/ 取回本机继续下游分析。"
