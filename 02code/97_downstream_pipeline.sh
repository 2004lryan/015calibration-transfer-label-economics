#!/usr/bin/env bash
# 97_downstream_pipeline.sh — 分片回收之后的下游分析链，一条命令跑完
#
# 前置：04outputs/62_crossover_engine.xlsx、04outputs/64_deep_transfer_server.xlsx、
#       04outputs/50_formal_multiseed_benchmark.xlsx 三份合并结果都已就位。
# 顺序取自 refine-logs/26深度臂训练缺陷与案例研究数据回溯.md 第五节，任一步失败即停——
# 半截结果混进 04outputs 会让后面的表和图各读到一个版本的数字。
#
# 用法:  bash 02code/97_downstream_pipeline.sh [unified|case|all]
set -uo pipefail
WHICH="${1:-all}"
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE"
mkdir -p 05logs
PY="${PYTHON:-python3}"

run() {
  local script="$1"; shift
  # 同一个脚本按参数跑多遍时（88 号按基准拆开），日志不能都写同一个文件——
  # 后一份会把前一份的报错盖掉，失败时 tail 出来的是别的基准的尾巴。
  local stem="${script%.py}${LOG_SUFFIX:-}"
  local log="05logs/97_${stem}.out"
  printf '  ▶ %-46s ' "$script"
  local t0=$SECONDS
  if "$PY" "02code/$script" "$@" > "$log" 2>&1; then
    printf '完成 %4ds\n' "$((SECONDS - t0))"
  else
    printf '失败 %4ds  → %s\n' "$((SECONDS - t0))" "$log"
    tail -20 "$log"
    exit 1
  fi
}

need() {
  for f in "$@"; do
    [[ -s "$f" ]] || { echo "缺前置文件：$f"; exit 1; }
  done
}

if [[ "$WHICH" == "unified" || "$WHICH" == "all" ]]; then
  echo "=== 统一基准线 ==="
  need 04outputs/62_crossover_engine.xlsx 04outputs/64_deep_transfer_server.xlsx
  run 63_crossover_analysis.py
  run 65_paper_analysis.py
  run 67_winners_curse_reanalysis.py
  run 68_normalized_hierarchical_reanalysis.py
  run 73_heuristic_map_lobo.py
  run 74_shift_structure_diagnosis.py
  run 75_nrmsep_table_columns.py
  run 86_selection_free_effect_sizes.py
  run 87_prescription_paired_and_deep_sanity.py
  # 88/89 在服务器上与分片同池跑过，本机只在缺输出时补跑。
  # 88 号按基准一份 xlsx——服务器上是 5 个进程并行写的，本机补跑串行写，文件名一样。
  for b in corn tablet mango ossl_mir apple; do
    [[ -s "04outputs/88_deep_indomain_sanity_${b}.xlsx" ]] && continue
    LOG_SUFFIX="_${b}"
    run 88_deep_indomain_sanity.py --device cpu --benchmarks "$b" \
        --out "04outputs/88_deep_indomain_sanity_${b}.xlsx"
  done
  LOG_SUFFIX=""
  [[ -s 04outputs/89_source_reference_baselines.xlsx ]] || run 89_source_reference_baselines.py
  run 80_source_sanity_table.py
  run 94_method_wise_table.py
  echo "  --- 图 ---"
  run 66_paper_figures.py
  run 76_revision_figures.py
  run 77_framework_figure.py
  run 78_map_and_regret_figure.py
fi

if [[ "$WHICH" == "case" || "$WHICH" == "all" ]]; then
  echo "=== 案例研究线 ==="
  need 04outputs/50_formal_multiseed_benchmark.xlsx
  run 56_post_experiment_pipeline.py
fi

echo "下游分析链跑完。接下来是改稿口径数字（26 号记录第六节）。"
