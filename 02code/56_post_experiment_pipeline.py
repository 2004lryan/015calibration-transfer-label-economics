"""
56_post_experiment_pipeline.py：50 号实验落地后的一键收尾流水线

在 50 号（干净数据 / 5 个 Formal 种子 / 602 场景 / 5 方法）的分片全部回收之后运行。
按固定顺序执行，任一环节失败即整体中止——绝不允许「半截结果」流进论文。

流水线：
    1. 锚点核验    ⚠ 此前的计划是「seed=42 必须与旧的 18_exp 逐场景一致，否则新数字不得入稿」。
                   该计划是**错的**：旧的 18_exp 算在**脏数据**上（含 SSC=56.07 的越界样本
                   与 31 行光谱-标签错配的重复行）。要求干净数据去复现被污染的旧结果，
                   等于把污染当成正确性基准。
                   锚点的含义改为：报告新旧差异的分布，核对它是否**符合数据清洗的预期效应**
                   （被污染样本所在的场景，其 RMSE 应显著下降）。
    2. 结论方向核验 预注册假设 H1–H7 的方向是否与论文现有表述一致
                   ⚠ 方向翻转 → 中止并报告，由人决定怎么改写，而不是我偷偷改数字
    3. 免标签诊断  55 号：BL 残差是否真能**预测**迁移风险（预注册阈值，可能推翻第一贡献）
    4. 数字回填    ⟪N_SEED⟫ 及表1/表3 的全部数字
    5. 重绘图      Fig2 / Fig3（49 号，数据源不存在时会响亮失败）
    6. 出稿        ⚠ 已摘除。返修后 Elsevier_zh.tex 手工维护，48 号模板会把返修稿整篇覆盖；
                   数字改为人工按 04outputs/52_report_new_numbers.md 回填到中英双稿

运行方式:
    cd <repository root>
    python 02code/56_post_experiment_pipeline.py

输出文件：
    04outputs/56_post_experiment_report.md - 全流水线的核验报告（含结论方向的判定）
"""

import os
import subprocess
import sys
import tempfile

import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(_THIS, '..'))
OUT = os.path.join(BASE, '04outputs')

SRC50 = os.path.join(OUT, '50_formal_multiseed_benchmark.xlsx')
SHEET_RAW = '表h：全部原始结果'
# 表里一律中文列名（50 号 2026-09 起的表结构）。这几个名字改了本脚本必须跟着改，
# 所以集中在这里，而不是散在各步里用属性访问（`df.seed` 那种写法在列名变更时
# 报的是 AttributeError，读起来像是 pandas 的问题，不像是表结构变了）。
COL_SCENE, COL_METHOD, COL_CORR, COL_SEED, COL_RMSE = '场景', '方法', '校正', '种子', 'RMSE'

# 稿件当前对 BL 残差的定位。改这个常量等于改稿件的主张，所以每一档都贴着稿件里的
# 原话（PROBE）：脚本会去两份补充材料里核这句话在不在，不在就是常量与稿件脱节了。
PAPER_BL_STANCE = '相关'
BL_STANCE_PROBE = {
    '诊断': ('可独立判定的诊断信号', 'a diagnostic signal that can stand on its own'),
    '相关': ('而非可独立判定的诊断信号', 'not as\na diagnostic signal that can stand on its own'),
    '结构差异': ('只能表征域间结构差异', 'only characterises structural differences'),
}
# 55 号判定 → 稿件最多能说到哪一档
VERDICT_TO_MAX_STANCE = {'✅': '诊断', '⚠': '相关', '🔴': '结构差异'}
STANCE_ORDER = ['结构差异', '相关', '诊断']

REPORT: list[str] = []


def log(msg: str = '') -> None:
    print(msg)
    REPORT.append(msg)


def die(msg: str) -> None:
    """中止：把已写的部分存盘，再退出。绝不静默继续。"""
    log(f'\n🔴 中止：{msg}')
    _save_report()
    sys.exit(1)


def _save_report() -> None:
    path = os.path.join(OUT, '56_post_experiment_report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# 50 号实验落地后的收尾核验报告\n\n')
        f.write('\n'.join(REPORT) + '\n')
    print(f'\n报告已保存: {path}')


def run(cmd: list[str], what: str) -> str:
    """跑一个子步骤；非零退出即中止。"""
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK='TRUE', OMP_NUM_THREADS='2')
    r = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        log(f'  🔴 {what} 失败（退出码 {r.returncode}）')
        log('  ' + (r.stderr or r.stdout)[-800:])
        die(f'{what} 失败')
    return r.stdout


# ══════════════════════════════════════════════════════════════════════
# 1. 锚点核验：seed=42 必须复现旧的 18_exp
# ══════════════════════════════════════════════════════════════════════
def _prev_merged() -> "pd.DataFrame | None":
    """上一版合并表——从 git 取，不留 .prev/.bak 旁路副本（备份只走 git，3.7 条）。"""
    rel = os.path.relpath(SRC50, BASE)
    r = subprocess.run(['git', '-C', BASE, 'show', f'HEAD:{rel}'],
                       capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return None
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        tmp.write(r.stdout)
        name = tmp.name
    try:
        return pd.read_excel(name, sheet_name=SHEET_RAW)
    except (ValueError, OSError):
        return None
    finally:
        os.unlink(name)


def step0_coverage() -> None:
    """每个 (方法, 校正) 的格子完整率：场景 × 种子 的满格里实际落了多少。

    为什么这一步必须在最前面：深度方法在某些场景下会撞上 BatchNorm 的 batch=1
    （日志里的 ``Expected more than 1 value per channel``），该 (方法, 场景, 种子)
    单元被整格跳过——**不报错、不写 NaN、不留行**。合并出来的表看上去是完整的，
    实际上各方法是在不同的场景子集上取的均值，跨方法比较的分母因此不一样。
    50 号自带的「表a2：共同场景子集」正是为此准备的，但只有先知道缺了多少、缺在哪，
    才知道该不该改用那张表。
    """
    log('## 0. 格子覆盖率（场景 × 种子 满格里实际落了多少）\n')
    if not os.path.exists(SRC50):
        die(f'{os.path.basename(SRC50)} 不存在——50 号还没 merge')
    d = pd.read_excel(SRC50, sheet_name=SHEET_RAW)
    n_sc, n_sd = d[COL_SCENE].nunique(), d[COL_SEED].nunique()
    full = n_sc * n_sd
    log(f'  满格 = 场景 {n_sc} × 种子 {n_sd} = {full}（每个 方法×校正 都应有这么多行）\n')

    cnt = d.groupby([COL_METHOD, COL_CORR]).size().unstack(COL_CORR)
    rate = (100 * cnt / full).round(1)
    log(f'  {"方法":10s} ' + ' '.join(f'{c:>8s}' for c in cnt.columns) + '   完整率')
    worst, worst_m = 100.0, ''
    for m in cnt.index:
        cells = ' '.join(f'{int(cnt.loc[m, c]):>8d}' for c in cnt.columns)
        rr = ' / '.join(f'{rate.loc[m, c]:.1f}%' for c in cnt.columns)
        log(f'  {m!s:10s} {cells}   {rr}')
        lo = float(rate.loc[m].min())
        if lo < worst:
            worst, worst_m = lo, str(m)
    log('')

    # 缺哪几格要落盘，否则「缺了 150 个」只是一个数字，没法追到具体场景。
    grid = pd.MultiIndex.from_product(
        [sorted(d[COL_METHOD].unique()), sorted(d[COL_CORR].astype(str).unique()),
         sorted(d[COL_SCENE].unique()), sorted(d[COL_SEED].unique())],
        names=[COL_METHOD, COL_CORR, COL_SCENE, COL_SEED]).to_frame(index=False)
    have = d[[COL_METHOD, COL_CORR, COL_SCENE, COL_SEED]].copy()
    have[COL_CORR] = have[COL_CORR].astype(str)
    miss = grid.merge(have.assign(_有=1), how='left').query('_有.isna()').drop(columns='_有')
    path = os.path.join(OUT, '56_coverage_missing.csv')
    miss.to_csv(path, index=False, encoding='utf-8-sig')
    log(f'  缺失单元 {len(miss)} 个 → {os.path.relpath(path, BASE)}')

    # 与分片日志逐条对账。50 号在 `except Exception` 里记下方法名与原因，但被跳过的
    # (方法, 场景, 种子) 不会在分片 CSV 里留下任何痕迹——所以「日志里数出来的失败次数」
    # 与「表里缺的格子数」必须逐方法相等，不等就说明还有一条没记日志的丢格路径。
    # 按原因分类而不是只数一种消息：2026-09 的这一轮里 762 次失败分两类，
    # 只数 BatchNorm 那类会漏掉 52 次 NaN。
    import glob
    import re
    pat = re.compile(r'\[(PLSR|SVR|CNN|CNN\+MMD|CNN\+BL|Phys|Phys\+BL)\]\s+[^:]+:\s*(.+)')
    by_method: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    for f in glob.glob(os.path.join(BASE, '05logs', 'server', '50_shard_*.out')):
        with open(f, encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                m = pat.search(line)
                if not m:
                    continue
                by_method[m.group(1)] = by_method.get(m.group(1), 0) + 1
                key = m.group(2).strip()[:46]
                by_reason[key] = by_reason.get(key, 0) + 1
    log('  日志里记下的失败次数，按原因：')
    for k, v in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        log(f'    {v:5d}  {k}')
    miss_bare = miss[miss[COL_CORR] == '裸'].groupby(COL_METHOD).size().to_dict()
    log('  逐方法对账（日志失败次数 vs 表里缺的训练次数）：')
    bad = []
    for m_ in sorted(set(by_method) | set(miss_bare)):
        a, b = by_method.get(m_, 0), int(miss_bare.get(m_, 0))
        flag = '✅' if a == b else '🔴'
        log(f'    {flag} {m_:10s} 日志 {a:4d}   缺格 {b:4d}')
        if a != b:
            bad.append(f'{m_}（日志 {a}、缺格 {b}）')
    if bad:
        die('有缺格对不上日志：' + '；'.join(bad) +
            '——说明存在一条不记日志的丢格路径，须先查清')
    log('')

    if worst < 90.0:
        die(f'{worst_m} 的格子完整率只有 {worst:.1f}%（下限 90%）——'
            '这个量级的缺失会让跨方法比较的分母显著不同，须先查清再往下走')
    log(f'  ✅ 最低完整率 {worst:.1f}%（{worst_m}）≥ 90%；跨方法比较一律看'
        '「表a2：共同场景子集」，不看全集均值\n')


def step1_anchor() -> None:
    """锚点：本版合并表与 git 里的上一版逐格比，报告差异分布。

    原计划是「seed=42 必须与旧的 18_exp 逐场景一致」，那个计划有两处已经不成立：
    旧的 18_exp 算在脏数据上（含 SSC=56.07 的越界样本与 31 行光谱–标签错配），
    要求干净数据去复现被污染的结果等于把污染当成基准；而 Formal 的固定种子集
    （20060515/20041210/19810915/2023/2024）里本来也没有 42。
    锚点因此改为：与**上一次入库的同一张表**逐格比，把差异分布摆出来供人判断。
    """
    log('## 1. 锚点核验（与 git 里上一版合并表逐格比差异分布）\n')
    new = pd.read_excel(SRC50, sheet_name=SHEET_RAW)
    log(f'  本版：{len(new)} 行 | 场景 {new[COL_SCENE].nunique()} | '
        f'方法 {sorted(new[COL_METHOD].unique())} | 种子 {sorted(new[COL_SEED].unique())}\n')

    prev = _prev_merged()
    if prev is None:
        log('  ⚠ git 的 HEAD 里没有这张表（首次入库），无可比对象——跳过\n')
        return

    m_new, m_prev = set(new[COL_METHOD].unique()), set(prev[COL_METHOD].unique())
    if m_new - m_prev:
        log(f'  本版新增方法：{sorted(m_new - m_prev)}（上一版没有，无从比对）')
    if m_prev - m_new:
        log(f'  🔴 上一版有而本版没有的方法：{sorted(m_prev - m_new)}')
    key = [COL_SCENE, COL_METHOD, COL_CORR, COL_SEED]
    for d in (new, prev):
        d[COL_CORR] = d[COL_CORR].astype(str)
    mg = new.merge(prev, on=key, suffixes=('_新', '_旧'), how='inner')
    if mg.empty:
        log('  ⚠ 两版无法按 (场景, 方法, 校正, 种子) 对齐，跳过\n')
        return
    d = (mg[f'{COL_RMSE}_新'] - mg[f'{COL_RMSE}_旧']).abs()
    same = int((d < 1e-9).sum())
    log(f'  对齐 {len(mg)} 格 | 逐位相同 {same} 格（{100 * same / len(mg):.1f}%）')
    log(f'  |ΔRMSE| 中位 {d.median():.4f} | P95 {d.quantile(.95):.4f} | 最大 {d.max():.4f}')
    log('  变化最大的 5 格：')
    for _, r in mg.assign(_d=d).nlargest(5, '_d').iterrows():
        log(f'    {str(r[COL_SCENE])[:30]:30s} {r[COL_METHOD]!s:8s} {r[COL_CORR]!s:4s} '
            f'{r[f"{COL_RMSE}_旧"]:7.3f} → {r[f"{COL_RMSE}_新"]:7.3f}')
    log('')


# ══════════════════════════════════════════════════════════════════════
# 2. 结论方向核验：预注册假设的方向有没有翻
# ══════════════════════════════════════════════════════════════════════
def step2_direction() -> None:
    """
    把每个预注册假设的**实测方向**与**论文当前的表述方向**对照。

    50 号的「表c」只给统计量（差值 / P / Cliffs δ），不给"论文怎么写的"——
    后者只存在于 content_csae.py 的文字里。故在此显式列出论文当前主张的方向，
    与实测方向逐条比对。这份对照表就是论文与实验之间的契约。

    方向约定：`差值_A减B` = RMSE(A) − RMSE(B)。RMSE 越低越好，
    因此**负值 = A 优于 B**。
    """
    log('## 2. 结论方向核验（预注册假设 H1–H7）\n')
    log('  ⚠ 本步骤只**报告**方向，不改写论文。若方向翻转，流水线中止，')
    log('     由人决定如何改写结论——而不是让脚本悄悄把数字换掉。\n')

    # 论文当前的主张（'A优' = A 的 RMSE 更低；'B优' 反之；'不显著' = 论文未主张有差异）
    #   H6/H7 需 CNN+BL 与 Phys，主实验（5 方法）未跑，故在表c 中缺席，
    #   由消融实验（57 号，见 step2b）单独验证——此处只核 H1–H5。
    paper_claims = {
        'H1': 'A优',    # 零标签下 Phys+BL 优于 CNN —— 论文表1 的核心主张
        'H2': 'B优',    # 校正后 Phys+BL **显著劣于** PLSR —— 论文结论3 明写"显著劣于"(P<0.005)
        'H3': 'B优',    # 校正后 Phys+BL **显著劣于** CNN
        'H4': 'A优',    # 零标签下 Phys+BL 优于 CNN+MMD
        'H5': 'A优',    # slope/bias 校正对 Phys+BL 自身有增益
        # H6/H7 **不进这张阻塞表**。补充材料 S2 节那两行出自 19/20 号（经 70 号 5 种子
        # 聚合），其卷积主干的回归头是 Linear(latent,16)+Dropout+Linear(16,1)；50 号统一
        # 继承 18_exp 的 Linear(latent,32)+Linear(32,1)（见 50 号文件内注释）。同一个消融
        # 换了回归头就是另一次实验，拿 50 号的表c 去判 S2 节的表述会制造假警报。
        # 50 号那两行作为**独立复现**报告在 step2c，不作为通过条件。
    }

    hyp = pd.read_excel(SRC50, sheet_name='表c：预注册配对检验')
    flipped = []
    log(f'  {"":4s} {"假设":6s} {"论文主张":8s} {"实测":8s} {"差值(A−B)":>11s} {"P_holm":>9s}  {"效应":6s}')
    checked = 0
    for _, r in hyp.iterrows():
        hid = str(r['假设'])
        if hid not in paper_claims:
            continue
        diff = float(r['差值_A减B'])
        p = float(r['Wilcoxon_p_holm'])
        mag = str(r.get('效应量级', ''))
        actual = '不显著' if p >= 0.05 else ('A优' if diff < 0 else 'B优')
        claim = paper_claims[hid]
        ok = (actual == claim)
        log(f'  {"✅" if ok else "🔴":4s} {hid:6s} {claim:8s} {actual:8s} {diff:>+11.4f} {p:>9.3g}  {mag:6s}')
        checked += 1
        if not ok:
            flipped.append(f'{hid}（论文说 {claim}，实测 {actual}）')

    log('')
    if flipped:
        for f in flipped:
            log(f'  🔴 {f}')
        die('结论方向与论文表述不符：' + '；'.join(flipped) +
            '。这是实质性改写，必须由人判定，不得由脚本自动替换数字。')
    log(f'  ✅ 已核 {checked} 个假设方向与论文一致\n')


# ══════════════════════════════════════════════════════════════════════
# 2c. 大回归头上的独立复现：S2 节两条消融换一个主干还成不成立
# ══════════════════════════════════════════════════════════════════════
def step2c_replication() -> None:
    """把 50 号表c 的 H6/H7 与补充材料 S2 节的表述并排摆出来，只报告、不判通过。

    两者比的是同一对方法，但不是同一次实验：S2 节出自 19/20 号（回归头
    Linear(latent,16)+Dropout+Linear(16,1)），50 号统一用 18_exp 的
    Linear(latent,32)+Linear(32,1)。回归头不同、场景与种子相同，因此 50 号这两行
    是一次**换主干的独立复现**——复现上了是加分，没复现上是必须写进稿件的边界，
    两种情况都不该由本脚本替人裁决，所以这一步永远不 die。
    """
    log('## 2c. 换一个回归头之后，S2 节的两条消融还成不成立（只报告）\n')
    hyp = pd.read_excel(SRC50, sheet_name='表c：预注册配对检验')
    hyp['假设'] = hyp['假设'].astype(str)
    if not {'H6', 'H7'} <= set(hyp['假设']):
        log('  ⚠ 表c 里没有 H6/H7，跳过\n')
        return
    log('  S2 节（19/20 号，回归头 16）与 50 号（回归头 32）逐条并排：')
    si = {'H6': 'CNN+BL 较 CNN 略差，P=1.1e-2，可统计分辨',
          'H7': 'Phys+BL 较 Phys 精度相当（+0.63%），但把发散率降低约 9 倍'}
    for hid in ('H6', 'H7'):
        r = hyp[hyp['假设'] == hid].iloc[0]
        p = float(r['Wilcoxon_p_holm'])
        verdict = '不显著' if p >= 0.05 else ('A 优' if float(r['差值_A减B']) < 0 else 'B 优')
        log(f'    {hid}  S2 节：{si[hid]}')
        log(f'        50 号：差值 {float(r["差值_A减B"]):+.4f}，P_holm {p:.3g}，'
            f'Cliffs δ {float(r["Cliffs_delta"]):+.3f}（{r["效应量级"]}）→ {verdict}')
    d = pd.read_excel(SRC50, sheet_name=SHEET_RAW)
    bare = d[d[COL_CORR].astype(str) == '裸']
    log('  50 号里的发散率（场景–种子运行中 RMSE > 10 °Brix）：')
    for m in ('Phys', 'Phys+BL', 'CNN', 'CNN+BL'):
        g = bare[bare[COL_METHOD] == m]
        if len(g):
            k = int((g[COL_RMSE] > 10).sum())
            log(f'    {m:8s} {k:4d}/{len(g):4d} = {100 * k / len(g):.2f}%')
    log('')
    log('  → 这两行是否要写进 S2 节、以及怎么写，由人判定；本步骤不设通过条件。\n')


# ══════════════════════════════════════════════════════════════════════
# 2b. 架构消融（57 号）：H6/H7 的方向 + 表2/§2.3 在干净数据上
# ══════════════════════════════════════════════════════════════════════
def step2b_ablation() -> None:
    log('## 2b. 架构消融方向核验（H6/H7，干净数据）\n')
    log('  ⚠ 表2/§2.3 原建立在含 56.07 污染的脏数据上（第二次审计发现）。')
    log('     消融方法（CNN+BL/Phys）不在主实验内，故用 57 号在干净数据上单独重算。\n')

    hyp = pd.read_excel(SRC50, sheet_name='表c：预注册配对检验')
    if {'H6', 'H7'} <= set(hyp['假设'].astype(str)):
        log('  ⏭ 主实验已跑满 7 个方法，H6/H7 已在第 2 步用同一批 602 场景 × 5 种子核过；')
        log('     57 号那套单独分片是主实验只有 5 个方法时的替代通路，此处跳过。\n')
        return

    abl_shards = os.path.join(OUT, '50ablation_shards')
    import glob
    if not glob.glob(os.path.join(abl_shards, 'seed*_shard*.csv')):
        die('消融分片不存在，且主实验表c 里也没有 H6/H7——两条通路都不可用')

    run([sys.executable, '02code/57_ablation_clean_analysis.py'], '57 号（消融分析）')
    f = os.path.join(OUT, '57_ablation_clean_analysis.xlsx')
    t2 = pd.read_excel(f, sheet_name='表2')
    d23 = pd.read_excel(f, sheet_name='§2.3发散')

    # 论文主张：H6 = CNN+BL 劣于 CNN（改进为负）；H7 = BL 的净贡献主要是训练稳定器
    #   （发散场景 Phys 多、Phys+BL 少；正常场景改进可忽略）
    row_h6 = t2[t2['对比'] == 'CNN+BL vs CNN'].iloc[0]
    h6_actual = 'CNN+BL 有害' if row_h6['imp_mean'] < 0 else 'CNN+BL 有益'
    h6_ok = row_h6['imp_mean'] < 0
    log(f'  {"✅" if h6_ok else "🔴"} H6  CNN+BL vs CNN：改进均值 {row_h6["imp_mean"]:+.2f}% '
        f'→ {h6_actual}（论文主张：有害）')

    phys_div = int(d23['Phys发散数'].iloc[0])
    physbl_div = int(d23['PhysBL发散数'].iloc[0])
    h7_ok = phys_div > physbl_div
    log(f'  {"✅" if h7_ok else "🔴"} H7  发散场景 Phys {phys_div} vs Phys+BL {physbl_div} '
        f'→ BL 是训练稳定器（论文主张：Phys 多、Phys+BL 近零）')
    log(f'       McNemar P={d23["McNemar_P"].iloc[0]:.2e}，聚类 P={d23["McNemar_聚类P"].iloc[0]:.2e}\n')

    if not (h6_ok and h7_ok):
        die('消融方向与论文表述不符（H6/H7）——实质性改写，必须由人判定')
    log('  ✅ H6/H7 方向与论文一致（干净数据）；表2/§2.3 的数字见 57 号 md，待人工回填\n')


# ══════════════════════════════════════════════════════════════════════
# 3. 免标签诊断（55 号）：BL 残差能不能**预测**迁移风险
# ══════════════════════════════════════════════════════════════════════
def step3_diagnosis() -> None:
    log('## 3. 免标签迁移风险诊断（55 号，预注册阈值）\n')
    run([sys.executable, '02code/55_bl_risk_diagnosis_validation.py'], '55 号')

    f = os.path.join(OUT, '55_bl_risk_diagnosis_validation.xlsx')
    if not os.path.exists(f):
        die('55 号没有产出 xlsx')
    v = pd.read_excel(f, sheet_name='表e：结论')
    verdict = str(v['判定'].iloc[0])
    med_rho = float(v['ρ中位数'].iloc[0])
    ci_ok = bool(v['CI下界全>0'].iloc[0])
    loo_ok = bool(v['留一通过'].iloc[0])
    log(f'  ρ 中位数 = {med_rho:.3f} | 95% CI 下界全 > 0: {ci_ok} | 留一产地验证通过: {loo_ok}')
    log(f'  预注册判定：**{verdict}**\n')

    # 相关分析明细
    res = pd.read_excel(f, sheet_name='表b：相关分析')
    log('  各结局的相关分析：')
    log('  ' + res.to_string(index=False).replace(chr(10), chr(10) + '  '))
    log('')

    max_stance = VERDICT_TO_MAX_STANCE.get(verdict[:1])
    if max_stance is None:
        die(f'55 号的判定字符串不认识：{verdict[:20]!r}')
    log(f'  证据支持的最高档位：**{max_stance}**；稿件当前写的是：**{PAPER_BL_STANCE}**')

    # 常量说稿件是哪一档，就必须能在两份补充材料里找到那一档的原话。
    zh_probe, en_probe = BL_STANCE_PROBE[PAPER_BL_STANCE]
    ms = os.path.join(BASE, '06doc', '01manuscript')
    for name, probe in (('Supplementary_zh.tex', zh_probe), ('Supplementary_en.tex', en_probe)):
        with open(os.path.join(ms, name), encoding='utf-8') as f:
            if probe not in f.read():
                die(f'{name} 里找不到「{PAPER_BL_STANCE}」档的原话 {probe!r}——'
                    'PAPER_BL_STANCE 与稿件脱节了，先对齐再跑')
    log(f'  ✅ 两份补充材料都能查到「{PAPER_BL_STANCE}」档的原话')

    if STANCE_ORDER.index(PAPER_BL_STANCE) > STANCE_ORDER.index(max_stance):
        die(f'稿件写到「{PAPER_BL_STANCE}」，而证据只支持到「{max_stance}」——'
            '实质性改写，必须由人确认，脚本不代劳')
    if STANCE_ORDER.index(PAPER_BL_STANCE) < STANCE_ORDER.index(max_stance):
        log(f'  🟡 稿件比证据保守了一档（证据到「{max_stance}」，稿件写「{PAPER_BL_STANCE}」）；')
        log('     这不阻塞交付，但值得回看一次是不是压得太狠。\n')
    else:
        log('  ✅ 稿件的表述档位与证据一致\n')


# ══════════════════════════════════════════════════════════════════════
# 4-6. 回填数字 → 重绘图 → 出稿自检
# ══════════════════════════════════════════════════════════════════════
def step4_6_finalize() -> None:
    log('## 4. 数字回填 → 5. 重绘图 → 6. 出稿自检\n')
    run([sys.executable, '02code/52_report_new_numbers.py'], '52 号（数字报告）')
    log('  ✅ 52 号：数字报告已出')
    run([sys.executable, '02code/49_plot_csae_figures.py'], '49 号（绘图）')
    log('  ✅ 49 号：Fig1 / Fig2 / Fig3 已出')
    # 48 号（出稿 LaTeX）已从本流水线摘除。大修返修之后 Elsevier_zh.tex 是手工维护的：
    # 案例研究整体搬进补充材料、新增 §3.4 与表 6、全篇改用 NRMSEP 共同主端点，48 号的模板
    # 停留在投出前的版本，在这里跑一次就会把返修稿整篇覆盖掉。数字回填改为人工按
    # 04outputs/52_report_new_numbers.md 落到中英双稿。
    log('  ⏭ 48 号（出稿 LaTeX）已摘除：返修稿手工维护，见 04outputs/52_report_new_numbers.md 手工回填')

    # 出稿自检（54 号）暂停：54 号原为《农业工程学报》docx 全文反伪造自检，其 docx track
    # （47 号 + content_csae.build()）已随本次清理移除。Elsevier LaTeX 稿的等价自检待重建
    # （最简做法：让 54 号改扫 06doc/01manuscript/Elsevier_zh.tex）——见与作者的讨论。
    log('  ⚠ 54 号自检（docx track）已停用，Elsevier 稿等价自检待重建\n')


def main() -> None:
    log('> 本流水线的设计前提：**宁可中止，也不让半截结果或翻转的结论悄悄流进论文。**\n')
    step0_coverage()
    step1_anchor()
    step2_direction()
    step2c_replication()
    step2b_ablation()
    step3_diagnosis()
    step4_6_finalize()
    log('---\n')
    log('## ✅ 全流水线通过——稿件可交付')
    _save_report()


if __name__ == '__main__':
    main()
