# Calibration transfer under limited target labels

Code for the manuscript

**"Calibration transfer under limited target labels: a five-benchmark comparison of
correction and representation methods"**
P. Li, Y. Li, L. Li, Y. Liu, Y. Feng, M. W. Akram, J. Guo, H. Huang
(submitted to *Chemometrics and Intelligent Laboratory Systems*).

This version of the repository corresponds to the revised manuscript. Every script
cited by number in the manuscript, the Supplementary Material and the responses to
the reviewers is in [`02code/`](02code/) under that number ("script 74" is
`02code/74_shift_structure_diagnosis.py`).

## The question

Near-infrared calibration models degrade across instruments, seasons, growing
regions and laboratories. The study asks, under controlled target-label budgets, when
a deep or physics-informed representation beats the simplest correction. One protocol
covers five benchmarks (corn, tablets, soil, mango, apples), four shift types and two
spectral modalities: 126 transfer tasks, target-label budgets n ∈ {0, 5, 10, 20, 40},
five fixed seeds, and errors reported in native units (RMSEP) and as
NRMSEP = RMSEP/σ<sub>y,tgt</sub> = 1/RPD.

## Findings reported in the manuscript

- At 0–40 target labels, label-free second-moment alignment (CORAL, used
  transductively) and a two-parameter slope/bias correction match or exceed the deep
  and physics-informed baselines each budget admits on the instrument, seasonal and
  cross-origin benchmarks. Cross-laboratory mid-infrared soil is the exception: there
  the deep family is ahead at every positive budget. A selection-free reanalysis
  agrees.
- The in-sample map of which method wins fails as a rule on a held-out benchmark
  (median regret 11.2%). The fixed prescription "CORAL at zero labels, slope/bias
  correction otherwise" needs no diagnosis of the shift and had the lowest
  leave-one-benchmark-out regret: 2.8% (5.2% with the apple standardisation refitted
  without the target domain), against 20.9% for a default deep model.
- Shift magnitude does not predict the crossover budget (R² ≤ 0.13). Matching the
  first two moments removes 92% of the source–target distance under instrument shift
  and 73–78% under the population shifts, and this share, not the distance,
  correlates with the CORAL gain (ρ = 0.50) and with the simple family's zero-label
  margin (ρ = 0.44). A label-free low-rank index tracks direct-transfer failure
  (ρ = 0.49, just under its pre-registered 0.5).
- In the apple case study, the typical-scenario edge of a Beer–Lambert-inspired
  architecture (Phys+BL) over a generic network reverses once the same labels fund a
  slope/bias correction.

**Scope.** Shift type and dataset are nearly collinear (season only in mango,
laboratory only in soil, origin/year only in apple), so there are five independent
benchmarks and the mechanism-level statements form an exploratory map. The
held-out regret is measured on a benchmark held out from the same five, not on data
acquired after the rule was fixed. CORAL and both label-free diagnostics need a batch
of unlabelled target spectra, and the season-wise apple standardisation and outlier
removal see the target spectra for every method. Section 4.4 of the manuscript lists
ten limitations.

## Repository layout

```
├── 02code/          analysis scripts, numbered as cited in the manuscript
│   └── tests/       pytest suite (69 tests; no data, no GPU)
├── data/DATA.md     dataset sources, licences, SHA-256 checksums, task matrix
├── docs/            data cards (metadata only, no spectra)
├── pyproject.toml   ruff / mypy / pytest configuration
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

Run every script from the repository root. The scripts locate their neighbours
through the directory name `02code/` and write results to `04outputs/` and logs to
`05logs/` next to it; both are created on first run and are git-ignored. Script
names start with digits, so they are run as files (`python 02code/NN_name.py`), not
imported as modules.

The code comments and log messages are in Chinese, as they were when the analyses
were run.

## Where each result comes from

Main text

| Item | Scripts |
|---|---|
| Table 1, benchmarks and the 126-task matrix | `61` |
| Table 2, method applicability | methods as implemented in `62` (classical) and `64` (deep) |
| Fig. 1, study overview | `77` |
| Fig. 2, the Phys+BL architecture | `49` (function `fig_arch`) |
| Section 3.1 and Table 3, family comparison | `86` (selection-free effect sizes), `75` (NRMSEP columns), `68` (normalised reanalysis), `101` (domain pair as the unit) |
| Fig. 3, label-budget reckoning curves | `66`, from the outputs of `65` and `63` |
| Table 4, leave-one-benchmark-out regret | `73`; paired comparison of prescriptions `87` |
| Fig. 4, in-sample frequency map and held-out regret | `78`, from the output of `73` |
| Table 5, label-free structure of the shift | `74` |
| Fig. 5, structure rather than magnitude of the shift | `76`, from `74` and `63` |
| Section 2.3, apple standardisation refitted without the target domain | `108` |

Supplementary Material

| Section | Scripts |
|---|---|
| S1–S2, apple data and preprocessing; Fig. S1 | `02`, `49`; standardisation sensitivity `106` (case study) and `108` (unified benchmark) |
| S8, Beer–Lambert decomposability; Fig. S2 | `03`, `28`, `35`, `55`, `71`; figure `49` |
| S9, five methods across 602 scenarios; Table S1, Figs. S3–S4 | `50`, `58`, `72`; clustering robustness `60sensitivity_domain_clustering` (cited as script 60); strict zero-label control `59`; outlier-threshold variants `02 --outlier_pct` with `104`/`105`; figures `49` |
| S10, ablations; Table S2 | `50` |
| S11, slope/bias correction at a matched label budget; Table S3, Fig. S5 | `50`; clustering robustness `60sensitivity_domain_clustering`; figure `49` |
| S13, implementation settings; Table S4 | `62`, `64` |
| S14, source-domain positive control; Tables S5–S6 | `88`, `89`, `80`; diagnosis of the deep training recipe `90`; update-budget sensitivity `102` (server runner `100`) |
| S15, the four public benchmarks alone; Table S7 | `47` |
| S16, data card, diagnostic settings, robustness; Tables S8–S9 | `61`, `74`, `86`, `75`, `101`, `110` (CORAL restricted to a calibration pool), `65` |

`83_revision_number_receipt.py` recomputes the numbers quoted in the manuscript from
the result files and prints a PASS/FAIL line for each. It reads the manuscript sources,
which are not part of this repository; it is included because `101` reuses its
per-task medians and method-family definitions.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10+; developed on 3.13
pip install -r requirements.txt
```

The canonical runs used the CPU (`--device cpu`); a GPU is optional. Check the
installation from the repository root:

```bash
pytest                                               # 69 tests
ruff check .
mypy 02code/tests 02code/fetch_datasets.py
```

## Data

No spectra are redistributed. Obtain the four public datasets from their official
sources ([`data/DATA.md`](data/DATA.md) gives URLs, licences and SHA-256 checksums)
and place them so that these paths resolve:

```
05data/036_corn_nir_3instruments/corn.mat
05data/037_tablet_nir_shootout2002/nir_shootout_2002.mat
05data/038_ossl_global_soil_spectral/ossl_all_L1_v1.2.csv.gz
05data/041_mango_anderson_crossseason/NAnderson2020MendeleyMangoNIRData.csv
```

`05data/` is resolved two levels above the repository root (constant `DATA05` at the
top of `02code/61_benchmark_datasets.py`). `python 02code/fetch_datasets.py --list`
prints the same pointers and checksums.

The apple spectra are internal data of Xinjiang Agricultural University and are not
public (see *Data availability*). The scripts expect the cleaned per-year tables
`03data/02_data_{2018,2019,2025}.csv` written by `02code/02_data_processing.py`. Model
inputs are read through the loaders of `02code/01_export_utils.py`
(`load_processed_dataframe` and the functions built on it), which apply the two
cleaning rules of Supplementary Section S1.

## Reproducing the results

Unified benchmark (formal seeds `20060515, 20041210, 19810915, 2023, 2024`):

```bash
python 02code/61_benchmark_datasets.py                  # load, unify, cache; 126-task registry
python 02code/62_crossover_engine.py --benchmarks corn,tablet,mango,apple,ossl_mir
bash   02code/92_run_deep_arm_local.sh                  # deep arm (script 64): 5 benchmarks x 5 seeds
python 02code/64_deep_transfer_server.py --merge
bash   02code/97_downstream_pipeline.sh unified         # 63, 65, 67, 68, 73-75, 80, 86-89, 94; Figs. 1, 3-5
```

Script 62 repeats each draw five times per seed by default (`--rep 5`), giving 25 runs
per cell. The remaining analyses run on their own (`47`, `101`, `102`, `106`, `108`,
`110`), some after a sharded run on a server:

- `95_server_run_all.sh` runs the deep arm, the case-study shards and the strict
  zero-label shards in one process pool on a multi-core machine and merges them.
- `100` reruns script 88 with a raised update budget, which `102` analyses; `105`,
  `107` and `109` run `104`, `106` and `108` on a server.
- `93`, `96`, `98` and `99` install the dependencies, deploy the code, launch the
  runs and fetch the results. `96`, `98`, `99` and `100` take `user@host port
  directory` as arguments; `105`, `107` and `109` read `SSC015_HOST`, `SSC015_PORT`
  and `SSC015_REMOTE`.

Apple case study (needs the restricted apple data):

```bash
python 02code/02_data_processing.py --year all          # cleaned per-year tables
bash   02code/91_run_case_study_local.sh                # script 50: 602 scenarios x 5 seeds, sharded
python 02code/50_formal_multiseed_benchmark.py --merge
# strict zero-label control (script 59), one call per shard i = 0..23:
python 02code/59_reviewer_experiments.py --device cpu --zero_epochs 200 --seed 42 --shard_id 0 --n_shards 24
python 02code/59_reviewer_experiments.py --merge
bash   02code/97_downstream_pipeline.sh case
python 02code/49_plot_csae_figures.py                   # Fig. 2 and the case-study figures
```

For the manuscript, Phys and Phys+BL were rerun with `103` after the correction of
their regression term (see the changes below) and merged with
`--merge --override_dir 04outputs/50_shards_physfix` (script 50) and
`--merge --override_dir 04outputs/59_shards_physfix` (script 59).

Each script's docstring states its exact run command, inputs and outputs.

## Changes relative to the initial release (2026-07-20)

The initial release (commit `294d191`) held the code of the submitted manuscript.
This version holds the code of the revised manuscript:

- **Deep baselines (`64`).** The encoder pools to 16 bins before flattening instead
  of averaging over the whole wavelength axis; training is a fixed budget of 2,000
  mini-batch updates instead of one full-batch update per epoch; all four deep source
  models stop early on the same source-domain hold-out of about one sixth (evaluated
  every 25 steps, patience 20 evaluations), which contains no target labels.
- **Apple data in the unified benchmark (`61`).** The apple spectra are read through
  the shared loader of `01_export_utils.py`, which also removes groups of bit-wise
  identical spectra with differing reference values, so the unified benchmark and the
  case study use the same data.
- **Crossover budget (`63`).** n\* is computed as the manuscript defines it: the simple
  family against the deep family, on the {0, 5, 10, 20, 40} grid, aggregating repeats
  by the median; tasks that have not crossed by 40 labels are right-censored.
- **Case-study physics network.** `13_model_physics_informed.py`, the module that
  `50` loads, is included; its regression term now compares predictions and
  reference values of the same shape. Phys and Phys+BL were rerun with the same
  seeds and splits (`103`, `104`, `105`). The initial release contained
  `13_model_physics_informed_v3.py`, which `50` does not load.
- **Added:** 49 scripts that the initial release did not contain (the revision's
  analyses, figure scripts and run scripts in the tables above, and the case-study
  scripts they depend on), direct standardisation in `62`, and the outlier-threshold
  and standardisation options of `02`.
- **Removed** (not used by the revised manuscript): `13_model_physics_informed_v3.py`,
  `36_mixed_effects_origin_year.py`, `46_slopebias_unified_comparison.py`,
  `60_decomposed_transfer.py`, `62_ablation_multiseed_analysis.py`.
- **Layout and documentation.** The scripts moved from `code/` to `02code/`, the
  directory name they resolve their neighbours by. The mango benchmark is documented
  with its 306 channels (285–1200 nm); the apple data card gives the analysis-set sizes
  of Supplementary Section S1.

## Data availability

- **Apple NIR spectra** (2018, 2019 and 2025 seasons; several origins; two instrument
  generations) are internal data of Xinjiang Agricultural University and have not
  been publicly released. Researchers who wish to reproduce or extend the experiments
  may request access from the corresponding author on a case-by-case basis, subject
  to the data-sharing policies of the university. The data cards in `docs/` give
  their structure and SHA-256 checksums.
- **Public benchmarks** (Cargill corn, IDRC 2002 tablet shoot-out, OSSL global soil
  spectral library, Anderson mango) are third-party datasets and are not re-hosted;
  [`data/DATA.md`](data/DATA.md) documents their sources, licences and checksums.

## Citation

```bibtex
@unpublished{li2026calibration,
  title  = {Calibration transfer under limited target labels: a five-benchmark
            comparison of correction and representation methods},
  author = {Li, Panlin and Li, Yuchang and Li, Longjie and Liu, Ya and Feng, Yutong
            and Akram, Muhammad Waqar and Guo, Junxian and Huang, Hua},
  note   = {Manuscript submitted to Chemometrics and Intelligent Laboratory Systems},
  year   = {2026}
}
```

## License

The code is released under the MIT License (see [`LICENSE`](LICENSE)). The datasets
remain under their own licences (see [`data/DATA.md`](data/DATA.md)). The manuscript
is not included in this repository.

## Contact

**Hua Huang** (corresponding author), huanghua@xjau.edu.cn
College of Mathematics and Physics, Xinjiang Agricultural University,
Urumqi 830052, China
