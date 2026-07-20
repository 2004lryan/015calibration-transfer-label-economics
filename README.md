# The label economics of calibration transfer

Code and reproducibility materials for the manuscript

**"The label economics of calibration transfer: a five-benchmark, dual-modality
evaluation of simple corrections versus deep and physics-informed representations"**
(under review, *Chemometrics and Intelligent Laboratory Systems*).

## Overview

Near-infrared (NIR) and mid-infrared (MIR) calibration models degrade sharply when
deployed across instruments, seasons, growing regions, or laboratories. The
calibration-transfer literature has grown from classical corrections (piecewise
direct standardization, slope/bias correction, second-order moment alignment) to
deep domain adaptation and physics-informed decomposition — but one practical
question has never been answered systematically: **given a concrete transfer and a
target-domain labelling budget you can afford, when is an expensive deep or
physics-informed representation actually worth it, and when does the simplest
classical correction already suffice?**

This repository reproduces an honest, multi-benchmark reckoning rather than the
promotion of a new algorithm. Under one unified protocol we evaluate **5
calibration-transfer benchmarks, 126 transfer tasks, 4 drift types, and 2 spectral
modalities**, with 5 fixed random seeds and target-set-clustered bootstrap tests
throughout.

Main findings reproduced here:

- **Simple wins at small budgets.** At target budgets of ≤5–10 labels, label-free
  second-order moment alignment (CORAL) and two-parameter slope/bias correction
  match or beat deep and physics-informed representations. The margin varies by
  drift type: simple methods lead throughout under population/season drift, while
  model updating only edges ahead at higher budgets under instrument/laboratory
  drift.
- **Zero-label alignment is viable for pure instrument drift.** CORAL lifts the
  median residual predictive deviation (RPD) from 0.61 to 2.01 with *no* target
  labels — reaching the common quantitative **screening** threshold (RPD ≥ 2), not
  industrial grading. Season, laboratory, and origin drift recover only partially
  (RPD 1.02–1.27).
- **A decision look-up table, not a law.** Across the 126 tasks the best method
  correlates empirically with (drift type × label budget) and can be tabulated as a
  training-free look-up table — fitted within this task batch, requiring prospective
  validation.
- **A negative result on scaling.** The crossover budget *n\** at which simple
  methods catch up with complex ones is **not** predictable from continuous drift
  magnitude (maximum mean discrepancy, Wasserstein distance, intrinsic dimension):
  log–log R² ≤ 0.25, and leave-one-benchmark-out extrapolation errors are large. We
  report this as an **empirical observation over 5 benchmarks and this metric
  family**, not as a non-existence proof.

An in-house apple soluble-solids-content (SSC) cross-origin benchmark serves as the
in-depth case study, in which a Beer–Lambert physics-informed decomposition
(Phys+BL) is tested and its boundary of validity honestly delimited.

### Scope and identifiability limits

Please read these before reusing the conclusions. With **n = 5** independent
benchmarks, drift mechanism and dataset are **near-collinear** — each drift class is
carried mainly by a single dataset, so "the split is driven by mechanism" and "the
split is driven by dataset" cannot be fully separated in this design. The reckoning
additionally involves a within-task-group winner's curse, transductive CORAL, and a
limited deep-model tuning budget. All conclusions are empirical observations across
these five benchmarks.

## Repository structure

```
├── code/
│   ├── 01_export_utils.py             # shared I/O, seeding, logging, statistics
│   │
│   │   # ─ main line: multi-benchmark reckoning ─
│   ├── 61_benchmark_datasets.py       # unify 5 benchmarks into one schema + task registry
│   ├── 62_crossover_engine.py         # classical transfer methods × label-budget sweep
│   ├── 64_deep_transfer_server.py     # deep baselines (CNN / DeepCORAL / DANN / Phys+BL), CUDA
│   ├── 63_crossover_analysis.py       # locate n*, fit scaling law, leave-one-benchmark-out test
│   ├── 65_paper_analysis.py           # merge classical + deep → main tables, paired tests
│   ├── 66_paper_figures.py            # Figures 3–5 (reckoning curves / decision table / n* scatter)
│   │
│   │   # ─ apple SSC case study ─
│   ├── 02_data_processing.py          # apple multi-year spectra cleaning and outlier removal
│   ├── 13_model_physics_informed_v3.py# differentiable Beer–Lambert forward model (Phys+BL)
│   ├── 50_formal_multiseed_benchmark.py# 5 methods × 2 corrections × 602 scenarios × 5 seeds
│   ├── 46_slopebias_unified_comparison.py # slope/bias correction under a unified split
│   ├── 57_ablation_clean_analysis.py  # architecture ablation on cleaned data
│   ├── 62_ablation_multiseed_analysis.py # multi-seed ablation with paired CIs and effect sizes
│   ├── 58_cluster_robust_and_strata.py# target-set-clustered bootstrap + stratified reanalysis
│   ├── 60_decomposed_transfer.py      # 9-domain transfer decomposed by drift type
│   ├── 36_mixed_effects_origin_year.py# mixed-effects model over origin and year
│   ├── 04_migration_scenarios.json    # the 602 apple transfer scenarios
│   ├── fetch_datasets.py              # sources + SHA-256 for the public external datasets
│   └── tests/                         # pytest suite for the core scripts (69 tests)
├── data/
│   └── DATA.md                        # dataset sources, licenses, availability, SHA-256
├── docs/                              # data cards (datasheets) — metadata only, no spectra
├── pyproject.toml                     # ruff / mypy / pytest configuration
├── requirements.txt                   # pinned dependencies
├── CITATION.cff
└── LICENSE
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -r requirements.txt
```

CUDA is required only for the deep baselines (`64_deep_transfer_server.py`); every
classical script runs on CPU.

To check the installation, run the test suite from the repository root. It needs
no data and no GPU, and covers the preprocessing, alignment, CORAL, drift-metric,
crossover-budget, scaling-fit and statistics helpers, plus the integrity of the
dataset registry:

```bash
pytest            # 69 tests
ruff check .      # lint
mypy code/tests code/fetch_datasets.py
```

## Data setup

No spectra are redistributed here. Obtain each public dataset from its official
source (see [`data/DATA.md`](data/DATA.md) for URLs, licenses and SHA-256
checksums) and place it so that the following paths resolve:

```
05data/036_corn_nir_3instruments/corn.mat
05data/037_tablet_nir_shootout2002/nir_shootout_2002.mat
05data/038_ossl_global_soil_spectral/ossl_all_L1_v1.2.csv.gz
05data/041_mango_anderson_crossseason/NAnderson2020MendeleyMangoNIRData.csv
```

`05data/` is resolved as a sibling two levels above the repository root; edit the
`DATA05` constant at the top of `code/61_benchmark_datasets.py` to point elsewhere.
Run `python code/fetch_datasets.py --list` to print the same pointers and checksums.

The **apple** benchmark is restricted-access internal data of Xinjiang Agricultural
University and is not distributed (see *Data availability* below).

## Reproducing the main results

```bash
# 1. Unify the benchmarks and build the transfer-task registry (126 tasks)
python code/61_benchmark_datasets.py

# 2. Classical calibration transfer across the label-budget grid n ∈ {0,5,10,20,40}
python code/62_crossover_engine.py --benchmarks corn,tablet,mango,apple,ossl_mir --rep 10

# 3. Deep and physics-informed baselines (CUDA; 5 fixed seeds)
python code/64_deep_transfer_server.py --device cuda \
    --seeds 20060515,20041210,19810915,2023,2024

# 4. Crossover budget n*, scaling-law fit, leave-one-benchmark-out extrapolation
python code/63_crossover_analysis.py

# 5. Merge into the paper tables (paired Wilcoxon, Cliff's δ, clustered bootstrap)
python code/65_paper_analysis.py

# 6. Figures 3–5 (Chinese and English versions)
python code/66_paper_figures.py
```

Apple case study (Figures 6–9, Tables 3–5):

```bash
python code/02_data_processing.py --year all       # cleaning; needs the restricted apple data
python code/50_formal_multiseed_benchmark.py       # 5 methods × 602 scenarios × 5 seeds (CUDA)
python code/46_slopebias_unified_comparison.py     # slope/bias correction contrast
python code/62_ablation_multiseed_analysis.py      # multi-seed ablation
python code/58_cluster_robust_and_strata.py        # clustered-bootstrap reanalysis
```

Each script writes tables to `04outputs/` and logs to `05logs/` (both created on
run and git-ignored). Every script carries a docstring header stating its exact run
command, inputs and outputs. Formal runs use the five fixed seeds
`[20060515, 20041210, 19810915, 2023, 2024]`.

**Note on file names.** Scripts are kept at their original numbered names so that
the paths in the docstrings, logs, and manuscript remain traceable. Because
`01_export_utils.py` is not a valid module name, every script loads it via
`importlib.util.spec_from_file_location` — run the scripts as files, not as modules.

## Data availability

- **Apple NIR spectra** (2018 / 2019 / 2025, multiple origins, two instrument
  generations) are internal data of Xinjiang Agricultural University and are **not
  publicly released**. They are available from the corresponding author on
  reasonable request, subject to the data-sharing policies of the university. This
  repository ships the data cards (`docs/`), including SHA-256 checksums, but not
  the spectra.
- **External datasets** (Cargill corn, IDRC 2002 tablet, OSSL global soil, Anderson
  mango) are third-party public datasets. We do **not** re-host them;
  [`data/DATA.md`](data/DATA.md) documents sources, licenses and expected SHA-256
  checksums.

## Citation

```bibtex
@article{li2026label,
  title   = {The label economics of calibration transfer: a five-benchmark,
             dual-modality evaluation of simple corrections versus deep and
             physics-informed representations},
  author  = {Li, Panlin and Li, Yuchang and Li, Longjie and Liu, Ya and Feng, Yutong
             and Akram, Muhammad Waqar and Guo, Junxian and Huang, Hua},
  journal = {Chemometrics and Intelligent Laboratory Systems (under review)},
  year    = {2026}
}
```

## License

Code is released under the **MIT License** (see [`LICENSE`](LICENSE)). All datasets
remain subject to their original licenses (see [`data/DATA.md`](data/DATA.md)). The
manuscript is under peer review and is not included in this repository.

## Contact

**Hua Huang** (corresponding author) — huanghua@xjau.edu.cn
College of Mathematics and Physics, Xinjiang Agricultural University,
Urumqi 830052, China
