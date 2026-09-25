# Data card: apple cross-origin NIR benchmark (SSC)

Metadata only — no spectra are distributed with this repository.

## Source and license

In-house measurements collected by Xinjiang Agricultural University (2018, 2019,
2025 harvests). **Restricted access**: not publicly released; available from the
corresponding author (huanghua@xjau.edu.cn) on reasonable request, subject to the
data-sharing policies of the university.

## Integrity

Checksums are of the cleaned per-year tables written by `02code/02_data_processing.py`
(SHA-256, first 16 hex characters):

| File | Rows | Columns | Bytes | SHA-256 (16) |
|---|---:|---:|---:|---|
| `02_data_2018.csv` | 567 | 1216 | 13,590,326 | `a248445fd874bbf0` |
| `02_data_2019.csv` | 604 | 1216 | 14,438,126 | `1495630ea432f4fb` |
| `02_data_2025.csv` | 571 | 232 | 2,573,222 | `bc6337e462fe5006` |

## Granularity and units

One row per measured apple. Columns: harvest year, growing origin, SSC content
(°Brix, reference value by digital refractometer), followed by per-wavelength
diffuse-reflectance channels.

## Instruments and spectral grid

Two instrument generations, which is why this benchmark carries compound drift:

- **Instrument A** (2018, 2019): 1213 channels, 590.67–1001.12 nm, spacing 0.31–0.37 nm
  (mean 0.34 nm).
- **Instrument B** (2025): 229 channels, 591.19–1001.12 nm, spacing 1.75–1.82 nm
  (mean 1.80 nm).

For cross-instrument transfer the two grids are aligned to a common 229-channel
grid by nearest-wavelength matching (`61_benchmark_datasets.py`).

## Sample statistics

Cleaned per-year tables (after the season-wise spectral outlier removal, before the
two rules applied when the data are read):

| Year | Instrument | n | Origins (n) | SSC mean ± sd | SSC range |
|---|---|---:|---|---|---|
| 2018 | A | 567 | Shaanxi 200, Shandong 186, Xinjiang 181 | 12.97 ± 2.36 | 8.23 – 56.07 |
| 2019 | A | 604 | Xinjiang 213, Shaanxi 203, Shandong 188 | 13.89 ± 1.69 | 10.12 – 18.26 |
| 2025 | B | 571 | Gansu 225, Shandong 180, Xinjiang 166 | 12.08 ± 2.02 | 5.62 – 37.12 |

Analysis set (what every experiment uses; Supplementary Section S1):

| Year | n | Origins (n) | SSC mean ± sd | SSC range |
|---|---:|---|---|---|
| 2018 | 535 | Shaanxi 195, Shandong 182, Xinjiang 158 | 12.88 ± 1.51 | 8.23 – 18.70 |
| 2019 | 574 | Xinjiang 195, Shaanxi 195, Shandong 184 | 13.84 ± 1.68 | 10.12 – 18.26 |
| 2025 | 570 | Gansu 225, Shandong 179, Xinjiang 166 | 12.03 ± 1.73 | 5.62 – 19.70 |

1679 fruit in **9 origin-year domains** (2018/2019 × {Shandong, Xinjiang, Shaanxi},
2025 × {Shandong, Xinjiang, Gansu}) of 158–225 fruit each, giving **72 ordered
transfer tasks** for SSC in the unified benchmark. The case study uses the 2018 and
2025 seasons: 1105 fruit in six origin-year domains.

## Cleaning and filtering

- **2025 facets.** Five facets were scanned per fruit (3228 facet spectra), and each
  image holds three fruit whose facets are interleaved. Facets are matched to their
  fruit by the fruit identifier recorded with the scan, not by acquisition order, and
  averaged into 635 fruit-level spectra; the reference value is the mean of the five
  facet readings.
- **Spectral outliers.** Within each season, on all origins and before any
  source/target split, the 10% of spectra farthest from the mean after band-wise
  standardisation are removed (63 in 2018, 67 in 2019, 64 in 2025). This gives the
  cleaned tables above.
- **Two rules applied when the data are read** (`load_processed_dataframe` in
  `02code/01_export_utils.py`, shared by every experiment):
  - reference SSC above 20 °Brix is removed as a refractometer misreading (56.07 °Brix
    in 2018, 37.12 °Brix in 2025; none in 2019);
  - groups of bit-wise identical spectra with differing reference values are removed
    in their entirety (31 rows in 2018, 30 in 2019, none in 2025), since at least one
    spectrum–label pairing in each group is wrong and it cannot be told which.
- The unified benchmark also skips an origin labelled "unknown" and any domain with
  fewer than 30 fruit (no domain of the analysis set is affected).
- Modelling input in the unified benchmark is standard normal variate (SNV); the case
  study uses the preprocessing of Supplementary Section S2.

## Split logic

Splits are by **domain** (origin × year × instrument): the source domain supplies the
calibration set of the source model. Each task draws a split in which the target domain is halved into a test set and a
calibration pool; each budget `n ∈ {0, 5, 10, 20, 40}` is drawn from the pool. The
classical methods repeat the draw five times per seed and the deep methods once
(25 and 5 runs per cell, five fixed seeds `[20060515, 20041210, 19810915, 2023,
2024]`), and per-task values are medians over the runs. Target labels never enter
model selection. Unlabelled target spectra do enter three steps: CORAL estimates its
moments from all target spectra, DANN and Deep CORAL use the unlabelled spectra of
the calibration pool, and on apple the season-wise outlier removal and band-wise
standardisation see every spectrum of the season, for every method alike (manuscript
Section 2.3).

The apple case study additionally uses 602 scenarios enumerated in
`02code/04_migration_scenarios.json`. This is a different accounting unit from the
72 unified transfer tasks and the two must not be pooled; the manuscript states
this explicitly.

## Known biases and limitations

- **Origin and year are confounded with instrument.** 2025 is the only
  instrument-B year and the only year containing Gansu, so "cross-origin",
  "cross-year" and "cross-instrument" effects are not fully separable.
- Sampling is unbalanced across origins and years.
- All fruit are Fuji apples; conclusions do not transfer to other cultivars
  without revalidation.
- Reference SSC is a destructive single-point refractometer reading per fruit and
  carries its own measurement error, which bounds the achievable RMSEP.

## Label reliability

SSC reference values are measured by digital refractometer immediately after
spectral acquisition, one reading per fruit. No repeated reference measurements
were taken, so reference repeatability is not quantified.

## Preprocessing pointer

`02code/02_data_processing.py` → cleaned per-year tables;
`02code/61_benchmark_datasets.py` → unified domain arrays and the transfer-task
registry. Raw sources are read read-only; nothing is written back to the source
tree.

## Ethics

Plant material only. No human subjects, no personally identifiable information, no
animal experimentation.
