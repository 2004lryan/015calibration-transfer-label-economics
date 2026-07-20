# Data card: apple cross-origin NIR benchmark (SSC)

Metadata only — no spectra are distributed with this repository.

## Source and license

In-house measurements collected by Xinjiang Agricultural University (2018, 2019,
2025 harvests). **Restricted access**: not publicly released; available from the
corresponding author (huanghua@xjau.edu.cn) on reasonable request, subject to the
data-sharing policies of the university.

## Integrity

Checksums are of the cleaned per-year tables written by `code/02_data_processing.py`
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

- **Instrument A** (2018, 2019): 1213 channels, 590.67–1001 nm, ~0.36 nm spacing.
- **Instrument B** (2025): 229 channels, 591.19 nm onwards, ~1.76 nm spacing.

For cross-instrument transfer the two grids are aligned to a common 229-channel
grid by nearest-wavelength matching (`61_benchmark_datasets.py`).

## Sample statistics

| Year | Instrument | n | Origins (n) | SSC mean ± sd | SSC range |
|---|---|---:|---|---|---|
| 2018 | A | 567 | Shaanxi 200, Shandong 186, Xinjiang 181 | 12.97 ± 2.36 | 8.23 – 56.07 |
| 2019 | A | 604 | Xinjiang 213, Shaanxi 203, Shandong 188 | 13.89 ± 1.69 | 10.12 – 18.26 |
| 2025 | B | 571 | Gansu 225, Shandong 180, Xinjiang 166 | 12.08 ± 2.02 | 5.62 – 37.12 |

After the out-of-range filter (see below) this yields **9 origin-year domains**
(2018/2019 × {Shandong, Xinjiang, Shaanxi}, 2025 × {Shandong, Xinjiang, Gansu}),
each with 166–225 samples, and **72 ordered transfer tasks** for SSC.

## Cleaning and filtering

- SSC values outside `5 < SSC < 20` °Brix are discarded as reference-instrument
  misreadings. The 2018 maximum of 56.07 °Brix and the 2025 maximum of 37.12 °Brix
  are refractometer errors removed by this filter; an earlier version of the
  analysis that ran on unfiltered data was superseded.
- Spectra with any non-finite channel are discarded.
- Bitwise-duplicate spectra are removed.
- Origin labelled "unknown" is excluded, as are domains with fewer than 30 samples.
- Preprocessing for modelling is standard normal variate (SNV) transformation.

## Split logic

Splits are by **domain** (origin × year × instrument): the source domain provides
the full calibration set, the target domain provides a fixed label budget
`n ∈ {0, 5, 10, 20, 40}` with the remainder held out for testing. No target test
sample participates in preprocessing fitting, feature selection, hyper-parameter
search, or model selection. Each task is repeated under 5 fixed random seeds
`[20060515, 20041210, 19810915, 2023, 2024]`.

The apple case study additionally uses 602 scenarios enumerated in
`code/04_migration_scenarios.json`. This is a different accounting unit from the
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

`code/02_data_processing.py` → cleaned per-year tables;
`code/61_benchmark_datasets.py` → unified domain arrays and the transfer-task
registry. Raw sources are read read-only; nothing is written back to the source
tree.

## Ethics

Plant material only. No human subjects, no personally identifiable information, no
animal experimentation.
