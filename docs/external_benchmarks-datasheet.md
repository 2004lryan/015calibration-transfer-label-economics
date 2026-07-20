# Data card: the four external calibration-transfer benchmarks

Metadata only — no spectra are distributed with this repository. Sources,
licenses and checksums are in [`../data/DATA.md`](../data/DATA.md).

## Cargill corn — instrument drift

- **Source**: Eigenvector Research public data sets. Free for research use per the
  publisher's terms. SHA-256 (16): `e28fd4be274a54ca`.
- **Composition**: 80 corn samples measured on 3 NIR instruments (m5, mp5, mp6),
  1100–2498 nm, 700 channels.
- **Labels**: moisture, oil, protein, starch (4 properties).
- **Drift**: instrument response drift. Samples are **paired** across instruments,
  which is what makes piecewise direct standardization (PDS) applicable here.
- **Tasks**: 6 ordered instrument pairs × 4 properties = 24.
- **Known bias**: only 80 samples per instrument, so per-task test sets are small
  and the crossover budget `n*` is estimated with correspondingly wide uncertainty.

## IDRC 2002 tablet ("shootout") — instrument drift

- **Source**: Eigenvector Research public data sets (IDRC 2002 Software Shootout).
  Free for research use per the publisher's terms. SHA-256 (16): `129a32ec9e194e56`.
- **Composition**: 655 pharmaceutical tablets measured on 2 transmission NIR
  instruments, 600–1898 nm, 650 channels.
- **Labels**: tablet weight, hardness, active ingredient content (3 properties).
- **Drift**: instrument drift. Paired across instruments.
- **Tasks**: 2 ordered instrument pairs × 3 properties = 6.
- **Known bias**: a manufactured, tightly controlled product; spectral variability
  is far lower than in the agricultural benchmarks, so results here are the
  best-case end of the difficulty range.

## OSSL global soil spectral library — laboratory drift

- **Source**: OpenGeoHub / Soil Spectroscopy, `ossl_all_L1_v1.2.csv.gz` (CC-BY
  4.0). SHA-256 (16): `2044946ad5a10af0`.
- **Composition**: the mid-infrared (MIR) subset, restricted to three sub-libraries
  built by different laboratories (KSSL, AFSIS1, CAF). High-dimensional absorbance
  spectra are binned to 340 equally spaced wavenumber bins. Each sub-library is
  capped at 700 samples to keep the pipeline within an 8 GB memory budget.
- **Labels**: total carbon (`c.tot`, ≈ soil organic carbon), sand content.
- **Drift**: cross-laboratory drift. Not paired.
- **Tasks**: 6 ordered laboratory pairs × 2 properties = 12.
- **Role**: the **only MIR** benchmark in this study, and therefore the only
  evidence for the laboratory-drift class. Any conclusion attributed to
  "laboratory drift" rests on this single dataset — see the identifiability
  caveat in the README.
- **Known bias**: sub-libraries differ in geography and soil type as well as
  laboratory, so laboratory effects are confounded with population effects. The
  700-sample cap subsamples each sub-library rather than using it in full.

## Anderson mango — season drift

- **Source**: Anderson et al. 2020, Mendeley Data (CC-BY 4.0), DOI
  `10.17632/46htwnp833`. SHA-256 (16): `86159e2df9244fa5`.
- **Composition**: intact mango fruit measured across 4 harvest seasons, Vis-NIR
  spectra binned to 103 channels.
- **Labels**: dry matter content (DM).
- **Drift**: season / population drift. Not paired.
- **Tasks**: 4 seasons taken pairwise and ordered × 1 property = 12.
- **Known bias**: season is confounded with cultivar mix, orchard, and instrument
  maintenance state across years; this is the only season-drift benchmark.

## Shared protocol

Every benchmark is mapped onto one schema by `code/61_benchmark_datasets.py`:

```python
dict(name, modality, shift_type, properties, paired,
     domains={domain_key: dict(X=(n, p), Y=(n, n_prop), wl=(p,))})
```

All tasks share the same evaluation pipeline: the source domain supplies the full
calibration set; the target domain supplies a fixed label budget
`n ∈ {0, 5, 10, 20, 40}` with the remainder held out for testing; results are
averaged over 5 fixed random seeds; and statistical tests cluster on the target
set. No target test sample participates in any fitting, selection or thresholding
step.
