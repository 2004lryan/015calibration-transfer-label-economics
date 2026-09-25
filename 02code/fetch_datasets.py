"""
fetch_datasets.py: pointers and SHA-256 checksums for the public external
datasets used in the multi-benchmark calibration-transfer reckoning.

We do NOT re-host third-party datasets. This script documents where to obtain
each dataset and the expected SHA-256 of the raw source file, so that a
reproduction can verify integrity against the values reported in the manuscript.
Download each dataset from its official source (respecting its own license),
place it under 05data/<id>/, then run 61_benchmark_datasets.py.

The apple near-infrared benchmark is NOT public (XJAU restricted access) and is
therefore not listed here; request it from the corresponding author.

Usage:
    python 02code/fetch_datasets.py --list
    python 02code/fetch_datasets.py --verify /path/to/05data
"""

from __future__ import annotations

import argparse
import hashlib
import os

# id -> (name, modality / drift axis, official source, license, relative path, SHA-256 [first 16 hex])
EXTERNAL_DATASETS = {
    "corn": (
        "Cargill corn (NIR, 3 instruments m5/mp5/mp6)",
        "NIR point spectra / instrument drift",
        "Eigenvector Research data sets: https://eigenvector.com/resources/data-sets/",
        "free for research use, per publisher terms",
        "036_corn_nir_3instruments/corn.mat",
        "e28fd4be274a54ca",
    ),
    "tablet": (
        "IDRC 2002 'shootout' tablet (NIR, 2 instruments)",
        "NIR point spectra / instrument drift",
        "Eigenvector Research data sets (Software Shootout): https://eigenvector.com/resources/data-sets/",
        "free for research use, per publisher terms",
        "037_tablet_nir_shootout2002/nir_shootout_2002.mat",
        "129a32ec9e194e56",
    ),
    "ossl": (
        "Open Soil Spectral Library (MIR, cross-laboratory sub-libraries)",
        "MIR spectra / laboratory drift",
        "OpenGeoHub / Soil Spectroscopy: https://soilspectroscopy.github.io/ossl-manual/",
        "CC-BY 4.0",
        "038_ossl_global_soil_spectral/ossl_all_L1_v1.2.csv.gz",
        "2044946ad5a10af0",
    ),
    "mango": (
        "Anderson 2020 mango dry matter (NIR, 4 harvest seasons)",
        "Vis-NIR point spectra / season drift",
        "Anderson et al. 2020, Mendeley Data: https://data.mendeley.com/datasets/46htwnp833",
        "CC-BY 4.0",
        "041_mango_anderson_crossseason/NAnderson2020MendeleyMangoNIRData.csv",
        "86159e2df9244fa5",
    ),
}


def _sha256_16(path: str) -> str:
    """Return the first 16 hex characters of a file's SHA-256 digest."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def list_datasets() -> None:
    print("External public datasets (download from the official source; do not re-host):\n")
    for did, (name, modality, source, lic, rel, sha) in EXTERNAL_DATASETS.items():
        print(f"[{did}] {name}")
        print(f"    modality : {modality}")
        print(f"    source   : {source}")
        print(f"    license  : {lic}")
        print(f"    place at : 05data/{rel}")
        print(f"    SHA-256  : {sha}\n")
    print("Apple NIR benchmark: not public (XJAU restricted access); request from the corresponding author.")


def verify(root: str) -> int:
    """Check every expected file under `root`; return the number of failures."""
    failures = 0
    for did, (_name, _modality, _source, _lic, rel, sha) in EXTERNAL_DATASETS.items():
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            print(f"[{did}] MISSING   {path}")
            failures += 1
            continue
        got = _sha256_16(path)
        if got == sha:
            print(f"[{did}] OK        {rel}")
        else:
            print(f"[{did}] MISMATCH  {rel}  expected {sha}, got {got}")
            failures += 1
    print(f"\n{len(EXTERNAL_DATASETS) - failures}/{len(EXTERNAL_DATASETS)} datasets verified.")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset pointers and integrity checks.")
    parser.add_argument("--list", action="store_true", help="list dataset sources, licenses and checksums")
    parser.add_argument("--verify", metavar="DIR", help="verify SHA-256 of the datasets under DIR (an 05data root)")
    args = parser.parse_args()

    if args.verify:
        raise SystemExit(1 if verify(args.verify) else 0)
    list_datasets()


if __name__ == "__main__":
    main()
