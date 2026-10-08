"""Integrity tests for the dataset registry shipped to reproducers.

These checksums are the repository's reproducibility promise (project conventions 4.5.1):
they are quoted in data/DATA.md and in the manuscript. A malformed or drifting
entry here silently breaks that promise, so the registry itself is tested.
"""

from __future__ import annotations

import os
import re

import pytest
from conftest import CODE_DIR, load_script

fd = load_script("fetch_datasets.py", "fetch_datasets")

SHA16 = re.compile(r"^[0-9a-f]{16}$")


def test_all_five_public_benchmarks_minus_the_restricted_one_are_registered() -> None:
    """Four external datasets; the apple benchmark is restricted and must be absent."""
    assert set(fd.EXTERNAL_DATASETS) == {"corn", "tablet", "ossl", "mango"}


def test_no_restricted_apple_data_is_listed() -> None:
    blob = repr(fd.EXTERNAL_DATASETS).lower()
    assert "apple" not in blob


def test_every_entry_has_the_documented_field_count() -> None:
    for did, entry in fd.EXTERNAL_DATASETS.items():
        assert len(entry) == 6, f"{did} has {len(entry)} fields, expected 6"


def test_checksums_are_well_formed_16_hex_prefixes() -> None:
    for did, (_n, _m, _s, _l, _p, sha) in fd.EXTERNAL_DATASETS.items():
        assert SHA16.match(sha), f"{did} has a malformed SHA-256 prefix: {sha!r}"


def test_checksums_are_unique() -> None:
    shas = [e[5] for e in fd.EXTERNAL_DATASETS.values()]
    assert len(set(shas)) == len(shas)


def test_every_source_is_an_https_url() -> None:
    for did, (_n, _m, source, _l, _p, _sha) in fd.EXTERNAL_DATASETS.items():
        assert "https://" in source, f"{did} has no resolvable source URL"


def test_every_entry_declares_a_license() -> None:
    for did, (_n, _m, _s, lic, _p, _sha) in fd.EXTERNAL_DATASETS.items():
        assert lic.strip(), f"{did} has an empty license field"


def test_relative_paths_are_dataset_scoped_and_not_absolute() -> None:
    for did, (_n, _m, _s, _l, rel, _sha) in fd.EXTERNAL_DATASETS.items():
        assert not os.path.isabs(rel), f"{did} uses an absolute path"
        assert ".." not in rel, f"{did} escapes the data root"
        assert rel.count("/") == 1, f"{did} should be <dataset_id>/<file>"


def test_registry_checksums_agree_with_data_md() -> None:
    """DATA.md and the script are two copies of one promise; they must not drift."""
    data_md = os.path.join(os.path.dirname(CODE_DIR), "data", "DATA.md")
    with open(data_md, encoding="utf-8") as fh:
        text = fh.read()
    for did, (_n, _m, _s, _l, _p, sha) in fd.EXTERNAL_DATASETS.items():
        assert sha in text, f"{did} checksum {sha} is missing from data/DATA.md"


def test_listing_runs_without_error(capsys: pytest.CaptureFixture[str]) -> None:
    fd.list_datasets()
    out = capsys.readouterr().out
    assert "corn" in out
    assert "restricted" in out.lower()


def test_verify_reports_failures_for_a_missing_tree(tmp_path: object) -> None:
    """verify() must count every absent dataset rather than silently passing."""
    failures = fd.verify(str(tmp_path))
    assert failures == len(fd.EXTERNAL_DATASETS)
