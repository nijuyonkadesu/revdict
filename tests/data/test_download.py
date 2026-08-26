import json
import gzip
from pathlib import Path

import pytest

from revdict.data import download


def test_download_publishes_validated_file_and_provenance(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("dataset", encoding="utf-8")
    destination = tmp_path / "cache" / "dataset.txt"

    info = download.download_cached_file(
        source.as_uri(),
        destination,
        validator=download.validate_nonempty_file,
        retries=1,
    )

    assert destination.read_text(encoding="utf-8") == "dataset"
    assert info["reused_cache"] is False
    provenance = json.loads(download.provenance_path(destination).read_text(encoding="utf-8"))
    assert provenance["sha256"] == download.sha256_file(destination)
    assert provenance["bytes"] == len("dataset")


def test_verified_cache_is_reused_without_reading_the_source_again(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("dataset", encoding="utf-8")
    destination = tmp_path / "dataset.txt"
    download.download_cached_file(
        source.as_uri(), destination, validator=download.validate_nonempty_file, retries=1
    )
    source.unlink()

    info = download.download_cached_file(
        source.as_uri(), destination, validator=download.validate_nonempty_file, retries=1
    )

    assert info["reused_cache"] is True
    assert destination.read_text(encoding="utf-8") == "dataset"


def test_failed_refresh_preserves_known_good_destination(tmp_path, monkeypatch):
    destination = tmp_path / "dataset.txt"
    destination.write_text("known-good", encoding="utf-8")

    def fail_open(*_args, **_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(download.urllib.request, "urlopen", fail_open)
    with pytest.raises(download.DatasetDownloadError, match="offline"):
        download.download_cached_file(
            "https://example.invalid/dataset",
            destination,
            validator=download.validate_nonempty_file,
            refresh=True,
            retries=1,
        )

    assert destination.read_text(encoding="utf-8") == "known-good"
    assert not destination.with_name("dataset.txt.download.part").exists()


def test_corrupt_cached_file_is_replaced_only_after_validation(tmp_path):
    source = tmp_path / "source.gz"
    source.write_bytes(b"\x1f\x8bvalid-enough-for-header-validation")
    destination = tmp_path / "dataset.gz"
    destination.write_text("corrupt", encoding="utf-8")

    download.download_cached_file(
        source.as_uri(), destination, validator=download.validate_gzip_header, retries=1
    )

    assert destination.read_bytes().startswith(b"\x1f\x8b")


def test_stale_provenance_checksum_causes_a_safe_redownload(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("fresh", encoding="utf-8")
    destination = tmp_path / "dataset.txt"
    destination.write_text("stale", encoding="utf-8")
    download.provenance_path(destination).write_text(
        json.dumps({"url": source.as_uri(), "bytes": 5, "sha256": "wrong"}),
        encoding="utf-8",
    )

    download.download_cached_file(
        source.as_uri(), destination, validator=download.validate_nonempty_file, retries=1
    )

    assert destination.read_text(encoding="utf-8") == "fresh"


def test_complete_gzip_validation_rejects_a_truncated_stream(tmp_path):
    path = tmp_path / "dataset.gz"
    with gzip.open(path, "wb") as f:
        f.write(b"dictionary data" * 100)
    path.write_bytes(path.read_bytes()[:-4])

    with pytest.raises((EOFError, OSError)):
        download.validate_gzip_stream(path)
