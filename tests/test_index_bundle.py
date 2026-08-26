import json
import os

import numpy as np
import pytest

from revdict import index_bundle


def _write_staged_bundle(root, headword="word"):
    staging = index_bundle.create_staging_index(root)
    embeddings = np.array([[1.0, 0.0]], dtype="float32")
    record_embedding_indices = np.array([0], dtype="int32")
    np.save(staging / "embeddings.npy", embeddings)
    np.save(staging / "record_embeddings.npy", record_embedding_indices)
    (staging / "metadata.jsonl").write_text(
        json.dumps(
            {
                "headword": headword,
                "pos": "noun",
                "definition": "a definition",
                "examples": [],
                "source": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (staging / "word_index.json").write_text(
        json.dumps({headword: [0]}), encoding="utf-8"
    )
    (staging / "literary_frequency.json").write_text("{}", encoding="utf-8")
    manifest = index_bundle.build_manifest(
        staging,
        record_count=1,
        definition_count=1,
        embeddings=embeddings,
        record_embedding_indices=record_embedding_indices,
    )
    index_bundle.write_manifest(staging, manifest)
    return staging, manifest


def test_legacy_layout_remains_available_without_a_current_pointer(tmp_path):
    for filename in index_bundle.REQUIRED_ARTIFACTS:
        (tmp_path / filename).touch()

    assert index_bundle.resolve_active_index_dir(tmp_path) == tmp_path
    assert index_bundle.index_layout_exists(tmp_path)


def test_validated_bundle_is_published_through_one_atomic_pointer(tmp_path):
    staging, manifest = _write_staged_bundle(tmp_path)
    index_bundle.validate_index_directory(staging, verify_hashes=True)

    published = index_bundle.publish_staged_index(tmp_path, staging, manifest)

    assert published == tmp_path / "versions" / manifest["build_id"]
    assert (tmp_path / "current").is_symlink()
    assert index_bundle.resolve_active_index_dir(tmp_path) == published
    assert index_bundle.index_layout_exists(tmp_path)


def test_failed_pointer_swap_keeps_the_previous_bundle_current(tmp_path, monkeypatch):
    first_staging, first_manifest = _write_staged_bundle(tmp_path, "first")
    first = index_bundle.publish_staged_index(tmp_path, first_staging, first_manifest)
    second_staging, second_manifest = _write_staged_bundle(tmp_path, "second")

    real_replace = os.replace

    def fail_current_swap(source, destination):
        if destination == tmp_path / "current":
            raise OSError("simulated pointer swap failure")
        return real_replace(source, destination)

    monkeypatch.setattr(index_bundle.os, "replace", fail_current_swap)
    with pytest.raises(OSError, match="simulated"):
        index_bundle.publish_staged_index(tmp_path, second_staging, second_manifest)

    assert index_bundle.resolve_active_index_dir(tmp_path) == first
    assert second_staging.is_dir()
    assert list(tmp_path.glob(".current-*")) == []


def test_validation_rejects_a_checksum_mismatch(tmp_path):
    staging, _manifest = _write_staged_bundle(tmp_path)
    with (staging / "metadata.jsonl").open("a", encoding="utf-8") as f:
        f.write("{}\n")

    with pytest.raises(index_bundle.IndexValidationError, match="size mismatch"):
        index_bundle.validate_index_directory(staging, verify_hashes=True)


def test_validation_rejects_an_out_of_range_embedding_mapping(tmp_path):
    staging, manifest = _write_staged_bundle(tmp_path)
    np.save(staging / "record_embeddings.npy", np.array([9], dtype="int32"))
    manifest = index_bundle.build_manifest(
        staging,
        record_count=1,
        definition_count=1,
        embeddings=np.array([[1.0, 0.0]], dtype="float32"),
        record_embedding_indices=np.array([9], dtype="int32"),
        build_id=manifest["build_id"],
    )
    index_bundle.write_manifest(staging, manifest)

    with pytest.raises(index_bundle.IndexValidationError, match="invalid vector reference"):
        index_bundle.validate_index_directory(staging)


def test_loaded_index_validation_rejects_out_of_range_word_reference(tmp_path):
    embeddings = np.array([[1.0]], dtype="float32")
    with pytest.raises(index_bundle.IndexValidationError, match="invalid row reference"):
        index_bundle.validate_loaded_index(
            tmp_path,
            embeddings,
            [{"headword": "word"}],
            {"word": [1]},
        )


def test_staging_cleanup_refuses_a_directory_outside_the_index_versions(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="Refusing"):
        index_bundle.discard_staging_index(tmp_path / "index", outside)


def test_staging_cleanup_refuses_a_symlink_even_inside_versions(tmp_path):
    versions = tmp_path / "index" / "versions"
    versions.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    staging_link = versions / ".staging-link"
    staging_link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="non-directory staging path"):
        index_bundle.discard_staging_index(tmp_path / "index", staging_link)
