# tests/test_picker.py
import pytest
import tempfile
from pathlib import Path

from revdict import picker
from revdict.picker import (
    PickerError,
    _render_exact_preview,
    format_candidate_line,
    parse_selection,
    run_picker,
    write_candidate_files,
)

_CANDIDATE_FIXTURE = [
    {
        "headword": "joyful",
        "pos": "adjective",
        "definition": "feeling great happiness",
        "examples": [],
        "label": "joy",
        "polarity": "positive",
        "relevance": 90,
    }
]


class _FakeCompletedProcess:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

_EXACT_MATCH_FIXTURE = {
    "headword": "happy",
    "senses": [
        {
            "pos": "adjective",
            "definition": "feeling great pleasure",
            "examples": ["a happy child"],
            "source": "wordnet",
            "synonyms": ["glad", "content"],
            "label": "joy",
            "polarity": "positive",
        },
        {
            "pos": "adjective",
            "definition": "willing to do something",
            "examples": [],
            "source": "wiktionary",
            "synonyms": None,
            "label": "neutral",
            "polarity": "neutral",
        },
    ],
}


def test_write_candidate_files_returns_one_line_per_candidate_plus_exact_match():
    with tempfile.TemporaryDirectory() as tmp:
        lines = write_candidate_files(Path(tmp), _CANDIDATE_FIXTURE, _EXACT_MATCH_FIXTURE)

        assert len(lines) == 2  # exact match + 1 candidate
        assert lines[0].startswith("★")
        assert (Path(tmp) / "0.txt").exists()
        assert (Path(tmp) / "1.txt").exists()


def test_write_candidate_files_with_no_exact_match_writes_only_candidates():
    with tempfile.TemporaryDirectory() as tmp:
        lines = write_candidate_files(Path(tmp), _CANDIDATE_FIXTURE, None)

        assert len(lines) == 1
        assert not lines[0].startswith("★")
        assert (Path(tmp) / "0.txt").exists()
        assert not (Path(tmp) / "1.txt").exists()


def test_format_candidate_line_has_five_tab_fields_and_marks_exact_match():
    line = format_candidate_line(
        "happy", "adjective", "feeling pleasure", "Joy", "positive", 92, index=3, is_exact=True
    )
    fields = line.split("\t")
    assert len(fields) == 5
    assert fields[-1] == "3"
    assert fields[0].startswith("★")


def test_format_candidate_line_truncates_long_definitions():
    long_definition = "x" * 200
    line = format_candidate_line(
        "word", "noun", long_definition, "neutral", "neutral", 50, index=0
    )
    gloss_field = line.split("\t")[1]
    assert len(gloss_field) < 100


def test_parse_selection_extracts_trailing_index_or_none_for_empty_input():
    line = format_candidate_line("joyful", "adjective", "x", "Joy", "positive", 80, index=5)
    assert parse_selection(line + "\n") == 5
    assert parse_selection("") is None
    assert parse_selection("   ") is None


def test_render_exact_preview_shows_real_per_sense_emotion_badge_and_synonyms():
    """Fix 1 + Fix 2: the exact-match preview pane must show each sense's real
    emotion tag (previously nothing was shown at all for the exact match) and
    synonyms when present, without a dangling "Synonyms:" label when absent."""
    preview = _render_exact_preview(_EXACT_MATCH_FIXTURE)

    assert "joy · positive" in preview
    assert "neutral · neutral" in preview
    assert "glad, content" in preview
    # The second sense has no synonyms -- must not print an empty label.
    assert "Synonyms: \n" not in preview
    assert "Synonyms:\n" not in preview
    assert preview.count("Synonyms:") == 1


def test_render_exact_preview_shows_stress_info_when_present():
    fixture = {
        "headword": "happy",
        "senses": [
            {
                "pos": "adjective",
                "definition": "feeling great pleasure",
                "examples": [],
                "source": "wordnet",
                "synonyms": None,
                "label": "joy",
                "polarity": "positive",
                "stress": "HAPpy",
            }
        ],
    }

    preview = _render_exact_preview(fixture)

    assert "Stress: HAPpy" in preview


def test_render_candidate_preview_omits_stress_line_when_absent():
    from revdict.picker import _render_candidate_preview

    candidate = dict(_CANDIDATE_FIXTURE[0])
    candidate["stress"] = None

    preview = _render_candidate_preview(candidate)

    assert "Stress:" not in preview


def test_render_candidate_preview_shows_synonyms_when_present():
    from revdict.picker import _render_candidate_preview

    candidate = dict(_CANDIDATE_FIXTURE[0])
    candidate["synonyms"] = ["glad", "content"]

    preview = _render_candidate_preview(candidate)

    assert "Synonyms: glad, content" in preview


def test_render_candidate_preview_omits_synonyms_line_when_absent():
    from revdict.picker import _render_candidate_preview

    candidate = dict(_CANDIDATE_FIXTURE[0])
    candidate["synonyms"] = None

    preview = _render_candidate_preview(candidate)

    assert "Synonyms:" not in preview


def test_run_picker_returns_none_when_fzf_binary_is_missing(monkeypatch):
    monkeypatch.setattr(picker.shutil, "which", lambda name: None)

    result = run_picker(_CANDIDATE_FIXTURE, None)

    assert result is None


def test_run_picker_returns_none_on_user_cancellation_without_raising(monkeypatch):
    """fzf's documented exit codes: 130 = interrupted (Ctrl-C or Esc). This
    must be treated as a quiet cancellation, not an error -- no fallback,
    no warning, just None."""
    monkeypatch.setattr(picker.shutil, "which", lambda name: "/usr/bin/fzf")
    monkeypatch.setattr(
        picker.subprocess,
        "run",
        lambda *a, **k: _FakeCompletedProcess(returncode=130, stdout=""),
    )

    result = run_picker(_CANDIDATE_FIXTURE, None)

    assert result is None


def test_run_picker_returns_none_on_no_match_exit_code(monkeypatch):
    """Exit code 1 = "no match" (e.g. user filtered to zero results and hit
    enter) -- also a quiet, non-error cancellation-like outcome."""
    monkeypatch.setattr(picker.shutil, "which", lambda name: "/usr/bin/fzf")
    monkeypatch.setattr(
        picker.subprocess,
        "run",
        lambda *a, **k: _FakeCompletedProcess(returncode=1, stdout=""),
    )

    result = run_picker(_CANDIDATE_FIXTURE, None)

    assert result is None


def test_run_picker_raises_picker_error_on_genuine_runtime_failure(monkeypatch):
    """A nonzero exit that isn't cancellation (e.g. fzf's exit code 2 "Error",
    observed empirically in this environment for "no controlling terminal")
    must be distinguishable from cancellation so the caller can fall back
    instead of silently returning nothing."""
    monkeypatch.setattr(picker.shutil, "which", lambda name: "/usr/bin/fzf")
    monkeypatch.setattr(
        picker.subprocess,
        "run",
        lambda *a, **k: _FakeCompletedProcess(
            returncode=2, stdout="", stderr="inappropriate ioctl for device"
        ),
    )

    with pytest.raises(PickerError) as excinfo:
        run_picker(_CANDIDATE_FIXTURE, None)

    assert excinfo.value.returncode == 2
    assert "ioctl" in excinfo.value.stderr


def test_run_picker_parses_a_normal_selection_on_success(monkeypatch):
    monkeypatch.setattr(picker.shutil, "which", lambda name: "/usr/bin/fzf")
    monkeypatch.setattr(
        picker.subprocess,
        "run",
        lambda *a, **k: _FakeCompletedProcess(returncode=0, stdout="whatever\t0\n"),
    )

    result = run_picker(_CANDIDATE_FIXTURE, None)

    assert result == "joyful"


def test_run_picker_pins_the_exact_match_with_a_real_emotion_badge_not_placeholders(monkeypatch):
    """Fix 1: the pinned exact-match fzf line previously hardcoded
    "exact match" / "n/a" instead of a real badge. Capture the actual input
    fed to fzf and assert it carries the real per-sense label/polarity."""
    captured = {}

    def fake_run(*args, **kwargs):
        captured["input"] = kwargs["input"]
        return _FakeCompletedProcess(returncode=0, stdout="whatever\t0\n")

    monkeypatch.setattr(picker.shutil, "which", lambda name: "/usr/bin/fzf")
    monkeypatch.setattr(picker.subprocess, "run", fake_run)

    result = run_picker(_CANDIDATE_FIXTURE, _EXACT_MATCH_FIXTURE)

    pinned_line = captured["input"].splitlines()[0]
    assert "joy · positive" in pinned_line
    assert "exact match" not in pinned_line
    assert "n/a" not in pinned_line
    assert result == "happy"


def test_build_live_session_args_includes_disabled_and_reload_bindings():
    args = picker.build_live_session_args(
        preview_dir=Path("/tmp/preview"),
        history_path=Path("/tmp/history"),
        python_executable="/usr/bin/python3",
    )

    assert "--disabled" in args
    joined = " ".join(args)
    assert "change:reload:sleep 0.1" in joined
    assert "/usr/bin/python3 -u -m revdict.cli --query-only {q}" in joined
    assert "start:reload:sleep 0.1" in joined


def test_build_live_session_args_does_not_use_fzfs_native_history_flag():
    args = picker.build_live_session_args(
        preview_dir=Path("/tmp/preview"),
        history_path=Path("/tmp/history"),
        python_executable="/usr/bin/python3",
    )

    joined = " ".join(args)
    assert "--history=" not in joined


def test_build_live_session_args_rebinds_esc_enter_and_arrow_keys():
    args = picker.build_live_session_args(
        preview_dir=Path("/tmp/preview"),
        history_path=Path("/tmp/history"),
        python_executable="/usr/bin/python3",
    )

    joined = " ".join(args)
    assert "esc:clear-query" in joined
    assert "enter:execute-silent(echo {q} >> /tmp/history)+clear-query" in joined
    assert "up:transform-query:tail -n 1 /tmp/history" in joined
    assert "down:clear-query" in joined
    assert "ctrl-p:up" in joined
    assert "ctrl-n:down" in joined


def test_build_live_session_args_uses_the_given_debounce_and_threshold():
    args = picker.build_live_session_args(
        preview_dir=Path("/tmp/preview"),
        history_path=Path("/tmp/history"),
        python_executable="/usr/bin/python3",
        debounce_seconds=0.25,
        layout_threshold_columns=80,
    )

    joined = " ".join(args)
    assert "sleep 0.25" in joined
    assert "<80(up,50%)" in joined


def test_build_live_session_args_preview_command_reads_from_the_preview_dir():
    args = picker.build_live_session_args(
        preview_dir=Path("/tmp/preview"),
        history_path=Path("/tmp/history"),
        python_executable="/usr/bin/python3",
    )

    joined = " ".join(args)
    assert "cat /tmp/preview/{5}.txt" in joined


def test_run_live_session_returns_none_when_fzf_binary_is_missing(monkeypatch):
    monkeypatch.setattr(picker.shutil, "which", lambda name: None)

    result = picker.run_live_session()

    assert result is None


def test_run_live_session_invokes_fzf_with_the_built_args_and_cleans_up(monkeypatch):
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs.get("env")
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(picker.shutil, "which", lambda name: "/usr/bin/fzf")
    monkeypatch.setattr(picker.subprocess, "run", fake_run)

    created_dirs = []
    real_mkdtemp = picker.tempfile.mkdtemp

    def tracking_mkdtemp(*a, **k):
        path = real_mkdtemp(*a, **k)
        created_dirs.append(path)
        return path

    monkeypatch.setattr(picker.tempfile, "mkdtemp", tracking_mkdtemp)

    picker.run_live_session()

    assert captured["args"][0] == "fzf"
    assert "--disabled" in captured["args"]
    assert captured["env"]["REVDICT_LIVE_PREVIEW_DIR"] == created_dirs[0]
    assert not picker.Path(created_dirs[0]).exists()  # cleaned up after fzf exits
