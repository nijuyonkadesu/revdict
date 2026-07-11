import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from revdict.paths import QUERY_HISTORY_PATH

# fzf's documented exit codes (see `man fzf` EXIT STATUS): 0 = normal, 1 = no
# match, 2 = error, 130 = interrupted (Ctrl-C or Esc). 1 and 130 both mean
# "the user didn't make a selection" (cancellation or filtering down to
# nothing) and should be treated as a quiet non-error. Anything else nonzero
# (observed empirically in this environment as 2, with stderr "inappropriate
# ioctl for device" when there's no controlling terminal) is a genuine
# runtime failure that must not be silently swallowed.
_CANCELLED_RETURN_CODES = {1, 130}


class PickerError(RuntimeError):
    """Raised when fzf ran but exited with a genuine runtime error -- as
    opposed to the user simply cancelling (Esc/Ctrl-C) or filtering to no
    matches. Callers should catch this and fall back to a non-interactive
    rendering rather than let the picker silently produce no output."""

    def __init__(self, returncode: int, stderr: str):
        super().__init__(f"fzf exited with code {returncode}: {stderr.strip() or '(no stderr)'}")
        self.returncode = returncode
        self.stderr = stderr


def format_candidate_line(
    headword: str,
    pos: str,
    definition: str,
    emotion_label: str,
    polarity: str,
    relevance: int,
    index: int,
    is_exact: bool = False,
) -> str:
    marker = "★" if is_exact else " "
    gloss = definition if len(definition) <= 70 else definition[:67] + "..."
    fields = [
        f"{marker} {headword}",
        f"({pos}) {gloss}",
        f"[{emotion_label} · {polarity}]",
        f"{relevance}%",
        str(index),
    ]
    return "\t".join(fields)


def parse_selection(fzf_stdout: str) -> int | None:
    line = fzf_stdout.strip()
    if not line:
        return None
    return int(line.rsplit("\t", 1)[1])


def _render_exact_preview(exact_match: dict) -> str:
    lines = [f"Exact match — {exact_match['headword']}", ""]
    for sense in exact_match["senses"]:
        lines.append(f"({sense['pos']}) {sense['definition']}")
        if sense.get("stress"):
            lines.append(f"Stress: {sense['stress']}")
        lines.append(f"Emotion: {sense['label']} · {sense['polarity']}")
        synonyms = sense.get("synonyms")
        if synonyms:
            lines.append(f"Synonyms: {', '.join(synonyms)}")
        for example in sense["examples"]:
            lines.append(f'    "{example}"')
        lines.append("")
    return "\n".join(lines)


def _render_candidate_preview(candidate: dict) -> str:
    lines = [
        f"{candidate['headword']} ({candidate['pos']})",
        "",
        candidate["definition"],
        "",
    ]
    if candidate.get("stress"):
        lines.append(f"Stress: {candidate['stress']}")
    lines.append(f"Emotion: {candidate['label']} · {candidate['polarity']}")
    lines.append(f"Match confidence: {candidate['relevance']}%")
    synonyms = candidate.get("synonyms")
    if synonyms:
        lines.append(f"Synonyms: {', '.join(synonyms)}")
    if candidate["examples"]:
        lines.append("")
        for example in candidate["examples"]:
            lines.append(f'"{example}"')
    return "\n".join(lines)


def write_candidate_files(
    tmp_path: Path, candidates: list[dict], exact_match: dict | None
) -> list[str]:
    """Writes one preview .txt file per row to tmp_path and returns the
    matching tab-delimited fzf input lines. Shared by run_picker's one-shot
    session (writes once, invokes fzf once) and the live session's
    change:reload path (invoked repeatedly against the same tmp_path as the
    query changes)."""
    lines = []
    index = 0

    if exact_match is not None:
        first_sense = exact_match["senses"][0]
        (tmp_path / f"{index}.txt").write_text(
            _render_exact_preview(exact_match), encoding="utf-8"
        )
        lines.append(
            format_candidate_line(
                exact_match["headword"],
                first_sense["pos"],
                first_sense["definition"],
                first_sense["label"],
                first_sense["polarity"],
                100,
                index=index,
                is_exact=True,
            )
        )
        index += 1

    for candidate in candidates:
        (tmp_path / f"{index}.txt").write_text(
            _render_candidate_preview(candidate), encoding="utf-8"
        )
        lines.append(
            format_candidate_line(
                candidate["headword"],
                candidate["pos"],
                candidate["definition"],
                candidate["label"],
                candidate["polarity"],
                candidate["relevance"],
                index=index,
            )
        )
        index += 1

    return lines


def run_picker(candidates: list[dict], exact_match: dict | None) -> str | None:
    if shutil.which("fzf") is None:
        return None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        lines = write_candidate_files(tmp_path, candidates, exact_match)

        input_text = "\n".join(lines) + "\n"
        result = subprocess.run(
            [
                "fzf",
                "--delimiter",
                "\t",
                "--with-nth=1,2,3,4",
                "--preview",
                f"cat {tmp_path}/{{5}}.txt",
                "--preview-window",
                "right:60%:wrap",
                "--bind",
                "?:toggle-preview",
            ],
            input=input_text,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            if result.returncode in _CANCELLED_RETURN_CODES:
                return None
            raise PickerError(result.returncode, result.stderr)

        selection_index = parse_selection(result.stdout)
        if selection_index is None:
            return None
        if exact_match is not None:
            if selection_index == 0:
                return exact_match["headword"]
            return candidates[selection_index - 1]["headword"]
        return candidates[selection_index]["headword"]


def build_live_session_args(
    preview_dir: Path,
    history_path: Path,
    python_executable: str,
    debounce_seconds: float = 0.1,
    layout_threshold_columns: int = 50,
) -> list[str]:
    """Builds the fzf argument list for the persistent live-typing session
    (see docs/superpowers/specs/2026-07-11-live-interactive-cli-design.md).
    Pure and side-effect-free so the exact bindings are unit-testable
    without a real terminal -- run_live_session is the thin wrapper that
    actually invokes this as a subprocess.

    layout_threshold_columns is in the units fzf itself uses for the
    `<SIZE_THRESHOLD` clause of --preview-window: the width of the PREVIEW
    WINDOW, not the full terminal (`man fzf`, PREVIEW WINDOW section). Since
    the base layout below is `right,50%`, the preview window is ~half the
    terminal's width, so this value should be about half of the real
    terminal-column width you want the stacked-layout switch to trigger at.
    Empirically confirmed in a real tmux/fzf session (task-5 manual
    validation): 50 here produces the layout switch at a real terminal
    width of 100 columns, matching the design spec's target. An earlier
    default of 100 was a bug -- it produced a real-world switch at ~200
    terminal columns, double what was intended, because it was mistakenly
    set as if the threshold applied to the full terminal width."""
    reload_command = (
        f"sleep {debounce_seconds}; "
        f"{python_executable} -u -m revdict.cli --query-only {{q}}"
    )
    return [
        "fzf",
        "--disabled",
        "--delimiter",
        "\t",
        "--with-nth=1,2,3,4",
        f"--history={history_path}",
        "--preview",
        f"cat {preview_dir}/{{5}}.txt",
        "--preview-window",
        f"right,50%,wrap,<{layout_threshold_columns}(up,50%)",
        "--bind",
        f"start:reload:{reload_command}",
        "--bind",
        f"change:reload:{reload_command}",
        "--bind",
        "esc:clear-query",
        "--bind",
        f"enter:execute-silent(echo {{q}} >> {history_path})+clear-query",
        "--bind",
        "ctrl-c:abort",
        "--bind",
        "ctrl-d:abort",
        "--bind",
        "up:prev-history",
        "--bind",
        "down:next-history",
        "--bind",
        "ctrl-p:up",
        "--bind",
        "ctrl-n:down",
        "--bind",
        "?:toggle-preview",
    ]


def run_live_session() -> None:
    if shutil.which("fzf") is None:
        return None

    QUERY_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUERY_HISTORY_PATH.touch(exist_ok=True)

    preview_dir = tempfile.mkdtemp(prefix="revdict-live-")
    try:
        args = build_live_session_args(
            preview_dir=Path(preview_dir),
            history_path=QUERY_HISTORY_PATH,
            python_executable=sys.executable,
        )
        subprocess.run(
            args,
            env={**os.environ, "REVDICT_LIVE_PREVIEW_DIR": preview_dir},
        )
    finally:
        shutil.rmtree(preview_dir, ignore_errors=True)
    return None
