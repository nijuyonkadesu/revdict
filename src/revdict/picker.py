import shutil
import subprocess
import tempfile
from pathlib import Path

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
        f"Emotion: {candidate['label']} · {candidate['polarity']}",
        f"Relative match: {candidate['relevance']}%",
    ]
    if candidate["examples"]:
        lines.append("")
        for example in candidate["examples"]:
            lines.append(f'"{example}"')
    return "\n".join(lines)


def run_picker(candidates: list[dict], exact_match: dict | None) -> str | None:
    if shutil.which("fzf") is None:
        return None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
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
