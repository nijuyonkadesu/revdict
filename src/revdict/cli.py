import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from revdict import category
from revdict import daemon
from revdict import picker
from revdict import sort
from revdict.paths import INDEX_DIR
from revdict.picker import PickerError, run_picker, write_candidate_files

console = Console()

LIVE_SESSION_TOP_N = 30


class _ArgumentError(Exception):
    """Raised instead of argparse's default sys.exit(2) on a usage error, so
    main() can return an ordinary exit code rather than aborting the whole
    process -- important both for real callers and for tests that invoke
    main() directly. --help's normal sys.exit(0) is left alone (that IS the
    correct behavior for real interactive use)."""

    def __init__(self, message: str, usage: str):
        super().__init__(message)
        self.message = message
        self.usage = usage


class _QuietArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _ArgumentError(message, self.format_usage())


def _build_index_parser() -> argparse.ArgumentParser:
    parser = _QuietArgumentParser(
        prog="revdict build-index", description="Build (or rebuild) the local search index."
    )
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    return parser


def _daemon_parser() -> argparse.ArgumentParser:
    parser = _QuietArgumentParser(
        prog="revdict daemon", description="Manage the background query daemon."
    )
    parser.add_argument("action", choices=["start", "stop", "status"])
    return parser


_ARPABET_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW",
}


def _validate_meter_pattern(value: str) -> str:
    """argparse type= callback: rejects a --meter value containing anything
    other than '/' and 'x' up front, via an ArgumentTypeError (argparse's
    own convention for type= validation failures, converted by this file's
    _QuietArgumentParser into the same clean error path as an invalid
    --sort/--category choice), rather than silently accepting garbage that
    would then just never match any real headword's meter string."""
    if not value or any(ch not in "/x" for ch in value):
        raise argparse.ArgumentTypeError(
            f"invalid meter pattern {value!r}: must contain only '/' and 'x'"
        )
    return value


def _query_parser() -> argparse.ArgumentParser:
    parser = _QuietArgumentParser(
        prog="revdict",
        description="Local offline reverse-dictionary CLI: look up a word, "
        "or describe a meaning to get candidate words.",
    )
    parser.add_argument(
        "query", nargs="*", help="Word or phrase to look up (omit to read from stdin)."
    )
    parser.add_argument(
        "-n", type=int, default=30, metavar="N", help="Number of candidates to show (default: 30)."
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Print a plain table instead of launching the fzf picker.",
    )
    parser.add_argument(
        "--sort",
        choices=list(sort.SORT_MODES),
        default=None,
        help='Sort order for results (default: relevance, i.e. "most similar").',
    )
    parser.add_argument(
        "--category",
        choices=list(category.CATEGORIES),
        default=None,
        help="Filter results by category (default: all).",
    )
    parser.add_argument(
        "--syllables", type=int, default=None, metavar="N",
        help="Filter results to headwords with exactly N syllables.",
    )
    parser.add_argument(
        "--primary-vowel", choices=list(_ARPABET_VOWELS), default=None, metavar="VOWEL",
        type=str.upper,
        help="Filter results to headwords whose primary-stressed vowel is VOWEL (an ARPAbet vowel symbol, e.g. AE).",
    )
    parser.add_argument(
        "--rhymes-with", default=None, metavar="WORD",
        help="Filter results to headwords that rhyme with WORD (resolved as a noun).",
    )
    parser.add_argument(
        "--sounds-like", default=None, metavar="WORD",
        help="Filter results to headwords that sound phonetically similar to WORD (resolved as a noun).",
    )
    parser.add_argument(
        "--meter", default=None, metavar="PATTERN", type=_validate_meter_pattern,
        help='Filter results to headwords matching a stress pattern of "/" (stressed) and "x" (unstressed) per syllable, e.g. "/x".',
    )
    return parser


def _index_exists() -> bool:
    return (INDEX_DIR / "embeddings.npy").exists()


def _fzf_missing() -> bool:
    return shutil.which("fzf") is None


def _print_no_index_error() -> None:
    console.print("[bold red]No index found.[/bold red] Run: [bold]revdict build-index[/bold]")


def _print_static_results(result: dict) -> None:
    if result["exact_match"] is not None:
        table = Table(title=f"Exact match — {result['exact_match']['headword']}")
        table.add_column("POS")
        table.add_column("Definition")
        table.add_column("Stress")
        table.add_column("Emotion")
        table.add_column("Synonyms")
        for sense in result["exact_match"]["senses"]:
            synonyms = sense.get("synonyms")
            stress_text = Text.from_ansi(sense["stress"]) if sense.get("stress") else ""
            table.add_row(
                sense["pos"],
                sense["definition"],
                stress_text,
                f"{sense['label']} · {sense['polarity']}",
                ", ".join(synonyms) if synonyms else "",
            )
        console.print(table)

    table = Table(title="Related words you might mean")
    table.add_column("#")
    table.add_column("Word")
    table.add_column("Definition")
    table.add_column("Stress")
    table.add_column("Emotion")
    table.add_column("Synonyms")
    table.add_column("Relevance")
    for position, candidate in enumerate(result["candidates"], start=1):
        stress_text = Text.from_ansi(candidate["stress"]) if candidate.get("stress") else ""
        synonyms = candidate.get("synonyms")
        table.add_row(
            str(position),
            candidate["headword"],
            candidate["definition"],
            stress_text,
            f"{candidate['label']} · {candidate['polarity']}",
            ", ".join(synonyms) if synonyms else "",
            f"{candidate['relevance']}%",
        )
    console.print(table)


def _local_search_fallback(
    query: str,
    top_n: int,
    sort_mode: str | None = None,
    category: str | None = None,
    syllables: int | None = None,
    primary_vowel: str | None = None,
    rhymes_with: str | None = None,
    sounds_like: str | None = None,
    meter: str | None = None,
) -> dict:
    from revdict.query_env import configure_offline_quiet_env

    configure_offline_quiet_env()
    from revdict import search as search_mod

    return search_mod.search(
        query,
        top_n=top_n,
        sort_mode=sort_mode,
        category=category,
        syllables=syllables,
        primary_vowel=primary_vowel,
        rhymes_with=rhymes_with,
        sounds_like=sounds_like,
        meter=meter,
    )


def _get_search_result(
    query: str,
    top_n: int,
    sort_mode: str | None = None,
    category: str | None = None,
    syllables: int | None = None,
    primary_vowel: str | None = None,
    rhymes_with: str | None = None,
    sounds_like: str | None = None,
    meter: str | None = None,
) -> dict:
    result = daemon.send_query(
        query, top_n, sort_mode=sort_mode, category=category, syllables=syllables,
        primary_vowel=primary_vowel, rhymes_with=rhymes_with, sounds_like=sounds_like, meter=meter,
    )
    if result is not None:
        return result
    if daemon.ensure_daemon_running():
        result = daemon.send_query(
            query, top_n, sort_mode=sort_mode, category=category, syllables=syllables,
            primary_vowel=primary_vowel, rhymes_with=rhymes_with, sounds_like=sounds_like, meter=meter,
        )
        if result is not None:
            return result
    return _local_search_fallback(
        query, top_n, sort_mode=sort_mode, category=category, syllables=syllables,
        primary_vowel=primary_vowel, rhymes_with=rhymes_with, sounds_like=sounds_like, meter=meter,
    )


def _build_index(skip_confirm: bool) -> None:
    from revdict.data.build_index import build

    build(skip_confirm=skip_confirm)

    if daemon.is_daemon_running():
        console.print(
            "[yellow]A revdict daemon is still running with the old index loaded — "
            "run `revdict daemon stop` so your next query picks up the refreshed "
            "data.[/yellow]"
        )


def _daemon_start() -> None:
    daemon.run_server()


def _daemon_stop() -> bool:
    return daemon.stop_daemon()


def _daemon_status() -> str:
    return daemon.daemon_status()


def _run_query(
    query: str,
    top_n: int,
    interactive: bool,
    sort_mode: str | None = None,
    category: str | None = None,
    syllables: int | None = None,
    primary_vowel: str | None = None,
    rhymes_with: str | None = None,
    sounds_like: str | None = None,
    meter: str | None = None,
) -> int:
    if not query.strip():
        console.print("[yellow]Please enter a word or phrase.[/yellow]")
        return 0

    result = _get_search_result(
        query, top_n, sort_mode=sort_mode, category=category, syllables=syllables,
        primary_vowel=primary_vowel, rhymes_with=rhymes_with, sounds_like=sounds_like, meter=meter,
    )

    if interactive:
        try:
            selected = run_picker(result["candidates"], result["exact_match"])
        except PickerError as error:
            console.print(
                f"[yellow]fzf exited unexpectedly (code {error.returncode}): "
                f"{error.stderr.strip() or 'no error output'}[/yellow]"
            )
            _print_static_results(result)
            return 0
        if selected is None and _fzf_missing():
            _print_static_results(result)
            return 0
        if selected:
            print(selected)
        return 0

    _print_static_results(result)
    return 0


def _run_query_only(query: str) -> int:
    if not query.strip():
        return 0

    preview_dir = Path(os.environ["REVDICT_LIVE_PREVIEW_DIR"])
    result = _get_search_result(query, LIVE_SESSION_TOP_N)
    lines = write_candidate_files(preview_dir, result["candidates"], result["exact_match"])
    for line in lines:
        print(line)
    return 0


def _run_jsonl_query(query: str) -> int:
    if not query.strip():
        return 0

    result = _get_search_result(query, LIVE_SESSION_TOP_N)
    rows = []
    if result["exact_match"] is not None:
        first_sense = result["exact_match"]["senses"][0]
        rows.append(
            {
                "headword": result["exact_match"]["headword"],
                "pos": first_sense["pos"],
                "definition": first_sense["definition"],
                "stress": first_sense.get("stress"),
                "label": first_sense["label"],
                "polarity": first_sense["polarity"],
                "synonyms": first_sense.get("synonyms") or [],
                "examples": first_sense["examples"],
                "relevance": 100,
                "is_exact": True,
            }
        )
    for candidate in result["candidates"]:
        rows.append(
            {
                "headword": candidate["headword"],
                "pos": candidate["pos"],
                "definition": candidate["definition"],
                "stress": candidate.get("stress"),
                "label": candidate["label"],
                "polarity": candidate["polarity"],
                "synonyms": candidate.get("synonyms") or [],
                "examples": candidate["examples"],
                "relevance": candidate["relevance"],
                "is_exact": False,
            }
        )
    for row in rows:
        print(json.dumps(row))
    return 0


_CLIPBOARD_TOOL_CANDIDATES = [
    ["wl-copy"],
    ["xclip", "-selection", "clipboard"],
    ["xsel", "--clipboard", "--input"],
    ["pbcopy"],
]


def _is_remote_session() -> bool:
    return bool(
        os.environ.get("TMUX")
        or os.environ.get("SSH_TTY")
        or os.environ.get("SSH_CONNECTION")
        or os.environ.get("SSH_CLIENT")
    )


def _build_osc52_sequence(text: str) -> str:
    """Builds the OSC 52 escape sequence that sets the terminal's
    clipboard to `text`. Pure and testable without a real tty --
    _copy_via_osc52 is the thin wrapper that actually writes this to
    /dev/tty (confirmed necessary during design: writing to stdout
    instead does not reach the pty tmux monitors for OSC 52)."""
    encoded = base64.b64encode(text.encode()).decode()
    return f"\x1b]52;c;{encoded}\x07"


def _copy_via_osc52(text: str) -> None:
    try:
        with open("/dev/tty", "w") as tty:
            tty.write(_build_osc52_sequence(text))
    except OSError:
        pass


def _copy_via_system_clipboard(text: str) -> None:
    for command in _CLIPBOARD_TOOL_CANDIDATES:
        if shutil.which(command[0]) is None:
            continue
        try:
            subprocess.run(command, input=text.encode(), check=True, timeout=2)
        except (subprocess.SubprocessError, OSError):
            continue
        return


def _run_copy_selection(headword: str) -> int:
    headword = headword.strip()
    if not headword:
        return 0
    if _is_remote_session():
        _copy_via_osc52(headword)
    else:
        _copy_via_system_clipboard(headword)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    try:
        if argv and argv[0] == "build-index":
            args = _build_index_parser().parse_args(argv[1:])
            _build_index(skip_confirm=args.yes)
            return 0

        if argv and argv[0] == "daemon":
            args = _daemon_parser().parse_args(argv[1:])
            if args.action == "start":
                _daemon_start()
                return 0
            if args.action == "stop":
                if _daemon_stop():
                    console.print("Daemon stopped.")
                else:
                    console.print("[yellow]Daemon was not running.[/yellow]")
                return 0
            console.print(_daemon_status())
            return 0

        if argv and argv[0] == "--query-only":
            query = argv[1] if len(argv) > 1 else ""
            return _run_query_only(query)

        if argv and argv[0] == "--jsonl-query":
            query = argv[1] if len(argv) > 1 else ""
            return _run_jsonl_query(query)

        if argv and argv[0] == "--copy-selection":
            headword = argv[1] if len(argv) > 1 else ""
            return _run_copy_selection(headword)

        if not argv:
            if not _index_exists():
                _print_no_index_error()
                return 1
            if sys.stdout.isatty():
                if _fzf_missing():
                    console.print(
                        "[yellow]Live mode requires fzf. Install it, or use "
                        "revdict \"your query\" for one-shot search.[/yellow]"
                    )
                    return 1
                picker.run_live_session()
                return 0
            query = console.input("[bold]> [/bold]")
            return _run_query(query, top_n=30, interactive=False)

        args = _query_parser().parse_args(argv)

        query = " ".join(args.query)

        if not _index_exists():
            _print_no_index_error()
            return 1

        interactive = not args.no_interactive and sys.stdout.isatty()
        return _run_query(
            query, args.n, interactive, sort_mode=args.sort, category=args.category,
            syllables=args.syllables, primary_vowel=args.primary_vowel, rhymes_with=args.rhymes_with,
            sounds_like=args.sounds_like, meter=args.meter,
        )
    except _ArgumentError as error:
        # markup=False: argparse's usage text contains literal square
        # brackets (e.g. "[query ...]") that Rich would otherwise try to
        # parse as its own markup tags, silently swallowing them.
        console.print(
            f"{error.usage.strip()}\nrevdict: error: {error.message}", style="red", markup=False
        )
        return 1
    except ValueError as error:
        # Surfaces search()/resolve_phonetic_target's deliberate fail-loud
        # ValueError (e.g. --rhymes-with/--sounds-like when stressmark is
        # missing) as a clean one-line message instead of an unhandled
        # traceback -- mirrors the _ArgumentError handler above.
        console.print(f"revdict: error: {error}", style="red")
        return 1


if __name__ == "__main__":
    sys.exit(main())
