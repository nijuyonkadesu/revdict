import os
import sys

# huggingface_hub/transformers snapshot these into module-level constants the
# moment they're first imported, so they must be set before that import
# happens anywhere in the process — not merely before model construction.
# `revdict.data.build_index` (imported below) transitively imports them, so
# this has to run first, at true module-load time, based on real sys.argv
# (not the `argv` parameter `main()` accepts for testability).
if not (len(sys.argv) > 1 and sys.argv[1] == "build-index"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import shutil  # noqa: E402

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from revdict import search as search_mod  # noqa: E402
from revdict.data.build_index import build  # noqa: E402
from revdict.paths import INDEX_DIR  # noqa: E402
from revdict.picker import PickerError, run_picker  # noqa: E402

console = Console()


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
        table.add_column("Emotion")
        table.add_column("Synonyms")
        for sense in result["exact_match"]["senses"]:
            synonyms = sense.get("synonyms")
            table.add_row(
                sense["pos"],
                sense["definition"],
                f"{sense['label']} · {sense['polarity']}",
                ", ".join(synonyms) if synonyms else "",
            )
        console.print(table)

    table = Table(title="Related words you might mean")
    table.add_column("#")
    table.add_column("Word")
    table.add_column("Definition")
    table.add_column("Emotion")
    table.add_column("Relevance")
    for position, candidate in enumerate(result["candidates"], start=1):
        table.add_row(
            str(position),
            candidate["headword"],
            candidate["definition"],
            f"{candidate['label']} · {candidate['polarity']}",
            f"{candidate['relevance']}%",
        )
    console.print(table)


def _run_query(query: str, top_n: int, interactive: bool) -> int:
    if not query.strip():
        console.print("[yellow]Please enter a word or phrase.[/yellow]")
        return 0

    result = search_mod.search(query, top_n=top_n)

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


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] == "build-index":
        build(skip_confirm="--yes" in argv)
        return 0

    if not argv:
        if not _index_exists():
            _print_no_index_error()
            return 1
        query = console.input("[bold]> [/bold]")
        return _run_query(query, top_n=10, interactive=sys.stdout.isatty())

    no_interactive = "--no-interactive" in argv
    args = [arg for arg in argv if arg != "--no-interactive"]

    top_n = 10
    if "-n" in args:
        position = args.index("-n")
        top_n = int(args[position + 1])
        args = args[:position] + args[position + 2 :]

    query = " ".join(args)

    if not _index_exists():
        _print_no_index_error()
        return 1

    interactive = not no_interactive and sys.stdout.isatty()
    return _run_query(query, top_n, interactive)


if __name__ == "__main__":
    sys.exit(main())
