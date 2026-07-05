import shutil
import sys

from rich.console import Console
from rich.table import Table

from revdict import daemon
from revdict.paths import INDEX_DIR
from revdict.picker import PickerError, run_picker

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


def _local_search_fallback(query: str, top_n: int) -> dict:
    from revdict.query_env import configure_offline_quiet_env

    configure_offline_quiet_env()
    from revdict import search as search_mod

    return search_mod.search(query, top_n=top_n)


def _get_search_result(query: str, top_n: int) -> dict:
    result = daemon.send_query(query, top_n)
    if result is not None:
        return result
    if daemon.ensure_daemon_running():
        result = daemon.send_query(query, top_n)
        if result is not None:
            return result
    return _local_search_fallback(query, top_n)


def _build_index(skip_confirm: bool) -> None:
    from revdict.data.build_index import build

    build(skip_confirm=skip_confirm)

    if "is running" in daemon.daemon_status():
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


def _run_query(query: str, top_n: int, interactive: bool) -> int:
    if not query.strip():
        console.print("[yellow]Please enter a word or phrase.[/yellow]")
        return 0

    result = _get_search_result(query, top_n)

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
        _build_index(skip_confirm="--yes" in argv)
        return 0

    if argv and argv[0] == "daemon":
        action = argv[1] if len(argv) > 1 else None
        if action == "start":
            _daemon_start()
            return 0
        if action == "stop":
            if _daemon_stop():
                console.print("Daemon stopped.")
            else:
                console.print("[yellow]Daemon was not running.[/yellow]")
            return 0
        if action == "status":
            console.print(_daemon_status())
            return 0
        console.print("[red]Usage: revdict daemon start|stop|status[/red]")
        return 1

    if not argv:
        if not _index_exists():
            _print_no_index_error()
            return 1
        query = console.input("[bold]> [/bold]")
        return _run_query(query, top_n=30, interactive=sys.stdout.isatty())

    no_interactive = "--no-interactive" in argv
    args = [arg for arg in argv if arg != "--no-interactive"]

    top_n = 30
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
