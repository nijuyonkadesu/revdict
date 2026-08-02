from revdict.progress import ProgressReporter, STAGES


def test_progress_reporter_emits_described_stage_lifecycle_events():
    """A broken stage id or lifecycle must never leave the UI guessing what ran."""
    events = []
    reporter = ProgressReporter(events.append)

    reporter.active("ready")
    reporter.completed("ready")
    reporter.skipped("rerank")

    assert events == [
        {"type": "stage", "id": "ready", "state": "active", "ordinal": 1, "total": 10, "label": "Ready search state"},
        {"type": "stage", "id": "ready", "state": "completed", "ordinal": 1, "total": 10, "label": "Ready search state"},
        {"type": "stage", "id": "rerank", "state": "skipped", "ordinal": 7, "total": 10, "label": "Rerank definitions"},
    ]


def test_progress_stage_catalog_is_the_ten_steps_presented_by_the_ui():
    assert [stage.id for stage in STAGES] == [
        "ready", "validate", "phonetics", "parse", "scope", "retrieve", "rerank", "filter", "enrich", "finalize"
    ]
