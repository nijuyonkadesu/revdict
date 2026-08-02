from revdict.progress import ProgressReporter, STAGES


def test_progress_reporter_emits_described_stage_lifecycle_events():
    """A broken stage id or lifecycle must never leave the UI guessing what ran."""
    events = []
    reporter = ProgressReporter(events.append)

    reporter.active("ready")
    reporter.completed("ready")
    reporter.skipped("filter")

    assert events == [
        {"type": "stage", "id": "ready", "state": "active", "ordinal": 1, "total": 9, "label": "Ready"},
        {"type": "stage", "id": "ready", "state": "completed", "ordinal": 1, "total": 9, "label": "Ready"},
        {"type": "stage", "id": "filter", "state": "skipped", "ordinal": 7, "total": 9, "label": "Resolving exact match and applying filters"},
    ]


def test_progress_stage_catalog_is_the_ten_steps_presented_by_the_ui():
    assert [stage.id for stage in STAGES] == [
        "ready", "validate", "phonetics", "parse", "scope", "retrieve", "filter", "enrich", "finalize"
    ]


def test_progress_reporter_can_update_the_detail_of_a_long_running_stage():
    events = []
    reporter = ProgressReporter(events.append)

    reporter.detail("ready", "Loading embedding index")

    assert events == [{
        "type": "stage", "id": "ready", "state": "active", "ordinal": 1,
        "total": 9, "label": "Ready", "detail": "Loading embedding index",
    }]
