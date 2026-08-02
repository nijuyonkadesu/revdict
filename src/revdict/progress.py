"""Shared, truthful progress events for daemon-backed searches."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable


@dataclass(frozen=True)
class SearchStage:
    id: str
    label: str


STAGES = (
    SearchStage("ready", "Ready"),
    SearchStage("validate", "Validating query & filters"),
    SearchStage("phonetics", "Resolving phonetic targets"),
    SearchStage("parse", "Parsing query syntax"),
    SearchStage("scope", "Determining candidate scope"),
    SearchStage("retrieve", "Retrieving candidates"),
    SearchStage("filter", "Resolving exact match and applying filters"),
    SearchStage("enrich", "Enriching, scoring and sorting results"),
    SearchStage("finalize", "Done"),
)
_STAGES_BY_ID = {stage.id: (ordinal, stage) for ordinal, stage in enumerate(STAGES, start=1)}


class ProgressReporter:
    """Turn internal stage transitions into stable UI/daemon event payloads."""

    def __init__(self, emit: Callable[[dict], None] | None = None) -> None:
        self._emit = emit

    def active(self, stage_id: str, detail: str | None = None) -> None:
        self._report(stage_id, "active", detail)

    def completed(self, stage_id: str) -> None:
        self._report(stage_id, "completed")

    def skipped(self, stage_id: str) -> None:
        self._report(stage_id, "skipped")

    def detail(self, stage_id: str, message: str) -> None:
        """Update a genuine sub-operation without fabricating another stage."""
        self._report(stage_id, "active", message)

    def _report(self, stage_id: str, state: str, detail: str | None = None) -> None:
        if self._emit is None:
            return
        ordinal, stage = _STAGES_BY_ID[stage_id]
        event = {"type": "stage", "id": stage.id, "state": state, "ordinal": ordinal, "total": len(STAGES), "label": stage.label}
        if detail:
            event["detail"] = detail
        self._emit(event)
