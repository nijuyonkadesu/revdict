"""Shared, truthful progress events for daemon-backed searches."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable


@dataclass(frozen=True)
class SearchStage:
    id: str
    label: str


STAGES = (
    SearchStage("ready", "Ready search state"),
    SearchStage("validate", "Validate query and filters"),
    SearchStage("phonetics", "Resolve phonetic targets"),
    SearchStage("parse", "Parse query syntax"),
    SearchStage("scope", "Determine candidate scope"),
    SearchStage("retrieve", "Retrieve candidates"),
    SearchStage("rerank", "Rerank definitions"),
    SearchStage("filter", "Resolve exact match and apply filters"),
    SearchStage("enrich", "Enrich, score, and sort results"),
    SearchStage("finalize", "Finalize the response"),
)
_STAGES_BY_ID = {stage.id: (ordinal, stage) for ordinal, stage in enumerate(STAGES, start=1)}


class ProgressReporter:
    """Turn internal stage transitions into stable UI/daemon event payloads."""

    def __init__(self, emit: Callable[[dict], None] | None = None) -> None:
        self._emit = emit

    def active(self, stage_id: str) -> None:
        self._report(stage_id, "active")

    def completed(self, stage_id: str) -> None:
        self._report(stage_id, "completed")

    def skipped(self, stage_id: str) -> None:
        self._report(stage_id, "skipped")

    def _report(self, stage_id: str, state: str) -> None:
        if self._emit is None:
            return
        ordinal, stage = _STAGES_BY_ID[stage_id]
        self._emit(
            {
                "type": "stage",
                "id": stage.id,
                "state": state,
                "ordinal": ordinal,
                "total": len(STAGES),
                "label": stage.label,
            }
        )
