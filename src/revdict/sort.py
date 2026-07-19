SORT_MODES = (
    "relevance",
    "alpha",
    "alpha_desc",
    "shortest",
    "longest",
    "most_common",
    "least_common",
)


def apply_sort(
    candidates: list[dict], sort_mode: str | None, literary_frequency: dict[str, float]
) -> list[dict]:
    if not sort_mode or sort_mode == "relevance":
        return candidates
    if sort_mode == "alpha":
        return sorted(candidates, key=lambda c: c["headword"].lower())
    if sort_mode == "alpha_desc":
        return sorted(candidates, key=lambda c: c["headword"].lower(), reverse=True)
    if sort_mode == "shortest":
        return sorted(candidates, key=lambda c: (len(c["headword"]), c["headword"].lower()))
    if sort_mode == "longest":
        return sorted(candidates, key=lambda c: (-len(c["headword"]), c["headword"].lower()))
    if sort_mode == "most_common":
        return sorted(
            candidates,
            key=lambda c: (
                -literary_frequency.get(c["headword"].lower(), 0.0),
                c["headword"].lower(),
            ),
        )
    if sort_mode == "least_common":
        return sorted(
            candidates,
            key=lambda c: (
                literary_frequency.get(c["headword"].lower(), 0.0),
                c["headword"].lower(),
            ),
        )
    raise ValueError(f"Unknown sort mode: {sort_mode!r}")
