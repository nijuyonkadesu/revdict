SORT_MODES = (
    "relevance",
    "alpha",
    "alpha_desc",
    "shortest",
    "longest",
    "most_common",
    "least_common",
    "most_formal",
    "oldest",
    "most_modern",
)

# Wiktionary sense tags treated as the informal end of the formal/informal
# register spectrum for --sort most_formal. Deliberately includes
# "informal" even though category.py's idiom_slang CATEGORY grouping
# excludes it (category.py:9-13, where "informal" is left out as "too
# broad" for that narrower category) -- a sort axis and a category filter
# are allowed to define "informal" differently; --category idiom_slang and
# --sort most_formal are not the same axis and must not be conflated.
_INFORMAL_REGISTER_TAGS = {"slang", "vulgar", "colloquial", "idiomatic", "informal"}

# Same vocabulary as category.py's "old" category (archaic/dated/obsolete/
# historical) -- reused here for the oldest/most_modern sort axis.
_OLD_REGISTER_TAGS = {"archaic", "dated", "obsolete", "historical"}


def _formality_rank(candidate: dict) -> int:
    """0 (formal, ranks first) / 1 (neutral -- no tag, or only an
    old-register/literary tag) / 2 (informal, ranks last). Real spot check
    against actual search() candidate pools (12 diverse queries, 360
    candidates checked against the real Wiktionary dump) found the
    explicit "formal" tag on only ~2% of candidates but an informal-family
    tag on a large share of register-rich queries (e.g. "toilet" surfaces
    khazi/biffy/pisser via their own slang senses) -- so this rank is
    built around demoting the common informal signal, not promoting the
    rare formal one."""
    tags = set(candidate.get("tags") or [])
    if "formal" in tags:
        return 0
    if tags & _INFORMAL_REGISTER_TAGS:
        return 2
    return 1


def _oldness_rank(candidate: dict) -> int:
    """0 (old-tagged, ranks first) / 1 (not old-tagged). Untagged
    candidates tie with each other in their original relevance order
    (Python's sorted() is stable and this uses no secondary key) -- this
    is deliberate: a word's matched sense is frequently untagged even when
    the same headword has a separate archaic sense elsewhere, so there is
    no reliable secondary signal to break the tie on. See this plan's
    Global Constraints for the measured tag-density numbers behind this
    call."""
    tags = set(candidate.get("tags") or [])
    return 0 if tags & _OLD_REGISTER_TAGS else 1


def _modernness_rank(candidate: dict) -> int:
    """Exact mirror of _oldness_rank -- not-old-tagged ranks first."""
    return 1 - _oldness_rank(candidate)


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
    if sort_mode == "most_formal":
        return sorted(candidates, key=_formality_rank)
    if sort_mode == "oldest":
        return sorted(candidates, key=_oldness_rank)
    if sort_mode == "most_modern":
        return sorted(candidates, key=_modernness_rank)
    raise ValueError(f"Unknown sort mode: {sort_mode!r}")
