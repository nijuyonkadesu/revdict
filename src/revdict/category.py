CATEGORIES = ("all", "noun", "adjective", "verb", "adverb", "idiom_slang", "old")

_POS_CATEGORIES = {"noun", "adjective", "verb", "adverb"}

# Confirmed against the real Wiktionary dump (2026-07 sample, ~371K
# senses): archaic 6003, dated 4139, obsolete 13038, historical 4918.
_OLD_TAGS = {"archaic", "dated", "obsolete", "historical"}

# Confirmed against the same sample: idiomatic 5789, slang 12203, vulgar
# 1580, colloquial 3450; pos values phrase 679, proverb 280. "informal"
# (7536, also real) is deliberately excluded -- it's a broad, common
# register marker that would dilute this category far past what
# "Idioms/Slang" is meant to mean.
_IDIOM_SLANG_TAGS = {"idiomatic", "slang", "vulgar", "colloquial"}
_IDIOM_SLANG_POS = {"phrase", "proverb"}


def matches_category(record: dict, category: str | None) -> bool:
    if not category or category == "all":
        return True
    if category in _POS_CATEGORIES:
        return record.get("pos") == category
    if category == "idiom_slang":
        return record.get("pos") in _IDIOM_SLANG_POS or bool(
            set(record.get("tags") or []) & _IDIOM_SLANG_TAGS
        )
    if category == "old":
        return bool(set(record.get("tags") or []) & _OLD_TAGS)
    raise ValueError(f"Unknown category: {category!r}")
