from dataclasses import dataclass, field

_WILDCARD_TRIGGER_CHARS = set("*?#@")


@dataclass
class ParsedQuery:
    mode: str  # "meaning" | "structural" | "combined" | "expand" | "phrase_contains"
    pattern_clauses: list[str] = field(default_factory=list)
    meaning_text: str | None = None
    expand_target: str | None = None
    phrase_word: str | None = None


def _split_pattern_clauses(text: str) -> list[str]:
    return [clause.strip() for clause in text.split(",") if clause.strip()]


def _looks_structural(text: str) -> bool:
    if "//" in text:
        return True
    if text.startswith("-") or text.startswith("+"):
        return True
    return any(char in _WILDCARD_TRIGGER_CHARS for char in text)


def parse_query(raw: str) -> ParsedQuery:
    text = raw.strip()

    if text.lower().startswith("expand:"):
        return ParsedQuery(mode="expand", expand_target=text[len("expand:"):].strip().lower())

    if text.startswith("**") and text.endswith("**") and len(text) > 4:
        return ParsedQuery(mode="phrase_contains", phrase_word=text[2:-2].strip().lower())

    if ":" in text:
        pattern_part, meaning_part = text.split(":", 1)
        pattern_part = pattern_part.strip()
        meaning_part = meaning_part.strip()
        if not pattern_part:
            return ParsedQuery(mode="meaning", meaning_text=meaning_part)
        return ParsedQuery(
            mode="combined",
            pattern_clauses=_split_pattern_clauses(pattern_part),
            meaning_text=meaning_part,
        )

    if _looks_structural(text):
        return ParsedQuery(mode="structural", pattern_clauses=_split_pattern_clauses(text))

    return ParsedQuery(mode="meaning", meaning_text=text)
