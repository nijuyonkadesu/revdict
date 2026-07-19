from dataclasses import dataclass, field


@dataclass
class ParsedQuery:
    mode: str  # "meaning" | "structural" | "combined" | "expand" | "phrase_contains"
    pattern_clauses: list[str] = field(default_factory=list)
    meaning_text: str | None = None
    expand_target: str | None = None
    phrase_word: str | None = None


def parse_query(raw: str) -> ParsedQuery:
    text = raw.strip()

    if ":" in text:
        pattern_part, meaning_part = text.split(":", 1)
        pattern_part = pattern_part.strip()
        meaning_part = meaning_part.strip()
        if not pattern_part:
            return ParsedQuery(mode="meaning", meaning_text=meaning_part)

    return ParsedQuery(mode="meaning", meaning_text=text)
