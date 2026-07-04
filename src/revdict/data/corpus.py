def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def merge_records(wordnet_records: list[dict], wiktionary_records: list[dict]) -> list[dict]:
    seen = {
        (_normalize(record["headword"]), _normalize(record["definition"]))
        for record in wordnet_records
    }
    merged = list(wordnet_records)
    for record in wiktionary_records:
        key = (_normalize(record["headword"]), _normalize(record["definition"]))
        if key in seen:
            continue
        seen.add(key)
        merged.append(record)
    return merged
