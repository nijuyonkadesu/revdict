try:
    import stressmark.engine as _engine
except ImportError:
    _engine = None


def is_available() -> bool:
    """True only when stressmark is installed AND new enough to expose
    .phonemes on its WordResult (Task 1 of this plan) -- an older
    stressmark install would otherwise silently produce phonetics-free
    results instead of a clear signal that an upgrade is needed."""
    if _engine is None:
        return False
    try:
        probe = _engine.WordResult("")
    except Exception:
        return False
    return hasattr(probe, "phonemes")


def _strip_stress(phoneme: str) -> str:
    return phoneme.rstrip("012")


def _phonetic_primary_index(phonemes: list[str]) -> int:
    """Mirrors stressmark.engine.stress_positions_for_pron's own
    fallback convention: find the first phoneme marked stress '1', or
    (if none is marked) the first VOWEL-bearing phoneme -- never an
    arbitrary raw-list index, which could land on a consonant. ARPAbet
    marks a stress digit only on vowel phonemes, so filtering to
    digit-ending phonemes IS filtering to vowels."""
    vowel_indices = [i for i, p in enumerate(phonemes) if p[-1].isdigit()]
    if not vowel_indices:
        return 0
    for i in vowel_indices:
        if phonemes[i][-1] == "1":
            return i
    return vowel_indices[0]


def resolve(word: str, pos: str) -> dict | None:
    """Full phonetic resolution for a single word -- used both at index-
    build time (every clean headword in the corpus) and at query time (an
    arbitrary --rhymes-with/--sounds-like target). Returns None, and never
    raises, when: stressmark is unavailable or too old (is_available() is
    False), `word` contains a space or hyphen (resolve_word_by_pos
    produces malformed syllable/stress data for both -- confirmed directly
    against real headwords like "kick the bucket" and "well-known" before
    this module was written, see the plan's Global Constraints), or any
    other unexpected failure occurs resolving this specific word. This
    must never crash a reindex, or a live query, over one weird headword.
    """
    if not is_available():
        return None
    if " " in word or "-" in word:
        return None
    try:
        result = _engine.resolve_word_by_pos(word, pos)
        phonemes = result.phonemes
        if not phonemes:
            return None
        syllable_count = len(result.syllables)
        idx = _phonetic_primary_index(phonemes)
        primary_vowel = _strip_stress(phonemes[idx])
        rhyme_key = " ".join(_strip_stress(p) for p in phonemes[idx:])
        stressed = {result.primary} | set(result.secondary)
        meter = "".join("/" if i in stressed else "x" for i in range(syllable_count))
        return {
            "syllable_count": syllable_count,
            "primary_vowel": primary_vowel,
            "rhyme_key": rhyme_key,
            "meter": meter,
            "phonemes": list(phonemes),
        }
    except Exception:
        return None
