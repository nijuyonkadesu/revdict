def _levenshtein(a: list[str], b: list[str]) -> int:
    """Standard edit distance over two phoneme-symbol sequences. Pure
    stdlib -- no new dependency, as the roadmap anticipated for this
    phase."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


SOUNDS_LIKE_THRESHOLD = 0.34
"""Normalized edit distance (edit_distance / max(len_a, len_b)) at or below
which two words are considered to "sound like" each other. Calibrated
against real ARPAbet phoneme sequences (stress digits stripped): true
homophones score 0.00 (night/knight, there/their, two/too, bear/bare,
sight/site all measured exactly 0.00); a one-phoneme substitution in a
short word like cat/bat measures 0.33 and is intentionally still a match;
a one-letter misspelling like elephant/elifant measures 0.14. Unrelated
words measure far higher: cat/elephant 0.86, phone/photograph 0.75. 0.34
sits just above the cat/bat case and well below every unrelated pair
tested."""


def matches_syllable_count(record: dict, syllables: int | None) -> bool:
    if syllables is None:
        return True
    phonetics = record.get("phonetics")
    return phonetics is not None and phonetics["syllable_count"] == syllables


def matches_primary_vowel(record: dict, vowel: str | None) -> bool:
    if not vowel:
        return True
    phonetics = record.get("phonetics")
    return phonetics is not None and phonetics["primary_vowel"] == vowel.upper()


def matches_rhyme(record: dict, target_rhyme_key: str | None) -> bool:
    if not target_rhyme_key:
        return True
    phonetics = record.get("phonetics")
    return phonetics is not None and phonetics["rhyme_key"] == target_rhyme_key


def matches_meter(record: dict, target_meter: str | None) -> bool:
    if not target_meter:
        return True
    phonetics = record.get("phonetics")
    return phonetics is not None and phonetics["meter"] == target_meter


def matches_sounds_like(record: dict, target_phonemes: list[str] | None) -> bool:
    if not target_phonemes:
        return True
    phonetics = record.get("phonetics")
    if phonetics is None:
        return False
    candidate = [p.rstrip("012") for p in phonetics["phonemes"]]
    target = [p.rstrip("012") for p in target_phonemes]
    distance = _levenshtein(candidate, target)
    return distance / max(len(candidate), len(target), 1) <= SOUNDS_LIKE_THRESHOLD
