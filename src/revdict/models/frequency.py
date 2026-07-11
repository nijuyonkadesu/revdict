from wordfreq import zipf_frequency

ZERO_FREQUENCY_THRESHOLD = 0.5


def is_essentially_unattested(word: str) -> bool:
    """True when `word` has no meaningful attestation across wordfreq's
    combined English source corpora (Wikipedia, subtitles, news, Google
    Books, web text, Twitter, Reddit) -- a reliable "this word is
    essentially unused" signal.

    Deliberately NOT used as a general familiarity ranking: general word
    frequency does not track literary/fiction register (verified during
    design -- "murmured" and "scowled" score lower than genuine
    Shakespeare-era archaisms like "wherefore"/"yonder" in wordfreq's
    blended corpus, since they're rare in casual speech/social media
    despite being fiction staples). This hard, narrow cutoff only fires
    for words that are close to actually unattested, not merely uncommon.
    """
    return zipf_frequency(word, "en") < ZERO_FREQUENCY_THRESHOLD
