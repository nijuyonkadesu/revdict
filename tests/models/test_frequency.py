from revdict.models.frequency import is_essentially_unattested


def test_is_essentially_unattested_true_for_a_word_with_zero_real_world_attestation():
    # Real example from investigation: a genuinely obscure Wiktionary dialectal
    # entry with zero attestation across wordfreq's combined source corpora.
    assert is_essentially_unattested("wealful") is True


def test_is_essentially_unattested_false_for_a_common_word():
    assert is_essentially_unattested("happy") is False


def test_is_essentially_unattested_false_for_a_fiction_common_but_casually_rare_word():
    # The whole reason this is a hard "< 0.5" cutoff and not a general
    # familiarity ranking: "murmured" is rare in casual/social-media text
    # (low zipf) but is NOT "essentially unattested" -- it's a fiction
    # staple. A general frequency boost would have incorrectly suppressed
    # words like this; the cutoff must stay narrow enough not to.
    assert is_essentially_unattested("murmured") is False
