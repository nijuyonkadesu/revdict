import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn

_POS_NAMES = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}


def _ensure_nltk_data() -> None:
    for package in ("wordnet", "sentiwordnet"):
        try:
            nltk.data.find(f"corpora/{package}")
        except LookupError:
            nltk.download(package, quiet=True)


def load_wordnet_senses() -> list[dict]:
    _ensure_nltk_data()
    records = []
    for synset in wn.all_synsets():
        pos = _POS_NAMES.get(synset.pos(), synset.pos())
        definition = synset.definition()
        examples = list(synset.examples())
        lemma_names = [name.replace("_", " ") for name in synset.lemma_names()]

        try:
            senti = swn.senti_synset(synset.name())
            sentiwordnet = {
                "pos": senti.pos_score(),
                "neg": senti.neg_score(),
                "obj": senti.obj_score(),
            }
        except (LookupError, ValueError):
            sentiwordnet = None

        for lemma in lemma_names:
            records.append(
                {
                    "headword": lemma,
                    "pos": pos,
                    "definition": definition,
                    "examples": examples,
                    "source": "wordnet",
                    "synset": synset.name(),
                    "synonyms": [name for name in lemma_names if name != lemma],
                    "sentiwordnet": sentiwordnet,
                }
            )
    return records
