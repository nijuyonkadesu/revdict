"""Evaluates revdict's real ranking quality against a growing corpus of
(uncommon word, natural-language description) cases pulled from real
creative-writing vocabulary notes.

This is a slow, model-in-the-loop evaluation tool, not a unit test -- it
loads the real embedder/reranker and the real built index, so it's run
manually (`.venv/bin/python scripts/evaluate_vocabulary_recovery.py`), not
as part of `pytest tests/`.

Cases live in tests/data/vocabulary_recovery/*.json, one file per source
vocabulary document, each shaped as:
    {"source": "<filename>", "cases": {"<word>": "<description>", ...}}

Reports two recall numbers per file and overall:
  - exact:    the target word itself appears in the top-N candidates.
  - synonym:  the target word OR one of its WordNet synonyms appears --
    a query with many valid answers shouldn't be marked "wrong" just
    because it returned a different, equally-good word (see the
    "svelte" -> slim/willowy finding in the design investigation).
Remaining misses (neither exact nor synonym) are printed for manual review,
since ranking quality beyond that is inherently a judgment call, not
something a script can grade on its own.
"""

import json
import sys
from pathlib import Path

from nltk.corpus import wordnet as wn

from revdict.search import search

TOP_N = 15
CASES_DIR = Path(__file__).resolve().parent.parent / "tests" / "data" / "vocabulary_recovery"


def wordnet_synonym_forms(word: str) -> set[str]:
    forms = {word.lower()}
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            forms.add(lemma.name().lower().replace("_", " "))
    return forms


def evaluate_case(word: str, query: str) -> dict:
    result = search(query, top_n=TOP_N)
    top_words = [c["headword"].lower() for c in result["candidates"]]
    if result["exact_match"] and result["exact_match"]["headword"].lower() == word.lower():
        top_words = [word.lower()] + top_words

    exact_rank = top_words.index(word.lower()) + 1 if word.lower() in top_words else None

    synonym_forms = wordnet_synonym_forms(word)
    synonym_rank = None
    for i, candidate in enumerate(top_words, 1):
        if candidate in synonym_forms:
            synonym_rank = i
            break

    return {
        "query": query,
        "exact_rank": exact_rank,
        "synonym_rank": synonym_rank,
        "top5": top_words[:5],
    }


def main() -> None:
    case_files = sorted(CASES_DIR.glob("*.json"))
    if not case_files:
        print(f"No case files found in {CASES_DIR}")
        sys.exit(1)

    total_exact = total_synonym = total_cases = 0
    for case_file in case_files:
        with case_file.open(encoding="utf-8") as f:
            data = json.load(f)
        source, cases = data["source"], data["cases"]

        exact_hits = synonym_hits = 0
        misses = []
        for word, query in cases.items():
            outcome = evaluate_case(word, query)
            if outcome["exact_rank"] is not None:
                exact_hits += 1
            elif outcome["synonym_rank"] is not None:
                synonym_hits += 1
            else:
                misses.append((word, outcome["top5"]))

        n = len(cases)
        either = exact_hits + synonym_hits
        print(f"=== {source} ({n} cases) ===")
        print(f"  exact recall:            {exact_hits}/{n}")
        print(f"  exact-or-synonym recall: {either}/{n}")
        if misses:
            print("  unresolved misses (manual review):")
            for word, top5 in misses:
                print(f"    {word:20} top5={top5}")
        print()

        total_exact += exact_hits
        total_synonym += either
        total_cases += n

    print(f"=== overall: {total_cases} cases across {len(case_files)} file(s) ===")
    print(f"  exact recall:            {total_exact}/{total_cases}")
    print(f"  exact-or-synonym recall: {total_synonym}/{total_cases}")


if __name__ == "__main__":
    main()
