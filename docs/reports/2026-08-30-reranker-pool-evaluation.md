# Real-corpus reranker pool evaluation

Date: 2026-08-30

Corpus: schema-v3 upgrade of build `980169f1b18f4fc499e6020e002632f3`,
1,267,250 sense records, 1,049,586 unique definition embeddings, and 787,587
headwords. Embedding retrieval was exhaustive over the complete float32
matrix. Each candidate pool used the unchanged cross-encoder, literary
frequency adjustment, and headword deduplication.

## Method

For each of ten reverse-dictionary prompts, retrieve the top 1,000 cosine
candidates once and evaluate prefixes of 75, 300, 600, and 1,000. The final
top ten from pool 1,000 is the comparison reference. This is a retrieval
regression measure, not a human relevance judgment: overlap indicates how
much of the deeper reranker's result each smaller pool recovers.

| Query | Pool 75 | Pool 300 | Pool 600 | Pool 1,000 |
|---|---:|---:|---:|---:|
| a feeling of intense joy | 7/10 | 9/10 | 10/10 | 10/10 |
| unable to be avoided | 9/10 | 9/10 | 10/10 | 10/10 |
| speaking in a way that is difficult to understand | 10/10 | 10/10 | 10/10 | 10/10 |
| a person who betrays a friend | 10/10 | 10/10 | 10/10 | 10/10 |
| light produced by living organisms | 8/10 | 9/10 | 9/10 | 10/10 |
| the smell after rain | 2/10 | 5/10 | 9/10 | 10/10 |
| to make less severe or painful | 10/10 | 10/10 | 10/10 | 10/10 |
| fear of confined spaces | 8/10 | 9/10 | 10/10 | 10/10 |
| existing everywhere at once | 9/10 | 10/10 | 10/10 | 10/10 |
| a temporary solution that avoids the real problem | 7/10 | 10/10 | 10/10 | 10/10 |
| **Total reference overlap** | **80/100** | **91/100** | **98/100** | **100/100** |

Pool 600 promoted candidates from below cosine rank 75 into the final UI for
seven of the ten prompts. Examples include `happiness` (rank 78) for intense
joy, `streetlight` (rank 191) for light produced by organisms, `jam` (rank
358) for confined spaces, and `temporize` (rank 287) for a temporary
workaround. This is the regression class that a pool of 75 cannot recover.

## Warm reranker cost

Five repeated runs of “a feeling of intense joy” on this machine measured:

| Pool | Median | Minimum |
|---|---:|---:|
| 75 | 63.2 ms | 57.5 ms |
| 300 | 191.6 ms | 188.5 ms |
| 600 | 386.7 ms | 385.0 ms |
| 1,000 | 650.9 ms | 646.0 ms |

The move from 75 to 600 costs about 324 ms in the median and recovers 18 more
of the 100 reference positions. Moving from 600 to 1,000 costs another 264 ms
for two positions. Production therefore uses 600 as the fixed minimum; a
future reduction should repeat this report against the real corpus and
provide equivalent or better recovery evidence.
