# LibReuseBench build report

## Drop funnel (reuse track)

- 01 fetched: 813 ext-lib seeds (+ 327 stdlib-only set aside)
- 02 candidates: 262 carried forward of 813 annotated
- 03 rewritten (faithful): 259 of 262
- 04 tests minted: 259
- 05 validated (dual-solvable): 212 of 246
- 06 reviewed: 400 keep, 1 update, 23 reject (individual verdicts across both reviewers), 0 pending of 212 (shipping 189)
- **final dataset: 189 problems** (test 179, dev 10)

**02 verdicts**

- drop: 551
- keep: 260
- maybe: 2

**03 drop reasons**

- unusable rewrite: 3

**05 drop reasons**

- output mismatch: 28
- runtime error: 2
- no library solution: 2
- prompt names ['nltk', 'sklearn']: 1
- prompt names ['pandas']: 1

**06 review verdicts**

- keep: 400
- reject: 23
- update: 1

**06 reviews by author**

- author_1: 212
- author_2: 212

**Final by category**

- stats: 12
- statistics: 12
- text-processing: 7
- data-aggregation: 6
- data-normalization: 3
- dataframe-construction: 3
- data-encoding: 3
- data-processing: 3
- data-transformation: 3
- text-processing-nlp: 3
- linear-algebra-stats: 2
- datetime-timezone-conversion: 2
- tabular-stats: 2
- json-serialization: 2
- data-scaling: 2
- string-manipulation: 2
- combinatorics: 2
- data-serialization: 2
- data-preprocessing: 2
- data-standardization: 2
- database-querying: 2
- stats-computation: 1
- data-generation-table: 1
- datetime-weekday-distribution: 1
- string-filtering-counting: 1
- data-extraction-parsing: 1
- data-processing-normalization: 1
- html-parsing: 1
- function-introspection-csv-export: 1
- datetime-timezone-stats: 1
- data-transformation-csv: 1
- text-processing-tokenization: 1
- text-analysis-vectorization: 1
- statistics-chi-square: 1
- statistics-correlation: 1
- text-processing-regex: 1
- dataframe-transformation: 1
- datetime-business-days: 1
- encoding-transformation: 1
- linear-regression: 1
- text-cleaning: 1
- symmetric-encryption: 1
- list-processing: 1
- data-transform: 1
- data-normalization-histogram: 1
- statistics-aggregation: 1
- linear-regression-forecast: 1
- text-nlp-stopwords: 1
- stats-normalization: 1
- statistics-json-analysis: 1
- dataframe-filtering: 1
- database-regex-filtering: 1
- datetime-math: 1
- json-processing-dataframe: 1
- statistics-standardization: 1
- linear-regression-stats: 1
- data-filtering-standardization: 1
- data-analysis-and-sorting: 1
- json-dataframe-stats: 1
- datetime-timezone-arithmetic: 1
- numerical-normalization: 1
- statistics-computation: 1
- log-parsing: 1
- combinatorics-arithmetic: 1
- text-tokenization: 1
- tabular-data: 1
- computational-geometry: 1
- tabular-computation: 1
- natural-language-processing: 1
- tabular-data-manipulation: 1
- regex-dataframe-extraction: 1
- dataframe-stats: 1
- dataframe-generation: 1
- array-processing-combinations: 1
- text-processing-frequency: 1
- statistics-calculation: 1
- text-tokenization-counting: 1
- stats-percentile: 1
- aggregation-stats: 1
- machine-learning-clustering: 1
- nlp-tfidf: 1
- text-processing-counting: 1
- time-series-aggregation: 1
- text-parsing-stats: 1
- spreadsheet-generation: 1
- csv-generation: 1
- matrix-concatenation-formatting: 1
- statistics-normalization: 1
- database-string-manipulation: 1
- spreadsheet-processing: 1
- csv-processing: 1
- array-manipulation: 1
- datetime-parsing: 1
- json-dataframe-manipulation: 1
- arithmetic-arrays: 1
- text-processing-tfidf: 1
- text-processing-word-frequency: 1
- data-processing-imputation: 1
- data-processing-tabular: 1
- statistics-regression: 1
- prime-filtering-with-sorting: 1
- euclidean-distance-calculation: 1
- cumulative-sum: 1
- statistical-analysis: 1
- random-matrix-generation: 1
- sql-to-csv: 1
- csv-diff: 1
- arithmetic: 1
- word-frequency: 1
- combinatorics-dataframe: 1
- csv-manipulation: 1
- data-transformation-encoding: 1
- matrix-combinatorics: 1
- text-frequency-count: 1
- stats-tabular: 1
- data-standardization-encoding: 1
- yaml-parsing-math: 1
- nlp-text-processing: 1
- normalization: 1
- string-analysis-entropy: 1
- regex-pattern-matching: 1
- dataframe-search: 1
- data-filtering-datetime: 1
- json-to-tabular: 1
- statistics-mode: 1
- nearest-neighbors-geo: 1
- string-decoding: 1
- data-transformation-scaling: 1
- numeric-scaling: 1
- stats-aggregation: 1
- stats-table: 1
- xml-parsing: 1
- log-parsing-csv-conversion: 1

**Final by seed library**

- pandas: 124
- numpy: 77
- sklearn: 38
- scipy: 18
- nltk: 13
- dateutil: 6
- pytz: 6
- regex: 3
- xlwt: 2
- bs4: 1
- cryptography: 1
- sympy: 1
- yaml: 1
- lxml: 1

## Control track

- 01 stdlib-only seeds: 327
- 02 candidates: 98 carried forward of 327 annotated
- 03 rewritten (faithful): 98 of 98
- 04 tests minted (already the fully validated pool - no dual-solvable stage for this track): 95 of 98
- 06 reviewed: 171 keep, 3 update, 16 reject (individual verdicts across both reviewers), 0 pending of 95 (shipping 79)
- **final control dataset: 79 problems**

**02 verdicts (control)**

- drop: 229
- keep: 98

**03 drop reasons (control)**

- (none)

**06 review verdicts (control)**

- keep: 171
- reject: 16
- update: 3

**06 reviews by author (control)**

- author_1: 95
- author_2: 95

**Control by category**

- string-processing: 6
- json-serialization: 3
- text-processing: 3
- text-analysis: 2
- function-introspection: 2
- csv-aggregation: 2
- cryptography-hmac: 2
- math-computation: 2
- csv-processing: 2
- threading-concurrency: 1
- string-algorithms: 1
- csv-writing: 1
- dict-manipulation-json: 1
- data-structure-manipulation: 1
- letter-to-number-product: 1
- dict-aggregation: 1
- combinatorics: 1
- url-parsing-hex-decoding: 1
- text-processing-stopwords: 1
- file-processing: 1
- csv-io: 1
- datetime-parsing-stats: 1
- string-processing-frequency: 1
- text-cleaning: 1
- combinatorics-factorial: 1
- text-encoding-unicode: 1
- database-operations: 1
- string-pattern-matching: 1
- list-merging-counting: 1
- dict-aggregation-math: 1
- path-validation: 1
- json-schema-validation: 1
- json-parsing: 1
- string-parsing-math: 1
- csv-formatting: 1
- json-parsing-validation: 1
- csv-to-json: 1
- json-parsing-text-processing: 1
- string-encoding: 1
- frequency-counting: 1
- combinatorics-statistics: 1
- dictionary-counting: 1
- text-processing-nlp: 1
- csv-encoding-conversion: 1
- text-parsing-counting: 1
- string-processing-hashing: 1
- dynamic-programming-optimization: 1
- log-parsing-formatting: 1
- data-processing: 1
- json-processing: 1
- text-processing-regex: 1
- string-frequency-counting: 1
- xml-to-csv-conversion: 1
- dict-optimization: 1
- frequency-analysis: 1
- string-parsing: 1
- datetime-parsing: 1
- combinatorics-counting: 1
- json-parsing-and-counting: 1
- regex-text-matching: 1
- json-validation: 1
- combinatorial-optimization: 1
- text-processing-frequency: 1
- text-processing-word-frequency: 1
