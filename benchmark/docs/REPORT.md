# LibReuseBench build report

## Drop funnel

- 01 fetched: 813 ext-lib seeds (+ 327 stdlib-only set aside)
- 02 candidates: 175 carried forward of 813 annotated
- 03 rewritten (faithful): 173 of 175
- 04 tests minted: 173
- 05 validated (dual-solvable): 139 of 172
- 06 reviewed: 139 keep, 0 update, 0 reject, 0 pending of 139 (shipping 139)
- **final dataset: 139 problems** (test 129, dev 10)

**02 verdicts**

- drop: 638
- keep: 175

**03 drop reasons**

- unusable rewrite: 2

**05 drop reasons**

- output mismatch: 23
- runtime error: 6
- solution used no library: 2
- prompt names ['nltk', 'sklearn']: 1
- prompt names ['pandas']: 1

**06 review verdicts**

- keep: 139

**Final by category**

- statistics: 11
- text-processing: 7
- data-aggregation: 5
- text-processing-nlp: 4
- data-normalization: 4
- data-transformation: 3
- statistics-computation: 2
- json-serialization: 2
- data-preprocessing: 2
- database-querying: 2
- combinatorics: 2
- string-manipulation: 2
- datetime-timezone-conversion: 2
- data-encoding: 2
- statistics-correlation: 2
- arithmetic: 1
- statistics-calculation: 1
- nlp-text-processing: 1
- data-transformation-scaling: 1
- statistics-aggregation: 1
- data-transformation-csv: 1
- combinatorics-arithmetic: 1
- text-processing-tokenization: 1
- text-processing-tfidf: 1
- symmetric-encryption: 1
- stats-aggregation: 1
- numerical-normalization: 1
- datetime-business-days: 1
- sql-to-csv: 1
- text-processing-word-frequency: 1
- csv-manipulation: 1
- regex-dataframe-extraction: 1
- datetime-timezone-stats: 1
- data-processing-normalization: 1
- statistics-mode: 1
- text-tokenization-counting: 1
- statistics-regression: 1
- dataframe-transformation: 1
- encoding-transformation: 1
- combinatorics-dataframe: 1
- statistical-analysis: 1
- machine-learning-clustering: 1
- log-parsing: 1
- statistics-json-analysis: 1
- euclidean-distance-calculation: 1
- arithmetic-arrays: 1
- data-standardization: 1
- dataframe-filtering: 1
- database-regex-filtering: 1
- aggregation-stats: 1
- data-serialization: 1
- statistics-standardization: 1
- prime-filtering-with-sorting: 1
- data-scaling: 1
- stats-normalization: 1
- text-processing-counting: 1
- string-pattern-matching: 1
- nlp-tfidf: 1
- data-standardization-encoding: 1
- regex-pattern-matching: 1
- statistics-normalization: 1
- time-series-aggregation: 1
- computational-geometry: 1
- linear-algebra-stats: 1
- text-similarity: 1
- stats-computation: 1
- yaml-parsing-math: 1
- data-extraction-parsing: 1
- text-processing-regex: 1
- descriptive-statistics: 1
- matrix-concatenation-formatting: 1
- data-processing-tabular: 1
- log-parsing-csv-conversion: 1
- database-string-manipulation: 1
- signal-processing-statistics: 1
- text-tokenization: 1
- datetime-math: 1
- text-cleaning: 1
- string-filtering-counting: 1
- datetime-timezone-arithmetic: 1
- datetime-trigonometry: 1
- list-processing: 1
- data-filtering-standardization: 1
- spreadsheet-processing: 1
- data-processing-imputation: 1
- datetime-weekday-distribution: 1
- data-analysis: 1
- text-analysis-vectorization: 1
- string-decoding: 1
- normalization: 1
- data-transformation-encoding: 1
- spreadsheet-generation: 1
- data-processing: 1
- natural-language-processing: 1
- nearest-neighbors-geo: 1
- data-filtering-datetime: 1
- array-processing-combinations: 1
- dataframe-generation: 1
- statistics-chi-square: 1
- text-processing-frequency: 1
- data-analysis-and-sorting: 1
- string-analysis-entropy: 1

**Final by seed library**

- pandas: 84
- numpy: 53
- sklearn: 28
- scipy: 15
- nltk: 13
- pytz: 7
- dateutil: 6
- regex: 3
- cryptography: 1
- sympy: 1
- levenshtein: 1
- yaml: 1
- xlwt: 1
