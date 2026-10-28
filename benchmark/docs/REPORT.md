# LibReuseBench build report

The splits reported here are the final ones, built from the tasks both human reviewers approved.

## The reuse track

- 01 seeds: 813
- 02 generated: 252 tasks kept, of 813 seeds tried
- 03 agent review: 234 keep, 0 fix, 18 reject, 0 not yet reviewed
- 04 human review: 441 keep, 1 update, 26 reject (individual verdicts across both reviewers), 0 tasks pending, 216 approved
- 05 shipped: test 216

**02 outcomes**

- skipped: 561
- kept: 252

**02 kept tasks by reuse tier**

- medium: 121
- low: 75
- high: 56

**03 blind solves, of 252 tasks**

- standard-library solve passed: 252
- free solve passed: 252
- free solve used a library: 63

**04 reviews by author**

- author_2: 234
- author_1: 234

**04 agreement between the reviewers** (update counts as keep)

- both keep: 216
- both reject: 8
- author_1 keeps, author_2 rejects: 7
- author_1 rejects, author_2 keeps: 3
- raw agreement: 95.7%; Cohen's kappa: 0.59

### The test split (216 tasks)

Reuse value: median 8 lines saved (-6 to 60).

**By reuse tier**

- medium: 106
- low: 58
- high: 52

**By library**

- pandas: 66
- sklearn: 65
- scipy: 27
- numpy: 21
- bs4: 11
- statsmodels: 4
- dateutil: 3
- nltk: 2
- matplotlib: 2
- xmltodict: 2
- cv2: 2
- Levenshtein: 1
- sympy: 1
- geopy: 1
- pytz: 1
- rsa: 1
- yaml: 1
- prettytable: 1
- textblob: 1
- pyquery: 1
- natsort: 1
- lxml: 1

**By domain**

- statistics: 81
- data wrangling: 50
- machine learning: 27
- text and NLP: 15
- parsing and markup: 15
- numeric and scientific: 11
- dates and times: 11
- linear algebra: 2
- images: 2
- cryptography and hashing: 1
- encoding and serialisation: 1

## The control track

- 01 seeds: 327
- 02 generated: 181 tasks kept, of 327 seeds tried
- 03 agent review: 96 keep, 0 fix, 84 reject, 1 not yet reviewed
  - not reviewed by the agent: BigCodeBench/644
- 04 human review: 178 keep, 0 update, 14 reject (individual verdicts across both reviewers), 0 tasks pending, 84 approved
- 05 shipped: control 84

**02 outcomes**

- kept: 181
- skipped: 146

**03 blind solves, of 180 tasks**

- free solve passed: 177
- standard-library solve passed: 175
- free solve used a library: 2

**04 reviews by author**

- author_2: 96
- author_1: 96

**04 agreement between the reviewers** (update counts as keep)

- both keep: 84
- both reject: 2
- author_1 keeps, author_2 rejects: 9
- author_1 rejects, author_2 keeps: 1
- raw agreement: 89.6%; Cohen's kappa: 0.25

### The control split (84 tasks)

**By domain**

- encoding and serialisation: 23
- text and NLP: 20
- cryptography and hashing: 15
- data wrangling: 10
- statistics: 5
- parsing and markup: 5
- other: 4
- numeric and scientific: 1
- dates and times: 1

## Reference solutions, through the evaluation harness

**The test split (216 tasks)**

- standard-library solution: passed 216 of 216, used a library 0 of 216
- library solution: passed 216 of 216, used a library 216 of 216
- wrong answer: passed 0 of 216, used a library 0 of 216
- packages installed: 25, failed to install: 0

**The control split (84 tasks)**

- standard-library solution: passed 84 of 84, used a library 0 of 84
- wrong answer: passed 0 of 84, used a library 0 of 84
- packages installed: 0, failed to install: 0
