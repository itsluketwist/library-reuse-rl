# Five-replica results

All 3,240 samples are scored: 216 test tasks × five replicas × three models. Direct API calls, high reasoning, temperature 1, top-p 1, 32,768 output tokens per call.

| Model | Replica passes (each /216) | Passes /1,080 | Pass@1 | Pass@5 | Mean thinking | Mean output | Imported | Static used | Mean LLOC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ML4 | 203, 204, 207, 206, 201 | 1021/1,080 | 94.54% | 99.07% | 13,050 | 13,403 | 21/1,080 | 21/1,080 (1.94%) | 30.3 |
| GLM-5.3 | 203, 205, 200, 204, 204 | 1016/1,080 | 94.07% | 99.07% | 1,041 | 1,759 | 34/1,080 | 34/1,080 (3.15%) | 27.7 |
| DeepSeek Flash | 215, 213, 214, 214, 215 | 1071/1,080 | 99.17% | 99.54% | 3,423 | 3,757 | 37/1,080 | 37/1,080 (3.43%) | 32.0 |

Mean output includes thinking and answer tokens from the accepted response. Every accepted response has usage. Partial/retried requests are outside these means. LLOC uses measurable code only: 1,048 ML4, 1,069 GLM and 1,079 DeepSeek submissions.

No exhausted API errors occurred. Capped/empty responses remain in the denominator, as does one GLM response not run because backports could not install. Static use means a possible imported-name reference, not execution coverage or absence of handwritten fallback code.

## Individual replicas

| Model | Replica | Passes /216 | Pass@1 | Imported | Used | Mean thinking | Mean output |
|---|---:|---:|---:|---:|---:|---:|---:|
| ML4 | 1 | 203/216 | 93.98% | 5 | 5 | 13,137 | 13,482 |
| ML4 | 2 | 204/216 | 94.44% | 4 | 4 | 12,781 | 13,139 |
| ML4 | 3 | 207/216 | 95.83% | 3 | 3 | 12,906 | 13,273 |
| ML4 | 4 | 206/216 | 95.37% | 4 | 4 | 12,887 | 13,249 |
| ML4 | 5 | 201/216 | 93.06% | 5 | 5 | 13,538 | 13,871 |
| GLM-5.3 | 1 | 203/216 | 93.98% | 6 | 6 | 861 | 1,575 |
| GLM-5.3 | 2 | 205/216 | 94.91% | 4 | 4 | 1,055 | 1,754 |
| GLM-5.3 | 3 | 200/216 | 92.59% | 8 | 8 | 1,262 | 2,060 |
| GLM-5.3 | 4 | 204/216 | 94.44% | 8 | 8 | 959 | 1,649 |
| GLM-5.3 | 5 | 204/216 | 94.44% | 8 | 8 | 1,068 | 1,755 |
| DeepSeek Flash | 1 | 215/216 | 99.54% | 6 | 6 | 3,730 | 4,071 |
| DeepSeek Flash | 2 | 213/216 | 98.61% | 8 | 8 | 3,853 | 4,183 |
| DeepSeek Flash | 3 | 214/216 | 99.07% | 6 | 6 | 3,665 | 4,007 |
| DeepSeek Flash | 4 | 214/216 | 99.07% | 8 | 8 | 3,237 | 3,574 |
| DeepSeek Flash | 5 | 215/216 | 99.54% | 9 | 9 | 2,629 | 2,948 |
