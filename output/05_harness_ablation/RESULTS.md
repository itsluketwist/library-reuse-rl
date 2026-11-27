# LibReuseBench results

Generated from the included records by `summarize.py`. One sample per task; all nine test conditions cover 216/216 tasks. Controls are separate.

## Test split

| Model | Arm | Passes / tasks | Pass@1 | Mean thinking | Mean output | Imports | Corrected use | Mean LLOC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ML4 | Direct API | 205/216 | 94.9% | 12,747 | 13,098 | 5/216 | 5/216 | 30.4 |
| ML4 | Vibe | 212/216 | 98.1% | 3,225 | 6,035 | 3/216 | 3/216 | 39.4 |
| ML4 | mini-swe | 213/216 | 98.6% | 2,583 | 6,059 | 17/216 | 17/216 | 56.3 |
| GLM-5.3 | Direct API | 208/216 | 96.3% | 785 | 1,473 | 7/216 | 7/216 | 27.8 |
| GLM-5.3 | Vibe | 213/216 | 98.6% | 2,266 | 4,588 | 4/216 | 4/216 | 37.4 |
| GLM-5.3 | mini-swe | 212/216 | 98.1% | 3,051 | 6,405 | 24/216 | 24/216 | 57.0 |
| DeepSeek Flash | Direct API | 212/216 | 98.1% | 3,851 | 4,169 | 9/216 | 9/216 | 31.0 |
| DeepSeek Flash | Vibe | 215/216 | 99.5% | 3,916 | 6,807 | 7/216 | 7/216 | 49.3 |
| DeepSeek Flash | mini-swe | 214/216 | 99.1% | 3,581 | 6,103 | 14/216 | 14/216 | 41.0 |

Output includes thinking, visible text and tool-call arguments, summed across reported calls. LLOC is logical lines in the final solution; its denominator can be smaller than 216. Static usage is a possible reference, not proof of execution.

## Analyzer correction, coverage and repetition

| Model | Arm | Original → corrected use | High-reuse used / 52 | Measurable LLOC | Calls with usage / recorded calls | Repeated thinking words | Repeated intent tasks |
|---|---|---:|---:|---:|---:|---:|---:|
| ML4 | Direct API | 5 → 5 | 2/52 | 213/216 | 216/216 | 4.20% | 94/216 |
| ML4 | Vibe | 1 → 3 | 2/52 | 216/216 | 1642/1646 | 0.30% | 0/216 |
| ML4 | mini-swe | 13 → 17 | 8/52 | 216/216 | 1840/1841 | 0.06% | 0/216 |
| GLM-5.3 | Direct API | 7 → 7 | 3/52 | 215/216 | 216/289 | 0.40% | 0/216 |
| GLM-5.3 | Vibe | 3 → 4 | 1/52 | 216/216 | 1478/2055 | 0.40% | 1/216 |
| GLM-5.3 | mini-swe | 12 → 24 | 13/52 | 216/216 | 2002/2400 | 0.02% | 0/216 |
| DeepSeek Flash | Direct API | 7 → 9 | 5/52 | 214/216 | 216/216 | 2.99% | 3/216 |
| DeepSeek Flash | Vibe | 2 → 7 | 3/52 | 216/216 | 2098/2098 | 0.09% | 0/216 |
| DeepSeek Flash | mini-swe | 11 → 14 | 9/52 | 216/216 | 1644/1645 | 0.05% | 0/216 |

All tasks have some reported output/thinking usage. Recorded attempts include transport/rate-limit retries, particularly for GLM; attempts without usage are not necessarily generated completions. Partial/deadline call usage is not estimated. Repetition ratios pool word counts. A repeated-intent flag is a heuristic, not proof of infinite generation.

## Separate source and runtime audit

| Model | Arm | Imported | Library-only, exercised | Mixed/partial, exercised | No library call on tests | Manual/mixed source | Any inactive import |
|---|---|---:|---:|---:|---:|---:|---:|
| ML4 | Direct API | 5 | 4 | 1 | 0 | 1 | 0 |
| ML4 | Vibe | 3 | 0 | 3 | 0 | 3 | 0 |
| ML4 | mini-swe | 17 | 1 | 10 | 6 | 15 | 6 |
| GLM-5.3 | Direct API | 7 | 5 | 1 | 1 | 2 | 1 |
| GLM-5.3 | Vibe | 4 | 2 | 2 | 0 | 2 | 0 |
| GLM-5.3 | mini-swe | 24 | 2 | 17 | 5 | 20 | 6 |
| DeepSeek Flash | Direct API | 9 | 7 | 2 | 0 | 2 | 0 |
| DeepSeek Flash | Vibe | 7 | 1 | 5 | 1 | 6 | 1 |
| DeepSeek Flash | mini-swe | 14 | 4 | 9 | 1 | 10 | 1 |

The 26 library-only, 50 mixed/partial-with-call and 14 no-call solutions partition the 90 imported programs. The 61 manual/mixed source labels and 15 inactive-import programs are different, overlapping measurements. Audit observations cover 1,022 saved test inputs; correctness scores are not replaced by the audit.

## Historical ML4 controls

Only ML4 ran the 84 control tasks. They are not included in the main model comparison or its plots.

| Arm | Passes / tasks | Pass@1 | Imported | Corrected used | Mean thinking | Mean output |
|---|---:|---:|---:|---:|---:|---:|
| Direct API | 82/84 | 97.6% | 0 | 0 | 7,305 | 7,486 |
| Vibe | 84/84 | 100.0% | 0 | 0 | 726 | 1,946 |
| mini-swe | 84/84 | 100.0% | 0 | 0 | 1,185 | 3,026 |
