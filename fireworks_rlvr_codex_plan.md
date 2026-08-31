# Fireworks RLVR Implementation Plan

Add a new top-level `fireworks/` directory to the existing project. Do not restructure or duplicate the rest of the project. This directory should contain all Fireworks-specific inference, RFT and evaluator code.

The first target is a **Qwen3-8B pilot**, followed later by Kimi K2.6 or another frontier model. Everything should therefore be model-agnostic and configured using model IDs rather than hard-coded to Qwen.

## 1. Directory structure

Create approximately:

```text
fireworks/
├── README.md
├── requirements.txt
├── .env.example
│
├── infer_base.py
├── infer_rft.py
├── inference_common.py
│
├── rft/
│   ├── prepare_dataset.py
│   ├── launch.py
│   │
│   └── evaluator/
│       ├── requirements.txt
│       ├── development/
│       │   └── sample.jsonl
│       │
│       ├── reward.py
│       ├── code_extraction.py
│       ├── library_extraction.py
│       │
│       └── tests/
│           └── pytest/
│               └── library_reuse/
│                   └── test_library_reuse.py
│
└── tests/
    └── ...
```

Do **not** make the top-level `fireworks/` directory a Python package named `fireworks`, because that risks shadowing Fireworks' own Python package.

Use an internal neutral module/package name if one becomes necessary.

## 2. Environment/configuration

Use:

```bash
FIREWORKS_API_KEY=...
```

from the environment.

Provide `.env.example`, but never commit a real key.

Use the Fireworks OpenAI-compatible endpoint:

```text
https://api.fireworks.ai/inference/v1
```

The normal OpenAI Python SDK is sufficient for inference.

Install at minimum:

```text
openai
pytest
requests
eval-protocol
```

Use the current Eval Protocol installation method recommended by Fireworks rather than pinning against an old `reward-kit` implementation.

## 3. Shared inference implementation

Implement the actual API logic once in:

```text
inference_common.py
```

and make both `infer_base.py` and `infer_rft.py` thin CLI wrappers around it.

The interface should accept at least:

```bash
python fireworks/infer_base.py \
    --model MODEL_ID \
    --input DATASET.jsonl \
    --output generations.jsonl
```

and:

```bash
python fireworks/infer_rft.py \
    --model FINE_TUNED_MODEL_ID \
    --input DATASET.jsonl \
    --output generations_rft.jsonl
```

Both scripts should use identical generation parameters unless deliberately overridden.

Support configurable:

```text
model
temperature
top_p
max_tokens
seed, if supported
number of samples
input path
output path
```

Do not bake experimental defaults into several different scripts.

## 4. Preserve the complete Fireworks response

For every generation, save something like:

```json
{
  "task_id": "...",
  "model": "...",
  "prompt": "...",
  "content": "...",
  "reasoning_content": "...",
  "finish_reason": "...",
  "usage": {},
  "generation_config": {}
}
```

In particular, preserve both:

```python
message.content
message.reasoning_content
```

where `reasoning_content` exists.

This is useful for later analysis even though the RL reward should ignore the reasoning.

Do not collapse the two fields together.

## 5. Base-model inference

`infer_base.py` should work with either serverless or on-demand Fireworks models.

### Serverless models

For example later:

```text
accounts/fireworks/models/kimi-k2p6
```

Call the API directly.

### On-demand models

For Qwen3-8B, assume an on-demand deployment has been created.

Document in the README how to create the deployment using either the Fireworks UI or `firectl`.

Do **not** automatically create a paid deployment every time `infer_base.py` runs.

Inference itself should otherwise use the same API.

## 6. Fine-tuned/RFT inference

`infer_rft.py` should use exactly the same input/output machinery as `infer_base.py`.

The principal difference is simply the model identifier:

```text
accounts/<ACCOUNT_ID>/models/<FINE_TUNED_MODEL_ID>
```

Fireworks RFT LoRA models should be treated as deployed models queried through the normal chat-completions API.

Document deployment commands in the README rather than silently creating paid resources.

The important experimental property is:

```text
base inference
and
RFT inference
```

must share exactly the same inference implementation and decoding configuration.

## 7. Prepare an RFT dataset adapter

Implement:

```text
rft/prepare_dataset.py
```

This should consume the **existing project dataset**, rather than creating another source of truth.

Its only job is to convert the relevant training split into the JSONL format expected by the Fireworks evaluator/RFT pipeline.

Each row should contain roughly:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "..."
    }
  ],
  "ground_truth": {
    "task_id": "...",
    "target_library": "..."
  }
}
```

Add whatever additional metadata the current experiment needs, but do not include executable tests or solutions unless required.

The training reward is intentionally based only on **structural library reuse**, not functional execution.

## 8. Evaluator architecture

Keep the evaluator modular:

```text
assistant response
       ↓
code_extraction.py
       ↓
library_extraction.py
       ↓
reward.py
       ↓
Eval Protocol wrapper
```

The evaluator **does not need to be one self-contained Python file**. Fireworks packages an evaluator source directory, so helper modules can be imported as long as they are included inside that uploaded evaluator directory.

For example:

```python
from code_extraction import extract_code
from library_extraction import analyse_library_usage
from reward import calculate_reward
```

is fine provided all three files are packaged under the evaluator source directory.

Since this project uses only Python stdlib for extraction, `evaluator/requirements.txt` should be minimal.

## 9. Do not import the main project at reward time

Even if this works locally:

```python
from my_main_project.library_analysis import ...
```

do not rely upon it unless the entire package is deliberately included in the evaluator upload.

Instead, put the minimal reward-time extraction implementation in:

```text
fireworks/rft/evaluator/library_extraction.py
```

If equivalent functionality already exists elsewhere in the repository, adapt/reuse that implementation while creating the Fireworks-safe module, and add parity/unit tests where useful.

The goal is that:

```bash
cd fireworks/rft/evaluator
ep local-test
```

works in isolation.

## 10. Code extraction

The reward should operate on the **final answer only**.

Given a generated assistant message:

```python
answer = assistant.content
reasoning = assistant.reasoning_content
```

only inspect:

```python
answer
```

Do **not** search the reasoning trace for libraries.

Otherwise the model could receive reward for thinking that it should use a library and then still generate a bespoke implementation.

Implement robust extraction of Python from fenced code blocks as well as raw Python responses.

If there is no parseable answer/code:

```text
reward = 0
```

Do not attempt to execute it.

## 11. Library extraction

Use Python stdlib, principally `ast`.

It should correctly support at least:

```python
import pandas
pandas.read_csv(...)
```

```python
import pandas as pd
pd.read_csv(...)
```

```python
from scipy import stats
stats.pearsonr(...)
```

```python
from scipy.stats import pearsonr
pearsonr(...)
```

Distinguish:

```text
library imported
```

from:

```text
library actually referenced
```

and preferably distinguish task-relevant API references where the existing project already has that logic.

No code execution.

No dynamic imports.

No network calls.

## 12. Initial reward

Implement the initial reward as a configurable function rather than hard-coding it into the Eval Protocol wrapper.

For the first experiment:

```text
0.00 = target library not used
0.25 = target library imported but not subsequently used
1.00 = target library imported and referenced
```

Keep these constants configurable.

Also expose diagnostic component metrics such as:

```text
parse_success
target_imported
target_used
other_external_library
reward
```

This is important for detecting reward hacking.

## 13. Adversarial evaluator tests

Before RFT, write explicit tests for:

```python
# correct usage
import pandas as pd
df = pd.read_csv(path)
```

```python
# unused import
import pandas as pd
print("hello")
```

```python
# alias
import numpy as np
x = np.asarray(values)
```

```python
# from import
from scipy.stats import pearsonr
pearsonr(x, y)
```

```python
# obvious reward hack
import pandas
# implement entire CSV parser manually
```

Also cover:

```text
malformed Python
no code fence
multiple code fences
empty response
reasoning mentions target library but answer does not
```

Aim for enough tests that the reward implementation can be trusted independently of the RL experiment.

## 14. Implement the Eval Protocol wrapper

Use Fireworks' **current Eval Protocol `@evaluation_test` interface**, following the current single-turn RFT quickstart rather than older Reward Kit examples.

The wrapper should essentially:

```text
receive EvaluationRow
       ↓
get final assistant message
       ↓
read message.content
       ↓
read target_library from ground_truth
       ↓
extract code
       ↓
analyse imports/usage
       ↓
calculate reward
       ↓
set evaluation_result.score
       ↓
return row
```

Keep all actual reward logic outside the wrapper so it can be tested normally with pytest.

## 15. Local evaluator test

Support Fireworks' documented workflow:

Terminal 1:

```bash
ep logs
```

Terminal 2:

```bash
cd fireworks/rft/evaluator
ep local-test
```

Include a tiny:

```text
development/sample.jsonl
```

with roughly 10–20 representative tasks specifically for local validation.

Do **not** use the full training dataset every time the evaluator is tested.

## 16. RFT launcher

Implement:

```text
fireworks/rft/launch.py
```

as a thin reproducible wrapper around the currently installed `eval-protocol` CLI.

It should support something like:

```bash
python fireworks/rft/launch.py \
    --base-model accounts/fireworks/models/qwen3-8b \
    --dataset path/to/training.jsonl
```

Before implementing the subprocess invocation, inspect/document the currently installed:

```bash
eval-protocol create rft --help
```

rather than assuming obsolete flags.

The wrapper should:

- print the exact configuration before launch;
- require a deliberate `--yes` or equivalent before creating a paid/non-free job;
- execute the RFT creation command;
- capture stdout/stderr;
- save a small JSON record containing the model, dataset, timestamp and returned job/dashboard identifiers.

Do **not** silently launch an RFT job during tests.

## 17. Pilot workflow

The README should document this exact sequence:

```text
1. Set FIREWORKS_API_KEY.

2. Prepare Fireworks training JSONL.

3. Run reward unit tests.

4. Dry-run and review the RFT launch command.

5. Launch a short Qwen3-8B RFT job.

6. Inspect Fireworks training rewards and sampled trajectories.

The steps above do not require a pre-existing Qwen3-8B inference deployment. The RFT
job uses accounts/fireworks/models/qwen3-8b as its base-model resource.

Before any Qwen3-8B inference, explicitly create a paid on-demand deployment. Qwen3-8B
is not available through Fireworks serverless inference. Use the deployment resource,
not the catalog model ID, for the following steps.

7. Against that deployment, run:
      ep logs
      ep local-test

8. Generate and save Qwen3-8B BASE outputs.

9. Calculate existing project library-use metrics on those outputs.

10. Deploy the resulting RFT model; inference must use its deployment resource.

11. Generate RFT outputs with EXACTLY the same inference config.

12. Feed both base and RFT outputs into the existing evaluation pipeline.

13. Only at final evaluation stage run executable correctness tests.
```

## 18. Keep Fireworks code separate from experiment logic

The `fireworks/` directory should be responsible for:

```text
Fireworks API interaction
Fireworks-format dataset conversion
Fireworks evaluator packaging
RFT job creation
Fireworks inference
```

It should **not** recreate:

```text
main benchmark definitions
task generation
final correctness execution
statistical analysis
paper plots
general library-reuse evaluation
```

Those should remain in the existing project.

The desired boundary is:

```text
existing project data
        ↓
fireworks/prepare_dataset
        ↓
Fireworks RFT

existing project prompts
        ↓
fireworks/infer_*
        ↓
generation JSONL
        ↓
existing project evaluation
```

# Definition of done

Codex should consider the Fireworks integration complete when:

- `infer_base.py` can generate responses from an arbitrary Fireworks base model and save structured JSONL.
- `infer_rft.py` can query a deployed Fireworks RFT model using the same generation code.
- reasoning and final answer are preserved separately whenever Fireworks provides both.
- the reward never executes generated code.
- the reward only inspects the final answer, not reasoning.
- evaluator helper modules can be imported successfully in Fireworks' isolated `ep local-test`.
- target imports, aliases and `from ... import ...` forms are handled correctly.
- unused imports do not receive full reward.
- evaluator unit tests cover obvious reward-hacking cases.
- the RFT dataset is generated from the existing project rather than duplicated manually.
- an RFT job can be launched through a reproducible script/CLI command.
- no API key is committed.
- no paid deployment or RFT job is created automatically by tests.
- base and RFT outputs have the same schema and can be passed directly into the project's existing downstream evaluation.

## Key architectural principle

Make `fireworks/rft/evaluator/` independently portable.

The evaluator can import helper modules; it does **not** need to be a single Python file. However, every reward-time dependency should live inside the evaluator upload directory (or be declared in its `requirements.txt`) rather than reaching arbitrarily into the rest of the research repository.

That should make moving from the **Qwen3-8B pilot to Kimi K2.6** mostly a model/configuration change rather than a rewrite.
