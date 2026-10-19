# lure: Library Usage REwards

Reward functions that score LLM-generated code for **reusing external libraries** instead of
reimplementing them, for reinforcement learning (e.g. GRPO) or evaluation.

- **Static and safe:** code is parsed with Python's `ast` module and never executed.
- **No dependencies:** only the standard library is used, and Python 3.10+ is supported.
- **Hard to fool:** a library only counts as used when the code actually references what it
  imported. Unused imports, names that are rebound, imports inside strings, code in the
  reasoning, and truncated code blocks earn no usage credit.

```bash
pip install lure
```

## Reward functions

````python
from lure import create_library_reward

reward_func = create_library_reward()

response = """<think>A CSV parser already exists, I should reuse one.</think>
```python
import pandas as pd

def load(path):
    return pd.read_csv(path)
```"""

reward_func(response, stop_reason="stop")  # 2.0
````

The reward is the sum of the weights of the components the response satisfies:

| Component | Default weight | Satisfied when |
|---|---|---|
| `correct_format` | 0.25 | the response has reasoning and an answer, and wasn't truncated (`stop_reason="length"`) |
| `passes_syntax_check` | 0.25 | the answer contains python code, and every code block is valid |
| `external_lib_imported` | 0.5 | the code imports an external (non-standard-library) library |
| `external_lib_used` | 1.0 | the code imports an external library and actually uses it |
| `appropriate_lib_used` | 0.0 | the code uses one of `appropriate_libraries` |

Every weight is a keyword argument, and setting a weight to 0 switches that component off:

```python
reward_func = create_library_reward(
    external_lib_used=1.0,
    appropriate_lib_used=0.5,
    appropriate_libraries=["requests", "httpx"],  # import or pypi names, any case
    local_modules=["app"],                        # the project's own code is never a library
    require_reasoning=False,                      # for non-thinking models
)

# local_modules and appropriate_libraries can also be set per response
reward_func(response, stop_reason="stop", appropriate_libraries=["pandas"])

# see which components a response satisfied, e.g. for logging
reward_func.components(response)

# the highest possible reward, for normalising to [0, 1]
reward_func.max_reward
```

When each dataset item has its own local modules or appropriate libraries, give them as a
dict keyed by item id, and pass each response's `item_id` when scoring it:

```python
reward_func = create_library_reward(
    appropriate_lib_used=0.5,
    local_modules={"task-1": ["app"], "task-2": ["server", "utils"]},
    appropriate_libraries={"task-1": ["requests", "httpx"], "task-2": ["pandas"]},
)

reward_func(response, stop_reason="stop", item_id="task-2")
```

A list passed with the response takes priority over the dict. An `item_id` that is missing
or not in the dict raises an error rather than using no names. Otherwise the project's own
imports would silently count as external libraries.

`stop_reason` is how generation ended, e.g. vLLM's `output.outputs[0].finish_reason` or the
OpenAI API's `choice.finish_reason`. If it's left out, the response is assumed not to be
truncated.

### With TRL's GRPOTrainer

```python
from lure import create_library_reward, create_trl_reward
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    reward_funcs=create_trl_reward(
        reward=create_library_reward(),
        eos_token_id=tokenizer.eos_token_id,  # detects truncated completions
    ),
    train_dataset=dataset,  # optional columns: local_modules, appropriate_libraries
    ...
)
```

If the reward was created with per-item dicts, each row's `item_id` is read from the dataset's
`id` column (change it with `item_id_column=...`).

## Parsing responses

`parse_response` exposes everything the reward is based on:

```python
from lure import parse_response

parsed = parse_response(response, stop_reason="stop", local_modules=["app"])

parsed.reasoning         # text before </think>
parsed.answer            # text after </think>
parsed.code_blocks       # the python code blocks in the answer, each analysed
parsed.external_imports  # frozenset({"pandas"})
parsed.external_used     # frozenset({"pandas"})
parsed.stdlib_imports    # frozenset()
parsed.valid             # True: there is code and it all parses
parsed.truncated         # False
parsed.longest_block     # the longest code block, e.g. to run it
```

The rules it follows:

- **Reasoning** is everything before the first `</think>` (also `</thinking>`, `</reasoning>`,
  `</thought>`, or your own tags, see below), and is never analysed. The opening tag is
  optional, because some chat templates put it in the prompt. An opening tag with no closing
  tag means there is no answer. A response with no tags at all is the answer, unless you pass
  `require_reasoning=True`: then it counts as unfinished reasoning (e.g. cut off before
  `</think>` when the template put `<think>` in the prompt), with no answer. The reward
  functions require reasoning by default.
- **Code** is every complete fenced block labelled `python`, `py` or `python3`, plus any
  unlabelled fenced block whose code parses as python (unlabelled blocks often hold shell
  commands, program output or directory trees). Other languages and unterminated fences are
  ignored. If the answer has no fences but is entirely valid python, the whole answer is
  used as the code.
- **Libraries** are top-level import names in their original case (e.g. `PIL`). The standard
  library, relative imports, `__future__`, and `local_modules` are never counted as external.
  A library is *used* when a single code block imports it and then references it.
  Referencing it in a different block doesn't count, and neither does a name rebound by
  assignment, loop variable, argument or similar.
- **Placeholder modules** aren't counted as libraries either, by default. These are names that
  stand for code the response doesn't have: a placeholder such as `your_module` or `my_app`, a
  generic project-module name such as `utils` or `models`, or a name like `data_utils` or
  `base_config`. A model that writes `from your_module import parse` beside its own code isn't
  reusing a library. Each analysed block lists them in `placeholder_imports`; pass
  `ignore_placeholders=False` (to `parse_response`, `analyse_code` or
  `create_library_reward`) to count them as before, and see `is_placeholder_module` for the
  exact rule.

Models that mark their reasoning with other tags can pass them as `(opening, closing)`
pairs. They are matched literally, ignoring case, and replace the defaults.
`create_library_reward` takes the same `reasoning_tags` argument:

```python
parse_response(response, reasoning_tags=[("[THINK]", "[/THINK]")])
parse_response(response, reasoning_tags=("<|begin_of_thought|>", "<|end_of_thought|>"))
```

The building blocks are also available on their own: `split_reasoning`, `extract_code_blocks`,
`analyse_code`, `is_placeholder_module`, and `pip_package` (e.g.
`pip_package("PIL") == "Pillow"`).

## Citation

Paper coming soon.

## Licence

lure is released under the MIT licence.
