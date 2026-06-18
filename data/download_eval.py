"""Download and format the code evaluation datasets used by the project.

Both datasets are reformatted into the same stdin/stdout schema, so the same
evaluator (src/response/evaluate.py) can run against either of them:

- LiveCodeBench (v5 + v6) - competitive programming problems, filtered to the
  subset solved by reading stdin and writing stdout (excludes the
  function-signature style LeetCode problems).
- APPS (introductory split) - a stdin/stdout dataset at a similar scope and
  difficulty to HumanEval/MBPP (no stdin/stdout reformatting of HumanEval/MBPP
  themselves exists, so APPS introductory is used instead for consistency of
  test format).

Usage: python data/download_eval.py
"""

import json
import random
from pathlib import Path

from datasets import Dataset, concatenate_datasets
from huggingface_hub import hf_hub_download
from llm_cgr import load_jsonl, save_jsonl


# formatted datasets are written here, ready for the evaluator to load directly
OUTPUT_DIR = Path(__file__).parent / "eval"

# commit hashes pinned for reproducibility
REVISIONS = {
    "livecodebench": "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
    "apps": "21e74ddf8de1a21436da12e3e653065c5213e9d1",
}

# used to deterministically sample apps down to the same size as livecodebench
SEED = 42

# apps's test cases are far less consistent in size than livecodebench's public
# ones - some are tens of MB, meant for performance stress-testing, which we
# don't need and don't want to commit to git. only apps needs capping (see below).
MAX_TEST_CASES = 20
MAX_TEST_CASE_CHARS = 20_000

# write a python program that reads stdin and writes stdout
PYTHON_INSTRUCTION = (
    "\n\nWrite a Python program that reads from stdin and writes to stdout."
)


def cap_test_cases(
    inputs: list[str],
    outputs: list[str],
) -> tuple[list[str], list[str]]:
    """Drop oversized test cases, then keep at most MAX_TEST_CASES of what remains.

    Returns the capped (inputs, outputs) lists.
    """
    kept = [
        (i, o)
        for i, o in zip(inputs, outputs)
        if len(i) <= MAX_TEST_CASE_CHARS and len(o) <= MAX_TEST_CASE_CHARS
    ]
    kept = kept[:MAX_TEST_CASES]
    return [i for i, _ in kept], [o for _, o in kept]


def is_stdin_only(
    public_test_cases: str,
) -> bool:
    """Check whether every public test case for a livecodebench problem uses stdin/stdout.

    Returns True if there is at least one test case and all of them are testtype "stdin".
    """
    tests = json.loads(public_test_cases) if public_test_cases else []
    return bool(tests) and all(t["testtype"] == "stdin" for t in tests)


def is_pure_stdin(
    row: dict,
) -> bool:
    """Check whether an apps problem is solved via stdin/stdout only (no function call).

    Returns True if the problem has no starter code and its tests have no fn_name set.
    """
    if (row["starter_code"] or "").strip():
        return False

    io = json.loads(row["input_output"]) if row["input_output"] else {}
    return bool(io.get("inputs")) and not io.get("fn_name")


def download_livecodebench() -> list[dict]:
    """Download livecodebench (v5 + v6) and filter to stdin/stdout-only problems.

    Source: https://huggingface.co/datasets/livecodebench/code_generation_lite

    The v5 and v6 releases are the test5.jsonl and test6.jsonl files in the repo.
    `datasets` 4.0+ dropped support for this repo's loading script (see
    https://github.com/LiveCodeBench/LiveCodeBench/issues/107), so the jsonl
    files are downloaded directly instead of using load_dataset. Each problem's
    public_test_cases is small (a few hundred chars, a handful per problem), so
    they aren't capped like apps's are.

    Returns the formatted livecodebench records.
    """
    lcb_datasets = []
    for filename in ["test5.jsonl", "test6.jsonl"]:
        jsonl_path = hf_hub_download(
            repo_id="livecodebench/code_generation_lite",
            filename=filename,
            repo_type="dataset",
            revision=REVISIONS["livecodebench"],
        )
        lcb_datasets.append(Dataset.from_json(jsonl_path))

    lcb = concatenate_datasets(lcb_datasets)
    print(f"loaded {len(lcb)} livecodebench problems (v5 + v6)")

    records = []
    for row in lcb:
        if not is_stdin_only(row["public_test_cases"]):
            continue

        tests = json.loads(row["public_test_cases"])
        records.append(
            {
                "prompt": row["question_content"] + PYTHON_INSTRUCTION,
                "inputs": [t["input"] for t in tests],
                "expected_outputs": [t["output"] for t in tests],
                "source": "livecodebench",
            }
        )

    print(f"  {len(records)} stdin/stdout records out of {len(lcb)} total problems")
    return records


def download_apps(
    num_records: int,
) -> list[dict]:
    """Download apps (introductory split) and filter to stdin/stdout-only problems.

    Source: https://huggingface.co/datasets/codeparrot/apps

    APPS is a stdin/stdout competitive-programming dataset; its introductory
    difficulty is comparable in scope to HumanEval/MBPP. Like livecodebench,
    this repo ships a loading script, so the jsonl file is downloaded directly.
    A small minority of apps problems are solved by implementing a named function
    rather than via stdin/stdout - those are excluded. The bulky solutions field
    isn't needed for evaluation, so it's dropped from the saved records.

    The filtered set is much bigger than livecodebench, so it's randomly
    downsampled (with a fixed seed, for reproducibility) to num_records, keeping
    the two evaluation sets comparably sized.

    Returns the formatted, downsampled apps records.
    """
    apps_path = hf_hub_download(
        repo_id="codeparrot/apps",
        filename="test.jsonl",
        repo_type="dataset",
        revision=REVISIONS["apps"],
    )
    apps = Dataset.from_json(apps_path)
    print(f"loaded {len(apps)} apps problems (all difficulties)")

    records = []
    for row in apps:
        if row["difficulty"] != "introductory" or not is_pure_stdin(row):
            continue

        io = json.loads(row["input_output"])
        inputs, expected_outputs = cap_test_cases(io["inputs"], io["outputs"])
        if not inputs:
            continue

        records.append(
            {
                "prompt": row["question"],
                "inputs": inputs,
                "expected_outputs": expected_outputs,
                "source": "apps_introductory",
            }
        )

    print(f"  found {len(records)} stdin/stdout records")
    records = random.Random(SEED).sample(records, num_records)
    print(f"  downsampled to {len(records)} records")
    return records


def main() -> None:
    """Download, format, and save the livecodebench and apps evaluation datasets."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lcb_records = download_livecodebench()
    save_jsonl(lcb_records, str(OUTPUT_DIR / "livecodebench.jsonl"))

    apps_records = download_apps(num_records=len(lcb_records))
    save_jsonl(apps_records, str(OUTPUT_DIR / "apps_introductory.jsonl"))

    print("\nsanity check:")
    for filename in ["livecodebench.jsonl", "apps_introductory.jsonl"]:
        records = load_jsonl(str(OUTPUT_DIR / filename))
        sample = records[0]
        print(f"\n{filename}: {len(records)} records")
        print(f"  prompt (first 200 chars): {sample['prompt'][:200]!r}")
        print(f"  num test cases: {len(sample['inputs'])}")


if __name__ == "__main__":
    main()
