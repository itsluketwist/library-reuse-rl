"""Download and format the regression tests of general coding ability used by the project.

- HumanEval+ is kept in its own function-call format (scored by src/eval/scoring/humaneval.py).
- LiveCodeBench: the easy and medium problems LiveCodeBench added in releases v5 and v6
  (test5.jsonl and test6.jsonl at the pinned revision, contests from 2024-09-22 to
  2025-04-06) that are solved by one named function - its LeetCode problems, asked for as a
  plain function as LibReuseBench's tasks are. Each is tested on its public and private test
  cases, in LibReuseBench's task format, so LibReuseBench's evaluator scores it - pass@k over
  several samples, and library use and code size, as for the benchmark itself. Answers are
  checked exactly, as LiveCodeBench checks them, so a problem that accepts an answer within a
  tolerance is left out.

Usage: python data/build/download_eval.py
"""

import ast
import base64
import io
import json
import pickle
import zlib

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from llm_cgr import save_jsonl

from data.build.paths import EVAL_DIR as OUTPUT_DIR


# commit hashes pinned for reproducibility
REVISIONS = {
    "livecodebench": "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
    # the same pin as library-adoption-rl and llm-training, so a task scores the same in each
    "humaneval_plus": "d32357c",
}

# livecodebench's private test cases run to megabytes, made to stress a solution's speed,
# which a regression check doesn't need - so each problem keeps at most this many, of at
# most this size
MAX_TEST_CASES = 20
MAX_TEST_CASE_CHARS = 20_000

# the livecodebench release that added each file's problems
LIVECODEBENCH_RELEASES = {
    "test5.jsonl": "release_v5",
    "test6.jsonl": "release_v6",
}

# the livecodebench problems kept: the easy and medium ones (an 8b model rarely solves a
# hard one, and spends its whole token budget trying)
LIVECODEBENCH_DIFFICULTIES = ("easy", "medium")

# how a problem says it accepts an answer within a tolerance. the answers are checked
# exactly, so these problems are left out
TOLERANCE_WORDING = "Answers within 10^"

# asks for the problem's function by name, as libreusebench's tasks do
FUNCTION_INSTRUCTION = "\n\nWrite a Python function `{signature}` that solves this."


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


class _PlainUnpickler(pickle.Unpickler):
    """
    Unpickle only plain values (strings, numbers, lists, dicts). Loading any class or
    function, which is how a pickle can run code, is refused - livecodebench's private test
    cases are downloaded data, pickled only to hold a JSON string.
    """

    def find_class(
        self,
        module: str,
        name: str,
    ) -> type:
        raise pickle.UnpicklingError(
            f"refusing to load {module}.{name} from a test file"
        )


def decode_private_tests(
    encoded: str,
) -> list[dict]:
    """
    Decode a livecodebench problem's private test cases: base64, then zlib, then a pickled
    JSON string (as livecodebench stores them), unpickled safely.

    Returns the test cases, as dicts with "input", "output" and "testtype".
    """
    raw = zlib.decompress(base64.b64decode(encoded))
    return json.loads(_PlainUnpickler(io.BytesIO(raw)).load())


def function_signature(
    starter_code: str,
    function_name: str,
) -> str:
    """
    Turn the method in livecodebench's starter code into a plain function's signature, e.g.
    "findXSum(nums, k, x)" from "class Solution: def findXSum(self, nums: List[int], ...".

    Returns the function's name and parameters.
    """
    # the starter code stops where the method's body would start
    tree = ast.parse(starter_code + "pass")
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    parameters = [arg.arg for arg in method.args.args[1:]]  # without self
    return f"{function_name}({', '.join(parameters)})"


def download_livecodebench() -> list[dict]:
    """
    Download livecodebench's v5 and v6 problems and keep the easy and medium ones that are
    solved by one named function and checked exactly.

    Source: https://huggingface.co/datasets/livecodebench/code_generation_lite

    `datasets` 4.0+ dropped support for this repo's loading script (see
    https://github.com/LiveCodeBench/LiveCodeBench/issues/107), so the jsonl files are
    downloaded directly. Each record holds the problem's function name and its tests, the
    public ones then the private ones (capped, see cap_test_cases), as the python literals
    LibReuseBench's function tasks hold, with rel_tol and abs_tol 0 so returned numbers
    must match exactly. It keeps the problem's difficulty, platform, contest date and release.

    Returns the livecodebench records.
    """
    records = []
    for filename, release in LIVECODEBENCH_RELEASES.items():
        jsonl_path = hf_hub_download(
            repo_id="livecodebench/code_generation_lite",
            filename=filename,
            repo_type="dataset",
            revision=REVISIONS["livecodebench"],
        )
        with open(jsonl_path) as f:
            for line in f:
                row = json.loads(line)
                public_tests = json.loads(row["public_test_cases"])
                if any(test["testtype"] != "functional" for test in public_tests):
                    continue
                if row["difficulty"] not in LIVECODEBENCH_DIFFICULTIES:
                    continue
                if TOLERANCE_WORDING in row["question_content"]:
                    continue

                function_name = json.loads(row["metadata"])["func_name"]
                signature = function_signature(
                    starter_code=row["starter_code"],
                    function_name=function_name,
                )
                tests = public_tests + decode_private_tests(row["private_test_cases"])
                # a test's input is a json value per argument, one per line, and its output
                # one json value - rewritten as python literals
                inputs, outputs = cap_test_cases(
                    inputs=[
                        repr([json.loads(value) for value in test["input"].split("\n")])
                        for test in tests
                    ],
                    outputs=[repr(json.loads(test["output"])) for test in tests],
                )
                records.append(
                    {
                        "task_id": f"LCB/{row['question_id']}",
                        "prompt": row["question_content"]
                        + FUNCTION_INSTRUCTION.format(signature=signature),
                        "function_name": function_name,
                        "test_inputs": inputs,
                        "expected_outputs": outputs,
                        "rel_tol": 0.0,
                        "abs_tol": 0.0,
                        "difficulty": row["difficulty"],
                        "platform": row["platform"],
                        "contest_date": row["contest_date"][:10],
                        # copied into each result's metadata, so results say which releases
                        # they cover
                        "dataset_version": release,
                        "source": "livecodebench",
                    }
                )

    print(f"  {len(records)} livecodebench problems")
    return records


def download_humaneval_plus() -> list[dict]:
    """Download HumanEval+ (all 164 tasks) in its own function-call format.

    Source: https://huggingface.co/datasets/evalplus/humanevalplus
    (Liu et al., NeurIPS 2023, Apache-2.0; built on OpenAI's HumanEval, MIT.)

    Each prompt is a function signature and docstring, given to the model as-is, as in
    library-adoption-rl. Each `test` defines a `check(candidate)` function holding the
    original HumanEval tests plus EvalPlus's extra ones, so nothing needs converting.
    One task, HumanEval/32, fails even its own canonical solution at this revision (its
    check() unpacks a float), so the best possible pass@1 is 163/164.

    Returns the HumanEval+ records.
    """
    rows = load_dataset(
        "evalplus/humanevalplus",
        split="test",
        revision=REVISIONS["humaneval_plus"],
    )
    print(f"loaded {len(rows)} humaneval+ problems")

    return [
        {
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "entry_point": row["entry_point"],
            "test": row["test"],
            "source": "humaneval_plus",
        }
        for row in rows
    ]


def main() -> None:
    """Download, format, and save the humaneval+ and livecodebench evaluation datasets."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_jsonl(download_humaneval_plus(), str(OUTPUT_DIR / "humaneval_plus.jsonl"))
    save_jsonl(download_livecodebench(), str(OUTPUT_DIR / "livecodebench.jsonl"))


if __name__ == "__main__":
    main()
