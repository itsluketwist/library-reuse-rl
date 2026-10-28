"""Score HumanEval+ responses: run each answer's function against the task's check()."""

import re
from typing import Any, Iterable

from libreusebench.environment import Environment, execute
from libreusebench.evaluate import _code_stats, plan_installs
from lure import CodeBlock, parse_response

from src.eval.scoring.base import _average, _library_statistics, _rate
from src.response.metrics import code_metrics


# the only third-party package the HumanEval+ tests import (163 of the 164 use it); the
# prompts and reference solutions use just the standard library
TEST_PACKAGE = "numpy"


def select_block(
    response: str,
    stop_reason: str,
    entry_point: str,
) -> CodeBlock | None:
    """
    Pick the code block to test: the longest one that defines the function under test.

    Answers often follow the function with a longer block of example calls or asserts, so
    the longest block alone can be one that never defines the function. When no block
    defines it, the longest block is used, and fails the test as it should.

    Returns the chosen CodeBlock, or None when the answer has no code.
    """
    parsed = parse_response(response, stop_reason=stop_reason)
    definition = re.compile(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", re.MULTILINE)
    defining = [block for block in parsed.code_blocks if definition.search(block.code)]
    if not defining:
        return parsed.longest_block
    return max(defining, key=lambda block: len(block.code.splitlines()))


def build_test_program(
    code: str,
    test: str,
    entry_point: str,
) -> str:
    """
    Build a program that runs a response's code, then calls the task's check() on its function.

    The response runs in its own namespace, so its names can't clash with the test's helpers
    (e.g. a helper of its own called `check`). Naming that namespace "solution" also means an
    `if __name__ == "__main__":` demo block in the response is skipped, as when imported.

    Returns the program's source code.
    """
    return (
        "_solution = {'__name__': 'solution'}\n"
        f"exec(compile({code!r}, '<solution>', 'exec'), _solution)\n"
        f"{test}\n"
        f"check(_solution[{entry_point!r}])\n"
    )


def score_humaneval(
    dataset: Iterable[dict],
    responses: list[list[tuple[str, str, int | None]]],
    python_executable: str | None = None,
    environment_root: str | None = None,
    environment_cache_dir: str | None = None,
) -> tuple[dict, list[dict]]:
    """
    Score HumanEval+ responses, every sample of every task, by running the answer's code block
    (see select_block) against the task's tests (original HumanEval cases plus EvalPlus's
    extras).

    The code runs in a venv built the way LibReuseBench builds one (see
    libreusebench.environment.Environment): disposable under `environment_root`, or reused
    from `environment_cache_dir` for debug runs. It gets numpy for the tests, and every
    package the answers import, so an answer that uses a library is judged on its code rather
    than on what happened to be installed. Each run uses LibReuseBench's sandbox, with its
    30 s time limit, its memory cap and numeric libraries held to one thread.

    Returns the analysis (pass@1, pass@k for k = samples per task, library use, code size as
    LibReuseBench measures it, the venv's packages) and one result per response.
    """
    del python_executable
    rows = list(dataset)
    # the pip packages the answers' code imports, as {package: {import names}}
    plan = plan_installs(
        {
            row["task_id"]: [response for response, _stop_reason, _tokens in samples]
            for row, samples in zip(rows, responses)
        }
    )

    results: list[dict[str, Any]] = []
    # passed samples per task, for pass@k
    task_passes: list[list[bool]] = []
    with Environment(
        cache_dir=environment_cache_dir,
        scratch_root=environment_root,
    ) as env:
        # a package that fails to install (e.g. a made-up one) only fails the answers that
        # import it - but without numpy every test would fail, which would look like a model
        # scoring zero rather than a broken environment
        install_failed = env.ensure_packages({TEST_PACKAGE, *plan})
        if TEST_PACKAGE in install_failed:
            raise RuntimeError(
                f"could not install {TEST_PACKAGE}, which the tests need"
            )
        # import each library once, so the first answer to use one isn't timed on it
        env.warm_up(
            {TEST_PACKAGE, *(name for names in plan.values() for name in names)}
        )

        for row, samples in zip(rows, responses):
            passes = []
            for sample, (response, stop_reason, completion_tokens) in enumerate(
                samples
            ):
                block = select_block(
                    response=response,
                    stop_reason=stop_reason,
                    entry_point=row["entry_point"],
                )
                # no code means nothing to run, so the sample fails
                passed, output = False, None
                if block is not None:
                    passed, output = execute(
                        code=build_test_program(
                            code=block.code,
                            test=row["test"],
                            entry_point=row["entry_point"],
                        ),
                        python_executable=env.python,
                    )
                passes.append(passed)
                # the tested block's size, measured as libreusebench measures its answers
                lloc, maintainability_index = (
                    code_metrics(block.code) if block is not None else (None, None)
                )
                results.append(
                    {
                        "task_id": row["task_id"],
                        "sample": sample,
                        "stop_reason": stop_reason,
                        "completion_tokens": completion_tokens,
                        "response": response,
                        "has_code": block is not None,
                        "passed": passed,
                        # the error message when the code failed (stdout when it passed)
                        "output": output,
                        "ext_libs": sorted(block.external_imports) if block else [],
                        "lloc": lloc,
                        "maintainability_index": maintainability_index,
                    }
                )
            task_passes.append(passes)

        # what the answers ran with, for the record (the venv may be deleted after this)
        environment = {
            "installed_packages": sorted({TEST_PACKAGE, *plan} - install_failed),
            "install_failed": sorted(install_failed),
            "distributions": env.installed_distributions(),
        }

    n_samples = len(task_passes[0])
    return {
        "task": "humaneval",
        "n_total": len(results),
        "n_tasks": len(task_passes),
        "n_passed": sum(result["passed"] for result in results),
        # the unbiased pass@1 is each task's fraction of passing samples, averaged
        "pass@1": _average([sum(passes) / len(passes) for passes in task_passes]),
        # with k equal to the samples per task, pass@k is the share of tasks solved at all
        f"pass@{n_samples}": _rate(sum(map(any, task_passes)), len(task_passes)),
        "n_no_code": sum(not result["has_code"] for result in results),
        "n_overflow": sum(result["stop_reason"] == "length" for result in results),
        **_library_statistics([set(result["ext_libs"]) for result in results]),
        # the same shape as libreusebench's, so the two can be read the same way
        "code_metrics": {
            "all": _code_stats(results),
            "passed": _code_stats([result for result in results if result["passed"]]),
        },
        "environment": environment,
    }, results
