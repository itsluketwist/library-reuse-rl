# Plan: stop reason on the parsed response, and in the format reward

Target: **main**, not the `sidekick` branch. Sketched 2026-09-04 while reviewing that branch.

## Idea

Give `ParsedResponse` a `stop_reason` field, so "how did generation end?" travels with the
response instead of being re-derived per consumer. The first-stage format reward then reads
it from the parsed object rather than reaching back into the raw training row.

## Why

Today the same question is answered two different ways, and neither knows about the other:

| | how it decides | vocabulary |
|---|---|---|
| eval | backend's `finish_reason` string → [`map_stop_reason`](src/eval/stop_reason.py) | `"eos"` / `"length"` / passthrough / `"unknown"` |
| grpo | `completion_ids[-1] != eos_token_id` ([`_is_truncated`](src/grpo/rewards/base.py#L144)) | bool |

They agree in meaning today, but nothing links them, and there are three concrete costs:

1. **`score_correct_format` takes `row` only to reach `completion_ids`.** Its other three
   arguments come from the parsed response; the row is threaded through purely for
   truncation. With `stop_reason` on the parsed object the signature collapses to
   `score_correct_format(reward, parsed)`.
2. **Truncation fails silently.** `_is_truncated` returns `False` when `eos_token_id` is
   `None` *or* `completion_ids` is missing — so if trl ever stops passing `completion_ids`,
   every truncated rollout quietly earns full format reward and nothing says so. There is a
   test pinning this fallback (`test_is_truncated_needs_both_eos_token_id_and_completion_ids`),
   so it is deliberate, but it is indistinguishable from "nothing was truncated".
3. **`n_overflow` and the format penalty are not provably the same notion.** Eval counts
   `stop_reason == "length"`; training penalises any non-eos ending. Those differ for a
   completion that stopped on a stop-string or content filter: eval keeps it visible as its
   own reason, training treats it as truncated.

## Shape of the change

- **`ParsedResponse.stop_reason: str`**, defaulting to `"unknown"` so the three existing
  `parse_response` call sites ([pipeline.py:36](src/response/pipeline.py#L36),
  [rewards/base.py:209](src/grpo/rewards/base.py#L209),
  [eval/scoring.py:258](src/eval/scoring.py#L258)) keep working untouched. Set by the
  caller via `parse_response(content, stop_reason=...)` — the parser never derives it,
  since it isn't in the text.
- **Move `stop_reason.py` from `src/eval/` to `src/response/`.** It is eval-only today,
  which is exactly why it currently sits there; once training consumes the same vocabulary
  it belongs beside `ParsedResponse`.
- **Add a token-side converter** next to `map_stop_reason`, e.g.
  `stop_reason_from_tokens(completion_ids, eos_token_id) -> str`, returning the same
  vocabulary (`"eos"` / `"length"` / `"unknown"` when either input is missing). This is the
  actual unification point: one vocabulary, two entry points (a backend string, or tokens).
- **Rewire the reward**: `LibraryReward.parse` attaches the stop reason, `_is_truncated`
  becomes `parsed.stop_reason == "length"` (or is dropped in favour of reading it inline),
  and `score_correct_format` loses its `row` argument.
- **Eval already has the value** — `scoring.py` carries `stop_reason` alongside each
  response and drops it into results; it would pass it into `parse_response` instead of
  keeping it in a parallel variable.

## The decision this forces

Narrowing "truncated" to `stop_reason == "length"` is **not reward-neutral**. Completions
that end for some other non-eos reason currently lose the `correct_format` component and
afterwards would keep it. That changes trained behaviour and breaks comparability with every
run so far.

Two options, pick deliberately:

- **Behaviour-preserving**: treat anything that is not `"eos"` as truncated. Pure refactor,
  existing runs stay comparable. Recommended as the first step.
- **Behaviour-changing**: penalise only `"length"`, and treat other reasons on their own
  terms. Cleaner semantics, but re-baseline the reward before comparing to old runs.

## Cost

Small — one dataclass field, one moved module, one new converter, one reward signature, plus
the tests for `_is_truncated` and `score_correct_format` in
`tests/test_grpo/test_rewards/`. The risk is not size but the reward-semantics decision
above; do it between experiment batches, not mid-sweep.

## Not doing

`fireworks/` is prototyping until better compute is sourced — it records `finish_reason` in
its generation records but its reward ignores it, and that stays as-is.
