#!/usr/bin/env bash
# copies what isambard produces but git doesn't track down into this repo: every job's logs,
# and the debug checks' outputs (scripts/submit_debug.sh). committed results - the base models'
# and training runs' evals - come through git instead. run from the repo root on your own
# machine; existing files are overwritten with the latest, and nothing here is deleted.
#
#   bash scripts/download_debug.sh

set -uo pipefail

HOST="u6yb.aip2.isambard"  # see ~/.ssh/config_clifton - run `clifton auth` first if refused
REMOTE_REPO="reuse"  # the repo's path on isambard, relative to the home directory

# every job's logs (debug checks, eval and training jobs) - all git-ignored
rsync -av "$HOST:$REMOTE_REPO/logs/" logs/

# the debug checks' outputs, all git-ignored (see .gitignore). -R recreates each path from the
# "/./" marker onwards, so everything lands in the same place here. a folder that doesn't exist
# yet is reported and skipped, and the rest still copies.
rsync -avR \
    "$HOST:$REMOTE_REPO/./output/debug" \
    "$HOST:$REMOTE_REPO/./output/qwen3-debug" \
    "$HOST:$REMOTE_REPO/./output/baselines/qwen3-debug" \
    "$HOST:$REMOTE_REPO/./output/*/*-scale" \
    .
