#!/usr/bin/env bash
# copies the debug runs' logs and results down from isambard into this repo (all git-ignored).
# run from the repo root on your own machine; existing files are overwritten with the latest.
#
#   bash scripts/download_debug.sh

set -uo pipefail

HOST="u6yb.aip2.isambard"  # see ~/.ssh/config_clifton - run `clifton auth` first if refused
REMOTE_REPO="reuse"  # the repo's path on isambard, relative to the home directory

# -R recreates each path from the "/./" marker onwards, so everything lands in the same place
# here. a folder that doesn't exist yet is reported and skipped, and the rest still copies.
rsync -avR \
    "$HOST:$REMOTE_REPO/./logs/debug" \
    "$HOST:$REMOTE_REPO/./output/debug" \
    "$HOST:$REMOTE_REPO/./output/qwen3-debug" \
    "$HOST:$REMOTE_REPO/./output/*/*-scale" \
    .
