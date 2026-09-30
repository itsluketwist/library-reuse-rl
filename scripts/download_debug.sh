#!/usr/bin/env bash
# copies every job's logs and results down from isambard into this repo: logs/ (debug checks,
# eval and training jobs) and output/ (debug outputs, base models' eval results, and training
# runs' metrics and epoch evals). run from the repo root on your own machine - existing files
# are overwritten with the latest, and nothing here is deleted.
#
#   bash scripts/download_debug.sh

set -uo pipefail

HOST="u6yb.aip2.isambard"  # see ~/.ssh/config_clifton - run `clifton auth` first if refused
REMOTE_REPO="reuse"  # the repo's path on isambard, relative to the home directory

# the trailing slashes copy each folder's contents into the matching folder here. rsync only
# transfers what has changed, so repeat runs are quick.
rsync -av "$HOST:$REMOTE_REPO/logs/" logs/
rsync -av "$HOST:$REMOTE_REPO/output/" output/
