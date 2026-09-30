#!/usr/bin/env bash
# copies what isambard produces but git doesn't track down into this repo: every job's logs,
# and the debug checks' outputs (scripts/submit_debug.sh). committed results - the base models'
# and training runs' evals - come through git instead. run from the repo root on your own
# machine; existing files are overwritten with the latest.
#
#   bash scripts/download_debug.sh         # download (or refresh) the local copies
#   bash scripts/download_debug.sh clean   # delete the local copies (lists them and asks first)

set -uo pipefail

HOST="u6yb.aip2.isambard"  # see ~/.ssh/config_clifton - run `clifton auth` first if refused
REMOTE_REPO="reuse"  # the repo's path on isambard, relative to the home directory

# the debug checks' outputs, all git-ignored (see .gitignore) - these paths are both what gets
# downloaded and what the clean option deletes, so the two can't drift apart
DEBUG_OUTPUTS=(
    "output/debug"
    "output/qwen3-debug"
    "output/baselines/qwen3-debug"
    "output/*/*-scale"
)

mode="${1:-download}"

case "$mode" in
    download)
        # every job's logs (debug checks, eval and training jobs) - all git-ignored
        rsync -av "$HOST:$REMOTE_REPO/logs/" logs/

        # -R recreates each path from the "/./" marker onwards, so everything lands in the
        # same place here. a folder that doesn't exist yet is reported and skipped, and the
        # rest still copies.
        sources=()
        for path in "${DEBUG_OUTPUTS[@]}"; do
            sources+=("$HOST:$REMOTE_REPO/./$path")
        done
        rsync -avR "${sources[@]}" .
        ;;
    clean)
        # only the local copies - the originals on isambard are untouched
        shopt -s nullglob
        candidates=(logs)
        for pattern in "${DEBUG_OUTPUTS[@]}"; do
            # unquoted, so patterns like output/*/*-scale expand to the matching folders
            for path in $pattern; do
                candidates+=("$path")
            done
        done
        leftovers=()
        for path in "${candidates[@]}"; do
            if [ -e "$path" ]; then
                leftovers+=("$path")
            fi
        done

        echo
        echo "Local copies of downloaded logs and debug outputs:"
        if [ ${#leftovers[@]} -eq 0 ]; then
            echo "  (no files)"
            exit 0
        fi
        du -sh "${leftovers[@]}" | sed 's/^/  /'
        echo

        read -r -p "Delete all of these? [y/N] " answer
        if [ "$answer" != "y" ]; then
            echo "Nothing deleted."
            exit 0
        fi
        rm -rf "${leftovers[@]}"
        echo "Done."
        ;;
    *)
        echo "Usage: bash scripts/download_debug.sh [download|clean]"
        exit 1
        ;;
esac
