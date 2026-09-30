#!/usr/bin/env bash
# copies the eval results on the full training split (library_usage_train) down from isambard.
# they are too large to commit (thousands of long responses per model), so they are git-ignored
# and fetched with this instead. run from the repo root on your own machine; existing files are
# overwritten with the latest.
#
#   bash scripts/download_train_results.sh         # download (or refresh) the local copies
#   bash scripts/download_train_results.sh clean   # delete the local copies (lists them and asks)

set -uo pipefail

HOST="u6yb.aip2.isambard"  # see ~/.ssh/config_clifton - run `clifton auth` first if refused
REMOTE_REPO="reuse"  # the repo's path on isambard, relative to the home directory

# the training split's result files: the analysis (.json) and every response (.generations.jsonl)
PATTERN="library_usage_train-*"

mode="${1:-download}"

case "$mode" in
    download)
        # walk every folder under output/ (--include='*/'), but copy only the training split's
        # result files, skipping everything else (--exclude='*') and leaving out folders with
        # none of them (--prune-empty-dirs) - wherever they are, e.g. output/baselines/{model}/
        # or a training run's epoch{n}/
        rsync -av --prune-empty-dirs \
            --include='*/' --include="$PATTERN" --exclude='*' \
            "$HOST:$REMOTE_REPO/output/" output/
        ;;
    clean)
        # only the local copies - the originals on isambard are untouched
        leftovers=()
        while IFS= read -r path; do
            leftovers+=("$path")
        done < <(find output -type f -name "$PATTERN" 2>/dev/null | sort)

        echo
        echo "Local copies of the training split's results:"
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
        rm -f "${leftovers[@]}"
        echo "Done."
        ;;
    *)
        echo "Usage: bash scripts/download_train_results.sh [download|clean]"
        exit 1
        ;;
esac
