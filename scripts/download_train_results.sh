#!/usr/bin/env bash
# copies the eval results on the full training split (library_usage_train), on the script
# pilot (library_usage_script_pilot), and every response on the full LibReuseBench v2 generation
# (libreusebench_v2_full), down from isambard. most are too large to commit (thousands of long
# responses per model), so they are git-ignored and fetched with this instead; the
# libreusebench_v2_full runs with 5 samples are committed too, and copying them is harmless.
# run from the repo root on your own machine; existing files are overwritten with the latest.
#
#   bash scripts/download_train_results.sh         # download (or refresh) the local copies
#   bash scripts/download_train_results.sh clean   # delete the local copies (lists them and asks)

set -uo pipefail

HOST="u6yb.aip2.isambard"  # see ~/.ssh/config_clifton - run `clifton auth` first if refused
REMOTE_REPO="reuse"  # the repo's path on isambard, relative to the home directory

# the result files to copy, each as the analysis (.json) and every response (.generations.jsonl):
# the full training split's, and the script pilot's (see data/build/library/build.py --pilot).
# for libreusebench_v2_full only the responses are copied: its analysis files are committed
PATTERNS=(
    "library_usage_train-*"
    "library_usage_script_pilot-*"
    "libreusebench_v2_full-*.generations.jsonl"
)

mode="${1:-download}"

# the patterns as rsync include rules, and as a find test matching any of them
includes=()
name_tests=()
for pattern in "${PATTERNS[@]}"; do
    includes+=(--include="$pattern")
    if [ ${#name_tests[@]} -gt 0 ]; then
        name_tests+=(-o)
    fi
    name_tests+=(-name "$pattern")
done

case "$mode" in
    download)
        # walk every folder under output/ (--include='*/'), but copy only the matching result
        # files, skipping everything else (--exclude='*') and leaving out folders with none of
        # them (--prune-empty-dirs) - wherever they are, e.g. output/baselines/{model}/ or a
        # training run's epoch{n}/
        rsync -av --prune-empty-dirs \
            --include='*/' "${includes[@]}" --exclude='*' \
            "$HOST:$REMOTE_REPO/output/" output/
        ;;
    clean)
        # only the local copies - the originals on isambard are untouched
        leftovers=()
        while IFS= read -r path; do
            leftovers+=("$path")
        done < <(find output -type f \( "${name_tests[@]}" \) 2>/dev/null | sort)

        echo
        echo "Local copies of the training split's and script pilot's results:"
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
