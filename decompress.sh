#!/usr/bin/env bash
# Unpack the WADI 2017 and 2019 zip files into the layout common.py expects:
#
#   data/raw/WADI.A1_9 Oct 2017/WADI_14days.csv, WADI_attackdata.csv, ...
#   data/raw/WADI.A2_19 Nov 2019/WADI_14days_new.csv, WADI_attackdataLABLE.csv, ...
#   data/cache/                      (empty, filled by common.load_year)
#
# Existing files are kept (safe to run again); --force overwrites them.
#
# usage: ./decompress.sh [--force]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="$HERE/data/raw"
CACHE="$HERE/data/cache"

ZIPS=(
    "WaDi.A1_9 Oct 2017.zip"
    "WaDi.A2_19 Nov 2019.zip"
)

overwrite="-n"                      # -n: never overwrite existing files
if [[ "${1:-}" == "--force" ]]; then
    overwrite="-o"                  # -o: overwrite without asking
elif [[ -n "${1:-}" ]]; then
    echo "usage: $0 [--force]" >&2
    exit 2
fi

if ! command -v unzip >/dev/null; then
    echo "error: 'unzip' is not installed (e.g. sudo apt install unzip)" >&2
    exit 1
fi

mkdir -p "$RAW" "$CACHE"

missing=0
for zip in "${ZIPS[@]}"; do
    path="$HERE/$zip"
    if [[ ! -f "$path" ]]; then
        echo "zip not found: $path" >&2
        missing=1
        continue
    fi
    echo "== $zip"
    # -x: skip the Windows folder metadata file
    unzip $overwrite "$path" -d "$RAW" -x "*/desktop.ini"
done

echo "done -> $RAW"
exit $missing
