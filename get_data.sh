#!/usr/bin/env bash
# Linux mirror of get_data.bat. Download 1 shard each of the rocket-science
# train/test splits (via the sibling mira env) and write data_paths.sh -- the
# Linux equivalent of data_paths.bat, sourced by train.sh. Edit SHARDS in
# mira/get_data.py for more.
#   RUN   env prefix (default "pixi run --frozen"; "" if torch on PATH)
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mira="$here/../mira"
[ -f "$mira/get_data.py" ] || { echo "ERROR: $mira/get_data.py not found (clone mira beside this repo)"; exit 1; }
RUN="${RUN:-pixi run --frozen}"

cd "$mira"
$RUN python "$mira/get_data.py" || { echo "ERROR: get_data.py failed"; exit 1; }

# get_data.py writes data_paths.bat (`set "VAR=val"`); translate to a POSIX
# data_paths.sh (`export VAR="val"`) that train.sh can source on Linux.
if [ -f "$mira/data_paths.bat" ]; then
    grep -E '^set "(TRAIN|TEST)_INDEX=' "$mira/data_paths.bat" \
        | sed -E 's/^set "([^=]+)=(.*)"$/export \1="\2"/' > "$mira/data_paths.sh"
    echo "wrote $mira/data_paths.sh:"
    cat "$mira/data_paths.sh"
else
    echo "WARNING: $mira/data_paths.bat not written -- get_data.py may have failed"
fi
