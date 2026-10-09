#!/bin/sh
set -eu
release_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export LD_LIBRARY_PATH="$release_dir/lib${KATAGO_LIBRARY_PATH:+:$KATAGO_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$release_dir/katago" "$@"
