#!/usr/bin/env bash
set -euo pipefail
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
args=()
setup=0
for arg in "$@"; do
  if [[ $arg == --setup-env ]]; then setup=1; else args+=("$arg"); fi
done
if ((setup)); then
  if ((EUID == 0)); then bash "$root/setup-linux.sh"; else sudo bash "$root/setup-linux.sh"; fi
  source /opt/katago-rtx50/env.sh
fi
python=''
for candidate in /usr/bin/python3 python3 /root/miniconda3/bin/python; do
  if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys, yaml; assert sys.version_info >= (3,10)' 2>/dev/null; then
    python=$candidate
    break
  fi
done
if [[ -z $python ]]; then
  echo 'Python 3.10+ with PyYAML is required. Run: sudo bash setup-linux.sh' >&2
  exit 2
fi
exec "$python" "$root/deploy_ikatago.py" "${args[@]}"
