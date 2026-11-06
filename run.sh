#!/usr/bin/env sh
# Set up (first run only) and start Text LLM Studio. Extra options go to run.py, e.g. ./run.sh --port 8000
cd "$(dirname "$0")" || exit 1
for py in python3 python; do
  if command -v "$py" >/dev/null 2>&1; then exec "$py" run.py "$@"; fi
done
echo "Python 3 was not found. Install it from https://www.python.org/downloads/ and try again." >&2
exit 1
