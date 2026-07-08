#!/usr/bin/env bash
# Local dev runner: mimics the container contract without Docker.
set -euo pipefail
cd "$(dirname "$0")"
set -a; source .env; set +a
export INPUT_PATH="${INPUT_PATH:-./local_io/tasks.json}"
export OUTPUT_PATH="${OUTPUT_PATH:-./local_io/results.json}"
export WORK_DIR="${WORK_DIR:-./local_io/work}"
mkdir -p local_io
exec python3 main.py
