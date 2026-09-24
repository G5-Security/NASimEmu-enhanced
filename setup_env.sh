#!/usr/bin/env bash
# Create the project venv and install all dependencies.
#
# Usage: ./setup_env.sh [python3.10-interpreter] [venv-dir]
#   e.g. ./setup_env.sh ~/.conda/envs/py310/bin/python venv
set -euo pipefail

PYTHON="${1:-python3.10}"
VENV="${2:-venv}"
cd "$(dirname "$0")"

"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10), f"need Python 3.10, got {sys.version}"'

"$PYTHON" -m venv "$VENV"
PIP="$VENV/bin/pip"

# pip >= 24.1 rejects gym 0.21.0's invalid metadata ("opencv-python>=3.").
"$PIP" install "pip<24.1"

# gym 0.21.0 only builds with old setuptools/wheel. Build it in an isolated
# env pinned to those, so the venv's own (newer) packaging is not used.
build_constraints="$(mktemp)"
trap 'rm -f "$build_constraints"' EXIT
printf 'setuptools<66\nwheel<0.40\n' > "$build_constraints"
PIP_CONSTRAINT="$build_constraints" "$PIP" install --use-pep517 gym==0.21.0

"$PIP" install -r requirements.txt
"$PIP" install -e .

"$VENV/bin/python" -c "import nasimemu, gym, torch, torch_scatter, torch_geometric; print('OK: gym', gym.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
