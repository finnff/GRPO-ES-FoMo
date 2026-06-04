#!/usr/bin/env bash
# Install the training stack into the CURRENT python env (conda or venv).
# Two pip invocations on purpose — see the note in requirements.txt.
#
#   conda create -n FoMo-RL python=3.11 && conda activate FoMo-RL
#   ./scripts/setup_env.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

pip install -r "$ROOT/requirements.txt"
pip install -r "$ROOT/requirements-verifiers.txt"
pip install -r "$ROOT/requirements-dev.txt"

# No editable install: grpo_es is a namespace package, imported by running
# run.py / pytest from the repo root (tests/conftest.py puts root on sys.path).

python -c "import trl, verifiers, datasets; print('ok:', trl.__version__, verifiers.__version__, datasets.__version__)"
