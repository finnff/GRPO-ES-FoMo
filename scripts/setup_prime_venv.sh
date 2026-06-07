#!/usr/bin/env bash
# Build the ISOLATED hub-env venv (.venv-prime) and install environments into it.
#
#   scripts/setup_prime_venv.sh primeintellect/gsm8k [owner/env ...]
#
# Hub environments are pip wheels with their own dependency trees (a fresh
# verifiers, a newer openai). Installing one into FoMo-RL would re-resolve the
# hand-pinned trl/verifiers/datasets combination (see requirements.txt), so
# hub deps live in their own venv and the trainer talks to it through a worker
# subprocess (scripts/prime_env_worker.py). NEVER pip-install a hub env into
# the training env.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv-prime"
PYVER="${PRIME_ENV_PYVER:-3.12}"

if ! command -v uv >/dev/null; then
    echo "error: uv not found on PATH ('pip install uv' or https://docs.astral.sh/uv/)" >&2
    exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
    uv venv --python "$PYVER" "$VENV"
fi

uv pip install --python "$VENV/bin/python" --upgrade verifiers prime

for env_id in "$@"; do
    # prime shells out to uv pip itself; VIRTUAL_ENV aims it at this venv.
    VIRTUAL_ENV="$VENV" "$VENV/bin/prime" env install "$env_id"
done

# The import that breaks in the training env must work here.
"$VENV/bin/python" -c "import verifiers; from verifiers import load_environment; print('ok: .venv-prime ready (verifiers ' + verifiers.__version__ + ')')"
