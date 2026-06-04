"""One verbosity switch for the whole run.

Called once from the entrypoint, before anything heavyweight loads, so the HF
environment toggles land in time. Everything goes through ``logging`` — our
package stays at INFO, third-party chatter is gated behind ``--verbose``.
"""

from __future__ import annotations

import logging
import os
import warnings

# setdefault, not assignment: an explicit env override from the caller wins.
_HF_ENV_DEFAULTS = {
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

_NOISY_LOGGERS = (
    "accelerate",
    "datasets",
    "filelock",
    "httpcore",
    "httpx",
    "huggingface_hub",
    "peft",
    "transformers",
    "trl",
    "urllib3",
)

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(*, verbose: bool = False) -> None:
    for key, value in _HF_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    if not verbose:
        # Set on the environment (not just logger levels) so any spawned
        # worker process inherits the hush.
        os.environ.setdefault("TQDM_DISABLE", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    root = logging.getLogger()
    root.setLevel(logging.INFO if verbose else logging.WARNING)
    formatter = logging.Formatter(_FORMAT)
    if root.handlers:
        for handler in root.handlers:
            handler.setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root.addHandler(handler)

    # Our own package stays informative regardless of --verbose.
    logging.getLogger("grpo_es").setLevel(logging.INFO)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.DEBUG if verbose else logging.ERROR)

    if verbose:
        return

    from transformers.utils import logging as hf_logging

    hf_logging.disable_default_handler()
    hf_logging.enable_propagation()
    hf_logging.set_verbosity_error()

    import datasets

    datasets.logging.set_verbosity_error()

    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()

    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
