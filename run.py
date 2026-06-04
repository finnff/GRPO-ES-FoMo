"""Entrypoint: ``python run.py --method grpo --task toy --model ...``."""

from __future__ import annotations

from grpo_es.config.logging_setup import configure_logging
from grpo_es.config.run_config import parse_args


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    configure_logging(verbose=cfg.verbose)
    # Import after logging is configured (the method modules pull in the
    # heavyweight HF stack), and only the leg being run.
    if cfg.method == "grpo":
        from grpo_es.methods.grpo import run_grpo

        run_grpo(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
