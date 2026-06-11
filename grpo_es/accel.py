"""Accelerator selection, in one place so the cuda/rocm/cpu choice can't drift.

PyTorch-ROCm exposes AMD GPUs through the *same* ``torch.cuda.*`` API as
NVIDIA — ``torch.cuda.is_available()`` is True on a working ROCm box and
``device_map="cuda"`` lands on the Radeon. So the GPU path here covers both
vendors; the only thing that differs is what we print, which matters when you
are bringing up an unsupported iGPU and need to see *which* backend bound.

``torch.version.hip`` is set (and ``torch.version.cuda`` is None) on a ROCm
build — that's how we tell AMD from NVIDIA without importing anything vendor
specific.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def gpu_available() -> bool:
    """True when a usable GPU is present — NVIDIA CUDA *or* AMD ROCm.

    Named separately from ``torch.cuda.is_available()`` only to document that
    the ROCm case rides the same call; the behaviour is identical."""
    return torch.cuda.is_available()


def is_rocm() -> bool:
    """True on a ROCm/HIP build of PyTorch (AMD), False on a CUDA build."""
    return torch.version.hip is not None


def describe() -> str:
    """One-line human label for the active accelerator, e.g.
    ``AMD ROCm 'AMD Radeon Graphics' via HIP 7.1 (bf16)`` — printed at startup
    so an AMD box doesn't silently report itself as plain 'cuda'."""
    if not gpu_available():
        return "CPU (fp32, no GPU detected)"
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover - driver/runtime hiccup
        name = "unknown GPU"
    if is_rocm():
        hip = (torch.version.hip or "").split("-")[0]
        return f"AMD ROCm '{name}' via HIP {hip} (bf16)"
    return f"NVIDIA CUDA '{name}' (bf16)"


def device_and_dtype() -> tuple[str, torch.dtype]:
    """``(device, dtype)`` for loading a model: bf16 on any GPU (CUDA/ROCm),
    fp32 on the CPU fallback. CPU bf16 generation is poorly supported, so the
    fallback drops to fp32 — a correctness path, not a fast one."""
    if gpu_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def log_active(context: str) -> bool:
    """Log the active accelerator for ``context`` and return whether a GPU was
    found (so callers can branch on bf16/use_cpu). Warns on the CPU fallback."""
    if gpu_available():
        logger.info("%s: %s", context, describe())
        return True
    logger.warning("%s: %s — slow, correctness fallback only.", context, describe())
    return False
