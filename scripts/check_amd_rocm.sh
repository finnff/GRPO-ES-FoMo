#!/usr/bin/env bash
# Bring-up doctor for the AMD/ROCm GPU path (e.g. Ryzen AI 350 -> Radeon iGPU).
#
# This project's GPU code path is vendor-agnostic: PyTorch-ROCm exposes AMD
# GPUs through the SAME torch.cuda API as NVIDIA, so a working ROCm install
# needs NO code change here -- torch.cuda.is_available() simply returns True.
# This script checks that the install actually binds the GPU and runs a real
# kernel on it; it does NOT install or modify torch.
#
# RDNA 3.5 iGPUs (Radeon 860M/890M on Ryzen AI 300/AI 350, gfx115x) are not on
# ROCm's official allowlist yet, so HSA wants a nudge to treat them as the
# nearest supported ISA. Export the override BEFORE running python/this script:
#
#     export HSA_OVERRIDE_GFX_VERSION=11.0.0   # treat gfx115x as gfx1100 (RDNA3)
#     ./scripts/check_amd_rocm.sh
#
# If the iGPU shares system RAM, raise the GTT/UMA limit in the kernel cmdline
# (amdgpu.gttsize=... / a larger UMA buffer in BIOS) so models fit.
set -euo pipefail

echo "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-<unset>}"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"
echo

command -v rocminfo >/dev/null 2>&1 && rocminfo 2>/dev/null | grep -E "Name:|gfx" | head || \
  echo "rocminfo not found (ROCm userspace not installed system-wide; the pip wheel may still work)"
echo

python - <<'PY'
import torch

hip = torch.version.hip
print(f"torch {torch.__version__}  | HIP build: {hip or '<none — this is a CUDA/CPU wheel, not ROCm>'}")
if not torch.version.hip:
    raise SystemExit("Not a ROCm build of torch. Install one: see scripts/check_amd_rocm.sh header.")

if not torch.cuda.is_available():
    raise SystemExit(
        "ROCm torch is installed but no GPU bound. Most common cause on an "
        "RDNA 3.5 iGPU: HSA_OVERRIDE_GFX_VERSION is unset/wrong. Try "
        "HSA_OVERRIDE_GFX_VERSION=11.0.0 and re-run."
    )

name = torch.cuda.get_device_name(0)
print(f"GPU bound: {name}")
# Prove a kernel actually runs on the device (init can lie; a matmul can't).
x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
y = (x @ x).float().sum().item()
print(f"bf16 matmul on device OK (checksum {y:.1f})")
print("\nGPU path is live. `python run.py --config configs/smoke_grpo.toml` will use it.")
PY
