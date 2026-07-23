#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime environment detection + auto-optimization — OS-agnostic, no user input.

Picks the best available device (CUDA > Apple MPS > CPU) and, on CPU, sets the torch
thread counts to the machine's core count so inference uses the hardware it's on without
the caller having to configure anything. Works on Linux / macOS / Windows.

  python -m shoshan.runtime      # prints the detected environment as JSON
"""
import os
import platform
from typing import Optional


def detect() -> dict:
    import torch
    cores = os.cpu_count() or 1
    gpu_mem = None
    if torch.cuda.is_available():
        device, accel = "cuda", torch.cuda.get_device_name(0)
        try:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory >> 30
        except Exception:
            pass
    elif getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)():
        device, accel = "mps", "Apple MPS"
    else:
        device, accel = "cpu", (platform.processor() or platform.machine() or "cpu")
    return {"os": platform.system(), "machine": platform.machine(),
            "python": platform.python_version(), "cores": cores,
            "torch": torch.__version__, "device": device,
            "accelerator": accel, "gpu_mem_gb": gpu_mem}


def configure(device: str = "auto", threads: Optional[int] = None, verbose: bool = True):
    """Resolve the device and tune threads for the detected machine. Returns (device, info)."""
    import torch
    info = detect()
    dev = info["device"] if device in (None, "auto") else device
    if dev == "cpu":
        n = threads or info["cores"]
        for setter in ("set_num_threads", "set_num_interop_threads"):
            try:
                getattr(torch, setter)(n if setter == "set_num_threads" else max(1, n // 2))
            except Exception:
                pass
        os.environ.setdefault("OMP_NUM_THREADS", str(n))
    if verbose:
        extra = f", {info['gpu_mem_gb']}GB" if info.get("gpu_mem_gb") else ""
        print(f"[shoshan] {info['os']}/{info['machine']} · {info['cores']} cores · "
              f"torch {info['torch']} · device={dev} ({info['accelerator']}{extra})")
    return dev, info


def main():
    import json
    print(json.dumps(detect(), indent=2))


if __name__ == "__main__":
    main()
