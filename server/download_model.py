"""
Run once on a fresh pod to download SD 3.5 Medium weights to /workspace.
setup.sh calls this automatically — it's a no-op if already cached.
Requires HF token at /workspace/key.txt (model is gated).
"""
import os
import torch
from diffusers import StableDiffusion3Pipeline
from pathlib import Path

MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
CACHE = Path("/workspace/models/sd35-medium")
HF_TOKEN_PATH = Path("/workspace/key.txt")

if CACHE.exists():
    print(f"Already cached at {CACHE} — nothing to do.")
else:
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            with open("/proc/1/environ", "rb") as f:
                for item in f.read().split(b"\0"):
                    if item.startswith(b"HF_TOKEN="):
                        token = item[9:].decode()
                        break
        except OSError:
            pass
    if not token and HF_TOKEN_PATH.exists():
        token = HF_TOKEN_PATH.read_text().strip()
    if not token:
        raise RuntimeError("HF token not found. Set HF_TOKEN as a RunPod secret or write it to /workspace/key.txt")
    print(f"Downloading {MODEL_ID} to {CACHE} (~10GB, this takes a few minutes)...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        token=token,
    )
    pipe.save_pretrained(CACHE)
    print("Done.")
