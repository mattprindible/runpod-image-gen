"""
Run this once on the pod to download model weights to the network volume.
After this, setup.sh will find them at /workspace/models/sdxl-turbo and
skip the download on future pod restarts.
"""
import torch
from diffusers import AutoPipelineForText2Image
from pathlib import Path

MODEL_ID = "stabilityai/sdxl-turbo"
CACHE = Path("/workspace/models/sdxl-turbo")

if CACHE.exists():
    print(f"Already cached at {CACHE} — nothing to do.")
else:
    print(f"Downloading {MODEL_ID} to {CACHE} ...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.save_pretrained(CACHE)
    print("Done.")
