import os
import torch
from diffusers import StableDiffusion3Pipeline
from pathlib import Path

MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
MODEL_CACHE = Path("/workspace/models/sd35-medium")
HF_TOKEN_PATH = Path("/workspace/key.txt")


def load_hf_token() -> str:
    # Prefer RunPod secret (injected as env var) over file
    if token := os.environ.get("HF_TOKEN"):
        return token
    if HF_TOKEN_PATH.exists():
        return HF_TOKEN_PATH.read_text().strip()
    raise RuntimeError("HF token not found. Set HF_TOKEN as a RunPod secret or write it to /workspace/key.txt")


def load_pipeline():
    if MODEL_CACHE.exists():
        source = str(MODEL_CACHE)
        token = None
    else:
        source = MODEL_ID
        token = load_hf_token()

    print(f"Loading model from: {source}")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        source,
        torch_dtype=torch.float16,
        token=token,
    )
    pipe = pipe.to("cuda")
    print("Model ready.")
    return pipe
