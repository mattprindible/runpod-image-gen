import torch
from diffusers import AutoPipelineForText2Image
from pathlib import Path

MODEL_ID = "stabilityai/sdxl-turbo"
# Network volume is mounted at /workspace - weights live here across pod restarts
MODEL_CACHE = Path("/workspace/models/sdxl-turbo")


def load_pipeline():
    source = str(MODEL_CACHE) if MODEL_CACHE.exists() else MODEL_ID
    print(f"Loading model from: {source}")

    pipe = AutoPipelineForText2Image.from_pretrained(
        source,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe = pipe.to("cuda")
    # Sliced attention reduces VRAM usage with minimal speed cost
    pipe.enable_attention_slicing()
    print("Model ready.")
    return pipe
