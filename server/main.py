import base64
import io
import time

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import load_pipeline

app = FastAPI()

# Allow the local UI (file://) to call this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipe = None


@app.on_event("startup")
async def startup():
    global pipe
    pipe = load_pipeline()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipe is not None}


class GenerateRequest(BaseModel):
    prompt: str
    steps: int = 1       # SDXL-Turbo is designed for 1-4 steps
    seed: int | None = None


@app.post("/generate")
def generate(req: GenerateRequest):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    generator = None
    if req.seed is not None:
        generator = torch.Generator("cuda").manual_seed(req.seed)

    start = time.time()
    result = pipe(
        prompt=req.prompt,
        num_inference_steps=req.steps,
        guidance_scale=0.0,   # SDXL-Turbo doesn't use classifier-free guidance
        generator=generator,
    )
    elapsed = time.time() - start

    image = result.images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "image": img_b64,
        "elapsed_seconds": round(elapsed, 3),
        "prompt": req.prompt,
        "steps": req.steps,
    }
