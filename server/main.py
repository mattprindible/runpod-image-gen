import io
import json
import math
import uuid
import queue
import zipfile
import asyncio
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from server.model import load_pipeline

executor = ThreadPoolExecutor(max_workers=1)
OUTPUT_DIR = Path("/workspace/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CFG_VALUES = [1.0, 3.0, 5.0, 7.0, 10.0, 15.0]

app = FastAPI(title="SD 3.5 Explorer")
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


@app.get("/")
async def root():
    return FileResponse("ui/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": pipe is not None}


# === Math utilities ===

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return (torch.dot(a, b) / (torch.norm(a) * torch.norm(b))).item()


def angle_degrees(a: torch.Tensor, b: torch.Tensor) -> float:
    sim = max(-1.0, min(1.0, cosine_sim(a, b)))
    return math.degrees(math.acos(sim))


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    v0_flat = v0.flatten()
    v1_flat = v1.flatten()
    v0_norm = v0_flat / torch.norm(v0_flat)
    v1_norm = v1_flat / torch.norm(v1_flat)
    dot = torch.clamp(torch.dot(v0_norm, v1_norm), -1.0, 1.0)
    theta = torch.acos(dot)
    if theta.abs() < 1e-6:
        result = (1 - t) * v0_flat + t * v1_flat
    else:
        sin_theta = torch.sin(theta)
        result = (
            torch.sin((1 - t) * theta) / sin_theta * v0_flat
            + torch.sin(t * theta) / sin_theta * v1_flat
        )
    magnitude = (torch.norm(v0_flat) + torch.norm(v1_flat)) / 2
    return (result / torch.norm(result) * magnitude).reshape(v0.shape)


# === Pipeline utilities ===

def decode_latents(latents: torch.Tensor):
    """Decode SD3 latents to PIL image (accounts for SD3's shift_factor)."""
    with torch.no_grad():
        scaled = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        decoded = pipe.vae.decode(scaled, return_dict=False)[0]
        return pipe.image_processor.postprocess(decoded, output_type="pil")[0]


def make_latents(seed: int, height: int, width: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    channels = pipe.transformer.config.in_channels
    return torch.randn(
        (1, channels, height // pipe.vae_scale_factor, width // pipe.vae_scale_factor),
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )


# === Sync generation functions (run in executor) ===

def run_generate(prompt, latents, steps, cfg, width, height):
    with torch.inference_mode():
        return pipe(
            prompt=prompt,
            latents=latents,
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=width,
            height=height,
        ).images[0]


def run_generate_from_embeds(embeds, neg_embeds, pooled, neg_pooled, latents, steps, cfg, width, height):
    with torch.inference_mode():
        return pipe(
            prompt_embeds=embeds,
            negative_prompt_embeds=neg_embeds,
            pooled_prompt_embeds=pooled,
            negative_pooled_prompt_embeds=neg_pooled,
            latents=latents,
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=width,
            height=height,
        ).images[0]


def run_denoising_steps(result_queue, prompt, latents, steps, cfg, width, height):
    prev_latents = [None]

    def callback(pipeline, step_index, timestep, callback_kwargs):
        current = callback_kwargs["latents"]
        latent_norm = round(current.norm().item(), 3)
        delta_norm = None
        if prev_latents[0] is not None:
            delta_norm = round((current - prev_latents[0]).norm().item(), 3)
        prev_latents[0] = current.clone()
        image = decode_latents(current)
        result_queue.put(("step", step_index, int(timestep.item()), image, latent_norm, delta_norm))
        return callback_kwargs

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            latents=latents,
            num_inference_steps=steps,
            guidance_scale=cfg,
            width=width,
            height=height,
            callback_on_step_end=callback,
        )
    result_queue.put(("done", steps, 0, result.images[0], None, None))


# === Main streaming endpoint ===

@app.get("/generate-stream")
async def generate_stream(
    prompt: str,
    mode: str = "denoising_steps",
    prompt_b: Optional[str] = None,
    num_frames: int = 6,
    walk_strength: float = 0.3,
    base_seed: Optional[int] = None,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.0,
    resolution: int = 512,
    slerp_embeds: bool = False,
    t_min: float = 0.0,
    t_max: float = 1.0,
):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if base_seed is None:
        base_seed = int(torch.randint(0, 2**32 - 1, (1,)).item())

    width = height = resolution
    batch_id = str(uuid.uuid4())[:8]
    batch_dir = OUTPUT_DIR / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "prompt": prompt,
        "prompt_b": prompt_b,
        "mode": mode,
        "num_frames": num_frames,
        "walk_strength": walk_strength,
        "base_seed": base_seed,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "resolution": resolution,
        "slerp_embeds": slerp_embeds,
        "t_min": t_min,
        "t_max": t_max,
    }
    (batch_dir / "params.json").write_text(json.dumps(params, indent=2))

    total = (
        num_inference_steps if mode == "denoising_steps"
        else len(DEFAULT_CFG_VALUES) if mode == "cfg_sweep"
        else num_frames
    )

    async def event_generator():
        yield f"event: start\ndata: {json.dumps({'batch_id': batch_id, 'total': total, 'params': params})}\n\n"

        loop = asyncio.get_event_loop()
        analysis = {"mode": mode, "batch_id": batch_id}

        if mode == "denoising_steps":
            latents = make_latents(base_seed, height, width)
            q = queue.Queue()
            future = loop.run_in_executor(
                executor, run_denoising_steps,
                q, prompt, latents, num_inference_steps, guidance_scale, width, height,
            )
            steps_data = []
            frame_idx = 0
            while True:
                try:
                    status, step, timestep, image, latent_norm, delta_norm = await loop.run_in_executor(None, q.get)
                    filename = f"step_{frame_idx:03d}.png"
                    image.save(batch_dir / filename)
                    step_entry = {"step": step, "timestep": timestep, "latent_norm": latent_norm, "delta_norm": delta_norm}
                    steps_data.append(step_entry)
                    frame_data = {
                        "frame": frame_idx, "total": num_inference_steps,
                        "step": step, "timestep": timestep,
                        "latent_norm": latent_norm, "delta_norm": delta_norm,
                        "url": f"/outputs/{batch_id}/{filename}",
                    }
                    yield f"event: frame\ndata: {json.dumps(frame_data)}\n\n"
                    frame_idx += 1
                    if status == "done":
                        break
                except Exception as e:
                    yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                    break
            await future
            analysis["steps"] = steps_data
            if steps_data:
                norms = [s["latent_norm"] for s in steps_data if s["latent_norm"]]
                deltas = [s["delta_norm"] for s in steps_data if s["delta_norm"]]
                analysis["summary"] = {
                    "peak_delta_norm": round(max(deltas), 3) if deltas else None,
                    "peak_delta_step": deltas.index(max(deltas)) + 1 if deltas else None,
                    "final_latent_norm": norms[-1] if norms else None,
                }

        elif mode == "cfg_sweep":
            latents = make_latents(base_seed, height, width)
            analysis["cfg_values"] = DEFAULT_CFG_VALUES
            for i, cfg in enumerate(DEFAULT_CFG_VALUES):
                try:
                    image = await loop.run_in_executor(
                        executor, run_generate,
                        prompt, latents.clone(), num_inference_steps, cfg, width, height,
                    )
                    filename = f"cfg_{cfg:.1f}.png"
                    image.save(batch_dir / filename)
                    yield f"event: frame\ndata: {json.dumps({'frame': i, 'total': len(DEFAULT_CFG_VALUES), 'cfg': cfg, 'url': f'/outputs/{batch_id}/{filename}'})}\n\n"
                except Exception as e:
                    yield f"event: error\ndata: {json.dumps({'error': str(e), 'frame': i})}\n\n"
                    break

        elif mode == "seed_walk":
            base_latents = make_latents(base_seed, height, width)
            frames_data = []
            for i in range(num_frames):
                try:
                    if i == 0:
                        latents = base_latents.clone()
                        sim = 1.0
                        ang = 0.0
                    else:
                        rand_latents = make_latents(base_seed + i * 1000, height, width)
                        latents = slerp(base_latents, rand_latents, walk_strength)
                        sim = round(cosine_sim(latents, base_latents), 4)
                        ang = round(angle_degrees(latents, base_latents), 2)
                    frames_data.append({"frame": i, "cosine_to_base": sim, "angle_degrees": ang})
                    image = await loop.run_in_executor(
                        executor, run_generate,
                        prompt, latents, num_inference_steps, guidance_scale, width, height,
                    )
                    filename = f"frame_{i:03d}.png"
                    image.save(batch_dir / filename)
                    yield f"event: frame\ndata: {json.dumps({'frame': i, 'total': num_frames, 'cosine_to_base': sim, 'angle_degrees': ang, 'url': f'/outputs/{batch_id}/{filename}'})}\n\n"
                except Exception as e:
                    yield f"event: error\ndata: {json.dumps({'error': str(e), 'frame': i})}\n\n"
                    break
            analysis["frames"] = frames_data
            if frames_data:
                max_angle = max(f["angle_degrees"] for f in frames_data)
                analysis["summary"] = {"max_angle_degrees": max_angle, "walk_strength": walk_strength}

        elif mode == "prompt_interpolation":
            if not prompt_b:
                yield f"event: error\ndata: {json.dumps({'error': 'prompt_b required for interpolation mode'})}\n\n"
                return

            with torch.no_grad():
                embeds_a, neg_a, pooled_a, neg_pooled_a = pipe.encode_prompt(
                    prompt=prompt, prompt_2=None, prompt_3=None,
                    device="cuda", num_images_per_prompt=1,
                    do_classifier_free_guidance=True, negative_prompt="",
                )
                embeds_b, neg_b, pooled_b, neg_pooled_b = pipe.encode_prompt(
                    prompt=prompt_b, prompt_2=None, prompt_3=None,
                    device="cuda", num_images_per_prompt=1,
                    do_classifier_free_guidance=True, negative_prompt="",
                )

            # Compute prompt geometry before generation starts
            pooled_sim = round(cosine_sim(pooled_a, pooled_b), 4)
            full_sim = round(cosine_sim(embeds_a.mean(dim=1), embeds_b.mean(dim=1)), 4)
            pooled_angle = round(angle_degrees(pooled_a, pooled_b), 2)
            analysis.update({
                "prompt_a": prompt,
                "prompt_b": prompt_b,
                "pooled_cosine_similarity": pooled_sim,
                "pooled_distance": round(1 - pooled_sim, 4),
                "pooled_angle_degrees": pooled_angle,
                "full_embedding_cosine_similarity": full_sim,
            })
            # Send prompt geometry immediately so it appears in UI before generation
            yield f"event: prompt_geometry\ndata: {json.dumps(analysis)}\n\n"

            fixed_latents = make_latents(base_seed, height, width)
            mix = slerp if slerp_embeds else lambda a, b, t: (1 - t) * a + t * b
            frames_data = []
            for i in range(num_frames):
                try:
                    t = t_min + (t_max - t_min) * i / max(num_frames - 1, 1)
                    dist_from_a = round(t * (1 - pooled_sim), 4)
                    dist_from_b = round((1 - t) * (1 - pooled_sim), 4)
                    frames_data.append({"frame": i, "t": round(t, 3), "dist_from_a": dist_from_a, "dist_from_b": dist_from_b})
                    image = await loop.run_in_executor(
                        executor, run_generate_from_embeds,
                        mix(embeds_a, embeds_b, t),
                        mix(neg_a, neg_b, t),
                        mix(pooled_a, pooled_b, t),
                        mix(neg_pooled_a, neg_pooled_b, t),
                        fixed_latents.clone(),
                        num_inference_steps, guidance_scale, width, height,
                    )
                    filename = f"interp_{i:03d}.png"
                    image.save(batch_dir / filename)
                    yield f"event: frame\ndata: {json.dumps({'frame': i, 'total': num_frames, 't': round(t, 3), 'dist_from_a': dist_from_a, 'dist_from_b': dist_from_b, 'url': f'/outputs/{batch_id}/{filename}'})}\n\n"
                except Exception as e:
                    yield f"event: error\ndata: {json.dumps({'error': str(e), 'frame': i})}\n\n"
                    break
            analysis["frames"] = frames_data

        # Write analysis and send final events
        (batch_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
        yield f"event: analysis\ndata: {json.dumps(analysis)}\n\n"
        yield f"event: done\ndata: {json.dumps({'batch_id': batch_id, 'total': total})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# === Analysis, image serving & download ===

@app.get("/batch/{batch_id}/analysis")
async def get_analysis(batch_id: str):
    path = OUTPUT_DIR / batch_id / "analysis.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Analysis not available yet")
    return json.loads(path.read_text())


@app.get("/outputs/{batch_id}/{filename}")
async def get_image(batch_id: str, filename: str):
    path = OUTPUT_DIR / batch_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type="image/png")


@app.get("/download/{batch_id}")
async def download_batch(batch_id: str):
    batch_dir = OUTPUT_DIR / batch_id
    if not batch_dir.exists():
        raise HTTPException(status_code=404, detail="Not found")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(batch_dir.glob("*.png")):
            zf.write(f, f.name)
        for name in ("params.json", "analysis.json"):
            if (batch_dir / name).exists():
                zf.write(batch_dir / name, name)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={batch_id}.zip"},
    )
