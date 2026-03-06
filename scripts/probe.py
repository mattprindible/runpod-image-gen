#!/usr/bin/env python3
"""
Probe script for SD 3.5 Explorer.

Runs a generation against the pod server, streams progress, downloads all
images and analysis data locally so Claude can read and evaluate the outputs.

Usage:
  python scripts/probe.py <SERVER_URL> --prompt "..." --mode denoising_steps
  python scripts/probe.py <SERVER_URL> --prompt "..." --prompt-b "..." --mode prompt_interpolation

Requires: pip install requests
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Missing dependency: pip install requests")
    sys.exit(1)


def run_probe(
    server_url, prompt, mode,
    prompt_b=None, num_frames=6, steps=20, cfg=7.0,
    resolution=512, walk_strength=0.3, seed=None,
    slerp_embeds=False, t_min=0.0, t_max=1.0,
    output_dir="/tmp/sd-probe",
):
    params = {
        "prompt": prompt,
        "mode": mode,
        "num_frames": num_frames,
        "num_inference_steps": steps,
        "guidance_scale": cfg,
        "resolution": resolution,
        "walk_strength": walk_strength,
    }
    if prompt_b:
        params["prompt_b"] = prompt_b
    if seed is not None:
        params["base_seed"] = seed
    if mode == "prompt_interpolation":
        params["t_min"] = t_min
        params["t_max"] = t_max
        if slerp_embeds:
            params["slerp_embeds"] = "true"

    print(f"Mode:    {mode}")
    print(f"Prompt:  {prompt}")
    if prompt_b:
        print(f"Prompt B: {prompt_b}")
    print(f"Server:  {server_url}")
    print()

    batch_id = None
    out = None
    image_urls = []
    frame_meta = []
    analysis = None
    start = time.time()

    try:
        with requests.get(
            f"{server_url}/generate-stream",
            params=params,
            stream=True,
            timeout=600,
        ) as resp:
            resp.raise_for_status()

            event_type = None
            for raw_line in resp.iter_lines(decode_unicode=True):
                line = raw_line.strip()
                if not line:
                    continue

                if line.startswith("event:"):
                    event_type = line[6:].strip()
                    continue

                if not line.startswith("data:"):
                    continue

                data = json.loads(line[5:].strip())

                if event_type == "start":
                    batch_id = data["batch_id"]
                    out = Path(output_dir) / batch_id
                    out.mkdir(parents=True, exist_ok=True)
                    print(f"Batch {batch_id} — {data['total']} frames")
                    print(f"Output: {out}/")
                    print()

                elif event_type == "prompt_geometry":
                    print("Prompt geometry (before generation):")
                    print(f"  pooled cosine similarity : {data.get('pooled_cosine_similarity')}")
                    print(f"  pooled distance          : {data.get('pooled_distance')}")
                    print(f"  angle (degrees)          : {data.get('pooled_angle_degrees')}°")
                    print(f"  full embedding similarity: {data.get('full_embedding_cosine_similarity')}")
                    print()

                elif event_type == "frame":
                    frame = data["frame"]
                    total = data["total"]
                    image_urls.append(data["url"])
                    frame_meta.append(data)

                    if mode == "denoising_steps":
                        delta = data.get("delta_norm", "—")
                        norm = data.get("latent_norm", "—")
                        print(f"  step {frame+1:>2}/{total}  t={data.get('timestep'):>4}  norm={norm}  Δ={delta}")
                    elif mode == "cfg_sweep":
                        print(f"  cfg {data['cfg']:>5}  ({frame+1}/{total})")
                    elif mode == "seed_walk":
                        print(f"  frame {frame+1}/{total}  angle={data.get('angle_degrees')}°  cosine={data.get('cosine_to_base')}")
                    elif mode == "prompt_interpolation":
                        print(f"  t={data['t']:.3f}  ({frame+1}/{total})  dist_from_a={data.get('dist_from_a')}  dist_from_b={data.get('dist_from_b')}")

                elif event_type == "analysis":
                    analysis = data
                    (out / "analysis.json").write_text(json.dumps(data, indent=2))

                elif event_type == "error":
                    print(f"\nServer error: {data.get('error')}", file=sys.stderr)

                elif event_type == "done":
                    elapsed = time.time() - start
                    print(f"\nGeneration complete ({elapsed:.1f}s)")

    except requests.RequestException as e:
        print(f"\nConnection error: {e}", file=sys.stderr)
        sys.exit(1)

    if not image_urls or out is None:
        print("No images received.", file=sys.stderr)
        sys.exit(1)

    # Download images
    print(f"\nDownloading {len(image_urls)} images...")
    for url in image_urls:
        filename = url.split("/")[-1]
        img_resp = requests.get(f"{server_url}{url}", timeout=60)
        img_resp.raise_for_status()
        (out / filename).write_bytes(img_resp.content)
        print(f"  {filename}")

    # Save frame metadata
    (out / "frames.json").write_text(json.dumps(frame_meta, indent=2))

    # Print analysis summary
    if analysis:
        print(f"\nAnalysis summary:")
        if mode == "denoising_steps" and "summary" in analysis:
            s = analysis["summary"]
            print(f"  peak Δ norm : {s.get('peak_delta_norm')} (step {s.get('peak_delta_step')})")
            print(f"  final norm  : {s.get('final_latent_norm')}")
        elif mode == "seed_walk" and "summary" in analysis:
            s = analysis["summary"]
            print(f"  max angle from base : {s.get('max_angle_degrees')}°")
        elif mode == "prompt_interpolation":
            print(f"  pooled distance : {analysis.get('pooled_distance')}")
            print(f"  angle           : {analysis.get('pooled_angle_degrees')}°")

    print(f"\nFiles saved to: {out}/")
    print(f"  analysis.json — batch metrics")
    print(f"  frames.json   — per-frame metadata")
    for url in image_urls:
        print(f"  {url.split('/')[-1]}")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe the SD 3.5 Explorer server")
    parser.add_argument("server", help="Pod URL, e.g. https://abc123-8000.proxy.runpod.net")
    parser.add_argument("--prompt", required=True, help="Primary prompt")
    parser.add_argument("--prompt-b", help="Second prompt (interpolation mode only)")
    parser.add_argument("--mode", default="denoising_steps",
                        choices=["denoising_steps", "cfg_sweep", "seed_walk", "prompt_interpolation"])
    parser.add_argument("--frames", type=int, default=6, help="Number of frames (walk/interpolation)")
    parser.add_argument("--steps", type=int, default=20, help="Inference steps per image")
    parser.add_argument("--cfg", type=float, default=7.0, help="Guidance scale")
    parser.add_argument("--resolution", type=int, default=512, choices=[512, 1024])
    parser.add_argument("--walk-strength", type=float, default=0.3)
    parser.add_argument("--seed", type=int, help="Base seed (omit for random)")
    parser.add_argument("--slerp-embeds", action="store_true", help="SLERP embeddings instead of LERP (interpolation mode)")
    parser.add_argument("--t-min", type=float, default=0.0, help="Start of t range (interpolation mode)")
    parser.add_argument("--t-max", type=float, default=1.0, help="End of t range (interpolation mode)")
    parser.add_argument("--output-dir", default="/tmp/sd-probe", help="Local directory for downloads")
    args = parser.parse_args()

    run_probe(
        server_url=args.server.rstrip("/"),
        prompt=args.prompt,
        mode=args.mode,
        prompt_b=args.prompt_b,
        num_frames=args.frames,
        steps=args.steps,
        cfg=args.cfg,
        resolution=args.resolution,
        walk_strength=args.walk_strength,
        seed=args.seed,
        slerp_embeds=args.slerp_embeds,
        t_min=args.t_min,
        t_max=args.t_max,
        output_dir=args.output_dir,
    )
