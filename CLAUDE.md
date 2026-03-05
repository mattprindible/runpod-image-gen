# CLAUDE.md — SD 3.5 Explorer

## What this project is

An empirical exploration harness for Stable Diffusion 3.5 Medium. The goal is not to build
a production image generator — it's to use generation as a tool for understanding the model's
internal geometry: how latent space is structured, how prompts relate to each other, how the
denoising process evolves over time.

The four exploration modes reflect this:
- **denoising_steps** — stream every denoising step as a frame; track latent norm and Δ norm
  to observe how structure crystallizes from noise
- **cfg_sweep** — fixed noise, 6 CFG values; observe how guidance strength shapes the output
- **seed_walk** — SLERP between noise tensors; explore the local geometry of latent space
  around a fixed prompt
- **prompt_interpolation** — encode two prompts, linearly interpolate embeddings frame by frame
  with fixed latents; measure cosine similarity and angle in embedding space before generation

The analysis layer (cosine similarity, latent norms, angles in degrees) is first-class — these
numbers are not decoration, they're the point. Claude should read and interpret them.

## Infrastructure philosophy

**GitHub is the single source of truth. The pod is a runner, not a workspace.**

- All code changes happen locally and are pushed to GitHub
- The pod only pulls and runs — never edit files directly on the pod
- If something is broken on the pod, fix it in the repo and `git pull`
- Pod disk holds only: model weights and generated outputs

This means the pod is fully disposable. Any pod of the right type can be bootstrapped from
scratch in under 5 minutes.

## Pod management

**Always use `runpodctl pod create`** — the GraphQL API returns SUPPLY_CONSTRAINT even when
GPUs are available. runpodctl is more reliable for pod creation.

**Always use SECURE cloud** — community cloud pods have unreliable SSH and don't properly
expose env vars. Not worth the cost savings.

Standard pod creation command:
```bash
runpodctl pod create \
  --template-id runpod-torch-v240 \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --cloud-type SECURE \
  --container-disk-in-gb 60 \
  --ports "22/tcp,8000/http" \
  --env '{"HF_TOKEN":"{{ RUNPOD_SECRET_HF_TOKEN }}"}' \
  --name sd35-explorer
```

Template `runpod-torch-v240` = `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
(Ubuntu 22.04, Python 3.11, PyTorch 2.4.0, CUDA 12.4.1 — pre-cached on RunPod hardware).

To find available GPUs: query the GraphQL API:
```bash
curl -s -X POST "https://api.runpod.io/graphql?api_key=$(grep apikey ~/.runpod/config.toml | sed 's/.*= *'"'"'//;s/'"'"'.*//')" \
  -H "Content-Type: application/json" \
  -d '{"query":"{ gpuTypes { id displayName memoryInGb secureCloud lowestPrice(input:{gpuCount:1}){uninterruptablePrice} } }"}' \
  | python3 -c "import json,sys; [print(f\"\${g['lowestPrice']['uninterruptablePrice']}/hr {g['memoryInGb']}GB {g['displayName']}\") for g in sorted(json.load(sys.stdin)['data']['gpuTypes'], key=lambda x: x.get('lowestPrice',{}).get('uninterruptablePrice') or 999) if g['secureCloud'] and g.get('lowestPrice',{}).get('uninterruptablePrice')]"
```

Pod lifecycle: **stop, don't terminate** to preserve the model weights on disk between sessions.
Restart a stopped pod:
```bash
curl -s -X POST "https://api.runpod.io/graphql?api_key=..." \
  -d '{"query":"mutation { podResume(input:{podId:\"<id>\", gpuCount:1}){id desiredStatus}}"}'
```

## SSH access

Secure cloud pods expose TCP port 22 with a public IP. Get the current IP:port after any start/restart:
```bash
runpodctl ssh info <pod_id>
```

Connect:
```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go -o PubkeyAcceptedAlgorithms=+ssh-rsa -p PORT root@IP
```

The port changes on every pod restart. Always run `runpodctl ssh info` first.

## Bootstrap (fresh pod)

```bash
GIT_TERMINAL_PROMPT=0 git clone https://github.com/mattprindible/runpod-image-gen.git /workspace/app
cd /workspace/app
pip install -r server/requirements.txt --quiet
nohup python3 server/download_model.py > /workspace/download.log 2>&1 &
# tail -f /workspace/download.log  (completes in ~30s, model is already on HuggingFace CDN)
nohup uvicorn server.main:app --host 0.0.0.0 --port 8000 > /workspace/server.log 2>&1 &
```

Health check: `curl https://<pod_id>-8000.proxy.runpod.net/health`

## Dev loop (iterating on code)

```bash
# Local: make changes, then:
git add -p && git commit -m "..." && git push

# On pod:
ssh ... "cd /workspace/app && git pull && pkill uvicorn; nohup uvicorn server.main:app --host 0.0.0.0 --port 8000 > /workspace/server.log 2>&1 &"
```

## Closed-loop testing

The probe script runs a generation, streams progress, and downloads all images and analysis
data locally so Claude can read and evaluate them directly:

```bash
python3 scripts/probe.py https://<pod_id>-8000.proxy.runpod.net \
  --prompt "..." \
  --mode denoising_steps \
  --steps 20 \
  --output-dir /tmp/sd-probe
```

After the probe, read the images with the Read tool and inspect analysis.json. The numbers
mean something — interpret the latent geometry, not just whether the image looks good.

## Key technical details

**HF Token / env vars**: RunPod injects Docker env vars into PID 1's environment, but SSH
sessions spawn a fresh shell that doesn't inherit them. `echo $HF_TOKEN` will be empty in
SSH even when the secret is correctly configured. The code reads `/proc/1/environ` as a
fallback — this is already handled in `server/model.py` and `server/download_model.py`.
To verify: `cat /proc/1/environ | tr "\0" "\n" | grep HF_TOKEN`

**Dependency pins** (see server/requirements.txt):
- `diffusers>=0.31.0,<0.32.0` — diffusers 0.32+ added Flash Attention 3 with @_custom_op
  annotations incompatible with PyTorch 2.4. 0.31 is the SD 3.5 release.
- `transformers<5.0.0` — 5.x removed MT5Tokenizer, breaking SD3's T5 encoder

**VAE decode for SD3** requires shift_factor:
```python
scaled = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
```

**Latent generation** uses `pipe.transformer.config.in_channels` (not a fixed number).

**Uvicorn** must be run as `uvicorn server.main:app` (package mode), not `server/main.py`.
This requires `server/__init__.py` to exist.

## File structure

```
server/
  __init__.py          # makes server/ a package for uvicorn
  main.py              # FastAPI app, all 4 exploration modes, SSE streaming
  model.py             # load_pipeline(), load_hf_token() with /proc/1/environ fallback
  download_model.py    # one-shot model download to /workspace/models/sd35-medium
  requirements.txt     # pinned deps, torch excluded (pre-installed in template)
ui/
  index.html           # served by FastAPI at /, all JS inline
scripts/
  probe.py             # CLI closed-loop tester; downloads images for Claude to read
```

## What NOT to do

- Don't edit files on the pod — fix in the repo and pull
- Don't terminate a pod with a downloaded model unless you mean to (stop instead)
- Don't use community cloud
- Don't use `diffusers` unpinned — it will pull 0.32+ and break on PyTorch 2.4
- Don't reinstall torch in requirements.txt — it's already in the template image
- Don't use the Ubuntu 24.04 template (v280) until PEP 668 situation is resolved
