# Offline Windows preparation and transfer

**English** | [简体中文](OFFLINE_WINDOWS.zh-CN.md) · [Documentation](README.md)

Offline use needs **Python dependencies, model files, local frontend assets and analysis data**, not only weights. Prepare/download while online, then test without external networking. Three.js 0.169.0 is already bundled under explorer/static/vendor/three.

## 1. Choose a mode and tools

| Mode | Requirements | Available behavior |
|---|---|---|
| Cache-only | Python environment, model config.json, bundled Three.js and reference caches | Map, cached lenses/connections, any separately supplied historical runs; no live generation or activations |
| Local inference | All cache-mode materials plus complete Qwen3.5-4B and tokenizer | Local chat, neuron interventions, captures and residual experiments, subject to memory/device limits |

Both modes import PyTorch/Transformers. EXPLORER_NO_MODEL disables weight loading, not those dependencies. Optional Claude/Codex agents and the online neuronary community are separate from offline local inference. The inherited vLLM sweep pipeline targets Linux/CUDA.

Use 64-bit Python 3.12 (same patch/architecture on preparation and target machines), uv, a WebGL/import-map browser such as Edge/Chrome, and the appropriate NVIDIA driver for CUDA. Keep full offline Python/driver installers and enough disk space for weights, wheels and temporary copies. Run commands from the root:

```powershell
Set-Location "D:\AI\neuron-explorer-lab"
py -3.12 --version
uv --version
tar --version
```

## 2. Prepare the Python environment online

```powershell
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:HF_DATASETS_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:UV_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:PIP_NO_INDEX -ErrorAction SilentlyContinue
uv sync --locked --python 3.12
.\.venv\Scripts\python.exe -c "import torch, transformers, numpy, fastapi, uvicorn; from transformers import Qwen3_5ForConditionalGeneration; print('torch:', torch.__version__); print('transformers:', transformers.__version__); print('imports OK')"
```

Use the lock first. Runtime hooks use Qwen3.5 internals, so independently upgrading torch/transformers may break capture/loading. Resolve missing wheels or version conflicts online before transfer.

Optional dependencies (choose the required set):

```powershell
uv sync --locked --python 3.12 --extra bitsandbytes
# Alternative:
uv sync --locked --python 3.12 --extra hqq
# Both quantization backends:
uv sync --locked --python 3.12 --extra quantization
# LoRA with full weights:
uv sync --locked --python 3.12 --extra lora
# Keep both types installed:
uv sync --locked --python 3.12 --extra quantization --extra lora
```

Repeat the needed extras on later sync commands. LoRA itself cannot be loaded with quantized weights in this app. Quantization requires NVIDIA CUDA; CPU/MPS use full weights. Activations, embeddings and other floating-point components still consume memory.

Check the actual PyTorch CUDA build:

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch; print('torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

nvidia-smi alone does not prove Python can use CUDA. If a different torch build is needed, use [PyTorch's official selector](https://pytorch.org/get-started/locally/) and apply its pip command to this environment after `python.exe -m ensurepip --upgrade`. Verify compatibility and inference again. Later uv sync/default uv run may replace a manually adjusted build; for the verified offline environment, launch its Python directly.

## 3. Download a pinned model

The inner directory contains actual files; the outer Qwen3.5-4B directory groups the base and optional my-lora adapter. Complete inference download, with recorded source revision:

```powershell
@'
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
repo = 'Qwen/Qwen3.5-4B'
revision = HfApi().model_info(repo).sha
folder = Path('models/Qwen3.5-4B/Qwen3.5-4B')
snapshot_download(repo, revision=revision, local_dir=folder)
(folder / 'offline-source.json').write_text(json.dumps({'repo': repo, 'revision': revision}, indent=2), encoding='utf-8')
print(folder, revision)
'@ | .\.venv\Scripts\python.exe -
```

For cache-only mode, add `allow_patterns=['config.json']` to snapshot_download instead of downloading the full model. If upgrading to complete inference later, fetch all files for the recorded revision. Validate config, tokenizer, weight index and every indexed shard before going offline. Preserve model LICENSE/NOTICE and the source record.

Optional R/J++ maps are downloaded separately:

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('bubskee/qwen3.5-4b-r-jpp-lenses', local_dir='explorer/cache/rjpp')"
```

## 4. Keep the correct frontend and analysis data

Copy the complete explorer/static/vendor/three directory. It is already configured locally; do not replace it with a CDN dependency.

The nine bundled reference caches are listed in upstream-cache-provenance.json. Historical rollouts additionally require matching runs/<name>.jsonl and runs/<name>.meta.json, rankings use the associated units file, embeddings need UMAP data, and per-layer calibration may need stats/. FFW web contexts need corpus.npz, urls.json and acts.npz in explorer/cache/ffw/. Raw R/J++ and S maps are optional downloads. Missing historical inputs can leave those panels empty without invalidating the basic map. Do not redistribute private runs or corpora as part of the public source release.

## 5. Launch without external networking

```powershell
$env:PYTHONUTF8 = "1"
$env:HF_HUB_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:UV_OFFLINE = "1"
$env:EXPLORER_MODEL = "models/Qwen3.5-4B/Qwen3.5-4B"
# Cache-only:
$env:EXPLORER_NO_MODEL = "1"
.\.venv\Scripts\python.exe explorer\server.py
```

For live inference, remove EXPLORER_NO_MODEL before starting instead:

```powershell
Remove-Item Env:EXPLORER_NO_MODEL -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe explorer\server.py
```

Any nonempty value, including `"0"`, disables the model. Open [localhost:8791](http://127.0.0.1:8791/). Live mode begins at idle; choose the weight format/compute precision and click Load. Status becomes ready on success. Cache-only mode reports no model. First attention analyses may take additional time.

BF16 is default. FP16 is available for devices without native BF16, but its smaller numerical range needs actual numerical validation. Quantized presets require the installed CUDA backend; they quantize selected weights only. The runtime chooses CUDA, then MPS, then CPU automatically; there is no forced-device environment variable. EXPLORER_QUANTIZATION and EXPLORER_COMPUTE_DTYPE override saved selections. See [experiments](EXPERIMENTS.md) for doses, batches and baseline semantics.

## 6. Build an offline wheel bundle on Windows

Use a Windows preparation machine matching the target Python/architecture and, ideally, GPU/driver. Do not create Windows wheels using the current macOS environment. Start with a fully installed and actually verified environment, including required optional packages:

```powershell
.\.venv\Scripts\python.exe -m ensurepip --upgrade
New-Item -ItemType Directory -Force ".\offline-cache\wheelhouse" | Out-Null
.\.venv\Scripts\python.exe -m pip freeze | Set-Content -Encoding UTF8 ".\offline-cache\requirements-offline.txt"
Get-Content ".\offline-cache\requirements-offline.txt"
```

Resolve editable installs, local disk/file references or Git dependencies into redistributable wheels first. Clear offline variables before downloading/building. Use the index corresponding to the actual installed torch build:

```powershell
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:HF_DATASETS_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:UV_OFFLINE -ErrorAction SilentlyContinue
Remove-Item Env:PIP_NO_INDEX -ErrorAction SilentlyContinue
$cudaBuild = .\.venv\Scripts\python.exe -c "import torch; print(torch.version.cuda or '')"
$wheelArgs = @("wheel", "--no-deps", "--wheel-dir", ".\offline-cache\wheelhouse", "-r", ".\offline-cache\requirements-offline.txt")
if ($cudaBuild) {
    $cudaTag = "cu" + $cudaBuild.Trim().Replace(".", "")
    $wheelArgs += @("--extra-index-url", "https://download.pytorch.org/whl/$cudaTag")
}
.\.venv\Scripts\python.exe -m pip @wheelArgs
if ($LASTEXITCODE -ne 0) { throw "Wheel preparation failed; do not transfer an incomplete package" }
```

--no-deps assumes the freeze includes all direct and indirect installed dependencies. For preview/special CUDA builds whose index does not match the derived tag, use the original official index. Do not replace a verified CUDA build with CPU packages just to make packaging succeed.

Rehearse a clean offline installation on the preparation machine:

```powershell
py -3.12 -m venv .offline-check
.\.offline-check\Scripts\python.exe -m pip --isolated install --no-index --only-binary=:all: --find-links ".\offline-cache\wheelhouse" -r ".\offline-cache\requirements-offline.txt"
if ($LASTEXITCODE -ne 0) { throw "Offline installation rehearsal failed" }
.\.offline-check\Scripts\python.exe -m pip check
.\.offline-check\Scripts\python.exe -c "import torch, transformers, fastapi, numpy; from transformers import Qwen3_5ForConditionalGeneration; print('offline imports OK'); print('CUDA:', torch.cuda.is_available())"
```

Then disconnect external networking and actually launch with .offline-check's Python. Imports alone do not test generation or the map. A virtual environment is not generally portable; transfer files and recreate the environment on the target, rather than copying .venv.

## 7. Transfer and install on the target

Carry Python/driver installers, explorer (including caches and vendor assets), sweep (server imports sweep/rank.py), model/tokenizer files, any optional runs/stats, offline-cache/wheelhouse and the frozen requirements, pyproject.toml, uv.lock and docs. Use media supporting files over 4 GB. Optionally record hashes:

```powershell
Get-ChildItem ".\models", ".\offline-cache\wheelhouse", ".\explorer\cache" -Recurse -File | Get-FileHash -Algorithm SHA256 | Export-Csv ".\offline-cache\sha256.csv" -NoTypeInformation -Encoding UTF8
```

That CSV uses preparation-machine absolute paths; map them to the equivalent relative files after transfer before comparing hashes. Recreate the target environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip --isolated install --no-index --only-binary=:all: --find-links ".\offline-cache\wheelhouse" -r ".\offline-cache\requirements-offline.txt"
if ($LASTEXITCODE -ne 0) { throw "Offline dependency installation failed" }
.\.venv\Scripts\python.exe -m pip check
```

The target does not need uv. Set the offline variables and run its Python directly. Separate offline distributions must preserve all model/package licenses and exclude personal data as appropriate.

## 8. Acceptance and troubleshooting

With all old servers stopped and external networking disconnected, verify config loading, a fresh browser window, local Three.js requests, cached lens lookup and available historical data. In inference mode also test short generation, token activations and batch save/use/stop/restart behavior. Use DevTools Disable cache if helpful, but do not use the browser's global Offline switch, which can block localhost itself.

| Symptom | Check |
|---|---|
| Empty map | WebGL, local vendor requests, console errors; serve through HTTP rather than opening explorer/static/index.html directly. |
| Missing config or offline download error | Current working directory, EXPLORER_MODEL and the inner config/tokenizer/shard paths; restart after changing environment variables. |
| CUDA unavailable | The project Python's torch build and torch.cuda.is_available(), not only nvidia-smi. |
| FP16/non-finite outputs | Try compatible compute precision, a short unsteered input and smaller intervention strength. |
| Host/GPU memory exhausted | Shorten text and chat; 1024-token activation/scoring arrays alone total about 2.25 GiB host memory, before extra captures/model costs. |
| Missing examples/embeddings/contexts | Check optional runs metadata, UMAP and FFW files. Basic caches cannot reconstruct historical generation. |
| Model busy | Stop generation and wait before changing weights/configuration. Computation on the shared model is serialized. |
| Port occupied | Set `$env:PORT = "8792"` before launch and visit that port. |
| No matching offline wheel | Match Python patch/architecture, supply all dependencies and Windows wheels, then repeat the clean preparation-machine rehearsal. |

Back up private names.json, neuron_batches.json, model_loading.json and offline-source.json separately. Ignored personal files are not backed up by a Git commit or source download. See [security](../SECURITY.md) and [third-party notices](../THIRD_PARTY_NOTICES.md).
