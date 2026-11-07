# Windows startup

**English** | [简体中文](START_WINDOWS.zh-CN.md) · [Documentation](README.md)

Initial package/model downloads require internet. After preparation, the local explorer can run offline at [localhost:8791](http://127.0.0.1:8791/).

## 1. Open PowerShell in the project folder

In File Explorer, open the folder containing README.md, pyproject.toml and explorer. Type `powershell` in the address bar. Run subsequent commands from this root.

## 2. Install dependencies

Install uv if needed, then reopen PowerShell:

```powershell
winget install --id=astral-sh.uv -e
uv sync --locked --python 3.12
```

Without winget, follow [uv's official Windows instructions](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer). The sync command creates `.venv`; activation is unnecessary. See the [offline guide](OFFLINE_WINDOWS.md) for CUDA and quantization. Optional LoRA dependencies: `uv sync --locked --python 3.12 --extra lora`.

## 3. Download model files

The inner Qwen3.5-4B directory holds the actual model, while the outer directory groups the base and optional adapters:

```text
neuron-explorer-lab/
├─ models/
│  └─ Qwen3.5-4B/
│     ├─ Qwen3.5-4B/       # config, tokenizer, complete weights
│     └─ my-lora/          # optional adapter
├─ explorer/
└─ README.md
```

For local chat and interventions, download the complete model:

```powershell
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B', local_dir='models/Qwen3.5-4B/Qwen3.5-4B')"
```

If already downloaded, verify config.json, tokenizer files, the weight index and every safetensors shard are directly in the inner directory. For cache-only browsing, download just the config:

```powershell
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-4B', local_dir='models/Qwen3.5-4B/Qwen3.5-4B', allow_patterns=['config.json'])"
```

## 4. Launch

Set these variables in the same window:

```powershell
$env:PYTHONUTF8 = "1"
$env:HF_HUB_OFFLINE = "1"
$env:EXPLORER_MODEL = "models/Qwen3.5-4B/Qwen3.5-4B"
```

For complete model inference:

```powershell
Remove-Item Env:EXPLORER_NO_MODEL -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe explorer\server.py
```

Or, for cache-only browsing:

```powershell
$env:EXPLORER_NO_MODEL = "1"
.\.venv\Scripts\python.exe explorer\server.py
```

After Uvicorn starts, open [localhost:8791](http://127.0.0.1:8791/) in Edge/Chrome. Complete-model mode first opens the loading dialog. Select full/quantized weights and BF16/FP16, then click **Load** to read weights. Loading progress/errors appear in the dialog; you can correct the selection and retry. Closing it permits cache browsing. Cache-only mode directly shows the map and “no model”. Keep PowerShell running; Ctrl+C stops the server.

## Optional: local LoRA

Install PEFT through the locked lora extra:

```powershell
uv sync --locked --python 3.12 --extra lora
```

Point EXPLORER_MODEL to the complete base used during training. A Base/pretrained adapter must not be silently paired with a different post-trained model. Choose **Full weights** and enter an adapter path such as `models\Qwen3.5-4B\my-lora`; it must contain adapter_config.json and adapter_model.safetensors and use the base tokenizer/chat format.

The adapter is merged in memory. Evaluation and conversation use the merged model. Each LoRA load runs an offline activation-range calibration with the bundled short English/Chinese text. Historical map sweeps remain original-model references; precomputed lens/connection features are restricted in LoRA mode. To return to the base, clear the adapter field and reload. The path defaults to empty after a server restart.

## Residual analysis and experiments

Run a short text and select a token. Residual analysis captures every layer; changing layers/stages reuses that capture. Local Δlogit/reader views are estimates, distinct from interventions that rerun the model. The residual tab supports up to eight independent texts and baseline/intervention comparison. See [the experiment guide](EXPERIMENTS.md) for token scopes, dose units, memory costs, batches and export.

The standalone `neuron-explorer-demo.html` uses simulated values and replies, without loading weights.

## Troubleshooting

| Problem | Check |
|---|---|
| `.venv\Scripts\python.exe` missing | Confirm the project root and a successful uv sync. |
| Model config not found | Run `Test-Path .\models\Qwen3.5-4B\Qwen3.5-4B\config.json`; check the inner directory. |
| Chat says model disabled | Download complete weights, stop the server, remove EXPLORER_NO_MODEL and restart. The string `"0"` still disables it. |
| Out of GPU memory | Begin with cache-only mode or short inputs; see the CUDA/quantization section of the offline guide. |

For numerical precision, data requirements, package transfer and more diagnostics, see [Offline Windows](OFFLINE_WINDOWS.md).
