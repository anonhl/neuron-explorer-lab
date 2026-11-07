# Neuron Explorer Lab

**English** | [简体中文](README.zh-CN.md)

A Windows-focused workbench for Qwen3.5-4B neuron and residual-stream experiments. This independently maintained fork of [lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) adds reusable interventions, multi-text comparisons, flexible model loading and local LoRA analysis.

[Repository](https://github.com/anonhl/neuron-explorer-lab) · [Documentation](docs/README.md) · [Changes](CHANGELOG.md) · [Contributing](CONTRIBUTING.md)

![Neuron explorer](explorer.png)

## What this fork adds

- **Residual experiments:** add a direction, remove a write, erase a projection or patch a state at a chosen layer, stage and token position; rerun the model to compare behavior.
- **Multi-text comparisons:** up to eight independent inputs; compare generation, next-token probabilities, KL divergence, target-continuation scores and activations; export JSON.
- **Reusable neuron batches:** save named groups of ordered operations with model/precision metadata; activate, replace or stop a batch.
- **Model-loading choices:** BF16/FP16, bitsandbytes/HQQ weight quantization, and optional local LoRA with full weights.
- **Local use and teaching:** Windows/offline guides, bundled Three.js, frequency inspection and a standalone simulated HTML demo.

The 3D explorer, language switching, basic neuron steering, three-column chat, original research and sweep pipeline originate upstream. This fork is not an official upstream release.

## Inspiration for residual interventions

One inspiration is research on **sodium oligomannate (GV-971, 甘露特钠)** and indirect regulation through the gut–brain axis. The design analogy treats the residual stream as a shared information environment: changing upstream writes can change what downstream neurons receive and how the model behaves. This is a conceptual inspiration, not an equivalence between biological and model mechanisms. [Original study](https://www.nature.com/articles/s41422-019-0216-x)

## Start here

| Goal | Guide |
|---|---|
| Try the UI without a model | Open `neuron-explorer-demo.html`; all values and replies are simulated |
| Install and launch on Windows | [Windows startup](docs/START_WINDOWS.md) |
| Prepare a fully offline machine | [Offline Windows guide](docs/OFFLINE_WINDOWS.md) |
| Use batches and residual experiments | [Experiment guide](docs/EXPERIMENTS.md) |
| Develop or prepare a release | [Contributing](CONTRIBUTING.md), [Releasing](docs/RELEASING.md) |
| See actual verification status | [Release checks](docs/RELEASE_CHECKS.md) |

Use **64-bit Python 3.12** for the documented setup (declared minimum: 3.11), uv, and a modern WebGL/import-map browser. Quantized loading requires NVIDIA CUDA; CPU/MPS use full weights. The inherited sweep scripts still target Linux/CUDA.

From the project root in PowerShell:

```powershell
uv sync --locked --python 3.12
# Optional: select the dependencies you need.
uv sync --locked --python 3.12 --extra quantization
uv sync --locked --python 3.12 --extra lora
# To keep both installed, put both --extra flags on the same command.
```

Weights are not distributed with the source. Follow the [startup guide](docs/START_WINDOWS.md) to download the model into `models/Qwen3.5-4B/Qwen3.5-4B`, then launch:

```powershell
$env:PYTHONUTF8 = "1"
$env:HF_HUB_OFFLINE = "1"
$env:EXPLORER_MODEL = "models/Qwen3.5-4B/Qwen3.5-4B"
Remove-Item Env:EXPLORER_NO_MODEL -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe explorer\server.py
```

Open [localhost:8791](http://127.0.0.1:8791/). Select a weight format and click **Load**; the server loads weights only after that action. For cache-only browsing, download the model config and set `EXPLORER_NO_MODEL="1"` as described in the guide.

## Loading and analysis

The dialog offers full BF16/FP16, bitsandbytes INT8/NF4/FP4 (including 4-bit double quantization), and HQQ 8/4/3/2/1-bit presets. Only selected linear weights are quantized; activations remain BF16/FP16. INT8 temporarily unpacks weights for floating-point multiplication, so smaller storage does not guarantee faster inference or a particular peak-memory footprint. Successful precision/preset choices are saved locally.

For LoRA, install `--extra lora`, select **Full weights**, and supply a local directory containing `adapter_config.json` and `adapter_model.safetensors`. Use a compatible base model. The adapter is merged in memory and activation calibration is session-specific; source weight files are not rewritten. LoRA is not combined with quantized loading in this app.

The explorer retains the upstream map, ranked units, lens/token search, live activations, steered chat, connections, rollout embeddings and optional agents. Historical rollouts, FFW contexts and raw R/J++ lens maps need additional data. The nine bundled reference caches are [verified against upstream](docs/upstream-cache-provenance.json).

## Limits and verification

- Quantization, FP16 and adapters can change results. Historical sweeps and indices describe the reference model; some original lens/connection features are restricted with LoRA.
- Local residual readouts are estimates. Causal experiments rerun baseline and intervention separately with fresh generation caches.
- Long activation captures use substantial host memory; weight quantization does not remove that cost. Start with short inputs.
- Optional Claude/Codex agents need separately installed/authenticated CLIs, may use online services and can access local files/commands. They are separate from offline Qwen inference. The server is for trusted single-user localhost use.

## Upstream research

The [original README](https://github.com/lyramakesmusic/neuron-explorer#readme), [research log](research_log.md) and [source notes](NOTES.md) document the upstream research and CUDA/vLLM sweep process. Original research records and executable agent prompts retain their English source text. Maintained English/Chinese guides are listed in the [documentation index](docs/README.md).

## License

Code is distributed under [MIT](LICENSE), retaining the original author's copyright. Models, datasets and bundled dependencies retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md). Source releases exclude weights, private adapters, personal configuration, chats and web corpora.
