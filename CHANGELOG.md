# Changelog

**English** | [简体中文](CHANGELOG.zh-CN.md)

## 0.1.0 — initial derivative release

Based on [lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), retaining the upstream MIT copyright and license.

### Added

- Windows startup and offline-transfer documentation, local Three.js assets, and a standalone simulated HTML demo.
- Deferred model loading with BF16/FP16 selection and bitsandbytes/HQQ weight-quantization presets.
- Optional local LoRA loading and session-specific activation calibration with full weights.
- Named neuron-configuration groups and ordered, reusable intervention batches.
- Separate evaluation and conversation chat modes, with batch saving from conversation mode.
- Frequency inspection linking generated words and connection-lens token occurrences.
- All-layer residual inspection, signed vector bands, contextual writes and local readout estimates.
- Up to eight independent texts with controlled residual direction, write removal, projection erasure and state-patching experiments; result export and cancellation.
- Unit tests, CI configuration, third-party attribution, cache provenance, and an allowlisted source-release builder with secret/file checks and checksums.
- Separate English/Chinese documentation and a short explanation of the GV-971 gut–brain-axis research analogy behind residual interventions.

### Known limits

- Static demo values are simulated, not Qwen measurements.
- Quantization and FP16 can change numerical behavior; LoRA needs a matching base model and is not combined with quantized loading.
- Historical sweeps and bundled indices describe the original reference model, not arbitrary adapters.
- Local residual readouts are estimates; the intervention experiment reruns the model separately.
- The optional agent integration is intended for a trusted local machine. The inherited sweep pipeline targets Linux/CUDA.
