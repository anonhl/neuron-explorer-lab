# Third-party notices

**English** | [简体中文](THIRD_PARTY_NOTICES.zh-CN.md)

The project MIT license does not replace the terms of third-party resources. Preserve their original copyright and permission notices when redistributing them.

## Original neuron-explorer

- Author / copyright: `Copyright (c) 2026 lyramakesmusic`.
- Source: <https://github.com/lyramakesmusic/neuron-explorer>.
- License: MIT; the complete original notice is retained in [LICENSE](LICENSE).
- Local modifications are attributed to anonhl in the same license file.
- Upstream reference inspected for this release preparation: `7ec363d2d9686802a1b4aa7c6d8a2627b1759159`. This is a comparison reference, not a claim about the original download's exact base commit.
- The original explorer, sweep scripts, research notes, image and bundled reference caches come from upstream. Windows/offline instructions, loading controls, neuron batches, frequency inspection, residual analysis/interventions, LoRA integration, tests and demo/release tooling include local changes. This derivative is not an official upstream release.

## Bundled Three.js

- Version: `three@0.169.0`, unmodified npm distribution.
- Source and integrity: [vendor README](explorer/static/vendor/README.md).
- Main license: MIT, `Copyright © 2010-2024 three.js authors`; see [complete notice](explorer/static/vendor/three/LICENSE).
- The full npm package also contains third-party examples, libraries, fonts and binaries with their own inline notices and accompanying licenses. They remain in their original directories. In particular, retain [the font license](explorer/static/vendor/three/examples/fonts/LICENSE) and [Droid's Apache notice](explorer/static/vendor/three/examples/fonts/droid/NOTICE). The package must not be relabeled wholesale as solely MIT.
- The portable demo embeds only the Three.js module and the selected controls/line addons used by the app. Its generated HTML includes both the original project's MIT notice and the Three.js MIT notice, so it can be shared independently.

## Bundled reference caches

The release tool includes only the nine upstream cache files listed in [cache provenance](docs/upstream-cache-provenance.json), and verifies each file's Git blob SHA-1 and SHA-256. These files are copied from the public upstream repository, not generated from the publisher's private sessions. The SHA-1 values identify upstream blobs; SHA-256 values check the release bytes.

`r_index.npz` and `jpp_index.npz` are derived using [bubskee/qwen3.5-4b-r-jpp-lenses](https://huggingface.co/bubskee/qwen3.5-4b-r-jpp-lenses), identified as Apache-2.0 in the upstream README and build script. Preserve that attribution and the [Apache-2.0 text](LICENSES/Apache-2.0.txt) with the derived indices. The original lens matrices are not bundled. This attribution records upstream provenance; it does not claim that every upstream artifact was independently relicensed by this fork.

Other bundled caches contain reference neuron statistics, layouts, connections, token strings and lens indices for the upstream Qwen model. They remain reference artifacts when a different quantization, precision or adapter is loaded. The release manifest records their precise bytes and origin.

## Model weights, adapters and optional datasets (not bundled)

| Resource | Source / policy |
|---|---|
| Qwen3.5-4B | [Official model](https://huggingface.co/Qwen/Qwen3.5-4B), Apache-2.0. Download separately and retain the model's LICENSE/NOTICE and model card when redistributing it. |
| R/J++ lens matrices | [Lens release](https://huggingface.co/bubskee/qwen3.5-4b-r-jpp-lenses), Apache-2.0 as identified upstream; download separately and preserve the release's own metadata/notices. |
| FineFineWeb | [m-a-p/FineFineWeb](https://huggingface.co/datasets/m-a-p/FineFineWeb). Optional user download; corpus, source URLs and activation captures are excluded from the release. Dataset and underlying web content terms must be checked before redistributing excerpts or derived data. No blanket MIT permission is asserted. |
| User LoRA adapters | Supplied by the user; not bundled. Follow each adapter's license, base-model terms and training-data restrictions. |
| vLLM steering fork | [RhizoNymph/vllm](https://github.com/RhizoNymph/vllm), downloaded by the optional Linux sweep setup, not vendored here. Preserve that project's license and notices in any separately distributed runtime. |

## Installed dependencies and agent tools

Python packages are installed through `pyproject.toml` / `uv.lock` and retain their own distribution licenses; their code is not copied into the source release. Examples include PyTorch, Transformers, Accelerate, PEFT, bitsandbytes, HQQ, NumPy, FastAPI, Uvicorn, datasets, sentence-transformers and umap-learn. A separate offline wheel bundle must retain the licenses shipped in its wheels and identify its exact dependency versions.

Claude and Codex CLIs, credentials and account services are not bundled. Users install/authenticate those optional tools separately under the providers' terms. Their names indicate integration compatibility, not endorsement.
