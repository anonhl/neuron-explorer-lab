# Release preparation

**English** | [简体中文](RELEASING.zh-CN.md) · [Documentation](README.md)

Repository: [anonhl/neuron-explorer-lab](https://github.com/anonhl/neuron-explorer-lab), an existing public fork of lyramakesmusic/neuron-explorer. Initial derivative version: 0.1.0.

## Identity and provenance

Repository name: `neuron-explorer-lab`. Display name: **Neuron Explorer Lab**.

Suggested GitHub description:

> Windows-focused neuron and residual-stream experiments for Qwen3.5-4B, with multi-text comparisons, reusable interventions, quantized loading, and local LoRA support. Based on neuron-explorer.

The local source directory has no .git history. The GitHub fork preserves upstream history; publish new changes on top of that history, not by force-replacing it with an unrelated initial commit. A prepared ZIP can be overlaid on a genuine clone, retaining the clone's .git. Review the actual changes before committing. RELEASE_MANIFEST.json is a release attachment inventory, not required source.

## Check and build

From the project root (PowerShell-compatible commands):

```powershell
python explorer/build_demo.py
python -m unittest discover -s tests -p test_demo_bundle.py -v
python -m unittest discover -s tests -p test_release.py -v
node tests/test_residual_vector.mjs
node tests/test_residual_lab.mjs
python tools/prepare_release.py --check
python tools/prepare_release.py
```

This creates dist/neuron-explorer-lab-0.1.0-source.zip and its .sha256 sidecar. The archive contains explicitly selected code/docs, the original third-party package, nine hash-verified reference caches and an exact per-file SHA-256 manifest. It excludes models, private configuration, sessions and corpora. The tool does not delete personal files. Do not commit generated archives into the source tree.

Full dependency checks and tests:

```powershell
uv lock --check
uv sync --locked --python 3.12 --group dev --extra quantization --extra lora
uv run --locked --group dev --extra quantization --extra lora python -m unittest discover -s tests -v
```

Record outcomes and omissions in [release checks](RELEASE_CHECKS.md) and its Chinese counterpart. CI checks source/demo and small-model/API tests.

## Distribution policy

- Retain the complete original MIT notice, copyright for lyramakesmusic, and the separate anonhl modification attribution.
- Preserve THIRD_PARTY_NOTICES.md, its translation, LICENSES/ and all original vendor notices.
- Include the nine recorded upstream reference caches, about 195.74 MiB in total. No individual file reaches GitHub's 100 MiB regular-file limit. Do not convert these to unconfigured LFS pointers that break source-ZIP downloads.
- Exclude weights, adapters, raw lens matrices, web corpora, offline wheels, personal SQLite, conversations and experiment exports. A separate offline bundle needs its own resource-license/privacy review.
- The standalone HTML may be attached separately. It embeds applicable MIT notices and clearly labels simulated data.
- Maintain language-switch links and both English/Chinese versions of user, contributor and release documents. Legal originals, upstream research records and executable prompts retain their original text.

## Before a release

- In a true Git checkout, check all local refs and inspect the actual diff. Pattern matching does not prove absence of every secret. If credentials were exposed, revoke/rotate before cleaning history.
- Confirm GitHub Actions actually passes. A workflow file alone is not a successful run.
- If public issues or private vulnerability reporting are desired, enable them in repository settings; do not claim they are enabled before checking.
- Publish v0.1.0, attaching the source ZIP, checksum and simulated HTML.

## Suggested release notes

Neuron Explorer Lab v0.1.0 is an independently maintained MIT-licensed derivative of lyramakesmusic/neuron-explorer.

It adds Windows/offline guides, BF16/FP16 and weight quantization, local LoRA with full weights, reusable neuron batches, all-layer residual inspection and up to eight independent texts in controlled residual-intervention experiments. A standalone simulated demo is included.

Weights are downloaded separately. Reference caches describe the original Qwen3.5-4B; precision, quantization and adapters can change results. Local readouts are distinguished from experiments that rerun the model. Installation/validation status is recorded in the repository. All values in the standalone demo are simulated.
