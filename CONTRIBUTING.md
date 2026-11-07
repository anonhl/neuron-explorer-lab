# Contributing

**English** | [简体中文](CONTRIBUTING.zh-CN.md)

Contributions to [anonhl/neuron-explorer-lab](https://github.com/anonhl/neuron-explorer-lab) are welcome. Explain the problem, expected behavior and reproduction steps. This fork focuses on Windows/local use and the added experiment workflows; credit the original research and explorer to upstream.

## Development and tests

Use Python 3.12 and uv from the repository root:

```powershell
uv sync --locked --python 3.12 --group dev --extra quantization --extra lora
uv run --locked --group dev --extra quantization --extra lora python -m unittest discover -s tests -v
node tests/test_residual_vector.mjs
node tests/test_residual_lab.mjs
```

The dev group supplies httpx for API tests. Quantization tests require both backends. LoRA support requires the lora extra. Keep the required extras on subsequent uv sync commands. Tests use small tensors, mocks and a tiny randomly initialized Qwen model; they do not download real weights.

For frontend checks, use Node.js 22.7 or newer, supporting ES-module detection without a package.json. No npm install or frontend bundler is required.

Standard-library-only checks:

```powershell
python -m unittest discover -s tests -p test_demo_bundle.py -v
python -m unittest discover -s tests -p test_release.py -v
python -m unittest discover -s tests -p test_neuron_batches.py -v
```

The last command skips runtime/API cases when dependencies are absent. Report skipped cases explicitly.

After UI changes, regenerate the demo and check release inputs:

```powershell
python explorer/build_demo.py
python tools/prepare_release.py --check
```

## Pull requests

- Explain the trigger, resulting behavior, validation and remaining limits.
- Test meaningful experiment properties, especially residual conservation, token positions, independent generation caches and hook cleanup.
- For dependency changes, run uv lock and commit both pyproject.toml and uv.lock. Do not use --frozen to hide an inconsistent lock.
- Update both language versions of maintained user/developer documents. Keep legal licenses, vendor resources, upstream research records and executable agent prompts in their original form.
- Preserve third-party notices and update THIRD_PARTY_NOTICES.md and its Chinese counterpart when adding assets.
- Exclude credentials, weights, private inputs, adapters, sessions and runtime caches.
- Claim hardware compatibility or performance only with recorded real-device evidence.

Contributions are distributed under the repository MIT license. Submit only material you are authorized to publish. See [documentation](docs/README.md) and [release preparation](docs/RELEASING.md).
