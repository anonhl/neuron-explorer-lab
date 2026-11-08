# Neuron batches and residual experiments

**English** | [简体中文](EXPERIMENTS.zh-CN.md) · [Documentation](README.md)

## Reusable neuron configurations

Open **neuron config** in the header. Create a named group, add operations (layer, neuron, add/multiply, signed dose and enabled state), choose the destination group and save a batch. **Add selected neuron** copies the current chat adjustment.

Each save records ordered operations, model/revision when available, weight format, compute precision, UTC time and notes. **Use** activates all enabled operations; using another batch replaces the previous batch. **Stop using** removes the batch effect. Repeated operations on the same neuron run in saved order. Additive doses use the chat residual-norm units; multiplication is a direct factor, and 0 ablates a neuron.

Chat's temporary operations stack on the active batch. The evaluation baseline therefore includes that batch; stop it to compare against the original model. **Chat only** combines signed adjustments into one streaming reply and offers **Save as batch**; **Evaluation** compares positive, baseline and reversed responses.

Groups/batches are saved as UTF-8 with atomic replacement in `explorer/cache/neuron_batches.json`. Browser reloads retain the active server session; server restart or weight reload clears activation, but saved records remain. Reload the matching model/revision/format/precision before reusing a batch. Older records without precision are treated as BF16. **Edit a copy** starts a new record; saving a copy under the same model settings requires an operation change. Right-click to rename/delete; deleting the active batch stops it. No weight file is modified.

## Observe the residual stream

Run a short input, select a token, then open residual analysis. Captures include every input token across all layers. Switching layer/stage/neuron reuses that capture. Inspect pre (layer input), mid (after attention/mixing, before MLP norm), post (layer output), attention/MLP updates, full coordinates and signed extrema.

The map shows grouped mean coordinates in overview and exact coordinates when expanded; all layers share the selected dimension window. A neuron's write is its actual activation times its down-projection column, usually affecting many dimensions. Vector sign, activation sign and token-score changes have separate meanings.

The local token Δlogit is a lens difference before/after removing a write. The next-layer reader estimate freezes its attention output and recomputes the norm/MLP. Neither reruns all later layers. JSON export contains raw all-layer residual vectors for the selected token. Reload weights or change a batch, then recapture.

Residual capture alone adds about 325 MiB host memory for 32 layers, 2560 dimensions, 1024 tokens and 16-bit values. Full activation/scoring arrays are larger; begin with short text.

## Run a controlled intervention

In **Run**, add up to eight independent texts. Inputs are not concatenated; the selected text controls the current token/activation/map view. Drafts are remembered in the browser, while large activation analysis is kept only for the active text.

Open the right-side **residual stream** tab between Chat and Details and prepare tokens. Choose a layer, pre/mid/post stage, strength and scope: each text's selected token, prompt end, all prompt tokens, or continued intervention during generation. Inputs are raw by default; enabling chat wrapping includes template tokens in position numbering. Each input is limited to 1024 model tokens, with an explicit error rather than truncation; generate 0–128 new tokens.

| Method | Effect |
|---|---|
| Add direction | Use a neuron output direction, a state difference between two inputs, or a full-dimensional JSON vector. Normalize it and add α times the current affected residual's L2 norm; negative α reverses it. This unit differs from chat neuron dose. |
| Remove write | Subtract a chosen multiple of the current neuron's, whole MLP's, attention head's or whole attention/mixing output. Neuron/MLP removal uses post; attention/head removal uses mid or post. |
| Erase projection | Remove the normalized direction component, with strength 0–1. |
| Patch state | Replace/blend a target with a source residual at the same layer/stage, strength 0–1. Select source and target token positions independently; matching indices do not imply semantic alignment. Continued patching reuses the same source vector. |

For every input the experiment runs baseline and intervention independently, with fresh caches and deterministic greedy generation. Both include any active neuron batch; only the intervention side receives the residual operation. Results show generated text, next-token probabilities, KL, activation changes at the final prompt position and per-layer residual norms. Each setting's target token position is retained in the export. Candidate answers are one per line and scored across the entire continuation, not only its first token (1–64 tokens each).

Stop an experiment to keep completed groups, or export inputs, settings, model metadata and results as JSON. Changing settings retains old results marked as a previous experiment. Rerun after changing model weights or the active batch. Hooks are scoped to the experiment; weight files are unchanged.

## Simulated practice

Open `neuron-explorer-demo.html` and choose **simulated mode → multi-text residual comparison** to practice. Its data and replies are browser-generated examples, not Qwen measurements. Rebuild it from current sources with `python explorer/build_demo.py`.


## Keep residual interventions in chat

Add neurons by IDs such as `L22 N2742, L23 N100`, or from the map. Each neuron has one independent entry, up to 32; adding the same ID again preserves its settings. Check multiple rows to include them, or select/clear all. Clicking a row edits it without clearing other checked rows. Selected IDs update with the checkboxes. Each entry keeps its own method, layer, neuron, stage, scope, target positions and strength. The text experiment applies all checked entries together; input formatting and generation options are shared.

**Keep residual interventions** snapshots the checked entries for subsequent evaluation and conversation turns, without requiring a map selection. The same button becomes **Stop interventions**; stopping also cancels an active chat reply. Editing settings does not change the held snapshot: stop and enable again to apply edits. Chat status and turn records show the number of held entries.

Only the configured evaluation column receives held residual interventions. The baseline and reversed-neuron columns have none; erasure and replacement are not forcibly inverted. To isolate residual effects, disable temporary neuron adjustments and compare the configured and baseline columns. Any active neuron batch still applies to all three columns. Columns generate separately using fresh caches under the same model configuration. Token scope has a separate absolute target position in the full chat prompt, including template and history. Other scopes adapt to each turn.

Neuron directions, write removal and custom directions need no analysis text; difference directions and state replacement need source texts. Reloading the page or changing model or neuron batch stops holding. This applies to requests from this page, leaves weights unchanged, and does not affect other browser pages. Copy icons beside chat neuron IDs copy the full `Llayer Nneuron` address.
