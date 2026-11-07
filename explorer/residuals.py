"""Contextual residual inspection. Local readouts are estimates, not causal rollouts."""
import numpy as np
from base64 import b64encode


def vector_data(vector, k=12, full=True):
    v = np.asarray(vector, dtype=np.float32)
    def side(indices):
        return [{"dim": int(i), "value": float(v[i])} for i in indices[:k]]
    order = np.argsort(v)
    result = {"norm": float(np.linalg.norm(v)),
              "positive": side(order[::-1][v[order[::-1]] > 0]),
              "negative": side(order[v[order] < 0])}
    if full:
        result["vector"] = v.tolist()
    return result


def residual_rows(trace, pos):
    """The previous block output IS this block's input; do not store it twice."""
    before = trace["input"][pos].float().numpy()
    rows = []
    for layer in sorted(trace["post"]):
        mid = trace["mid"][layer][pos].float().numpy()
        post = trace["post"][layer][pos].float().numpy()
        rows.append({"layer": layer, "pre": before, "mid": mid, "post": post,
                     "attention": mid - before, "mlp": post - mid})
        before = post
    return rows


def inspect_residual(rt, snapshot, pos, layer, n=-1):
    import torch
    from quantization import linear_weight
    with rt.lock, torch.inference_mode():
        if snapshot.get("model_key") != rt._weight_cache_key("residual"):
            raise ValueError("Residual capture is stale; run the text again.")
        captured_batch, current_batch = snapshot.get("batch") or {}, rt.batch_info() or {}
        if any(captured_batch.get(key) != current_batch.get(key) for key in ("id", "applied_at", "operations")):
            raise ValueError("Residual capture is stale; run the text again.")
        rows = residual_rows(snapshot["residuals"], pos)
        row = rows[layer]
        A, S = snapshot["acts"], snapshot["score"]
        candidates = np.argsort(-np.abs(S[layer, pos]))[:12]
        if n < 0:
            n = int(candidates[0])
        columns = linear_weight(rt.lm.layers[layer].mlp.down_proj)
        d = columns[:, n].float()
        writer_norms = columns[:, candidates.tolist()].float().norm(dim=0).cpu().tolist()
        del columns
        activation = float(A[layer, pos, n])
        write = d * activation
        hidden_dtype = rt.model.dtype
        def tensor(v):
            return torch.as_tensor(v, device=rt.dev, dtype=hidden_dtype)
        def logits(v):
            return rt.model.lm_head(rt.lm.norm(v[None, None]))[0, 0].float()
        def token_rows(values, ids):
            return [{"id": int(i), "token": rt.tok.decode([int(i)]), "score": float(values[i])} for i in ids]
        def poles(values, k=8):
            positive = [i for i in values.topk(min(k, values.numel())).indices.tolist() if values[i] > 0]
            negative = [i for i in (-values).topk(min(k, values.numel())).indices.tolist() if values[i] < 0]
            return {"positive": token_rows(values, positive), "negative": token_rows(values, negative)}
        stages = {}
        for name in ("pre", "mid", "post"):
            v = row[name]
            out = vector_data(v)
            z = logits(tensor(v))
            out["tokens"] = token_rows(z, z.topk(min(8, z.numel())).indices.tolist())
            stages[name] = out
        post = tensor(row["post"])
        # The difference uses a common context and the same final norm. It is still a local
        # lens estimate: later blocks have not been rerun as they would be in an ablation.
        local_delta = logits(post) - logits(post - write.to(post.dtype))
        contribution = vector_data(write.cpu().numpy())
        contribution["tokens"] = poles(local_delta)
        selected = {"layer": layer, "n": n, "act": activation, "score": float(S[layer, pos, n]),
                    "write": contribution}
        writers = []
        for i, column_norm in zip(candidates, writer_norms):
            a = float(A[layer, pos, i])
            norm = column_norm * abs(a)
            writers.append({"layer": layer, "n": int(i), "act": a,
                            "score": float(S[layer, pos, i]), "write_norm": norm})
        readers = []
        if layer + 1 < len(rows):
            next_layer = rt.lm.layers[layer + 1]
            mid = tensor(rows[layer + 1]["mid"])
            # Freeze the next attention output, remove this write on the identity residual
            # path, then recompute the next norm + MLP. This is a local estimate only.
            x = next_layer.post_attention_layernorm(torch.stack([mid, mid - write.to(mid.dtype)]))
            mlp = next_layer.mlp
            g, u = mlp.gate_proj(x), mlp.up_proj(x)
            pair = (mlp.act_fn(g) * u).float()
            for l, i, mode, value in rt._batch_push:
                if l == layer + 1:
                    if mode == "mul": pair[:, i] *= value
                    else: pair[:, i] += value
            delta = pair[0] - pair[1]
            top = [i for i in delta.abs().topk(min(8, delta.numel())).indices.tolist() if delta[i] != 0]
            readers = [{"layer": layer + 1, "n": int(i), "act": float(A[layer + 1, pos, i]),
                        "score": float(S[layer + 1, pos, i]), "local_delta": float(delta[i])} for i in top]
        summary = [{"layer": r["layer"], **{key: float(np.linalg.norm(r[key]))
                   for key in ("pre", "mid", "post", "attention", "mlp")},
                   "bins": {key: [float(x.mean()) for x in np.array_split(r[key], min(64, len(r[key])))]
                            for key in ("pre", "mid", "post")}} for r in rows]
        map_vectors = np.stack([rows[0]["pre"], *[r[key] for r in rows for key in ("mid", "post")]]).astype("<f4")
        return {"ready": True, "capture": snapshot["capture"], "pos": pos,
                "token": snapshot["toks"][pos], "layer": layer, "hidden": len(row["pre"]),
                "layers": summary, "stages": stages, "selected": selected,
                "map_vectors": {"shape": list(map_vectors.shape), "dtype": "float32-le",
                                "data": b64encode(map_vectors.tobytes()).decode("ascii")},
                "writers": writers, "readers": readers,
                "updates": {key: vector_data(row[key]) for key in ("attention", "mlp")},
                "batch": snapshot.get("batch")}
