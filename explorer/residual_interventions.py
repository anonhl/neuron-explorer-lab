"""Scoped residual experiments and held chat interventions; model weights are unchanged."""
import math
from contextlib import AbstractContextManager, ExitStack

import torch
import torch.nn.functional as F

from quantization import linear_weight

MAX_TEXTS = 8
MAX_TOKENS = 1024


def validate_experiment(body, layers, hidden, inter):
    if not isinstance(body, dict):
        raise ValueError("Experiment must be an object.")
    if "plans" in body:
        plans = body["plans"]
        if not isinstance(plans, list) or not 1 <= len(plans) <= 32 or any(not isinstance(p, dict) for p in plans):
            raise ValueError("Provide between 1 and 32 residual settings.")
        shared = {k: v for k, v in body.items() if k not in ("plan", "plans")}
        checked = [validate_experiment({**shared, "plan": p}, layers, hidden, inter) for p in plans]
        return {**checked[0], "plans": [item["plan"] for item in checked]}
    texts = body.get("texts")
    if not isinstance(texts, list) or not 1 <= len(texts) <= MAX_TEXTS:
        raise ValueError("Provide between 1 and 8 input texts.")
    seen = set()
    for row in texts:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or row["id"] in seen:
            raise ValueError("Every input needs a unique text ID.")
        seen.add(row["id"])
        if not isinstance(row.get("text"), str) or not row["text"].strip() or len(row["text"]) > 50000:
            raise ValueError("Each input must contain 1 to 50000 characters.")
    p = dict(body.get("plan") or {})
    def integer(name, default, limit):
        value = p.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < limit:
            raise ValueError(f"Invalid residual {name}.")
        p[name] = value
    integer("layer", 0, layers)
    integer("neuron", 0, inter)
    if p.get("stage") not in ("pre", "mid", "post"):
        raise ValueError("Choose pre, mid or post.")
    if p.get("method") not in ("add", "remove", "erase", "patch"):
        raise ValueError("Unknown residual intervention method.")
    if p.get("scope") not in ("token", "last", "prompt", "continuation", "all"):
        raise ValueError("Unknown residual token scope.")
    strength = p.get("strength", 1.)
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or abs(strength) > 10:
        raise ValueError("Residual strength must be finite and between -10 and 10.")
    p["strength"] = float(strength)
    if p["method"] in ("erase", "patch") and not 0 <= strength <= 1:
        raise ValueError("Erasure and replacement strength must be between 0 and 1.")
    if p["method"] == "remove":
        if p.get("component") not in ("neuron", "attention", "mlp", "head"):
            raise ValueError("Choose a residual writer to remove.")
        if p["stage"] == "pre" or (p["component"] in ("neuron", "mlp") and p["stage"] != "post"):
            raise ValueError("Remove MLP/neuron writes at post; attention/head writes at mid or post.")
        if p["component"] == "head":
            integer("head", 0, 256)
    if p["method"] in ("add", "erase"):
        if p.get("direction") not in ("neuron", "difference", "custom"):
            raise ValueError("Choose a residual direction source.")
        if p["direction"] == "custom":
            v = p.get("vector")
            if not isinstance(v, list) or len(v) != hidden or any(
                    isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in v):
                raise ValueError(f"Custom direction must contain {hidden} finite numbers.")
    if p["method"] == "patch" or (p["method"] in ("add", "erase") and p.get("direction") == "difference"):
        if p.get("source") not in seen:
            raise ValueError("Select a source text from this experiment.")
        if p["method"] != "patch":
            if p.get("reference") not in seen or p["reference"] == p["source"]:
                raise ValueError("A difference direction needs two distinct source texts.")
    for name in ("source_pos", "reference_pos"):
        value = p.get(name, -1)
        if isinstance(value, bool) or not isinstance(value, int) or value < -1:
            raise ValueError("Invalid source token position.")
        p[name] = value
    positions = p.get("positions", {})
    if not isinstance(positions, dict) or any(k not in seen or isinstance(v, bool) or not isinstance(v, int) or v < 0
                                               for k, v in positions.items()):
        raise ValueError("Invalid per-text token positions.")
    if p["scope"] == "token" and any(row["id"] not in positions for row in texts):
        raise ValueError("Choose a target token for every input text.")
    p["positions"] = dict(positions)
    if p["method"] != "remove":
        p.pop("component", None)
        p.pop("head", None)
    if p["method"] not in ("add", "erase"):
        p.pop("direction", None)
        p.pop("vector", None)
    max_new = body.get("max_new", 48)
    if isinstance(max_new, bool) or not isinstance(max_new, int) or not 0 <= max_new <= 128:
        raise ValueError("Generate between 0 and 128 new tokens.")
    if not isinstance(body.get("chat", False), bool):
        raise ValueError("Chat formatting must be true or false.")
    targets = body.get("targets", [])
    if not isinstance(targets, list) or len(targets) > 8 or any(not isinstance(t, str) or not t or len(t) > 200 for t in targets):
        raise ValueError("Provide up to 8 target continuations, each at most 200 characters.")
    return {**body, "plan": p, "max_new": max_new, "targets": targets}


def tokenize_texts(rt, texts, chat=False):
    rows = []
    for row in texts:
        ids = rt.chat_ids(row["text"]) if chat else rt.tok(row["text"]).input_ids
        if not ids or len(ids) > MAX_TOKENS:
            raise ValueError(f"Text {row['id']} must contain 1 to {MAX_TOKENS} model tokens; shorten it before running.")
        units = rt.token_units(ids)
        rows.append({**row, "ids": ids, "tokens": [{"text": text, "positions": pos} for text, pos in units]})
    return rows


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _replace(out, h):
    return (h, *out[1:]) if isinstance(out, tuple) else h


class ResidualHooks(AbstractContextManager):
    """A request owns these hooks under Runtime.lock; exiting always removes every hook.

    Mid interventions change the mixing branch before its residual addition, so both the
    MLP reader and the residual bypass receive the change. Editing the norm input alone
    would leave the bypass unchanged and would not be a residual-stream intervention.
    """
    def __init__(self, rt, plan, prompt_len, position, vector=None, observe=True):
        self.rt, self.plan, self.prompt_len, self.position = rt, plan, prompt_len, position
        self.vector, self.observe, self.offset = vector, observe, 0
        self.handles, self.stages, self.layers, self.neurons = [], {}, {}, {}
        self.pre = self.attention = self.mlp = self.neuron = self.head = None

    def _edit(self, h, write=None):
        p = self.plan
        if not p.get("enabled") or p["strength"] == 0:
            return h
        indices = torch.arange(h.shape[1], device=h.device) + self.offset
        scope = p["scope"]
        mask = {"token": indices == self.position, "last": indices == self.prompt_len - 1,
                "prompt": indices < self.prompt_len, "continuation": indices >= self.prompt_len - 1,
                "all": torch.ones_like(indices, dtype=torch.bool)}[scope]
        if not mask.any():
            return h
        x = h[:, mask].float()
        v = self.vector.to(h.device).float() if self.vector is not None else None
        amount = p["strength"]
        if p["method"] == "add":
            # Per-position relative L2 dose. Direction magnitude cannot silently change the dose.
            y = x + amount * x.norm(dim=-1, keepdim=True) * v
        elif p["method"] == "erase":
            y = x - amount * (x * v).sum(-1, keepdim=True) * v
        elif p["method"] == "patch":
            y = torch.lerp(x, v.expand_as(x), amount)
        else:
            if write is None:
                raise ValueError("The requested residual write was not captured.")
            y = x - amount * write[:, mask].float()
        if not torch.isfinite(y).all() or y.abs().max() > torch.finfo(h.dtype).max:
            raise ValueError("Residual intervention exceeds the model's numeric range; reduce the strength.")
        result = h.clone()
        result[:, mask] = y.to(h.dtype)
        return result

    def _save(self, stage, h):
        j = self.position - self.offset
        if 0 <= j < h.shape[1]:
            self.stages[stage] = h[0, j].detach().float().cpu().clone()

    def __enter__(self):
        rt, p = self.rt, self.plan
        chosen = p["layer"]
        try:
            neuron_direction = rt.down_col(chosen, p["neuron"]).detach() if p.get("component") == "neuron" else None
            for i, layer in enumerate(rt.lm.layers):
                if not self.observe and i != chosen:
                    continue
                def pre_hook(mod, args, kwargs, i=i):
                    h = kwargs.get("hidden_states", args[0] if args else None)
                    if i == chosen:
                        if p["stage"] == "pre":
                            h = self._edit(h)
                        self.pre = h
                        self._save("pre", h)
                        if p["stage"] == "pre" and p.get("enabled"):
                            if "hidden_states" in kwargs:
                                return args, {**kwargs, "hidden_states": h}
                            return (h, *args[1:]), kwargs
                self.handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True, prepend=True))
                def post_hook(mod, args, out, i=i):
                    h = _hidden(out)
                    if i == chosen:
                        if p["stage"] == "post":
                            write = {"neuron": self.neuron, "mlp": self.mlp, "attention": self.attention,
                                     "head": self.head}.get(p.get("component"))
                            h = self._edit(h, write)
                        self._save("post", h)
                    j = self.position - self.offset
                    if self.observe and 0 <= j < h.shape[1]:
                        self.layers[i] = float(h[0, j].float().norm())
                    return _replace(out, h)
                self.handles.append(layer.register_forward_hook(post_hook, prepend=True))
                if self.observe or (i == chosen and p.get("component") == "neuron"):
                    def neurons_hook(mod, args, i=i):
                        h = args[0]
                        j = self.position - self.offset
                        if self.observe and 0 <= j < h.shape[1]:
                            self.neurons[i] = h[0, j].detach().float().cpu().clone()
                        if i == chosen and p.get("component") == "neuron":
                            d = neuron_direction.to(device=h.device, dtype=h.dtype)
                            self.neuron = h[..., p["neuron"], None] * d
                    # Runs after the persistent neuron-batch hook, including its modifications.
                    self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(neurons_hook))
            layer = rt.lm.layers[chosen]
            attention = layer.self_attn if hasattr(layer, "self_attn") else layer.linear_attn
            def attention_hook(mod, args, out):
                a = _hidden(out)
                self.attention = a
                mid = self.pre + a
                if p["stage"] == "mid":
                    write = self.head if p.get("component") == "head" else a
                    edited = self._edit(mid, write)
                    if edited is not mid:
                        a = a + (edited - mid)
                        mid = self.pre + a
                self._save("mid", mid)
                return _replace(out, a)
            self.handles.append(attention.register_forward_hook(attention_hook))
            if p.get("component") == "mlp":
                self.handles.append(layer.mlp.register_forward_hook(lambda mod, args, out: setattr(self, "mlp", out)))
            if p.get("component") == "head":
                projection = attention.o_proj if hasattr(attention, "o_proj") else attention.out_proj
                count = rt.n_heads(chosen)
                if not 0 <= p["head"] < count:
                    raise ValueError(f"Layer {chosen} has {count} attention heads.")
                width = projection.in_features // count
                start = p["head"] * width
                weight = linear_weight(projection)[:, start:start + width].detach()
                def head_hook(mod, args):
                    self.head = F.linear(args[0][..., start:start + width], weight.to(args[0].dtype))
                self.handles.append(projection.register_forward_pre_hook(head_hook))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def run_experiment(rt, body, cancel):
    """Yields immutable completed rows; all model work is serialized and cancellation is checked per token."""
    with rt.lock, torch.inference_mode():
        rt.ensure_loaded()
        p = body["plan"]
        plans = body.get("plans", [p])
        rows = tokenize_texts(rt, body["texts"], body.get("chat", False))
        by_id = {r["id"]: r for r in rows}
        vectors = []
        for plan in plans:
            for row in rows:
                if plan["scope"] == "token" and plan["positions"][row["id"]] >= len(row["ids"]):
                    raise ValueError(f"Target token is outside text {row['id']}.")
            if cancel.is_set():
                return
            # Resolve every donor before installing any intervention, using the same tokenized inputs.
            vectors.append(prepare_chat_residual(rt, {**body, "plan": plan}, rows=by_id))
        eos = {x for x in (rt.tok.eos_token_id, rt.tok.convert_tokens_to_ids("<|im_end|>")) if isinstance(x, int) and x >= 0}
        target_ids = [(text, rt.tok(text, add_special_tokens=False).input_ids) for text in body["targets"]]
        if any(not ids or len(ids) > 64 for _, ids in target_ids):
            raise ValueError("Each target continuation must contain 1 to 64 tokens.")
        def run(row, enabled):
            ids = row["ids"]
            position = len(ids) - 1
            with ExitStack() as stack:
                # Observe the next-token prediction position once, after all edits at each layer.
                # Per-setting token positions are used only to target the edits.
                observers = stack.enter_context(ResidualHooks(rt, {**p, "enabled": False}, len(ids), position))
                hooks = []
                for plan, vector in reversed(list(zip(plans, vectors))):
                    target = plan["positions"][row["id"]] if plan["scope"] == "token" else position
                    hooks.append(stack.enter_context(ResidualHooks(
                        rt, {**plan, "enabled": enabled}, len(ids), target, vector, observe=False)))
                out = rt.model(input_ids=torch.tensor([ids], device=rt.dev), use_cache=True, logits_to_keep=1)
                lp = out.logits[0, -1].float().log_softmax(-1).cpu()
                if not torch.isfinite(lp).all():
                    raise ValueError("Model produced non-finite logits; reduce the intervention strength.")
                generated = []
                for step in range(body["max_new"]):
                    if cancel.is_set():
                        return None
                    if not torch.isfinite(out.logits[0, -1]).all():
                        raise ValueError("Model produced non-finite logits; reduce the intervention strength.")
                    token = int(out.logits[0, -1].argmax())
                    if token in eos:
                        break
                    generated.append(token)
                    if step + 1 < body["max_new"]:
                        for hook in [observers, *hooks]:
                            hook.offset = len(ids) + len(generated) - 1
                        out = rt.model(input_ids=torch.tensor([[token]], device=rt.dev),
                                       past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
                # Teacher-force entire target continuations from a fresh prefill/cache per target.
                targets = []
                observers.observe = False
                for text, suffix in target_ids:
                    if cancel.is_set():
                        return None
                    for hook in [observers, *hooks]:
                        hook.offset = 0
                    out = rt.model(input_ids=torch.tensor([ids], device=rt.dev), use_cache=True, logits_to_keep=1)
                    score = 0.
                    for k, token in enumerate(suffix):
                        term = float(out.logits[0, -1].float().log_softmax(-1)[token])
                        if not math.isfinite(term):
                            raise ValueError("Model produced non-finite logits; reduce the intervention strength.")
                        score += term
                        if k + 1 < len(suffix):
                            for hook in [observers, *hooks]:
                                hook.offset = len(ids) + k
                            out = rt.model(input_ids=torch.tensor([[token]], device=rt.dev),
                                           past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
                    targets.append({"text": text, "tokens": len(suffix), "logprob": score})
                values, top = lp.topk(min(8, lp.numel()))
                public = {"output": rt.tok.decode(generated, skip_special_tokens=True),
                          "generated_tokens": len(generated), "targets": targets,
                          "top": [{"id": int(i), "token": rt.tok.decode([int(i)]), "prob": float(v.exp())}
                                  for i, v in zip(top, values)]}
                return public, lp, observers.layers, observers.neurons
        yield {"type": "start", "total": len(rows), "batch": rt.batch_info(),
               "model_key": rt._weight_cache_key("residual"), "compute_dtype": rt.compute_dtype,
               "quantization": rt.quantization, "lora": rt.lora_path, "plan": p, "plans": plans}
        for row in rows:
            if cancel.is_set():
                return
            yield {"type": "progress", "id": row["id"], "phase": "baseline"}
            baseline = run(row, False)
            if baseline is None or cancel.is_set():
                return
            yield {"type": "progress", "id": row["id"], "phase": "intervened"}
            intervened = run(row, True)
            if intervened is None or cancel.is_set():
                return
            base, bp, bn, ba = baseline
            alt, ap, an, aa = intervened
            changes = ap.exp() - bp.exp()
            top = changes.abs().topk(min(10, changes.numel())).indices
            neurons = []
            for layer in sorted(ba.keys() & aa.keys()):
                d = aa[layer] - ba[layer]
                for n in d.abs().topk(min(3, d.numel())).indices.tolist():
                    if float(d[n]) != 0:
                        neurons.append({"layer": layer, "n": n, "baseline": float(ba[layer][n]),
                                        "intervened": float(aa[layer][n]), "delta": float(d[n])})
            neurons.sort(key=lambda r: -abs(r["delta"]))
            yield {"type": "result", "id": row["id"], "label": row.get("label", row["id"]), "text": row["text"],
                   "position": len(row["ids"]) - 1,
                   "positions": [plan["positions"][row["id"]] if plan["scope"] == "token" else len(row["ids"]) - 1 for plan in plans],
                   "input_tokens": len(row["ids"]), "baseline": base, "intervened": alt,
                   "kl": max(0., float((bp.exp() * (bp - ap)).sum())),
                   "changes": [{"token": rt.tok.decode([int(i)]), "baseline": float(bp[i].exp()),
                                "intervened": float(ap[i].exp()), "delta": float(changes[i])} for i in top],
                   "layers": [{"layer": l, "baseline": bn[l], "intervened": an[l]} for l in sorted(bn)],
                   "neurons": neurons[:16]}


def prepare_chat_residual(rt, body, rows=None):
    """Resolve source vectors before installing generation hooks; caller holds the runtime lock."""
    p = body["plan"]
    needs_source = p["method"] == "patch" or p.get("direction") == "difference"
    if rows is None:
        rows = {r["id"]: r for r in tokenize_texts(rt, body["texts"], body.get("chat", False))} if needs_source else {}
    def capture(key, position):
        ids = rows[key]["ids"]
        position = len(ids) - 1 if position == -1 else position
        if not 0 <= position < len(ids):
            raise ValueError("Source token is outside the selected source text.")
        with ResidualHooks(rt, {**p, "enabled": False}, len(ids), position, observe=False) as hooks:
            rt.lm(input_ids=torch.tensor([ids], device=rt.dev), use_cache=False)
        return hooks.stages[p["stage"]]
    if p["method"] == "remove":
        return None
    if p["method"] == "patch":
        return capture(p["source"], p["source_pos"])
    if p["direction"] == "neuron":
        vector = rt.down_col(p["layer"], p["neuron"]).detach().float().cpu()
    elif p["direction"] == "custom":
        vector = torch.tensor(p["vector"], dtype=torch.float32)
    else:
        vector = capture(p["source"], p["source_pos"]) - capture(p["reference"], p["reference_pos"])
    if not torch.isfinite(vector).all() or float(vector.norm()) < 1e-12:
        raise ValueError("Direction is zero or invalid; choose a different direction or source pair.")
    return vector / vector.norm()
