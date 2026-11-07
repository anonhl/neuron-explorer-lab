"""Small local tensors only: no model downloads or CUDA hardware needed.

Run: uv run --locked --group dev --extra quantization --extra lora python -m unittest discover -s tests -v
The dev group supplies httpx for API tests; no model weights are downloaded.
"""
import importlib
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorer"))

import numpy as np
import torch
import torch.nn.functional as F

import quantization as quant
import runtime


class WeightOnlyTests(unittest.TestCase):
    def test_all_presets_keep_bf16_compute(self):
        self.assertEqual(len({f.id for f in quant.FORMATS}), len(quant.FORMATS))
        self.assertEqual({f.bits for f in quant.FORMATS if f.backend == "hqq"}, {1, 2, 3, 4, 8})
        for fmt in quant.FORMATS:
            with self.subTest(fmt=fmt.id):
                config = quant.make_quantization_config(fmt.id)
                if fmt.backend == "bitsandbytes":
                    self.assertEqual(config.load_in_8bit, fmt.bits == 8)
                    self.assertEqual(config.load_in_4bit, fmt.bits == 4)
                    self.assertEqual(config.bnb_4bit_compute_dtype, torch.bfloat16)
                    self.assertEqual(config.bnb_4bit_use_double_quant, fmt.double_quant)
                    self.assertIn("lm_head", config.llm_int8_skip_modules)
                elif fmt.backend == "hqq":
                    self.assertEqual(config.quant_config["weight_quant_params"]["nbits"], fmt.bits)
                    self.assertIn("lm_head", config.skip_modules)
                else:
                    self.assertIsNone(config)

    def test_invalid_format_and_missing_backend(self):
        with self.assertRaises(ValueError):
            quant.weight_format("hqq-7")
        for dev in ("cpu", "mps"):
            with self.assertRaisesRegex(ValueError, "CUDA"):
                quant.validate_environment("hqq-4", dev)
        with patch.object(quant, "find_spec", return_value=None):
            with self.assertRaisesRegex(ValueError, "uv sync --extra bitsandbytes"):
                quant.validate_environment("bnb-nf4", "cuda")
        quant.validate_environment("none", "cpu")

    def test_all_presets_accept_fp16_compute(self):
        for fmt in quant.FORMATS:
            with self.subTest(fmt=fmt.id):
                config = quant.make_quantization_config(fmt.id, "float16")
                if fmt.backend == "bitsandbytes":
                    self.assertEqual(config.bnb_4bit_compute_dtype, torch.float16)
        for value in ("fp16", "float32", "", None, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                quant.make_quantization_config("none", value)

    def test_fp16_quantized_weights_keep_fp16_activations(self):
        from bitsandbytes.nn import Linear4bit, Linear8bitLt
        from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
        for fmt in quant.FORMATS:
            if fmt.backend == "none":
                continue
            with self.subTest(fmt=fmt.id):
                if fmt.backend == "hqq":
                    quant.make_quantization_config(fmt.id, "float16")  # select the PyTorch backend
                    layer = HQQLinear(torch.nn.Linear(64, 32, bias=False),
                                      quant_config=BaseQuantizeConfig(nbits=fmt.bits, group_size=64, axis=1),
                                      compute_dtype=torch.float16, device="cpu")
                elif fmt.bits == 8:
                    layer = Linear8bitLt(64, 32, bias=False, has_fp16_weights=False).to(torch.float16).to("cpu")
                else:
                    layer = Linear4bit(64, 32, bias=False, compute_dtype=torch.float16,
                                      quant_type=fmt.quant_type, compress_statistics=fmt.double_quant).to(torch.float16).to("cpu")
                quant.prepare_weight_only_model(torch.nn.Sequential(layer), fmt.id, "float16")
                x = torch.randn(1, 3, 64, dtype=torch.float16)
                before = x.clone()
                rt = runtime.Runtime.__new__(runtime.Runtime)
                rt._capture_heads, rt._heads = {"writes": {}, "y": {}}, None
                layer.register_forward_pre_hook(rt._head_hook(0, 4))
                with torch.inference_mode():
                    actual = layer(x)
                    weight = quant.linear_weight(layer)
                    expected = F.linear(x, weight)
                self.assertEqual(weight.dtype, torch.float16)
                self.assertEqual(actual.dtype, torch.float16)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue(torch.equal(x, before))
                torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)
                self.assertEqual(rt._capture_heads["writes"][0].shape, (3, 4))

    def test_bnb_four_bit_storage_and_float_activations(self):
        from bitsandbytes.nn import Linear4bit
        for kind in ("nf4", "fp4"):
            for double_quant in (False, True):
                with self.subTest(kind=kind, double_quant=double_quant):
                    layer = Linear4bit(64, 32, bias=False, compute_dtype=torch.bfloat16,
                                       quant_type=kind, compress_statistics=double_quant).to(torch.bfloat16).to("cpu")
                    x = torch.randn(2, 64, dtype=torch.bfloat16)
                    before = x.clone()
                    with torch.inference_mode():
                        expected = F.linear(x, quant.linear_weight(layer).to(x.dtype))
                        actual = layer(x)
                    self.assertEqual(actual.dtype, torch.bfloat16)
                    self.assertTrue(torch.equal(x, before))
                    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)
                    self.assertLess(layer.weight.numel(), 64 * 32)

    def test_bnb_int8_bypasses_activation_quantization(self):
        from bitsandbytes.nn import Linear8bitLt
        layer = Linear8bitLt(64, 32, bias=True, has_fp16_weights=False).to(torch.bfloat16).to("cpu")
        model = torch.nn.Sequential(layer)
        quant.prepare_weight_only_model(model, "bnb-int8")
        self.assertEqual(layer.weight.dtype, torch.int8)
        x = torch.randn(2, 64, dtype=torch.bfloat16)
        before = x.clone()
        # Standard LLM.int8 quantizes x inside matmul; that path must never run.
        with patch("bitsandbytes.matmul", side_effect=AssertionError("activation quantization used")):
            with torch.inference_mode():
                actual = model(x)
                expected = F.linear(x, quant.linear_weight(layer), layer.bias)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(x, before))
        torch.testing.assert_close(actual, expected)

    def test_hqq_storage_and_explorer_head_capture(self):
        from hqq.core.quantize import HQQBackend, HQQLinear, BaseQuantizeConfig
        HQQLinear.set_backend(HQQBackend.PYTORCH)
        for bits in (8, 4, 3, 2, 1):
            with self.subTest(bits=bits):
                layer = HQQLinear(torch.nn.Linear(64, 32, bias=False),
                                  quant_config=BaseQuantizeConfig(nbits=bits, group_size=64, axis=1),
                                  compute_dtype=torch.bfloat16, device="cpu")
                x = torch.randn(1, 3, 64, dtype=torch.bfloat16)
                before = x.clone()
                rt = runtime.Runtime.__new__(runtime.Runtime)
                rt._capture_heads = {"writes": {}, "y": {}}
                rt._heads = None
                layer.register_forward_pre_hook(rt._head_hook(0, 4))
                with torch.inference_mode():
                    actual = layer(x)
                    weight = quant.linear_weight(layer)
                    expected = F.linear(x, weight)
                self.assertEqual(weight.shape, (32, 64))
                self.assertEqual(actual.dtype, torch.bfloat16)
                self.assertTrue(torch.equal(x, before))
                torch.testing.assert_close(actual, expected)
                writes = torch.einsum("thd,ohd->tho", x[0].view(3, 4, 16), weight.view(32, 4, 16)).float().norm(dim=-1)
                torch.testing.assert_close(rt._capture_heads["writes"][0], writes)


def fake_model():
    layer = torch.nn.Module()
    layer.mlp = torch.nn.Module()
    layer.mlp.down_proj = torch.nn.Linear(64, 32)
    layer.self_attn = torch.nn.Module()
    layer.self_attn.o_proj = torch.nn.Linear(32, 32)
    layer.self_attn.head_dim = 16
    layer.post_attention_layernorm = torch.nn.LayerNorm(32)
    model = SimpleNamespace(model=SimpleNamespace(visual=object(), language_model=SimpleNamespace(layers=[layer])))
    model.eval = lambda: model
    return model


class AgentPrecisionTests(unittest.TestCase):
    def test_default_run_metadata_uses_current_precision(self):
        import agent_mcp
        for dtype, label in (("bfloat16", "bf16"), ("float16", "fp16")):
            with self.subTest(dtype=dtype), \
                 patch.object(agent_mcp, "get", return_value={"compute_dtype": dtype, "quantization": "hqq-4"}), \
                 patch.object(agent_mcp, "neuronary", return_value={}) as publish:
                agent_mcp.call("neuronary_reproduce", {"id": "test", "output": "example"})
                setup = publish.call_args.args[2]["setup"]
                self.assertTrue(setup.startswith(label + " "))
                self.assertIn("weights=hqq-4", setup)
        with patch.object(agent_mcp, "get") as get, patch.object(agent_mcp, "neuronary", return_value={}) as publish:
            agent_mcp.call("neuronary_reproduce", {"id": "test", "output": "example", "setup": "custom setup"})
            get.assert_not_called()
            self.assertEqual(publish.call_args.args[2]["setup"], "custom setup")


class RuntimeLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cache = patch.object(runtime, "CACHE", self.temp.name)
        self.cache.start()
        self.env = patch.dict(os.environ, {"EXPLORER_QUANTIZATION": "none", "EXPLORER_COMPUTE_DTYPE": ""})
        self.env.start()
        self.rt = runtime.Runtime(self.temp.name)

    def tearDown(self):
        self.rt._db.close()
        self.env.stop()
        self.cache.stop()
        self.temp.cleanup()

    def test_optional_lora_is_merged_and_can_be_removed(self):
        path = Path(self.temp.name) / "adapter with spaces"
        path.mkdir()
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            (path / name).touch()
        original, merged, restored = fake_model(), fake_model(), fake_model()
        merge = Mock(return_value=merged)
        config = SimpleNamespace(peft_type="LORA", base_model_name_or_path=runtime.MODEL)
        peft = SimpleNamespace(
            PeftConfig=SimpleNamespace(from_pretrained=Mock(return_value=config)),
            PeftModel=SimpleNamespace(from_pretrained=Mock(
                return_value=SimpleNamespace(merge_and_unload=merge))),
        )
        with patch.dict(sys.modules, {"peft": peft}), \
             patch.object(runtime, "NO_MODEL", False), \
             patch.object(self.rt, "device", return_value="cpu"), \
             patch.object(runtime.AutoTokenizer, "from_pretrained", return_value=object()), \
             patch.object(self.rt, "_load_weights", side_effect=[original, restored]):
            self.rt.select_quantization("none", "float16", str(path))
            self.assertIs(self.rt.model, merged)
            self.assertEqual(self.rt.loading_info()["lora_path"], str(path.resolve()))
            peft.PeftModel.from_pretrained.assert_called_once_with(
                original, str(path.resolve()), is_trainable=False, local_files_only=True)
            merge.assert_called_once_with(safe_merge=True)
            with self.assertRaisesRegex(ValueError, "local LoRA folder"):
                self.rt.select_quantization("none", "float16", str(path / "missing"))
            self.assertIs(self.rt.model, merged)
            self.rt.lora_act_scale = {"tokens": 10}
            self.rt.select_quantization("none", "float16", "")
            self.assertIsNone(self.rt.lora_act_scale)
            self.assertIs(self.rt.model, restored)
            self.assertIsNone(self.rt.loading_info()["lora_path"])

    def test_offline_calibration_measures_the_last_partial_chunk(self):
        self.rt.model = fake_model()
        self.rt.lm = self.rt.model.model.language_model
        width = self.rt.lm.layers[0].mlp.down_proj.in_features
        self.rt.tok = Mock(return_value=SimpleNamespace(input_ids=[1, 2, 3, 4, 5]), eos_token_id=0)
        first, last = torch.full((width,), 2.), torch.full((width,), 3.)
        with patch.object(self.rt, "all_activations", side_effect=[
            {0: (first, None, -last, None)}, {0: (last, None, -first, None)},
        ]) as measure:
            scale = self.rt.build_act_scale(seq=4, texts=["sample"])
        self.assertEqual(scale["tokens"], 6)
        np.testing.assert_array_equal(scale["pos"][0], last.numpy())
        np.testing.assert_array_equal(scale["neg"][0], -last.numpy())
        measure.assert_called_with([5, 0], skip_first=True)

    def test_quantization_passed_to_loader_without_changing_dtype(self):
        for fmt in quant.FORMATS:
            with self.subTest(fmt=fmt.id):
                self.rt.model = None
                self.rt.quantization = fmt.id
                config = object() if fmt.id != "none" else None
                with patch.object(runtime.Runtime, "device", return_value="cpu"), \
                     patch.object(runtime, "NO_MODEL", False), \
                     patch.object(runtime, "validate_environment"), \
                     patch.object(runtime, "make_quantization_config", return_value=config), \
                     patch.object(runtime, "prepare_weight_only_model"), \
                     patch.object(runtime.AutoTokenizer, "from_pretrained", return_value=object()), \
                     patch.object(runtime.Qwen3_5ForConditionalGeneration, "from_pretrained", return_value=fake_model()) as load:
                    self.rt.ensure_loaded()
                kwargs = load.call_args.kwargs
                self.assertEqual(kwargs["dtype"], torch.bfloat16)
                self.assertEqual(kwargs.get("quantization_config"), config)
                self.assertEqual(self.rt.status, "ready")
                self.assertTrue(self.rt.lm.layers[0].mlp.down_proj._forward_pre_hooks)

    def test_missing_cuda_preserves_working_model(self):
        original = self.rt.model = object()
        with patch.object(runtime.Runtime, "device", return_value="cpu"), patch.object(runtime, "NO_MODEL", False):
            with self.assertRaises(ValueError):
                self.rt.select_quantization("hqq-4")
        self.assertIs(self.rt.model, original)
        self.assertEqual(self.rt.quantization, "none")

    def test_fp16_precision_only_switch_reloads_and_persists(self):
        self.rt.model = object()
        self.rt.resid_p50[0] = 1.
        self.rt._active_batch = {"id": "old"}
        with patch.object(runtime.Runtime, "device", return_value="cpu"), \
             patch.object(runtime, "NO_MODEL", False), \
             patch.object(runtime.AutoTokenizer, "from_pretrained", return_value=object()), \
             patch.object(runtime.Qwen3_5ForConditionalGeneration, "from_pretrained", return_value=fake_model()) as load:
            self.rt.select_quantization("none", "float16")
        load.assert_called_once()
        self.assertEqual(load.call_args.kwargs["dtype"], torch.float16)
        self.assertEqual(self.rt.loading_info()["activation_dtype"], "float16")
        self.assertIsNone(self.rt.batch_info())
        self.assertFalse(self.rt.resid_p50)
        saved = json.loads(Path(self.temp.name, "model_loading.json").read_text())
        self.assertEqual(saved, {"quantization": "none", "compute_dtype": "float16"})
        restarted = runtime.Runtime(self.temp.name)
        try:
            self.assertEqual(restarted.compute_dtype, "float16")
        finally:
            restarted._db.close()

    def test_saved_precision_defaults_and_environment_override(self):
        path = Path(self.temp.name, "model_loading.json")
        for saved in ({"quantization": "none"}, {"compute_dtype": "invalid"}, []):
            with self.subTest(saved=saved):
                path.write_text(json.dumps(saved))
                rt = runtime.Runtime(self.temp.name)
                self.assertEqual(rt.compute_dtype, "bfloat16")
                rt._db.close()
        path.write_text(json.dumps({"quantization": "none", "compute_dtype": "float16"}))
        with patch.dict(os.environ, {"EXPLORER_COMPUTE_DTYPE": "bfloat16"}):
            rt = runtime.Runtime(self.temp.name)
            self.assertEqual(rt.compute_dtype, "bfloat16")
            rt._db.close()

    def test_invalid_precision_preserves_working_model(self):
        original = self.rt.model = object()
        with patch.object(runtime, "NO_MODEL", False):
            with self.assertRaisesRegex(ValueError, "compute precision"):
                self.rt.select_quantization("none", "fp16")
        self.assertIs(self.rt.model, original)
        self.assertEqual(self.rt.compute_dtype, "bfloat16")

    def test_fp16_precision_is_passed_to_every_quantized_loader(self):
        self.rt.compute_dtype = "float16"
        for fmt in quant.FORMATS:
            with self.subTest(fmt=fmt.id):
                self.rt.model = None
                self.rt.quantization = fmt.id
                with patch.object(runtime.Runtime, "device", return_value="cuda"), \
                     patch.object(runtime, "NO_MODEL", False), \
                     patch.object(runtime, "validate_environment"), \
                     patch.object(runtime, "make_quantization_config") as config, \
                     patch.object(runtime, "prepare_weight_only_model") as prepare, \
                     patch.object(runtime.AutoTokenizer, "from_pretrained", return_value=object()), \
                     patch.object(runtime.Qwen3_5ForConditionalGeneration, "from_pretrained", return_value=fake_model()) as load:
                    self.rt.ensure_loaded()
                self.assertEqual(load.call_args.kwargs["dtype"], torch.float16)
                config.assert_called_once_with(fmt.id, "float16")
                prepare.assert_called_once_with(self.rt.model, fmt.id, "float16")

    def test_failed_load_is_reported_and_retry_recovers(self):
        with patch.object(runtime.Runtime, "device", return_value="cpu"), \
             patch.object(runtime, "NO_MODEL", False), \
             patch.object(runtime.AutoTokenizer, "from_pretrained", return_value=object()), \
             patch.object(runtime.Qwen3_5ForConditionalGeneration, "from_pretrained", side_effect=OSError("out of memory")):
            with self.assertRaises(OSError):
                self.rt.ensure_loaded()
        self.assertEqual(self.rt.status, "error")
        self.assertEqual(self.rt.load_error, "out of memory")
        self.assertIsNone(self.rt.model)
        with patch.object(runtime.Qwen3_5ForConditionalGeneration, "from_pretrained") as load:
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                self.rt.ensure_loaded()
            load.assert_not_called()
        with patch.object(runtime.Runtime, "device", return_value="cpu"), \
             patch.object(runtime, "NO_MODEL", False), \
             patch.object(runtime.AutoTokenizer, "from_pretrained", return_value=object()), \
             patch.object(runtime.Qwen3_5ForConditionalGeneration, "from_pretrained", return_value=fake_model()):
            self.rt.select_quantization("none")
        self.assertEqual(self.rt.status, "ready")
        self.assertIsNone(self.rt.load_error)
        self.assertTrue(Path(self.temp.name, "model_loading.json").exists())

    def test_weight_progress_counts_completed_items_and_clears_on_failure(self):
        module = runtime.core_model_loading

        def load(*args, **kwargs):
            self.assertEqual(self.rt.loading_info()["progress"],
                             {"phase": "weights", "completed": None, "total": None})
            for i, _ in enumerate(module.tqdm(range(3), desc="Loading weights")):
                self.assertEqual(self.rt.loading_info()["progress"],
                                 {"phase": "weights", "completed": i, "total": 3})
            self.assertEqual(self.rt.loading_info()["progress"]["completed"], 3)
            raise OSError("checkpoint read failed")

        with patch.object(runtime, "NO_MODEL", False), \
             patch.object(runtime.Runtime, "device", return_value="cpu"), \
             patch.object(runtime.AutoTokenizer, "from_pretrained", return_value=object()), \
             patch.object(runtime.Qwen3_5ForConditionalGeneration, "from_pretrained", side_effect=load), \
             patch.object(module, "tqdm", side_effect=lambda items, **kwargs: items) as tqdm:
            with self.assertRaisesRegex(OSError, "checkpoint read failed"):
                self.rt.ensure_loaded()
            self.assertIs(module.tqdm, tqdm)
        self.assertEqual(self.rt.load_error, "checkpoint read failed")
        self.assertIsNone(self.rt.loading_info()["progress"])

    def test_weight_cache_keys_are_isolated(self):
        keys = set()
        for dtype in quant.COMPUTE_DTYPES:
            self.rt.compute_dtype = dtype
            for fmt in quant.FORMATS:
                self.rt.quantization = fmt.id
                keys.add(self.rt._weight_cache_key("1:2:12"))
        self.assertEqual(len(keys), len(quant.FORMATS) * len(quant.COMPUTE_DTYPES))
        self.rt.quantization, self.rt.compute_dtype = "none", "bfloat16"
        self.assertEqual(self.rt._weight_cache_key("1:2:12"), "1:2:12")


class LoadingApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.temp = tempfile.TemporaryDirectory()
        cfg = SimpleNamespace(num_hidden_layers=2, hidden_size=32, intermediate_size=64,
                              layer_types=["linear_attention", "full_attention"], num_attention_heads=2,
                              num_key_value_heads=1, linear_num_key_heads=1, linear_num_value_heads=2)
        import transformers
        with patch.object(runtime, "CACHE", cls.temp.name), \
             patch.dict(os.environ, {"EXPLORER_QUANTIZATION": "none", "EXPLORER_COMPUTE_DTYPE": "bfloat16"}), \
             patch.object(transformers.AutoConfig, "from_pretrained", return_value=SimpleNamespace(get_text_config=lambda: cfg)):
            cls.server = importlib.import_module("server")
        cls.client = TestClient(cls.server.app, base_url="http://127.0.0.1:8791")

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.server.rt._db.close()
        cls.temp.cleanup()

    def test_api_lists_both_backends_and_every_precision(self):
        result = self.client.get("/api/model/loading")
        self.assertEqual(result.status_code, 200)
        data = result.json()
        self.assertEqual(data["activation_dtype"], "bfloat16")
        self.assertEqual(data["compute_dtypes"], ["bfloat16", "float16"])
        self.assertEqual({fmt["backend"] for fmt in data["options"]}, {"none", "bitsandbytes", "hqq"})
        self.assertEqual(len(data["options"]), len(quant.FORMATS))

    def test_api_rejects_invalid_precision(self):
        self.assertEqual(self.client.post("/api/model/loading", json={"quantization": "hqq-7"}).status_code, 400)
        self.assertEqual(self.client.post("/api/model/loading", json={"quantization": "none", "compute_dtype": "float32"}).status_code, 400)
        self.assertEqual(self.client.post("/api/model/loading", json={}).status_code, 422)

    def test_api_passes_selected_compute_precision(self):
        with patch.object(self.server.rt, "select_quantization") as select, \
             patch.object(self.server, "_prepare_live_model"):
            response = self.client.post("/api/model/loading", json={"quantization": "none", "compute_dtype": "float16"})
        self.assertEqual(response.status_code, 200)
        select.assert_called_once_with("none", "float16", None)

    def test_api_forwards_optional_lora_path(self):
        adapter = r"C:\模型\我的 LoRA"
        with patch.object(self.server.rt, "select_quantization") as select, \
             patch.object(self.server, "_prepare_live_model"):
            response = self.client.post("/api/model/loading", json={
                "quantization": "none", "compute_dtype": "float16", "lora_path": adapter,
            })
        self.assertEqual(response.status_code, 200)
        select.assert_called_once_with("none", "float16", adapter)

    def test_lora_calibration_is_used_for_token_activation_ranking(self):
        server, rt = self.server, self.server.rt
        scale = {"pos": np.array([[2., 10.]]), "neg": np.array([[-2., -10.]]), "tokens": 6}
        baseline = {"pos": np.array([[10., 2.]]), "neg": np.array([[-10., -2.]]), "tokens": 6}
        acts = np.array([[[0., 0.], [4., 5.]]], dtype=np.float32)
        tok = Mock(return_value=SimpleNamespace(input_ids=[1, 2]))
        with patch.object(rt, "lora_path", "adapter"), patch.object(rt, "lora_act_scale", None), \
             patch.object(rt, "_load_state"), patch.object(rt, "tok", tok), \
             patch.object(rt, "ensure_loaded"), patch.object(rt, "vocab_strings"), \
             patch.object(rt, "build_act_scale", return_value=scale) as calibrate, \
             patch.object(rt, "full_activations", return_value=(acts, {"model_key": "residual", "batch": None})), \
             patch.object(rt, "token_units", return_value=[("我", [0]), ("猫", [1])]), \
             patch.object(server, "_act", baseline), patch.object(server, "_infer", {}):
            self.assertIsNone(server.active_act_scale())
            server._prepare_live_model()
            self.assertTrue(calibrate.call_args.kwargs["texts"])
            self.assertEqual(rt.status, "ready")
            result = self.client.post("/api/infer", json={"text": "我猫"}).json()
            self.assertEqual(result["tokens"], ["我", "猫"])
            ranked = self.client.get("/api/infer/view?pos=1&min=0").json()["results"]
            self.assertEqual([r["n"] for r in ranked], [0, 1])
            self.assertEqual(ranked[0]["score"], 2)
            self.assertEqual(ranked[0]["act"], 4)
            self.assertEqual(server.act_range(0, 0)["source"], "LoRA calibration")
            self.assertEqual(self.client.get("/api/infer/paths?pos=1").content, b"")
        self.assertIsNone(rt.lora_act_scale)

    def test_startup_and_live_requests_wait_for_explicit_loading(self):
        from fastapi.testclient import TestClient

        server = self.server
        cache = self.enterContext(tempfile.TemporaryDirectory())
        with patch.object(runtime, "CACHE", cache), \
             patch.object(runtime, "NO_MODEL", False), patch.object(server, "NO_MODEL", False):
            rt = runtime.Runtime(cache, load_on_demand=server.rt.load_on_demand)
            self.addCleanup(rt._db.close)
            with patch.object(server, "rt", rt), patch.object(rt, "device", return_value="cpu"), \
                 patch.object(server, "LENS_INDEX", __file__), patch.object(server, "ACT_SCALE", __file__), \
                 patch.object(rt, "vocab_strings", return_value=[]), \
                 patch.object(runtime.AutoTokenizer, "from_pretrained", return_value=object()) as tokenizer, \
                 patch.object(runtime.Qwen3_5ForConditionalGeneration, "from_pretrained", return_value=fake_model()) as load, \
                 TestClient(server.app, base_url="http://127.0.0.1:8791") as client:
                self.assertEqual(client.get("/").status_code, 200)
                self.assertEqual(client.get("/api/model/loading").json()["status"], "idle")
                self.assertEqual(client.post("/api/reads", json={"layer": 0, "n": 0, "text": "hello"}).status_code, 409)
                self.assertEqual(client.get("/api/fires", params={"text": "hello"}).status_code, 409)
                tokenizer.assert_not_called()
                load.assert_not_called()

                result = client.post("/api/model/loading", json={"quantization": "none", "compute_dtype": "float16"})
                self.assertEqual(result.status_code, 200, result.text)
                load.assert_called_once()
                self.assertEqual(load.call_args.kwargs["dtype"], torch.float16)
                self.assertEqual(client.get("/api/status").json()["status"], "ready")

    def test_busy_model_does_not_switch(self):
        with self.server.rt.lock:
            response = self.client.post("/api/model/loading", json={"quantization": "none"})
        self.assertEqual(response.status_code, 503)
        self.assertFalse(self.server._model_switching)

    def test_switch_blocks_other_inference_requests(self):
        with patch.object(self.server, "_model_switching", True):
            self.assertEqual(self.client.get("/api/infer/view").status_code, 409)
            self.assertEqual(self.client.get("/api/status").status_code, 200)

    def test_status_reports_preparation_failure_without_blocking_cached_routes(self):
        rt = self.server.rt

        def prepare_vocab():
            status = self.client.get("/api/status").json()
            self.assertEqual(status["status"], "loading")
            self.assertEqual(status["loading"]["progress"]["phase"], "vocab")
            raise OSError("cannot read vocabulary")

        try:
            with patch.object(rt, "ensure_loaded"), patch.object(rt, "vocab_strings", side_effect=prepare_vocab), \
                 patch.object(self.server, "LENS_INDEX", __file__):
                with self.assertRaisesRegex(OSError, "cannot read vocabulary"):
                    self.server._prepare_live_model()
            status = self.client.get("/api/status").json()
            self.assertEqual(status["status"], "error")
            self.assertEqual(status["error"], "cannot read vocabulary")
            self.assertIsNone(status["loading"]["progress"])
            self.assertEqual(self.client.get("/api/units?layer=0").status_code, 200)
        finally:
            rt.set_load_state("idle")

    def test_successful_switch_clears_stale_captures(self):
        self.server._infer["old"] = True
        self.server._hprof["old"] = True
        with patch.object(self.server.rt, "select_quantization") as select, \
             patch.object(self.server, "_prepare_live_model"):
            response = self.client.post("/api/model/loading", json={"quantization": "hqq-4"})
        self.assertEqual(response.status_code, 200)
        select.assert_called_once_with("hqq-4", None, None)
        self.assertFalse(self.server._infer)
        self.assertFalse(self.server._hprof)
        self.assertFalse(self.server._model_switching)

    def test_disconnected_request_does_not_release_loading_guard(self):
        started, release = threading.Event(), threading.Event()

        def slow_load(_, compute_dtype=None, lora_path=None):
            started.set()
            if not release.wait(3):
                raise RuntimeError("test timed out")

        async def run():
            task = asyncio.create_task(self.server.set_model_loading(self.server.ModelLoadingRequest(quantization="none")))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(self.server._model_switching)
            finally:
                release.set()
            async def finished():
                while self.server._model_switching:
                    await asyncio.sleep(0.01)
            await asyncio.wait_for(finished(), 2)

        with patch.object(self.server.rt, "select_quantization", side_effect=slow_load), \
             patch.object(self.server, "_prepare_live_model"):
            asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
