"""Batch persistence, matching and real small-tensor hook tests; no model downloads required."""
import importlib
import importlib.util
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
from neuron_batches import BatchStore, validate_operations


def operation(**changes):
    return {"layer": 0, "n": 1, "mode": "add", "dose": 0.8, "enabled": True, **changes}


class BatchStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name, "配置", "neuron_batches.json")
        self.store = BatchStore(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, group_id, **kw):
        return self.store.save(group_id=group_id, name="数学实验", notes="第一批修改", model="test-model",
                               model_revision="abc", quantization="none",
                               operations=validate_operations([operation(**kw)], 2, 4))

    def test_groups_and_batches_survive_reopening(self):
        group = self.store.create_group("推理能力")
        second_group = self.store.create_group("写作风格")
        batch = self.save(group["id"], dose=-0.8)
        copy = self.save(second_group["id"])
        reopened = BatchStore(self.path)
        data = reopened.list()
        self.assertEqual([g["name"] for g in data["groups"]], ["推理能力", "写作风格"])
        self.assertEqual(reopened.get(batch["id"]), batch)
        self.assertNotEqual(batch["id"], copy["id"])
        self.assertEqual(batch["group_id"], group["id"])
        self.assertEqual(batch["operations"][0]["unit"], "residual_norm")
        self.assertTrue(batch["created_at"].endswith("+00:00"))
        self.assertEqual(batch["model_revision"], "abc")
        self.assertEqual(batch["quantization"], "none")
        self.assertEqual(batch["compute_dtype"], "bfloat16")
        self.assertIn("推理能力", self.path.read_text(encoding="utf-8"))

    def test_user_must_create_and_select_group(self):
        with self.assertRaisesRegex(ValueError, "existing group"):
            self.save("missing")
        for name in ("", "  ", "x" * 121):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.store.create_group(name)
        self.store.create_group(" Test ")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.store.create_group("test")

    def test_failed_write_preserves_previous_data(self):
        group = self.store.create_group("实验")
        self.save(group["id"])
        original = self.path.read_bytes()
        with patch("neuron_batches.os.replace", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                self.save(group["id"], dose=2)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_concurrent_saves_keep_every_batch(self):
        from concurrent.futures import ThreadPoolExecutor
        group = self.store.create_group("并发")
        with ThreadPoolExecutor(max_workers=4) as pool:
            saved = list(pool.map(lambda _: self.save(group["id"]), range(12)))
        self.assertEqual(len(self.store.list()["batches"]), 12)
        self.assertEqual(len({b["id"] for b in saved}), 12)

    def test_invalid_values_and_units_are_rejected(self):
        for change in ({"layer": -1}, {"layer": 2}, {"n": 4}, {"n": True}, {"layer": 0.0},
                       {"dose": float("nan")}, {"dose": float("inf")}, {"dose": True},
                       {"mode": "set"}, {"enabled": "yes"}, {"unit": "raw"}, {"enabled": False}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_operations([operation(**change)], 2, 4)
        for ops in ([], [operation()] * 1001):
            with self.assertRaises(ValueError):
                validate_operations(ops, 2, 4)

    def test_damaged_file_is_not_silently_overwritten(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("[]", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.store.create_group("new")
        self.assertEqual(json.loads(self.path.read_text()), [])


HAS_RUNTIME = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "numpy"))


@unittest.skipUnless(HAS_RUNTIME, "Runtime dependencies are not installed")
class BatchRuntimeTests(unittest.TestCase):
    def setUp(self):
        import torch
        import runtime
        self.torch, self.runtime = torch, runtime
        self.temp = tempfile.TemporaryDirectory()
        self.cache = patch.object(runtime, "CACHE", self.temp.name)
        self.cache.start()
        with patch.dict(os.environ, {"EXPLORER_QUANTIZATION": "none", "EXPLORER_COMPUTE_DTYPE": "bfloat16"}):
            self.rt = runtime.Runtime(self.temp.name)
        layer = SimpleNamespace(mlp=SimpleNamespace(down_proj=torch.nn.Linear(4, 2, bias=False)))
        with torch.no_grad():
            layer.mlp.down_proj.weight.fill_(1)
            layer.mlp.down_proj.weight[:, 1] = torch.tensor([2., 0.])
        self.rt.lm = SimpleNamespace(layers=[layer])
        self.rt.model, self.rt.dev = object(), "cpu"
        self.rt.resid_p50[0] = 10.
        self.loaded = patch.object(self.rt, "ensure_loaded")
        self.loaded.start()

    def tearDown(self):
        self.loaded.stop()
        self.rt._db.close()
        self.cache.stop()
        self.temp.cleanup()

    def batch(self, *ops, **changes):
        return {"id": "batch1", "name": "测试", "model": self.runtime.MODEL, "quantization": "none",
                "operations": validate_operations(list(ops) or [operation()], 1, 4), **changes}

    def run_hook(self, tensor):
        changed = self.rt._hook(0)(None, (tensor,))
        return changed[0] if changed else tensor

    def test_batch_persists_across_forwards_and_captures_modified_values(self):
        torch = self.torch
        self.rt.use_batch(self.batch(operation()))  # .8 * residual 10 / column norm 2 = +4
        before = torch.ones(3, 2, 4, dtype=torch.bfloat16)
        captured = {}
        self.rt._read, self.rt._capture_full = ({0: [1]}, captured), {}
        for _ in range(2):
            after = self.run_hook(before)
            self.assertEqual(after.dtype, torch.bfloat16)
            torch.testing.assert_close(after[:, :, 1], torch.full((3, 2), 5., dtype=torch.bfloat16))
            torch.testing.assert_close(captured[0][0], torch.full((2,), 5.))
            torch.testing.assert_close(self.rt._capture_full[0][:, 1], torch.full((2,), 5.))
        torch.testing.assert_close(before, torch.ones_like(before))

    def test_read_multiple_neurons_in_one_forward(self):
        torch = self.torch
        self.rt.chat_ids = Mock(return_value=[1, 2])
        self.rt.token_units = Mock(return_value=[("第一", [0]), ("第二", [1])])

        def forward(**_):
            self.rt._hook(0)(None, (torch.tensor([[[1., 2., 3.], [4., 5., 6.]]]),))
            self.rt._hook(1)(None, (torch.tensor([[[7., 8., 9.], [10., 11., 12.]]]),))

        self.rt.lm = Mock(side_effect=forward)
        results = self.rt.read_many([(0, 2), (1, 1), (0, 0)], "示例")
        self.assertEqual([(row["layer"], row["n"]) for row in results], [(0, 2), (1, 1), (0, 0)])
        self.assertEqual([[u["act"] for u in row["units"]] for row in results], [[3., 6.], [8., 11.], [1., 4.]])
        self.rt.lm.assert_called_once()
        self.assertIsNone(self.rt._read)
        self.assertEqual(self.rt.read(0, 2, "示例"), {"units": results[0]["units"]})

    def test_batch_applies_to_all_comparison_rows_with_temporary_push_on_top(self):
        torch = self.torch
        self.rt.use_batch(self.batch(operation(), operation(n=0, mode="mul", dose=2)))
        self.rt._push = [self.rt._push_entry(0, 1, .2, "add", (1, 0, -1)),
                         self.rt._push_entry(0, 0, -3, "mul", (1, 0, -1))]
        after = self.run_hook(torch.ones(3, 2, 4))
        torch.testing.assert_close(after[:, 0, 1], torch.tensor([6., 5., 4.]))
        torch.testing.assert_close(after[:, 0, 0], torch.tensor([-6., 2., 6.]))
        self.rt._push = None  # Same cleanup as steer_multi's finally.
        self.assertEqual(float(self.run_hook(torch.ones(1, 1, 4))[0, 0, 1]), 5.)

    def test_conversation_generates_one_row_with_both_signed_adjustments(self):
        torch = self.torch
        self.rt.tok = SimpleNamespace(pad_token_id=0, convert_tokens_to_ids=lambda _: 3,
                                      decode=lambda ids, **_: "reply")
        self.rt.messages_ids = lambda _: [1]

        def forward(input_ids, **kwargs):
            self.assertEqual(input_ids.shape, (1, 1))
            activations = self.run_hook(torch.ones(1, 1, 4))
            torch.testing.assert_close(activations[0, 0, :2], torch.tensor([-2., 3.]))
            logits = [-10., -10., 10., -10.] if kwargs.get("past_key_values") is None else [-10., -10., -10., 10.]
            return SimpleNamespace(logits=torch.tensor([[logits]]), past_key_values=object())

        self.rt.model = Mock(side_effect=forward, lm_head=SimpleNamespace(out_features=4))
        pushes = [(0, 0, -2, "mul"), (0, 1, .4, "add")]
        messages = [[{"role": "user", "content": "Hello"}]]
        self.assertEqual(list(self.rt.steer_multi(pushes, messages, max_new=2, repetition_penalty=1.1)),
                         [(0, "reply"), (None, None)])
        self.assertIsNone(self.rt._push)

    def test_replacement_order_disabled_operations_and_stop(self):
        torch = self.torch
        self.rt.use_batch(self.batch(operation()))
        self.rt.use_batch(self.batch(operation(mode="mul", dose=0), operation(dose=-.4),
                                           operation(mode="mul", dose=-2), operation(dose=100, enabled=False)))
        self.assertEqual(float(self.run_hook(torch.ones(1, 1, 4))[0, 0, 1]), 4.)
        self.rt.clear_batch()
        self.assertIsNone(self.rt.batch_info())
        self.assertEqual(float(self.run_hook(torch.ones(1, 1, 4))[0, 0, 1]), 1.)

    def test_failed_application_preserves_active_batch(self):
        self.rt.use_batch(self.batch(operation()))
        previous = self.rt.batch_info()
        for batch in (self.batch(model="different"), self.batch(quantization="hqq-4"), self.batch(operation(dose=1e100))):
            with self.assertRaises(ValueError):
                self.rt.use_batch(batch)
            self.assertEqual(self.rt.batch_info(), previous)
        copied = self.rt.batch_info()
        copied["name"] = "modified"
        self.assertEqual(self.rt.batch_info(), previous)

    def test_fp16_batch_capture_range_and_legacy_matching(self):
        torch = self.torch
        self.rt.compute_dtype = "float16"
        # A legacy batch has no precision field and belongs to the original BF16 model.
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.rt.use_batch(self.batch())
        self.rt.use_batch(self.batch(compute_dtype="float16"))
        captured = {}
        self.rt._read = ({0: [1]}, captured)
        after = self.run_hook(torch.ones(1, 2, 4, dtype=torch.float16))
        self.assertEqual(after.dtype, torch.float16)
        torch.testing.assert_close(after[:, :, 1], torch.full((1, 2), 5., dtype=torch.float16))
        torch.testing.assert_close(captured[0][0], torch.full((2,), 5.))
        previous = self.rt.batch_info()
        with self.assertRaisesRegex(ValueError, "activation range"):
            self.rt.use_batch(self.batch(operation(mode="mul", dose=70000), compute_dtype="float16"))
        self.assertEqual(self.rt.batch_info(), previous)

    def test_weight_reload_stops_active_batch(self):
        self.rt.use_batch(self.batch(operation()))
        with patch.object(self.runtime, "NO_MODEL", False), patch.object(self.runtime, "validate_environment"), \
             patch.object(self.runtime, "make_quantization_config"), patch.object(self.runtime.torch.cuda, "is_available", return_value=False), \
             patch.object(self.runtime.torch.backends.mps, "is_available", return_value=False):
            self.rt.select_quantization("hqq-4")
        self.assertIsNone(self.rt.batch_info())
        self.assertEqual(self.rt._batch_push, ())

    def test_calibration_ignores_active_and_temporary_interventions(self):
        torch = self.torch
        self.rt.use_batch(self.batch(operation(mode="mul", dose=2)))
        previous_batch = self.rt._batch_push
        previous_push = self.rt._push = [(0, 1, "add", torch.tensor([1.]))]
        self.rt.resid_p50.clear()
        self.rt.tok = lambda *a, **kw: SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]]))
        def forward(**kwargs):
            self.assertEqual(self.rt._batch_push, ())
            self.assertIsNone(self.rt._push)
            self.rt._seen["resid"] = torch.tensor([10., 20., 30.])
        self.rt.lm = forward
        self.assertEqual(self.rt.layer_resid_p50(0), 20.)
        self.assertEqual(self.rt._batch_push, previous_batch)
        self.assertIs(self.rt._push, previous_push)
        self.assertNotIn("want_resid", self.rt._seen)


@unittest.skipUnless(HAS_RUNTIME and importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"),
                     "API dependencies are not installed")
class BatchApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import runtime
        import transformers
        cls.runtime = runtime
        cls.import_temp = tempfile.TemporaryDirectory()
        cfg = SimpleNamespace(num_hidden_layers=2, hidden_size=32, intermediate_size=64,
                              layer_types=["linear_attention", "full_attention"], num_attention_heads=2,
                              num_key_value_heads=1, linear_num_key_heads=1, linear_num_value_heads=2)
        with patch.object(runtime, "CACHE", cls.import_temp.name), \
             patch.object(transformers.AutoConfig, "from_pretrained", return_value=SimpleNamespace(get_text_config=lambda: cfg)):
            cls.server = importlib.import_module("server")

    @classmethod
    def tearDownClass(cls):
        cls.import_temp.cleanup()

    def setUp(self):
        from fastapi.testclient import TestClient
        self.temp = tempfile.TemporaryDirectory()
        with patch.object(self.runtime, "CACHE", self.temp.name), \
             patch.dict(os.environ, {"EXPLORER_QUANTIZATION": "none", "EXPLORER_COMPUTE_DTYPE": "bfloat16"}):
            self.rt = self.runtime.Runtime(self.temp.name)
        self.patches = [patch.object(self.server, "rt", self.rt),
                        patch.object(self.server, "batches", BatchStore(Path(self.temp.name, "batches.json"))),
                        patch.object(self.server, "NO_MODEL", False), patch.object(self.server, "MODEL_REVISION", None)]
        for p in self.patches:
            p.start()
        self.client = TestClient(self.server.app, base_url="http://127.0.0.1:8791")

    def tearDown(self):
        self.client.close()
        for p in reversed(self.patches):
            p.stop()
        self.rt._db.close()
        self.temp.cleanup()

    def save_batch(self):
        group = self.client.post("/api/neuron-batches/groups", json={"name": "我的分组"}).json()
        payload = {"group_id": group["id"], "name": "批次1", "model": self.server.MODEL,
                   "quantization": "none", "operations": [operation()]}
        response = self.client.post("/api/neuron-batches", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json(), payload

    def test_read_activation_addresses_are_validated_and_deduplicated(self):
        url = "/api/reads"
        with patch.object(self.rt, "read_many", return_value=[]) as read:
            for neurons in ([], [{"layer": 0, "n": True}], [{"layer": 2, "n": 1}], [{"layer": 0, "n": 64}]):
                self.assertEqual(self.client.post(url, json={"neurons": neurons, "text": "hi"}).status_code, 400)
            read.assert_not_called()

            neurons = [{"layer": 1, "n": 5}, {"layer": 0, "n": 2}, {"layer": 1, "n": 5}]
            response = self.client.post(url, json={"neurons": neurons, "text": "hi"})
            self.assertEqual(response.json(), {"results": []})
            read.assert_called_once_with([(1, 5), (0, 2)], "hi", True)

    def test_group_save_list_and_reopen(self):
        batch, _ = self.save_batch()
        data = self.client.get("/api/neuron-batches").json()
        self.assertEqual(data["groups"][0]["name"], "我的分组")
        self.assertEqual(data["batches"][0], batch)
        self.assertIsNone(data["active_batch"])
        with patch.object(self.server, "batches", BatchStore(Path(self.temp.name, "batches.json"))):
            self.assertEqual(self.client.get("/api/neuron-batches").json()["batches"][0], batch)

    def test_reused_batch_requires_an_operation_change(self):
        original, payload = self.save_batch()
        draft = {**payload, "name": "副本", "notes": "修改备注"}

        response = self.client.post("/api/neuron-batches", json={**draft, "source_batch_id": original["id"]})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Change a neuron operation", response.json()["error"])

        with patch.object(self.rt, "batch_info", return_value=original):
            response = self.client.post("/api/neuron-batches", json=draft)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.server.batches.list()["batches"], [original])

        response = self.client.post("/api/neuron-batches", json={**draft, "source_batch_id": original["id"],
                                                                 "operations": [operation(dose=-0.4)]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotEqual(response.json()["id"], original["id"])
        self.assertEqual(self.server.batches.get(original["id"]), original)

    def test_rename_and_delete_active_batch(self):
        batch, _ = self.save_batch()
        path = f"/api/neuron-batches/{batch['id']}"
        self.rt._active_batch = {**batch, "applied_at": "unchanged"}
        self.rt._batch_push = ((0, 1, "add", 0.8),)
        self.server._infer["old"], self.server._hprof["old"] = True, True

        self.assertEqual(self.client.patch(path, json={"name": " "}).status_code, 400)
        response = self.client.patch(path, json={"name": " 新名称 "})
        self.assertEqual(response.status_code, 200, response.text)
        renamed = {**batch, "name": "新名称"}
        self.assertEqual(self.server.batches.get(batch["id"]), renamed)
        self.assertEqual(self.rt.batch_info(), {**renamed, "applied_at": "unchanged"})
        self.assertTrue(self.rt._batch_push)
        self.assertTrue(self.server._infer)
        self.assertTrue(self.server._hprof)

        with self.rt.lock:
            self.assertEqual(self.client.delete(path).status_code, 409)
        with patch("neuron_batches.os.replace", side_effect=OSError("disk unavailable")):
            self.assertEqual(self.client.delete(path).status_code, 500)
        self.assertEqual(self.server.batches.get(batch["id"]), renamed)
        self.assertEqual(self.rt.batch_info()["id"], batch["id"])

        response = self.client.delete(path)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(response.json()["active_batch"])
        self.assertFalse(self.rt._batch_push)
        self.assertFalse(self.server._infer)
        self.assertFalse(self.server._hprof)
        data = self.server.batches.list()
        self.assertEqual(data["batches"], [])
        self.assertEqual(data["groups"][0]["id"], batch["group_id"])
        self.assertEqual(self.client.delete(path).status_code, 404)
        self.assertEqual(self.client.patch(path, json={"name": "找不到"}).status_code, 404)

    def test_invalid_groups_model_revision_and_parameters(self):
        batch, payload = self.save_batch()
        for changes, status in (({"group_id": "bad"}, 400), ({"operations": [operation(n=-1)]}, 400),
                                ({"model": "bad"}, 409), ({"quantization": "hqq-4"}, 409),
                                ({"model_revision": "bad"}, 409), ({"compute_dtype": "float16"}, 409)):
            with self.subTest(changes=changes):
                self.assertEqual(self.client.post("/api/neuron-batches", json={**payload, **changes}).status_code, status)
        with patch.object(self.rt, "quantization", "hqq-4"), patch.object(self.rt, "use_batch") as use:
            self.assertEqual(self.client.post(f"/api/neuron-batches/{batch['id']}/use", json={}).status_code, 409)
            use.assert_not_called()

    def test_fp16_batch_save_and_matching(self):
        legacy, payload = self.save_batch()
        self.rt.compute_dtype = "float16"
        response = self.client.post("/api/neuron-batches", json={**payload, "compute_dtype": "float16"})
        self.assertEqual(response.status_code, 200, response.text)
        batch = response.json()
        self.assertEqual(batch["compute_dtype"], "float16")
        self.assertEqual(self.client.get("/api/neuron-batches").json()["compute_dtype"], "float16")
        with patch.object(self.rt, "use_batch") as use:
            self.assertEqual(self.client.post(f"/api/neuron-batches/{legacy['id']}/use", json={}).status_code, 409)
            use.assert_not_called()
            self.assertEqual(self.client.post(f"/api/neuron-batches/{batch['id']}/use", json={}).status_code, 200)
            use.assert_called_once()

    def test_use_stop_status_and_stale_capture_invalidation(self):
        batch, _ = self.save_batch()
        def apply(saved):
            self.rt._active_batch = saved
        self.server._infer["old"], self.server._hprof["old"] = True, True
        with patch.object(self.rt, "use_batch", side_effect=apply):
            response = self.client.post(f"/api/neuron-batches/{batch['id']}/use", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get("/api/status").json()["active_batch"]["id"], batch["id"])
        self.assertFalse(self.server._infer)
        self.assertFalse(self.server._hprof)
        self.assertEqual(self.client.post("/api/neuron-batches/stop", json={}).status_code, 200)
        self.assertIsNone(self.rt.batch_info())
        self.assertEqual(len(self.client.get("/api/neuron-batches").json()["batches"]), 1)

    def test_busy_switching_missing_batch_and_offline_model(self):
        batch, _ = self.save_batch()
        with self.rt.lock:
            self.assertEqual(self.client.post(f"/api/neuron-batches/{batch['id']}/use", json={}).status_code, 409)
        self.assertEqual(self.client.post("/api/neuron-batches/missing/use", json={}).status_code, 404)
        with patch.object(self.server, "_model_switching", True):
            self.assertEqual(self.client.post(f"/api/neuron-batches/{batch['id']}/use", json={}).status_code, 409)
        with patch.object(self.server, "NO_MODEL", True):
            self.assertEqual(self.client.post(f"/api/neuron-batches/{batch['id']}/use", json={}).status_code, 400)

    def test_chat_accepts_multiple_operations_and_legacy_single_neuron(self):
        history = [{"user": "Previous", "replies": ["Up", "Baseline", "Down"]}]
        operations = [operation(dose=-0.25), operation(layer=1, n=5, mode="mul", dose=2),
                      operation(n=9, enabled=False)]
        with patch.object(self.rt, "steer_multi", return_value=iter([(0, "First"), (None, None)])) as run:
            response = self.client.post("/api/steer", json={"prompt": "Continue", "operations": operations,
                                                           "history": history})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn('"t": "First"', response.text)
            pushes, rows, *_ = run.call_args.args
            self.assertEqual(pushes, [(0, 1, -0.25, "add"), (1, 5, 2.0, "mul")])
            self.assertEqual([row[1]["content"] for row in rows], history[0]["replies"])
        with patch.object(self.rt, "steer_multi", return_value=iter([(None, None)])) as run:
            response = self.client.post("/api/steer", json={"prompt": "Hello", "layer": "0", "n": "1", "dose": "0.5"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(run.call_args.args[0], [(0, 1, 0.5, "add")])
        with patch.object(self.rt, "steer_multi") as run:
            response = self.client.post("/api/steer", json={"prompt": "Hello", "operations": [operation(n=-1)]})
            self.assertEqual(response.status_code, 400)
            run.assert_not_called()

    def test_chat_groups_replay_their_own_number_of_replies(self):
        operations = [operation(dose=0.8), operation(layer=1, n=5, dose=-0.4)]
        history = [{"user": "Previous", "replies": ["Only reply"]}]
        payload = {"prompt": "Continue", "operations": operations, "history": history}
        with patch.object(self.rt, "steer_multi", return_value=iter([(0, "Next"), (None, None)])) as run:
            response = self.client.post("/api/steer", json={**payload, "group": "conversation"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn('"t": "Next"', response.text)
            pushes, rows, *_ = run.call_args.args
            self.assertEqual(pushes, [(0, 1, 0.8, "add"), (1, 5, -0.4, "add")])
            self.assertEqual(rows, [[{"role": "user", "content": "Previous"},
                                     {"role": "assistant", "content": "Only reply"},
                                     {"role": "user", "content": "Continue"}]])
        with patch.object(self.rt, "steer_multi") as run:
            response = self.client.post("/api/steer", json={**payload, "group": "evaluation"})
            self.assertEqual(response.status_code, 400)
            run.assert_not_called()
        with patch.object(self.rt, "steer_multi", return_value=iter([(None, None)])) as run:
            response = self.client.post("/api/steer", json={"group": "evaluation", "prompt": "Hello",
                                                           "operations": [operation(dose=0.8)]})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(len(run.call_args.args[1]), 3)

    def test_conversation_accepts_empty_operations_only_for_plain_chat(self):
        payload = {"prompt": "Hello", "operations": []}
        with patch.object(self.rt, "steer_multi", return_value=iter([(0, "Hi"), (None, None)])) as run:
            response = self.client.post("/api/steer", json={**payload, "group": "conversation"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn('"t": "Hi"', response.text)
            self.assertEqual(run.call_args.args[:2], ([], [[{"role": "user", "content": "Hello"}]]))
        with patch.object(self.rt, "steer_multi") as run:
            for group in ("evaluation", None):
                with self.subTest(group=group):
                    response = self.client.post("/api/steer", json={**payload, "group": group})
                    self.assertEqual(response.status_code, 400)
            run.assert_not_called()

    def test_exact_activation_lookup_includes_weak_and_zero_scores(self):
        import numpy as np
        score = np.zeros((2, 2, 64), dtype=np.float32)
        score[0, 1, 1], score[1, 0, 2], score[1, 1, 2] = -0.25, 0.1, 0.05
        capture = {"score": score, "acts": score * 2, "toks": ["a", "b"]}
        with patch.object(self.server, "_infer", capture), patch.object(self.server, "lens_index", return_value=None):
            filters = {"addresses": "1:2,0:1,0:3,1:2", "lo": 5, "min": 2, "sign": "+", "limit": 1}
            response = self.client.get("/api/infer/view", params=filters)
            self.assertEqual(response.status_code, 200, response.text)
            rows = response.json()["results"]
            self.assertEqual([(r["layer"], r["n"], r["token"]) for r in rows], [(1, 2, "a"), (0, 1, "b"), (0, 3, "a")])
            self.assertEqual(rows[1]["act"], -0.5)
            self.assertEqual(rows[2]["act"], 0)
            response = self.client.get("/api/infer/view", params={**filters, "pos": 1})
            self.assertEqual([r["token"] for r in response.json()["results"]], ["b", "b", "b"])
            self.assertEqual(self.client.get("/api/infer/view", params={"addresses": "2:3"}).status_code, 400)
            normal = self.client.get("/api/infer/view", params={"sign": "+", "min": 0.06}).json()
            self.assertEqual([(r["layer"], r["n"]) for r in normal["results"]], [(1, 2)])


if __name__ == "__main__":
    unittest.main()
