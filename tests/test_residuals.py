"""Residual conservation and contextual neuron writes on a real, tiny hybrid Qwen model."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'explorer'))
HAS_DEPS = all(importlib.util.find_spec(m) for m in ('torch', 'transformers', 'numpy'))


@unittest.skipUnless(HAS_DEPS, 'Runtime dependencies are not installed')
class ResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        import runtime
        from transformers import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig, Qwen3_5ForConditionalGeneration
        cls.np, cls.torch, cls.runtime = np, torch, runtime
        torch.manual_seed(14)
        config = Qwen3_5Config(
            text_config=Qwen3_5TextConfig(hidden_size=32, intermediate_size=48, num_hidden_layers=32,
                num_attention_heads=2, num_key_value_heads=1, head_dim=16, vocab_size=64,
                linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
                layer_types=['linear_attention', 'full_attention'] * 16,
                rope_parameters={'rope_type': 'default', 'rope_theta': 10000., 'partial_rotary_factor': .5,
                                 'mrope_section': [2, 1, 1]}).to_dict(),
            vision_config=Qwen3_5VisionConfig(depth=1, hidden_size=32, intermediate_size=48, num_heads=2,
                out_hidden_size=32, num_position_embeddings=16).to_dict())
        cls.model = Qwen3_5ForConditionalGeneration(config).eval()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cache = patch.object(self.runtime, 'CACHE', self.temp.name)
        self.cache.start()
        self.rt = self.runtime.Runtime(self.temp.name)
        self.rt.model, self.rt.lm, self.rt.dev = self.model, self.model.model.language_model, 'cpu'
        self.rt.tok = type('Tokens', (), {'decode': lambda _, ids: f'token-{ids[0]}'})()
        self.hooks = []
        for i, layer in enumerate(self.rt.lm.layers):
            self.hooks += [layer.mlp.down_proj.register_forward_pre_hook(self.rt._hook(i)),
                           layer.register_forward_pre_hook(self.rt._resid_input_hook(i), with_kwargs=True),
                           layer.post_attention_layernorm.register_forward_hook(self.rt._mlp_in_hook(i)),
                           layer.register_forward_hook(self.rt._resid_hook(i))]
        self.ids = [3, 8, 10]

    def tearDown(self):
        for h in self.hooks: h.remove()
        self.rt._db.close()
        self.cache.stop()
        self.temp.cleanup()

    def snapshot(self):
        acts, trace = self.rt.full_activations(self.ids, residuals=True)
        return {'capture': 'test', 'model_key': trace['model_key'], 'residuals': trace, 'batch': trace['batch'],
                'acts': acts, 'score': acts / .01, 'toks': ['one', 'two', 'three']}

    def test_held_residual_changes_chat_logits_and_cleans_up(self):
        from residual_interventions import validate_experiment
        self.rt.messages_ids = lambda messages: [3, 8, 10]
        self.rt.tok = type('Tokens', (), {
            'pad_token_id': 0, 'convert_tokens_to_ids': lambda _, token: 63,
            'decode': lambda _, ids, **kw: ' '.join(map(str, ids))})()
        body = validate_experiment({
            'texts': [{'id': 'a', 'text': 'A'}], 'max_new': 0,
            'plan': dict(layer=31, neuron=0, method='add', direction='custom',
                         vector=[1.] + [0.] * 31, stage='post', scope='continuation', strength=.5)
        }, 32, 32, 48)
        # A custom direction needs no source tokenizer; the source texts remain validation metadata.
        self.rt.tok.__class__.__call__ = lambda _, text: SimpleNamespace(input_ids=[3, 8, 10])
        self.rt.token_units = lambda ids: [('A', list(range(len(ids))))]
        logits = []
        handle = self.model.register_forward_hook(lambda mod, args, output: logits.append(output.logits.detach().clone()))
        counts = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in self.model.modules()]
        try:
            list(self.rt.steer_multi([], [[{'role': 'user', 'content': 'A'}]], max_new=1))
            base = logits[-1]
            list(self.rt.steer_multi([], [[{'role': 'user', 'content': 'A'}]], max_new=1, residuals=[body]))
            self.assertFalse(self.torch.allclose(base, logits[-1]))
            self.assertEqual(counts, [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in self.model.modules()])
            list(self.rt.steer_multi([], [[{'role': 'user', 'content': 'A'}]], max_new=1))
            self.torch.testing.assert_close(base, logits[-1])
            generator = self.rt.steer_multi([], [[{'role': 'user', 'content': 'A'}]], max_new=3, residuals=[body])
            next(generator)
            generator.close()
            self.assertEqual(counts, [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in self.model.modules()])
            self.assertIsNone(self.rt._push)
        finally:
            handle.remove()

    def test_all_32_layers_and_residual_conservation(self):
        from residuals import residual_rows
        snap = self.snapshot()
        self.assertEqual(snap['acts'].shape, (32, 3, 48))
        rows = residual_rows(snap['residuals'], 2)
        self.assertEqual([r['layer'] for r in rows], list(range(32)))
        for l, row in enumerate(rows):
            w = self.rt.lm.layers[l].mlp.down_proj.weight.detach().numpy()
            self.np.testing.assert_allclose(row['post'] - row['mid'], snap['acts'][l, 2] @ w.T, atol=2e-8)
            if l:
                self.np.testing.assert_array_equal(row['pre'], rows[l - 1]['post'])
        self.assertIsNone(self.rt._capture_residuals)

    def test_five_selected_layers_are_jointly_intervened_and_baseline_stays_clean(self):
        import threading
        from residual_interventions import run_experiment, validate_experiment
        self.rt.tok = type('Tokens', (), {
            'eos_token_id': 63,
            '__call__': lambda _, text, **kw: SimpleNamespace(input_ids=[3, 8, 10]),
            'decode': lambda _, ids, **kw: ' '.join(map(str, ids)),
            'convert_tokens_to_ids': lambda _, text: 63})()
        self.rt.token_units = lambda ids: [('A', list(range(len(ids))))]
        plans = [self.residual_plan(layer=l, neuron=l % 48, direction='neuron', scope='all', strength=.4)
                 for l in (2, 7, 13, 21, 31)]
        body = validate_experiment({'texts': [{'id': 'a', 'text': 'A'}], 'plans': plans, 'max_new': 0}, 32, 32, 48)
        counts = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in self.model.modules()]
        events = list(run_experiment(self.rt, body, threading.Event()))
        result = next(e for e in events if e['type'] == 'result')
        self.assertGreater(result['kl'], 0)
        self.assertEqual([p['layer'] for p in events[0]['plans']], [2, 7, 13, 21, 31])
        # The layers before the first intervention preserve their residual norms.
        for row in result['layers'][:2]:
            self.assertAlmostEqual(row['baseline'], row['intervened'])
        self.assertEqual(counts, [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in self.model.modules()])
        zero = validate_experiment({**body, 'plans': [{**p, 'strength': 0} for p in plans]}, 32, 32, 48)
        result = next(e for e in run_experiment(self.rt, zero, threading.Event()) if e['type'] == 'result')
        self.assertEqual(result['kl'], 0)

    def test_signed_write_and_local_token_delta_without_rerunning_model(self):
        from residuals import inspect_residual
        snap = self.snapshot()
        with patch.object(self.rt.lm, 'forward', side_effect=AssertionError('Must reuse capture')):
            view = inspect_residual(self.rt, snap, 2, 5, 7)
        self.np.testing.assert_allclose(view['selected']['write']['vector'],
            snap['acts'][5, 2, 7] * self.rt.down_col(5, 7).detach().numpy())
        self.assertEqual(len(view['layers']), 32)
        from base64 import b64decode
        from residuals import residual_rows
        packet = view['map_vectors']
        self.assertEqual(packet['shape'], [65, 32])
        vectors = self.np.frombuffer(b64decode(packet['data']), dtype='<f4').reshape(packet['shape'])
        for l, row in enumerate(residual_rows(snap['residuals'], 2)):
            self.np.testing.assert_array_equal(vectors[l * 2], row['pre'])
            self.np.testing.assert_array_equal(vectors[l * 2 + 1], row['mid'])
            self.np.testing.assert_array_equal(vectors[l * 2 + 2], row['post'])
        for name in ('pre', 'mid', 'post'):
            self.assertEqual(len(view['stages'][name]['vector']), 32)
        tokens = view['selected']['write']['tokens']
        self.assertTrue(all(t['score'] > 0 for t in tokens['positive']))
        self.assertTrue(all(t['score'] < 0 for t in tokens['negative']))
        self.assertTrue(all(r['layer'] == 6 for r in view['readers']))
        self.assertTrue(all(t['value'] > 0 for t in view['selected']['write']['positive']))
        self.assertTrue(all(t['value'] < 0 for t in view['selected']['write']['negative']))

    def test_disabled_neuron_has_no_write_no_token_effect_no_reader_effect(self):
        from residuals import inspect_residual
        self.rt._batch_push = ((0, 7, 'mul', 0.),)
        view = inspect_residual(self.rt, self.snapshot(), 2, 0, 7)
        self.assertEqual(view['selected']['act'], 0.)
        self.assertEqual(view['selected']['write']['norm'], 0.)
        self.assertEqual(view['selected']['write']['tokens'], {'positive': [], 'negative': []})
        self.assertTrue(all(r['local_delta'] == 0 for r in view['readers']))

    def test_last_layer_and_stale_weights_or_batch(self):
        from residuals import inspect_residual
        snap = self.snapshot()
        self.assertEqual(inspect_residual(self.rt, snap, 2, 31)['readers'], [])
        with patch.object(self.rt, '_weight_cache_key', return_value='other'):
            with self.assertRaisesRegex(ValueError, 'stale'):
                inspect_residual(self.rt, snap, 2, 1)
        with patch.object(self.rt, 'batch_info', return_value={'id': 'other'}):
            with self.assertRaisesRegex(ValueError, 'stale'):
                inspect_residual(self.rt, snap, 2, 1)

    def test_capture_cleans_up_after_forward_error(self):
        with patch.object(self.rt.lm, 'forward', side_effect=RuntimeError('test failure')):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                self.rt.full_activations(self.ids, residuals=True)
        self.assertIsNone(self.rt._capture_residuals)
        self.assertIsNone(self.rt._capture_full)
        self.assertIsNone(self.rt._capture_heads)

    def residual_plan(self, **changes):
        return {'layer': 5, 'stage': 'post', 'method': 'add', 'scope': 'last', 'strength': .1,
                'direction': 'neuron', 'neuron': 7, 'positions': {}, 'enabled': True, **changes}

    def forward_with_residual(self, plan, vector=None, ids=None):
        from residual_interventions import ResidualHooks
        ids = ids or self.ids
        with self.torch.inference_mode(), ResidualHooks(self.rt, plan, len(ids), len(ids) - 1, vector) as hooks:
            out = self.rt.model(input_ids=self.torch.tensor([ids]), use_cache=False).logits.clone()
        return out, hooks

    def test_residual_zero_dose_all_stages_and_cleanup_on_exception(self):
        from residual_interventions import ResidualHooks
        with self.torch.inference_mode():
            original = self.rt.model(input_ids=self.torch.tensor([self.ids]), use_cache=False).logits
        v = self.torch.ones(32) / 32 ** .5
        for stage in ('pre', 'mid', 'post'):
            out, hooks = self.forward_with_residual(self.residual_plan(stage=stage, strength=0), v)
            self.torch.testing.assert_close(out, original, atol=0, rtol=0)
            self.assertEqual(hooks.handles, [])
        count = len(self.rt.lm.layers[5]._forward_hooks)
        with self.assertRaisesRegex(RuntimeError, 'forced'):
            with ResidualHooks(self.rt, self.residual_plan(), 3, 2, v):
                raise RuntimeError('forced')
        self.assertEqual(len(self.rt.lm.layers[5]._forward_hooks), count)

    def test_actual_neuron_write_removal_matches_neuron_ablation(self):
        removed, _ = self.forward_with_residual(self.residual_plan(method='remove', component='neuron', strength=1., scope='all'))
        self.rt._batch_push = ((5, 7, 'mul', 0.),)
        try:
            with self.torch.inference_mode():
                ablated = self.rt.model(input_ids=self.torch.tensor([self.ids]), use_cache=False).logits
        finally:
            self.rt._batch_push = ()
        self.torch.testing.assert_close(removed, ablated, atol=2e-6, rtol=2e-5)

    def test_mid_intervention_changes_both_mlp_input_and_residual_bypass(self):
        _, hooks = self.forward_with_residual(self.residual_plan(method='remove', component='attention', stage='mid', strength=1.))
        self.torch.testing.assert_close(hooks.stages['mid'], hooks.stages['pre'], atol=2e-8, rtol=2e-6)
        with self.torch.inference_mode():
            layer = self.rt.lm.layers[5]
            mid = hooks.stages['mid']
            expected = mid + layer.mlp(layer.post_attention_layernorm(mid))
        self.torch.testing.assert_close(hooks.stages['post'], expected, atol=2e-8, rtol=2e-6)

    def test_patch_identity_and_projection_formula_at_last_layer(self):
        vector = self.torch.ones(32) / 32 ** .5
        base, captured = self.forward_with_residual(self.residual_plan(enabled=False, layer=31))
        donor = captured.stages['post']
        same, _ = self.forward_with_residual(self.residual_plan(layer=31, method='patch', strength=1.), donor)
        self.torch.testing.assert_close(same, base, atol=0, rtol=0)
        edited, hooks = self.forward_with_residual(self.residual_plan(layer=31, method='erase', strength=1.), vector)
        expected = donor - self.torch.dot(donor, vector) * vector
        self.torch.testing.assert_close(hooks.stages['post'], expected)
        self.assertLess(abs(float(self.torch.dot(hooks.stages['post'], vector))), 1e-7)
        self.assertFalse(self.torch.equal(edited[:, -1], base[:, -1]))
        self.torch.testing.assert_close(edited[:, :-1], base[:, :-1], atol=0, rtol=0)

    def test_multiple_text_experiments_full_candidate_scores_and_no_leaked_hooks(self):
        import threading
        from residual_interventions import run_experiment, validate_experiment
        class Tokens:
            eos_token_id = 63
            def __call__(self, text, **kwargs):
                return SimpleNamespace(input_ids={'A': [3, 8, 10], 'B': [3, 11, 9, 12], 'answer': [5, 6]}[text])
            def decode(self, ids, **kwargs):
                return ' '.join(f'token-{i}' for i in ids)
            def convert_tokens_to_ids(self, _): return 63
        self.rt.tok = Tokens()
        body = validate_experiment({'texts': [{'id': 'a', 'label': 'A', 'text': 'A'}, {'id': 'b', 'label': 'B', 'text': 'B'}],
            'plan': self.residual_plan(method='patch', source='a', source_pos=-1, strength=1.),
            'max_new': 2, 'targets': ['answer']}, 32, 32, 48)
        events = list(run_experiment(self.rt, body, threading.Event()))
        rows = [e for e in events if e['type'] == 'result']
        self.assertEqual([r['id'] for r in rows], ['a', 'b'])
        self.assertEqual([r['input_tokens'] for r in rows], [3, 4])
        self.assertEqual(rows[0]['kl'], 0.)
        self.assertEqual(rows[0]['baseline'], rows[0]['intervened'])
        self.assertGreater(rows[1]['kl'], 0.)
        self.assertEqual(rows[0]['baseline']['targets'][0]['tokens'], 2)
        with self.torch.inference_mode():
            out = self.rt.model(input_ids=self.torch.tensor([[3, 8, 10, 5, 6]]), use_cache=False).logits
            score = float(out[0, 2].log_softmax(-1)[5] + out[0, 3].log_softmax(-1)[6])
        self.assertAlmostEqual(rows[0]['baseline']['targets'][0]['logprob'], score, places=5)
        self.assertIsNone(self.rt._capture_residuals)
        self.assertIsNone(self.rt._push)

    def test_residual_validation_rejects_ambiguous_and_invalid_requests(self):
        from residual_interventions import validate_experiment
        body = {'texts': [{'id': 'a', 'text': 'A'}], 'plan': self.residual_plan(), 'max_new': 0}
        for change in ({'strength': float('nan')}, {'method': 'patch', 'source': 'missing'},
                       {'method': 'remove', 'component': 'mlp', 'stage': 'mid'}, {'scope': 'token'},
                       {'direction': 'custom', 'vector': [1, 2]}, {'layer': -1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_experiment({**body, 'plan': {**body['plan'], **change}}, 32, 32, 48)

    def test_head_and_mlp_write_removal_for_both_attention_types(self):
        for l in (4, 5):
            layer = self.rt.lm.layers[l]
            attention = layer.self_attn if hasattr(layer, 'self_attn') else layer.linear_attn
            proj = attention.o_proj if hasattr(attention, 'o_proj') else attention.out_proj
            width = proj.in_features // self.rt.n_heads(l)
            for component in ('head', 'mlp'):
                with self.subTest(layer=l, component=component):
                    removed, _ = self.forward_with_residual(self.residual_plan(layer=l, method='remove',
                        component=component, head=0, stage='mid' if component == 'head' else 'post', strength=1., scope='all'))
                    def zero_head(mod, args):
                        h = args[0].clone(); h[..., :width] = 0
                        return (h,)
                    handle = (proj.register_forward_pre_hook(zero_head) if component == 'head' else
                              layer.mlp.register_forward_hook(lambda mod, args, out: self.torch.zeros_like(out)))
                    try:
                        with self.torch.inference_mode():
                            expected = self.rt.model(input_ids=self.torch.tensor([self.ids]), use_cache=False).logits
                    finally:
                        handle.remove()
                    self.torch.testing.assert_close(removed, expected, atol=2e-6, rtol=2e-5)

    def test_token_scope_pre_and_continuation_cache_positions(self):
        from residual_interventions import ResidualHooks
        v = self.torch.ones(32) / 32 ** .5
        p = self.residual_plan(stage='pre', scope='token', positions={'a': 1})
        with self.torch.inference_mode():
            with ResidualHooks(self.rt, {**p, 'enabled': False}, 3, 1, v) as base:
                self.rt.model(input_ids=self.torch.tensor([self.ids]), use_cache=False)
            with ResidualHooks(self.rt, p, 3, 1, v) as alt:
                self.rt.model(input_ids=self.torch.tensor([self.ids]), use_cache=False)
        expected = base.stages['pre'] + .1 * base.stages['pre'].norm() * v
        self.torch.testing.assert_close(alt.stages['pre'], expected)
        h = self.torch.tensor([[[3., 4.]]])
        with self.torch.inference_mode():
            last = ResidualHooks(self.rt, self.residual_plan(scope='last'), 3, 2, self.torch.tensor([1., 0.]))
            repeat = ResidualHooks(self.rt, self.residual_plan(scope='continuation'), 3, 2, self.torch.tensor([1., 0.]))
            last.offset = repeat.offset = 3
            self.torch.testing.assert_close(last._edit(h), h, atol=0, rtol=0)
            self.torch.testing.assert_close(repeat._edit(h), self.torch.tensor([[[3.5, 4.]]]))

    @unittest.skipUnless(importlib.util.find_spec('fastapi') and importlib.util.find_spec('httpx'), 'API dependencies missing')
    def test_api_residual_experiment_tokens_stream_and_error_cleanup(self):
        import importlib
        import json
        import transformers
        from fastapi.testclient import TestClient
        cfg = self.model.config.text_config
        with patch.object(transformers.AutoConfig, 'from_pretrained', return_value=SimpleNamespace(get_text_config=lambda: cfg)):
            server = importlib.import_module('server')
        self.rt.tok = type('Tokens', (), {
            'eos_token_id': 63,
            '__call__': lambda _, text, **kwargs: SimpleNamespace(input_ids=[3, 8, 10] if text == 'A' else [3, 11]),
            'decode': lambda _, ids, **kwargs: ' '.join(f'token-{i}' for i in ids),
            'convert_tokens_to_ids': lambda _, text: 63})()
        self.rt.token_units = lambda ids: [(f'token-{i}', [t]) for t, i in enumerate(ids)]
        body = {'texts': [{'id': 'a', 'label': '甲', 'text': 'A'}, {'id': 'b', 'label': '乙', 'text': 'B'}],
                'plan': self.residual_plan(strength=0), 'max_new': 0}
        with patch.object(server, 'rt', self.rt), patch.object(server, 'ARCH', {'n_layers': 32, 'hidden': 32, 'inter': 48}), \
             TestClient(server.app, base_url='http://127.0.0.1:8791') as client:
            out = client.post('/api/residual/tokens', json={'texts': body['texts']})
            self.assertEqual(out.status_code, 200, out.text)
            self.assertEqual([r['count'] for r in out.json()['texts']], [3, 2])
            response = client.post('/api/residual/experiment', json=body)
            self.assertEqual(response.status_code, 200, response.text)
            events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
            self.assertEqual([e['id'] for e in events if e['type'] == 'result'], ['a', 'b'])
            self.assertEqual(events[-1]['type'], 'done')
            self.assertTrue(all(e['kl'] == 0 for e in events if e['type'] == 'result'))
            self.assertEqual(client.post('/api/residual/experiment', json={**body, 'plan': {**body['plan'], 'layer': 32}}).status_code, 400)
            broken = {**body, 'plan': {**body['plan'], 'direction': 'custom', 'vector': [0.] * 32}}
            response = client.post('/api/residual/experiment', json=broken)
            events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
            self.assertEqual([e['type'] for e in events], ['error', 'done'])
            self.assertIsNone(self.rt._capture_residuals)

    @unittest.skipUnless(importlib.util.find_spec('fastapi') and importlib.util.find_spec('httpx'), 'API dependencies missing')
    def test_api_full_layer_capture_export_validation_and_invalidation(self):
        import importlib
        import transformers
        from fastapi.testclient import TestClient
        cfg = self.model.config.text_config
        with patch.object(transformers.AutoConfig, 'from_pretrained', return_value=SimpleNamespace(get_text_config=lambda: cfg)):
            server = importlib.import_module('server')
        self.rt.tok = type('Tokens', (), {
            '__call__': lambda _, text: SimpleNamespace(input_ids=[3, 8, 10]),
            'decode': lambda _, ids: f'token-{ids[0]}'})()
        self.rt.token_units = lambda ids: [(f'token-{i}', [t]) for t, i in enumerate(ids)]
        scale = {'pos': self.np.ones((32, 48)), 'neg': -self.np.ones((32, 48))}
        with patch.object(server, 'rt', self.rt), patch.object(server, 'ARCH', {'n_layers': 32, 'inter': 48}), \
             patch.object(server, 'active_act_scale', return_value=scale), patch.object(server, '_infer', {}), \
             TestClient(server.app, base_url='http://127.0.0.1:8791') as client:
            capture = client.post('/api/infer', json={'text': 'test'}).json()
            self.assertTrue(capture['ready'])
            params = {'capture': capture['capture'], 'pos': 2, 'layer': 31}
            result = client.get('/api/infer/residual', params=params)
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(len(result.json()['layers']), 32)
            exported = client.get('/api/infer/residual/export', params={'capture': capture['capture'], 'pos': 2})
            self.assertEqual(exported.status_code, 200, exported.text)
            self.assertEqual(len(exported.json()['layers']), 32)
            self.assertEqual(len(exported.json()['layers'][31]['post']), 32)
            for change in ({'pos': 3}, {'layer': 32}, {'n': 48}):
                self.assertEqual(client.get('/api/infer/residual', params={**params, **change}).status_code, 400)
            self.assertEqual(client.get('/api/infer/residual', params={**params, 'pos': -1}).status_code, 422)
            client.post('/api/infer', json={'text': 'new'})
            self.assertEqual(client.get('/api/infer/residual', params=params).status_code, 409)
            server._infer.clear()
            self.assertEqual(client.get('/api/infer/residual/export', params=params).status_code, 409)


if __name__ == '__main__':
    unittest.main()
