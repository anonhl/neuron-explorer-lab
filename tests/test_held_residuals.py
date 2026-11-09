"""Request validation and comparison routing without loading model dependencies."""
import ast
import asyncio
import json
import threading
from contextlib import ExitStack, nullcontext
import math
from pathlib import Path
from types import SimpleNamespace
from types import MethodType
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1] / 'explorer'


def load_function(filename, name, namespace, class_name=None):
    tree = ast.parse((ROOT / filename).read_text(encoding='utf-8'))
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), namespace)
    return namespace[name]


class HeldResidualTests(unittest.TestCase):
    def setUp(self):
        self.ns = {'math': math, 'MAX_TEXTS': 8}
        load_function('residual_interventions.py', 'validate_experiment', self.ns)
        self.ns['ARCH'] = {'n_layers': 32, 'hidden': 4, 'inter': 48}
        self.validate = load_function('server.py', 'validate_held_residuals', self.ns)

    def request(self, **changes):
        return {'texts': [{'id': 'a', 'text': 'hello'}],
                'plan': {'layer': 3, 'neuron': 7, 'stage': 'post', 'method': 'add',
                         'scope': 'continuation', 'direction': 'neuron', 'strength': .2, **changes}}

    def test_multiple_independent_settings_and_invalid_second_setting(self):
        result = self.validate([self.request(), self.request(layer=4, neuron=8, strength=-.3)])
        self.assertEqual([r['plan']['strength'] for r in result], [.2, -.3])
        with self.assertRaises(ValueError):
            self.validate([self.request(), self.request(neuron=48)])
        self.assertEqual(self.validate([]), [])
        with self.assertRaises(ValueError):
            self.validate([self.request()] * 33)

    def test_chat_api_passes_held_settings_in_both_modes_without_neuron_pushes(self):
        calls = []
        def generate(pushes, rows, max_new, cancel, **kwargs):
            calls.append((pushes, rows, kwargs))
            yield 0, 'reply'
            yield None, None
        self.ns.update({
            'Request': object, 'asyncio': asyncio, 'threading': threading, 'json': json,
            'rt': SimpleNamespace(steer_multi=generate),
            'StreamingResponse': lambda events, **kw: events,
            'JSONResponse': lambda body, **kw: (body, kw),
            'validate_held_residuals': self.validate,
            'validate_operations': lambda *a: (_ for _ in ()).throw(AssertionError('empty operations must bypass neuron validation')),
        })
        endpoint = load_function('server.py', 'steer', self.ns)
        async def run(group):
            class Request:
                async def json(inner):
                    return {'group': group, 'operations': [], 'residuals': [self.request(layer=l, neuron=l) for l in range(5)], 'prompt': 'hello'}
                async def is_disconnected(inner):
                    return False
            events = await endpoint(Request())
            return [event async for event in events]
        for group, count in [('evaluation', 3), ('conversation', 1)]:
            with self.subTest(group=group):
                events = asyncio.run(run(group))
                self.assertIn('reply', events[0])
                pushes, rows, kwargs = calls[-1]
                self.assertEqual(pushes, [])
                self.assertEqual(len(rows), count)
                self.assertEqual(kwargs['residuals'][0]['plan']['strength'], .2)
                self.assertEqual([item['plan']['layer'] for item in kwargs['residuals']], list(range(5)))

    def test_token_scope_requires_explicit_chat_position(self):
        for position in (None, True, -1, 1.5):
            with self.subTest(position=position), self.assertRaises(ValueError):
                self.validate([self.request(scope='token', positions={'a': 0}, chat_position=position)])
        self.assertEqual(self.validate([self.request(scope='token', positions={'a': 0}, chat_position=3)])[0]['plan']['chat_position'], 3)

    def test_comparison_preserves_settings_and_maps_rows(self):
        steer = load_function('runtime.py', 'steer_multi', {}, 'Runtime')
        calls = []
        residuals = [self.request(), self.request(layer=4)]
        def single(*args):
            calls.append(args)
            yield 0, 'reply'
            yield None, None
        fake = SimpleNamespace(steer_multi=single, ensure_loaded=lambda: None, lock=nullcontext())
        out = list(steer(fake, [(3, 7, .8, 'add'), (4, 8, 2., 'mul')], [['a'], ['b'], ['c']], residuals=residuals))
        self.assertEqual(out, [(0, 'reply'), (1, 'reply'), (2, 'reply'), (None, None)])
        self.assertEqual([c[0][0][2] for c in calls], [.8, 0., -.8])
        self.assertEqual([c[0][1][2] for c in calls], [2., 1., -2.])
        self.assertIs(calls[0][-1], residuals)
        self.assertIsNone(calls[1][-1])
        self.assertIsNone(calls[2][-1])

    def test_experiment_validates_every_selected_plan(self):
        validate = self.ns['validate_experiment']
        body = self.request()
        plans = [self.request(layer=l)['plan'] for l in range(5)]
        result = validate({**body, 'plans': plans}, 32, 4, 48)
        self.assertEqual([p['layer'] for p in result['plans']], list(range(5)))
        for invalid in ([], plans + [{'layer': 32}], plans * 7, [None]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate({**body, 'plans': invalid}, 32, 4, 48)
        self.assertNotIn('plans', validate(body, 32, 4, 48))  # Legacy single setting stays supported.

    def test_experiment_modes_are_validated_and_legacy_requests_default_to_joint(self):
        validate = self.ns['validate_experiment']
        body = self.request()
        self.assertEqual(validate(body, 32, 4, 48)['mode'], 'joint')
        for mode in ('joint', 'independent'):
            for plans in (None, [body['plan'], self.request(layer=4)['plan']]):
                request = {**body, 'mode': mode, **({'plans': plans} if plans else {})}
                self.assertEqual(validate(request, 32, 4, 48)['mode'], mode)
        for mode in ('unknown', '', None, False, [], {}):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate({**body, 'mode': mode}, 32, 4, 48)

    def test_independent_experiment_isolates_five_settings_across_five_texts(self):
        """Run actual orchestration with instrumented hooks; numerical tests use the tiny Qwen model."""
        active, forwards, prepared = [], [], []
        state = SimpleNamespace(fail_neuron=None)
        class Hook:
            def __init__(self, rt, plan, prompt_len, position, vector=None, **kwargs):
                self.plan, self.position = plan, position
                self.layers, self.neurons = {}, {}
            def __enter__(self): active.append(self); return self
            def __exit__(self, *args): active.remove(self)
        logits = MagicMock()
        lp = logits[0, -1].float().log_softmax(-1).cpu()
        lp.numel.return_value = 1
        lp.topk.return_value = ([], [])
        changes = lp.exp() - lp.exp()
        changes.numel.return_value = 1
        changes.abs().topk.return_value = SimpleNamespace(indices=[])
        def forward(**kwargs):
            edits = [(h.plan['neuron'], h.position) for h in active if h.plan['enabled']]
            forwards.append((kwargs['input_ids'][0], edits))
            if any(neuron == state.fail_neuron for neuron, _ in edits):
                raise RuntimeError('example failure')
            # Each measured run must create its own prefill, without sharing a generation cache.
            self.assertNotIn('past_key_values', kwargs)
            return SimpleNamespace(logits=logits)
        def prepare(rt, body, **kwargs):
            self.assertEqual(active, [])
            prepared.append(body['plan']['neuron'])
        torch = SimpleNamespace(inference_mode=nullcontext, tensor=lambda data, **kw: data,
                                isfinite=lambda value: SimpleNamespace(all=lambda: True))
        self.ns.update({'torch': torch, 'ExitStack': ExitStack, 'ResidualHooks': Hook,
                        'prepare_chat_residual': prepare, 'MAX_TOKENS': 1024})
        load_function('residual_interventions.py', 'tokenize_texts', self.ns)
        run = load_function('residual_interventions.py', 'run_experiment', self.ns)
        tokenizer = SimpleNamespace(eos_token_id=63, convert_tokens_to_ids=lambda _: 63,
                                    decode=lambda ids, **kw: str(ids))
        class Tokens:
            eos_token_id = 63
            convert_tokens_to_ids = staticmethod(tokenizer.convert_tokens_to_ids)
            decode = staticmethod(tokenizer.decode)
            def __call__(self, text, **kw): return SimpleNamespace(input_ids=[int(text), 10])
        rt = SimpleNamespace(lock=threading.RLock(), ensure_loaded=lambda: None, model=forward,
                             tok=Tokens(), token_units=lambda ids: [(str(ids), [0, 1])], dev='cpu',
                             batch_info=lambda: None, _weight_cache_key=lambda _: 'test',
                             compute_dtype='float32', quantization='none', lora_path=None)
        texts = [{'id': str(i), 'text': str(i)} for i in range(5)]
        plans = [self.request(layer=i, neuron=i, scope='token', positions={r['id']: i % 2 for r in texts})['plan']
                 for i in range(5)]
        body = self.ns['validate_experiment']({'mode': 'independent', 'texts': texts, 'plans': plans, 'max_new': 0}, 32, 4, 48)
        events = list(run(rt, body, threading.Event()))
        rows = [e for e in events if e['type'] == 'result']
        self.assertEqual(events[0]['total'], 25)
        self.assertEqual(events[0]['text_count'], 5)
        self.assertEqual(len(events[0]['groups']), 5)
        self.assertEqual(prepared, list(range(5)))
        self.assertEqual(len(forwards), 30)  # Five clean baselines, plus 25 independent interventions.
        self.assertEqual(sum(not edits for _, edits in forwards), 5)
        self.assertEqual([edits for _, edits in forwards if edits], [[(i, i % 2)] for i in range(5) for _ in texts])
        for i in range(5):
            group_rows = [r for r in rows if r['group_id'] == f'setting-{i + 1}']
            self.assertEqual([r['id'] for r in group_rows], [r['id'] for r in texts])
            self.assertTrue(all(r['plan_indices'] == [i] and r['positions'] == [i % 2] for r in group_rows))
        self.assertEqual(active, [])

        forwards.clear()
        joint = list(run(rt, {**body, 'mode': 'joint'}, threading.Event()))
        self.assertEqual(joint[0]['total'], 5)
        self.assertEqual(len([e for e in joint if e['type'] == 'result']), 5)
        self.assertTrue(all(len(edits) in (0, 5) for _, edits in forwards))
        self.assertEqual(active, [])

        cancel = threading.Event()
        generator = run(rt, body, cancel)
        for event in generator:
            if event['type'] == 'result': break
        cancel.set()
        self.assertEqual(list(generator), [])
        self.assertEqual(active, [])
        state.fail_neuron = 2
        with self.assertRaisesRegex(RuntimeError, 'example failure'):
            list(run(rt, body, threading.Event()))
        self.assertEqual(active, [])

    def test_comparison_cancellation_stops_remaining_rows(self):
        steer = load_function('runtime.py', 'steer_multi', {}, 'Runtime')
        cancel = SimpleNamespace(is_set=lambda: True)
        calls = []
        def single(*args):
            calls.append(args)
            yield None, None
        list(steer(SimpleNamespace(steer_multi=single, ensure_loaded=lambda: None, lock=nullcontext()), [], [[], [], []], cancel=cancel, residuals=[self.request()]))
        self.assertEqual(len(calls), 1)

    def test_five_runtime_hooks_reach_forward_in_chat_and_only_configured_evaluation_row(self):
        """Execute the actual generation setup against an instrumented model, without torch installed."""
        active, forwards = [], []
        class Tensor(list):
            def cumsum(self, *a): return self
            def __sub__(self, other): return self
            def clamp(self, **kw): return self
        class Hook:
            def __init__(self, rt, plan, prompt_len, position, vector, **kw):
                self.address = (plan['layer'], plan['neuron'])
                self.position = position
            def __enter__(self): active.append(self); return self
            def __exit__(self, *args): active.remove(self)
        class Model:
            def __call__(self, **kw):
                forwards.append((sorted(h.address for h in active), [h.position for h in active]))
                return SimpleNamespace()
        torch = SimpleNamespace(tensor=lambda data, **kw: Tensor(data),
                                Generator=lambda: SimpleNamespace(manual_seed=lambda seed: None),
                                inference_mode=nullcontext)
        steer = load_function('runtime.py', 'steer_multi', {'torch': torch, 'ExitStack': ExitStack, '__package__': ''}, 'Runtime')
        fake = SimpleNamespace(ensure_loaded=lambda: None, lock=nullcontext(), model=Model(), dev='cpu',
                               messages_ids=lambda messages: [3, 8, 10],
                               tok=SimpleNamespace(pad_token_id=0, convert_tokens_to_ids=lambda token: 63),
                               _push_entry=lambda *a: a)
        fake.steer_multi = MethodType(steer, fake)
        module = SimpleNamespace(ResidualHooks=Hook, prepare_chat_residual=lambda rt, body: body['plan']['neuron'])
        # A token index left over from previous edits must not affect other scopes.
        residuals = [self.request(layer=l, neuron=l, chat_position=100000) for l in range(5)]
        with patch.dict('sys.modules', {'residual_interventions': module}):
            list(fake.steer_multi([], [['hello']], max_new=0, residuals=residuals))
            self.assertEqual(forwards[-1], ([(l, l) for l in range(5)], [2] * 5))
            self.assertEqual(active, [])
            forwards.clear()
            list(fake.steer_multi([], [['a'], ['b'], ['c']], max_new=0, residuals=residuals))
            self.assertEqual([len(addresses) for addresses, positions in forwards], [5, 0, 0])
            self.assertEqual(active, [])
            def broken(**kw): raise RuntimeError('forward failed')
            fake.model = broken
            with self.assertRaisesRegex(RuntimeError, 'forward failed'):
                list(fake.steer_multi([], [['hello']], max_new=0, residuals=residuals))
            self.assertEqual(active, [])
            self.assertIsNone(fake._push)


if __name__ == '__main__':
    unittest.main()
