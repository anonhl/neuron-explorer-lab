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
from unittest.mock import patch

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
