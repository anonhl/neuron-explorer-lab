"""Grouped context reads must reuse captures and preserve active single-text state."""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest


class CachedContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'explorer/server.py').read_text(encoding='utf-8'))
        self.inspections = {}
        self.calls = []
        self.restores = []
        self.snapshots = {text: {'ids': [0, 1], 'toks': [text, '!'], 'capture': text + '-original',
            'units': [(text, [0]), ('!', [1])], 'residuals': {}} for text in ['a', 'b']}
        def summary(key):
            return self.snapshots.get(key[0])
        def restore(key, include_heads=True):
            self.restores.append(key)
            self.assertFalse(include_heads)
            return (self.snapshots[key[0]], {'private-heads': True}) if key[0] in self.snapshots else None
        def inspect(rt, snapshot, pos, layer, n):
            self.calls.append((snapshot['capture'], pos, layer, n))
            return {'capture': snapshot['capture'], 'pos': pos, 'token': snapshot['toks'][pos]}
        cache = SimpleNamespace(summary=summary, restore=restore, inspected=self.inspections.get,
            save_inspection=lambda key, value: self.inspections.__setitem__(key, value))
        active = {'capture': 'active-original'}
        self.ns = dict(Request=object, asyncio=asyncio, _inference_cache=cache,
            _configure_inference_cache=lambda: None, ARCH={'n_layers': 2, 'inter': 10},
            inspect_residual=inspect, rt=SimpleNamespace(heads='active-heads'), _infer=active,
            JSONResponse=lambda body, status_code: (status_code, body))
        for name in ['infer_contexts', 'cached_text_tokens']:
            fn = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == name)
            fn.decorator_list = []
            exec(compile(ast.Module(body=[fn], type_ignores=[]), '<cached-context>', 'exec'), self.ns)

    async def request(self, name, **body):
        async def json():
            return body
        return await self.ns[name](SimpleNamespace(json=json))

    async def test_grouped_reads_are_separate_and_do_not_replace_active_capture(self):
        result = await self.request('infer_contexts', texts=['a', 'b'])
        self.assertEqual([row['capture'] for row in result['contexts']], ['a-original', 'b-original'])
        self.assertEqual([row['text'] for row in result['contexts']], ['a', 'b'])
        self.assertEqual(self.ns['_infer']['capture'], 'active-original')
        self.assertEqual(self.ns['rt'].heads, 'active-heads')
        await self.request('infer_contexts', texts=['a', 'b'])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(self.restores), 2)

    async def test_token_selection_only_reads_requested_text(self):
        result = await self.request('infer_contexts', texts=['a'], pos=0)
        self.assertEqual(result['contexts'][0]['residual']['token'], 'a')
        self.assertEqual(self.calls, [('a-original', 0, 0, -1)])

    async def test_missing_text_never_runs_model(self):
        result = await self.request('infer_contexts', texts=['missing'])
        self.assertFalse(result['ready'])
        self.assertEqual(self.restores, [])
        self.assertEqual(self.calls, [])

    async def test_token_groups_read_only_small_summaries(self):
        result = await self.request('cached_text_tokens', texts=['a', 'b', 'missing'])
        self.assertEqual([row['tokens'] for row in result['texts']], [['a', '!'], ['b', '!'], []])
        self.assertEqual([row['ready'] for row in result['texts']], [True, True, False])
        self.assertEqual(self.restores, [])
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
