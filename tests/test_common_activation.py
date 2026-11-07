"""Common activation selection without loading model weights."""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest


class CommonActivationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'explorer/server.py').read_text(encoding='utf-8'))
        fn = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == 'infer_common')
        fn.decorator_list = []
        self.seen = []
        async def score(text):
            raise AssertionError("Filtering cached activations must never run the model")
        self.summaries = {text: {"score": text} for text in ('a', 'b', 'c')}
        async def view(snapshot, **kwargs):
            self.seen.append(kwargs)
            values = {'a': [(1, 4), (2, 3), (3, 2)], 'b': [(1, 2), (2, 5)], 'c': [(2, 4)]}[snapshot['score']]
            return {'results': [dict(layer=0, n=n, score=value) for n, value in values]}
        self.ns = dict(Request=object, rt=SimpleNamespace(tok=object(), token_units=lambda ids: [('t', [0])]),
                       active_act_scale=lambda: True, score_text=score, _infer_view=view,
                       _configure_inference_cache=lambda: None,
                       _inference_cache=SimpleNamespace(summary=lambda key: self.summaries.get(key[0])),
                       ARCH={'n_layers': 2, 'inter': 500}, JSONResponse=lambda body, status_code: (status_code, body))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<common>', 'exec'), self.ns)

    async def test_returning_to_unchanged_text_reuses_capture(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'explorer/server.py').read_text(encoding='utf-8'))
        fn = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == 'infer')
        fn.decorator_list = []
        snapshot = {'capture': 'original-a', 'units': [('a-token', [0])]}
        self.ns.update(asyncio=asyncio, _infer_lock=asyncio.Lock(),
            _inference_cache=SimpleNamespace(retain=lambda keys: None, restore=lambda key: (snapshot, {'ids': [1]})))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<infer>', 'exec'), self.ns)
        async def json():
            return {'text': 'a', 'texts': ['a', 'b']}
        for _ in range(3):
            result = await self.ns['infer'](SimpleNamespace(json=json))
            self.assertEqual(result['capture'], 'original-a')
            self.assertEqual(result['tokens'], ['a-token'])

    async def run_request(self, **body):
        async def json():
            return body
        return await self.ns['infer_common'](SimpleNamespace(json=json))

    async def test_intersection_uses_weakest_peak_and_full_candidates(self):
        result = await self.run_request(texts=['a', 'b'])
        self.assertEqual([(r['n'], r['score']) for r in result['results']], [(2, 3), (1, 2)])
        self.assertEqual(result['total'], 2)
        self.assertTrue(all(k['limit'] == 1000 for k in self.seen))

    async def test_all_texts_and_address_filter(self):
        result = await self.run_request(texts=['a', 'b', 'c'], addresses=[{'layer': 0, 'n': 1}])
        self.assertEqual(result['results'], [])
        result = await self.run_request(texts=['a', 'b', 'c'])
        self.assertEqual([r['n'] for r in result['results']], [2])

    async def test_missing_text_does_not_run_model(self):
        result = await self.run_request(texts=['a', 'new text'])
        self.assertEqual(result['message'], 'results.commonPending')
        self.assertEqual(self.seen, [])

    async def test_repeated_filter_only_reads_cached_results(self):
        for _ in range(3):
            result = await self.run_request(texts=['a', 'b'])
            self.assertEqual(result['total'], 2)

    async def test_requires_multiple_filled_texts(self):
        for texts in [['a'], ['a', ' '], ['a'] * 9]:
            self.assertEqual((await self.run_request(texts=texts))[0], 400)


if __name__ == '__main__':
    unittest.main()
