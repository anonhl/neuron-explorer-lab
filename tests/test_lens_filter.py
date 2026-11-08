"""Lens filtering on synthetic activations, without loading model weights."""
import ast
import asyncio
import base64
from pathlib import Path
import re
from types import SimpleNamespace
import unittest

import numpy as np


class LensFilterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.vocab = ['other', 'Cute', 'Indifferent', '健康', '乐观']
        up = np.zeros((1, 8, 7), dtype=np.int32)
        down = np.zeros_like(up)
        up[0, 0, :2] = [1, 2]
        up[0, 1, 0] = 1
        up[0, 2, 0] = 2
        down[0, 3, :2] = [1, 2]
        down[0, 4, 0] = 1
        up[0, 4, :2] = [1, 2]  # Opposite polarity cannot supply a missing term.
        down[0, 5, 0] = 2
        up[0, 6, 6] = 1  # Only the first six lens tokens are searched.
        up[0, 7, :2] = [3, 4]
        scores = np.array([[[8, 7, 6, -5, -4, -3, 2, 1]]], dtype=np.float32)
        self.snapshot = dict(score=scores, acts=scores, toks=['input'])
        self.ns = dict(np=np, re=re, asyncio=asyncio, _infer=self.snapshot,
                       lens_index=lambda: dict(up=up, down=down), names=lambda: {},
                       rt=SimpleNamespace(vocab_strings=lambda: self.vocab))
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'explorer/server.py').read_text(encoding='utf-8'))
        for name in ['_infer_view', 'infer_view']:
            fn = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == name)
            fn.decorator_list = []
            exec(compile(ast.Module(body=[fn], type_ignores=[]), '<lens-filter>', 'exec'), self.ns)

    async def view(self, **kwargs):
        return await self.ns['infer_view'](min=0.01, **kwargs)

    async def test_default_matches_either_term_in_active_polarity(self):
        result = await self.view(q='Cute,Indifferent')
        self.assertEqual([r['n'] for r in result['results']], [0, 1, 2, 3, 4, 5])
        self.assertEqual(result['total'], 6)

    async def test_all_requires_every_term_in_same_neuron_and_polarity(self):
        result = await self.view(q='Cute,Indifferent', q_all=True)
        self.assertEqual([r['n'] for r in result['results']], [0, 3])
        self.assertEqual(result['total'], 2)
        marks = np.frombuffer(base64.b64decode(result['map']), dtype=np.float16)
        np.testing.assert_array_equal(marks, [8, 0, 0, -5, 0, 0, 0, 0])

    async def test_all_keeps_sign_and_layer_filters(self):
        result = await self.view(q='Cute,Indifferent', q_all=True, sign='-')
        self.assertEqual([r['n'] for r in result['results']], [3])
        self.assertEqual((await self.view(q='Cute,Indifferent', q_all=True, lo=1))['total'], 0)

    async def test_chinese_commas_whitespace_case_and_duplicates(self):
        result = await self.view(q=' , CUTE， indifferent , cute， ', q_all=True)
        self.assertEqual([r['n'] for r in result['results']], [0, 3])
        result = await self.view(q='健康， 乐观', q_all=True)
        self.assertEqual([r['n'] for r in result['results']], [7])

    async def test_unknown_term_excludes_all_only_in_all_mode(self):
        self.assertEqual((await self.view(q='Cute,missing', q_all=True))['total'], 0)
        self.assertEqual((await self.view(q='Cute,missing'))['total'], 4)

    async def test_empty_query_does_not_filter_and_single_term_is_unchanged(self):
        self.assertEqual((await self.view(q=' ,， ', q_all=True))['total'], 8)
        for q in ['Cute', 'Indifferent']:
            any_result = await self.view(q=q)
            all_result = await self.view(q=q, q_all=True)
            self.assertEqual(all_result, any_result)


if __name__ == '__main__':
    unittest.main()
