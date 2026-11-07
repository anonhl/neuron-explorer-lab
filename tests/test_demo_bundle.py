"""The downloadable demo must contain the current UI and work without external modules."""
import base64
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StaticDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (ROOT / 'neuron-explorer-demo.html').read_text(encoding='utf-8')
        cls.imports = json.loads(re.search(r'<script type="importmap">([\s\S]*?)</script>', cls.html)[1])['imports']

    def test_every_module_is_embedded_and_imports_resolve_offline(self):
        for name, url in self.imports.items():
            with self.subTest(module=name):
                self.assertTrue(url.startswith('data:text/javascript;base64,'))
                source = base64.b64decode(url.split(',', 1)[1]).decode('utf-8')
                pattern = r'''(?:^\s*import\s+[\w*$,{}\s]+?\s+from\s*|\bimport\s*\(\s*)(["'])([^"'\r\n]+)\1'''
                for _, specifier in re.findall(pattern, source, re.MULTILINE):
                    self.assertIn(specifier, set(self.imports))

    def test_current_residual_ui_is_included_between_chat_and_details(self):
        self.assertLess(self.html.index('data-tab="chat"'), self.html.index('data-tab="residual"'))
        self.assertLess(self.html.index('data-tab="residual"'), self.html.index('data-tab="details"'))
        self.assertIn('id="inf-add-text"', self.html)
        self.assertIn('id="tab-residual"', self.html)
        self.assertIn('demo-residual-lab', self.imports)
        self.assertIn("case '/api/residual/tokens'", self.html)
        self.assertIn("case '/api/residual/experiment'", self.html)

    def test_demo_labels_do_not_claim_real_model_measurements(self):
        labels = base64.b64decode(self.imports['demo-i18n'].split(',', 1)[1]).decode('utf-8')
        self.assertIn('不加载模型、不调用推理服务', labels)
        self.assertIn('不是 Qwen 的真实测量结果', labels)
        self.assertIn('simulated: true', self.html)


if __name__ == '__main__':
    unittest.main()
