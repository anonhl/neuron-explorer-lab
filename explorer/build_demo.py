"""Build the portable, offline demo from the current UI. Works on Windows/macOS/Linux.

Run from the project root: python explorer/build_demo.py
"""
import base64
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'explorer' / 'static'
DEMO = ROOT / 'explorer' / 'demo'
MODULES = ('app', 'map3d', 'scatter', 'i18n', 'model-loading', 'neuron-batches',
           'frequency', 'residual-panel', 'residual-vector', 'residual-lab')
ADDONS = ('controls/OrbitControls.js', 'lines/LineSegments2.js',
          'lines/LineSegmentsGeometry.js', 'lines/LineMaterial.js')


def rewrite_ui(source):
    return re.sub(r'''(\bfrom\s*|\bimport\s*\(\s*)(["'])\./([a-z0-9-]+)\.js(?:\?[^"']*)?\2''',
                  lambda m: f'{m[1]}{m[2]}demo-{m[3]}{m[2]}', source)


def data_url(source):
    return 'data:text/javascript;base64,' + base64.b64encode(source.encode('utf-8')).decode('ascii')


def build():
    imports = {'three': data_url((STATIC / 'vendor/three/build/three.module.js').read_text(encoding='utf-8'))}
    for name in MODULES:
        source = rewrite_ui((STATIC / f'{name}.js').read_text(encoding='utf-8'))
        if name == 'i18n':
            overrides = (DEMO / 'i18n-overrides.js').read_text(encoding='utf-8')
            source = source.replace('const DICT =', overrides + '\nconst DICT =', 1)
        imports['demo-' + name] = data_url(source)
    addons_root = STATIC / 'vendor/three/examples/jsm'
    for name in ADDONS:
        path = addons_root / name
        source = path.read_text(encoding='utf-8')
        def rewrite_addon(match):
            target = (path.parent / match[3]).resolve().relative_to(addons_root.resolve()).as_posix()
            return f'{match[1]}"three/addons/{target}"'
        source = re.sub(r'''(\bfrom\s*)(["'])(\.{1,2}/[^"']+)["']''',
                        rewrite_addon, source)
        imports['three/addons/' + name] = data_url(source)
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    html = re.sub(r'<title[^>]*>.*?</title>', '<title>神经元探索器 · 模拟练习</title>', html)
    css = (STATIC / 'style.css').read_text(encoding='utf-8') + '\n' + (DEMO / 'guide.css').read_text(encoding='utf-8')
    html = re.sub(r'<link rel="stylesheet"[^>]*>', lambda _: '<style>\n' + css + '\n</style>', html)
    html = re.sub(r'<script type="importmap">[\s\S]*?</script>',
                  lambda _: '<script type="importmap">\n' + json.dumps({'imports': imports}, separators=(',', ':')) + '\n</script>', html)
    html = html.replace('<header>', '<header>\n<button id="demo-help-toggle" type="button" aria-expanded="false" aria-controls="demo-help">模拟模式 <span aria-hidden="true">ⓘ</span></button>', 1)
    html = html.replace('</header>', '</header>\n' + (DEMO / 'guide.html').read_text(encoding='utf-8'), 1)
    mock = (DEMO / 'mock-api.js').read_text(encoding='utf-8')
    boot = (DEMO / 'boot.js').read_text(encoding='utf-8')
    html = re.sub(r'<script type="module">[\s\S]*?</script>',
                  lambda _: '<script>\n' + mock + '\n</script>\n<script type="module">\n' + boot + '\n</script>', html)
    # The portable HTML redistributes upstream code and Three.js on its own.
    # Keep their complete permission notices inside the artifact, too.
    notices = '\n\n'.join((ROOT / name).read_text(encoding='utf-8') for name in
                          ('LICENSE', 'explorer/static/vendor/three/LICENSE'))
    if '--' in notices:
        raise ValueError('License text cannot be embedded safely in an HTML comment.')
    html = '<!--\nNeuron Explorer portable demo: third-party and upstream license notices\n' + notices + '\n-->\n' + html
    destination = ROOT / 'neuron-explorer-demo.html'
    destination.write_text(html, encoding='utf-8')
    print(f'Built {destination.name}: {destination.stat().st_size:,} bytes; all UI modules and Three.js embedded.')


if __name__ == '__main__':
    build()
