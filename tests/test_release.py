"""Release tests: private data exclusion, cache integrity and credential redaction."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_release', ROOT / 'tools/prepare_release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def test_private_files_and_unknown_caches_never_enter_source_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('README.md', 'explorer/runtime.py', 'sweep/run_layer.sh',
                         'explorer/cache/vocab.json', 'explorer/cache/neuron_batches.json',
                         'explorer/cache/agent/auth.json', 'explorer/names.json',
                         'explorer/agent_prompt.local.md', 'models/private.safetensors',
                         '.env', '.env.local', 'docs/.env/key.txt', 'personal-notes.md',
                         'logs/private.txt', 'dist/old.zip', 'explorer/__pycache__/secret.py',
                         'scripts/mirror.py', 'scripts/private.py'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('private fixture', encoding='utf-8')
            with patch.object(release, 'cache_metadata', return_value={'files': [{'path': 'explorer/cache/vocab.json'}]}):
                selected = {p.relative_to(root).as_posix() for p in release.collect_files(root)}
            self.assertEqual(selected, {'README.md', 'explorer/runtime.py', 'sweep/run_layer.sh',
                                        'explorer/cache/vocab.json', 'scripts/mirror.py'})

    def test_symlink_cannot_pull_external_data_into_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'explorer').mkdir()
            try:
                (root / 'explorer/linked.py').symlink_to(root / 'outside.py')
            except OSError:
                self.skipTest('Symlink creation not available for this account')
            with patch.object(release, 'cache_metadata', return_value={'files': []}):
                with self.assertRaisesRegex(ValueError, 'Symlinks'):
                    release.collect_files(root)

    def test_secret_findings_only_return_category_and_line(self):
        token = 'gh' + 'p_' + 'A' * 36
        found = release.secret_findings('safe\n' + token)
        self.assertEqual(found, [('GitHub token', 2)])
        self.assertNotIn(token, str(found))
        self.assertTrue(release.secret_findings('api_key = "' + 'Z' * 28 + '"'))
        self.assertEqual(release.secret_findings('token = os.environ.get("HF_TOKEN")'), [])

    def test_release_inputs_and_reference_caches_validate(self):
        files, _ = release.validate(ROOT)
        names = {p.relative_to(ROOT).as_posix() for p in files}
        self.assertIn('LICENSE', names)
        self.assertIn('sweep/run_layer.sh', names)
        self.assertIn('explorer/static/vendor/three/examples/fonts/droid/NOTICE', names)
        self.assertNotIn('explorer/cache/connections.sqlite', names)
        self.assertEqual(len(json.loads((ROOT / 'docs/upstream-cache-provenance.json').read_text())['files']), 9)

    def test_archive_manifest_and_checksum_describe_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'README.md'
            source.write_text('公开源码\n', encoding='utf-8')
            output = root / 'release.zip'
            with patch.object(release, 'validate', return_value=([source], 'not available')), \
                 patch.object(release, 'cache_metadata', return_value={'reference_commit': 'fixture'}):
                release.build_release(root, output)
                first = output.read_bytes()
                release.build_release(root, output)
            self.assertEqual(first, output.read_bytes(), 'Identical inputs produce identical ZIP bytes')
            self.assertTrue(output.with_name('release.zip.sha256').read_text().startswith(hashlib.sha256(first).hexdigest()))
            with zipfile.ZipFile(output) as archive:
                prefix = 'neuron-explorer-lab-0.1.0/'
                manifest = json.loads(archive.read(prefix + 'RELEASE_MANIFEST.json'))
                self.assertEqual(len(archive.namelist()), 2)
                data = archive.read(prefix + 'README.md')
                self.assertEqual(manifest['files'], [{'path': 'README.md', 'size': len(data),
                                                       'sha256': hashlib.sha256(data).hexdigest()}])


if __name__ == '__main__':
    unittest.main()
