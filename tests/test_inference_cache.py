"""Session cache reuse, invalidation and bounded memory ownership."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'explorer'))
from inference_cache import InferenceCache


def fail_head_restore():
    raise AssertionError('Context reads must not load attention-head tensors')


class UnrestorableHeads:
    def __reduce__(self):
        return fail_head_restore, ()


class InferenceCacheTests(unittest.TestCase):
    def setUp(self):
        self.cache = InferenceCache()
        self.addCleanup(self.cache.directory.cleanup)
        self.cache.configure('model-1')

    def test_switch_back_restores_exact_capture_without_inference(self):
        self.cache.save(('a', False), {'capture': 'a-original', 'acts': [1, 2]}, {'peak': 2}, {'ids': [1]})
        self.cache.save(('b', False), {'capture': 'b-original'}, {'peak': 3}, {'ids': [2]})
        snapshot, heads = self.cache.restore(('a', False))
        self.assertEqual(snapshot, {'capture': 'a-original', 'acts': [1, 2]})
        self.assertEqual(heads, {'ids': [1]})
        self.assertEqual(self.cache.summary(('a', False)), {'peak': 2})
        # Only a disk path and compact summary are retained per text.
        self.assertIsInstance(self.cache.entries[('a', False)][0], Path)

    def test_edits_prune_old_versions_without_evicting_unchanged_text(self):
        self.cache.save(('a', False), {'capture': 'stable'}, {}, None)
        for version in range(20):
            key = (f'b-{version}', False)
            self.cache.retain({('a', False), key})
            self.cache.save(key, {'capture': version}, {}, None)
        self.assertEqual(self.cache.restore(('a', False))[0]['capture'], 'stable')
        self.assertEqual(len(self.cache.entries), 2)

    def test_context_reads_skip_full_attention_head_tensors(self):
        self.cache.save(('a', False), {'capture': 'a'}, {}, UnrestorableHeads())
        self.assertEqual(self.cache.restore(('a', False), include_heads=False), ({'capture': 'a'}, None))

    def test_model_change_invalidates_cache(self):
        self.cache.save(('a', False), {}, {}, None)
        self.cache.configure('model-1')
        self.assertIsNotNone(self.cache.restore(('a', False)))
        self.cache.configure('model-2')
        self.assertIsNone(self.cache.restore(('a', False)))
        self.assertEqual(list(Path(self.cache.directory.name).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
