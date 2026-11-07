"""Session-local inference results: compact summaries in RAM, full captures on disk."""
from collections import OrderedDict
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from uuid import uuid4


class InferenceCache:
    def __init__(self, limit=8):
        self.limit = limit
        self.directory = TemporaryDirectory(prefix='neuron-inference-')
        self.entries = OrderedDict()
        self.context = None
        self.inspections = OrderedDict()

    def configure(self, context):
        if context != self.context:
            self.clear()
            self.context = context

    def clear(self):
        for path, _ in self.entries.values():
            path.unlink(missing_ok=True)
        self.entries.clear()
        self.inspections.clear()

    def save(self, key, snapshot, summary, heads):
        path = Path(self.directory.name) / (uuid4().hex + '.pickle')
        with path.open('wb') as stream:
            pickle.dump(snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(heads, stream, protocol=pickle.HIGHEST_PROTOCOL)
        if key in self.entries:
            self.entries.pop(key)[0].unlink(missing_ok=True)
        self.entries[key] = (path, summary)
        while len(self.entries) > self.limit:
            self.entries.popitem(last=False)[1][0].unlink(missing_ok=True)

    def retain(self, keys):
        for key in list(self.entries):
            if key not in keys:
                self.entries.pop(key)[0].unlink(missing_ok=True)

    def restore(self, key, include_heads=True):
        if key not in self.entries:
            return None
        self.entries.move_to_end(key)
        with self.entries[key][0].open('rb') as stream:
            snapshot = pickle.load(stream)
            return snapshot, pickle.load(stream) if include_heads else None

    def summary(self, key):
        entry = self.entries.get(key)
        return entry[1] if entry else None

    def inspected(self, key):
        if key not in self.inspections:
            return None
        self.inspections.move_to_end(key)
        return self.inspections[key]

    def save_inspection(self, key, result):
        self.inspections[key] = result
        self.inspections.move_to_end(key)
        while len(self.inspections) > 16:
            self.inspections.popitem(last=False)
