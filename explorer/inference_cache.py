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

    def configure(self, context):
        if context != self.context:
            self.clear()
            self.context = context

    def clear(self):
        for path, _ in self.entries.values():
            path.unlink(missing_ok=True)
        self.entries.clear()

    def save(self, key, snapshot, summary, heads):
        path = Path(self.directory.name) / (uuid4().hex + '.pickle')
        with path.open('wb') as stream:
            pickle.dump((snapshot, heads), stream, protocol=pickle.HIGHEST_PROTOCOL)
        if key in self.entries:
            self.entries.pop(key)[0].unlink(missing_ok=True)
        self.entries[key] = (path, summary)
        while len(self.entries) > self.limit:
            self.entries.popitem(last=False)[1][0].unlink(missing_ok=True)

    def retain(self, keys):
        for key in list(self.entries):
            if key not in keys:
                self.entries.pop(key)[0].unlink(missing_ok=True)

    def restore(self, key):
        if key not in self.entries:
            return None
        self.entries.move_to_end(key)
        with self.entries[key][0].open('rb') as stream:
            return pickle.load(stream)

    def summary(self, key):
        entry = self.entries.get(key)
        return entry[1] if entry else None
