"""Portable, atomic storage for reusable neuron intervention batches (no model dependency)."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from uuid import uuid4


def validate_operations(operations, n_layers, intermediate_size):
    if not isinstance(operations, list) or not 1 <= len(operations) <= 1000:
        raise ValueError("A batch must contain between 1 and 1000 operations.")
    result = []
    for op in operations:
        if not isinstance(op, dict):
            raise ValueError("Invalid neuron operation.")
        layer, n, mode, dose = (op.get(k) for k in ("layer", "n", "mode", "dose"))
        if type(layer) is not int or not 0 <= layer < n_layers:
            raise ValueError(f"Layer must be an integer from 0 to {n_layers - 1}.")
        if type(n) is not int or not 0 <= n < intermediate_size:
            raise ValueError(f"Neuron must be an integer from 0 to {intermediate_size - 1}.")
        if mode not in ("add", "mul"):
            raise ValueError("Operation mode must be add or mul.")
        if type(dose) not in (int, float) or not math.isfinite(dose):
            raise ValueError("Dose must be a finite number.")
        enabled = op.get("enabled", True)
        if type(enabled) is not bool:
            raise ValueError("Enabled must be true or false.")
        unit = "residual_norm" if mode == "add" else "multiplier"
        if op.get("unit", unit) != unit:
            raise ValueError("Unsupported dose unit.")
        result.append({"layer": layer, "n": n, "mode": mode, "dose": float(dose),
                       "enabled": enabled, "unit": unit})
    if not any(op["enabled"] for op in result):
        raise ValueError("Enable at least one operation before saving.")
    return result


class BatchStore:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()

    def _read(self):
        try:
            with self.path.open(encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {"version": 1, "groups": [], "batches": []}
        if (not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("batches"), list)
                or not isinstance(data.get("groups"), list)):
            raise ValueError("Unsupported neuron batch file format.")
        return data

    def _write(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Close before replace: Windows cannot replace a file with an open temporary handle.
        fd, temporary = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def list(self):
        with self.lock:
            return deepcopy(self._read())

    def create_group(self, name):
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 120:
            raise ValueError("Group name must contain between 1 and 120 characters.")
        name = name.strip()
        with self.lock:
            data = self._read()
            if any(g["name"].casefold() == name.casefold() for g in data["groups"]):
                raise ValueError("A group with this name already exists.")
            group = {"id": uuid4().hex, "name": name, "created_at": datetime.now(timezone.utc).isoformat()}
            data["groups"].append(group)
            self._write(data)
        return deepcopy(group)

    def get(self, batch_id):
        with self.lock:
            batches = self._read()["batches"]
            return batches[self._position(batches, batch_id)]

    @staticmethod
    def _position(batches, batch_id):
        for index, batch in enumerate(batches):
            if batch["id"] == batch_id:
                return index
        raise KeyError(batch_id)

    @staticmethod
    def _batch_name(name):
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 120:
            raise ValueError("Batch name must contain between 1 and 120 characters.")
        return name.strip()

    def rename(self, batch_id, name):
        name = self._batch_name(name)
        with self.lock:
            data = self._read()
            batches = data["batches"]
            batch = batches[self._position(batches, batch_id)]
            batch["name"] = name
            self._write(data)
            return batch

    def delete(self, batch_id):
        with self.lock:
            data = self._read()
            batches = data["batches"]
            batch = batches.pop(self._position(batches, batch_id))
            self._write(data)
            return batch

    def save(self, *, group_id, name, notes, model, model_revision, quantization, operations, compute_dtype="bfloat16",
             source_batch_id=None):
        name = self._batch_name(name)
        if not isinstance(notes, str) or len(notes) > 2000:
            raise ValueError("Notes must contain at most 2000 characters.")
        batch = {"id": uuid4().hex, "group_id": group_id, "name": name, "notes": notes.strip(), "model": model,
                 "model_revision": model_revision, "quantization": quantization, "compute_dtype": compute_dtype,
                 "created_at": datetime.now(timezone.utc).isoformat(), "operations": deepcopy(operations)}
        with self.lock:
            data = self._read()
            if not any(g["id"] == group_id for g in data["groups"]):
                raise ValueError("Select an existing group before saving.")
            if source_batch_id is not None:
                source = next((saved for saved in data["batches"] if saved["id"] == source_batch_id), None)
                if source is None:
                    raise ValueError("Original batch could not be found.")
                if (source["model"] == model and source.get("model_revision") == model_revision
                        and source["quantization"] == quantization and source.get("compute_dtype", "bfloat16") == compute_dtype
                        and source["operations"] == operations):
                    raise ValueError("Change a neuron operation before saving a new batch.")
            data["batches"].insert(0, batch)
            self._write(data)
        return deepcopy(batch)
