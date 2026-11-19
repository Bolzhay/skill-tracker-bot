"""A tiny in-memory stand-in for the slice of the Firestore API bot/db.py uses,
so DB logic can be tested without credentials, network or quota."""

import copy
import itertools
import types
import uuid
from datetime import datetime, timedelta, timezone

from firebase_admin import firestore

_clock = itertools.count()
_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _resolve(data: dict) -> dict:
    """Real Firestore swaps the SERVER_TIMESTAMP sentinel for the commit time;
    an ever-increasing fake clock keeps order_by("created_at") deterministic."""
    return {
        key: (_EPOCH + timedelta(microseconds=next(_clock)) if value is firestore.SERVER_TIMESTAMP else value)
        for key, value in data.items()
    }


class FakeSnapshot:
    def __init__(self, ref, data):
        self.reference = ref
        self.id = ref.id
        self._data = data
        self.exists = data is not None

    def to_dict(self):
        return copy.deepcopy(self._data) if self._data is not None else None


class FakeDocRef:
    def __init__(self, client, path):
        self._client = client
        self.path = path
        self.id = path[-1]

    def get(self):
        return FakeSnapshot(self, self._client.store.get(self.path))

    def set(self, data, merge=False):
        data = _resolve(data)
        if merge and self.path in self._client.store:
            self._client.store[self.path].update(data)
        else:
            self._client.store[self.path] = data

    def update(self, data):
        if self.path not in self._client.store:
            raise KeyError(f"No document to update: {self.path}")
        self._client.store[self.path].update(_resolve(data))

    def collection(self, name):
        return FakeCollection(self._client, self.path + (name,))

    def collections(self):
        depth = len(self.path)
        names = sorted(
            {p[depth] for p in self._client.store if len(p) > depth and p[:depth] == self.path}
        )
        return [FakeCollection(self._client, self.path + (name,)) for name in names]


class FakeQuery:
    def __init__(self, snapshots, field, direction, limit=None):
        self._snapshots = snapshots
        self._field = field
        self._direction = direction
        self._limit = limit

    def limit(self, count):
        return FakeQuery(self._snapshots, self._field, self._direction, count)

    def stream(self):
        # Like Firestore, documents missing the ordered field are excluded.
        docs = [s for s in self._snapshots if self._field in s.to_dict()]
        docs.sort(key=lambda s: s.to_dict()[self._field], reverse=self._direction == "DESCENDING")
        return iter(docs[: self._limit] if self._limit else docs)


class FakeCollection:
    def __init__(self, client, path):
        self._client = client
        self.path = path
        self.id = path[-1]

    def document(self, doc_id=None):
        return FakeDocRef(self._client, self.path + (doc_id or uuid.uuid4().hex[:20],))

    def _snapshots(self):
        depth = len(self.path)
        paths = sorted(p for p in self._client.store if len(p) == depth + 1 and p[:depth] == self.path)
        return [FakeDocRef(self._client, p).get() for p in paths]

    def stream(self):
        return iter(self._snapshots())

    def order_by(self, field, direction="ASCENDING"):
        return FakeQuery(self._snapshots(), field, direction)

    def count(self):
        total = float(len(self._snapshots()))
        return types.SimpleNamespace(get=lambda: [[types.SimpleNamespace(value=total)]])


class FakeBatch:
    def __init__(self):
        self._ops = []

    def set(self, ref, data):
        self._ops.append(lambda: ref.set(data))

    def update(self, ref, data):
        self._ops.append(lambda: ref.update(data))

    def commit(self):
        for op in self._ops:
            op()


class FakeClient:
    def __init__(self):
        self.store = {}

    def collection(self, name):
        return FakeCollection(self, (name,))

    def batch(self):
        return FakeBatch()

    def recursive_delete(self, ref):
        depth = len(ref.path)
        for path in [p for p in self.store if p[:depth] == ref.path]:
            del self.store[path]
