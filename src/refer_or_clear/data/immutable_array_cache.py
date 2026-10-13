"""Pure, single-process caching of authenticated pre-augmentation uint8 arrays."""

from collections import OrderedDict
from collections.abc import Callable, Mapping
import hashlib
import json

import numpy as np


def _json_value(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("binding keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError("binding must contain only JSON values")


def _binding_key(binding: Mapping) -> bytes:
    if not isinstance(binding, Mapping) or not binding:
        raise ValueError("binding must be a nonempty mapping")
    encoded = json.dumps(
        _json_value(binding), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).digest()


class ImmutableArrayCache:
    """LRU cache with a strict retained-array-payload byte budget.

    The caller supplies a complete JSON binding including raw/product hashes,
    saved registration, preprocessing, and every other output-affecting input.
    This helper neither infers authority nor authenticates files itself.
    ``validate`` must authenticate current inputs on EVERY fetch and raise on
    failure, returning None or True on success. ``build`` returns named uint8
    arrays before augmentation. Both callbacks must be deterministic and must
    not mutate the binding. Only callers perform decoding or use private RNG.

    Stored payloads are immutable bytes, never aliases of builder arrays.
    Every return contains independent writable copies for augmentation.
    ``max_bytes`` bounds retained array bytes, not Python bookkeeping, transient
    construction/copies, or process RSS. Empty arrays are rejected so entry
    count is also bounded by the payload budget. Zero disables retention;
    oversized entries bypass retention without evicting useful entries.
    This class is not a concurrent or multiprocessing cache.
    """

    def __init__(self, *, max_bytes: int):
        if type(max_bytes) is not int:
            raise TypeError("max_bytes must be an integer")
        if max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative")
        self._max_bytes = max_bytes
        self._current_bytes = 0
        self._entries = OrderedDict()

    @property
    def current_bytes(self) -> int:
        """Total retained immutable array payload bytes."""
        return self._current_bytes

    def __len__(self) -> int:
        return len(self._entries)

    def fetch(
        self,
        binding: Mapping,
        *,
        validate: Callable[[], object],
        build: Callable[[], Mapping[str, np.ndarray]],
        validate_hits: bool = True,
    ) -> dict[str, np.ndarray]:
        """Return defensive copies of all planes.

        ``validate`` always runs before a cold build. With ``validate_hits``
        (the default) it also runs before every warm lookup; with
        ``validate_hits=False`` a warm hit trusts the inputs authenticated at
        build time, so the caller must guard its authorities separately.
        """
        if type(validate_hits) is not bool:
            raise TypeError("validate_hits must be a boolean")
        key = _binding_key(binding)
        if validate_hits or key not in self._entries:
            validation = validate()
            if validation is not None and validation is not True:
                raise ValueError("input validation must succeed explicitly or raise")
            if _binding_key(binding) != key:
                raise ValueError("binding changed during validation")

        if key in self._entries:
            frozen = self._entries[key]
            self._entries.move_to_end(key)
        else:
            arrays = build()
            if _binding_key(binding) != key:
                raise ValueError("binding changed during construction")
            if not isinstance(arrays, Mapping) or not arrays:
                raise ValueError("build must return a nonempty array mapping")
            planes = []
            for name, array in arrays.items():
                if not isinstance(name, str) or not name:
                    raise TypeError("array names must be nonempty strings")
                if not isinstance(array, np.ndarray) or array.dtype != np.uint8:
                    raise TypeError("build must return uint8 numpy arrays")
                if not array.size:
                    raise ValueError("empty arrays cannot be cached")
                planes.append((name, tuple(array.shape), array.tobytes(order="C")))
            frozen = tuple(planes)
            size = sum(len(payload) for name, shape, payload in frozen)
            if size <= self._max_bytes:
                while self._current_bytes + size > self._max_bytes:
                    unused_key, evicted = self._entries.popitem(last=False)
                    self._current_bytes -= sum(len(payload) for name, shape, payload in evicted)
                self._entries[key] = frozen
                self._current_bytes += size

        return {
            name: np.frombuffer(payload, dtype=np.uint8).reshape(shape).copy()
            for name, shape, payload in frozen
        }
