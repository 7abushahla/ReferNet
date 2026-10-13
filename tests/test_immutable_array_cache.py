import hashlib
import importlib
import random
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cache_type(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "src"))

    def create(**kwargs):
        module_path = ROOT / "src/refer_or_clear/data/immutable_array_cache.py"
        assert module_path.is_file(), "immutable array cache helper is missing"
        return importlib.import_module(
            "refer_or_clear.data.immutable_array_cache"
        ).ImmutableArrayCache(**kwargs)

    return create


def binding():
    return {
        "raw_sha256": "a" * 64,
        "product_sha256": "b" * 64,
        "registration": {"inverse": [1, 0, 0, 0, 1, 0]},
        "preprocess": {"canvas": [4, 4], "mode": "RGB"},
    }


def canvases():
    return {
        "query": np.arange(48, dtype=np.uint8).reshape(4, 4, 3),
        "reference": np.full((4, 4, 3), 17, dtype=np.uint8),
        "zone": np.eye(4, dtype=np.uint8),
        "overlap": np.ones((4, 4), dtype=np.uint8),
    }


def test_hit_is_byte_identical_and_validates_before_lookup(cache_type):
    cache = cache_type(max_bytes=128)
    events = []
    original = canvases()

    def validate():
        events.append("validate")

    def build():
        events.append("build")
        return original

    first = cache.fetch(binding(), validate=validate, build=build)
    second = cache.fetch(binding(), validate=validate, build=build)
    assert events == ["validate", "build", "validate"]
    assert cache.current_bytes == 128
    assert len(cache) == 1
    for name in original:
        assert first[name].dtype == second[name].dtype == np.uint8
        assert first[name].shape == second[name].shape == original[name].shape
        assert first[name].tobytes() == second[name].tobytes() == original[name].tobytes()


def test_validate_hits_false_authenticates_only_before_cold_build(cache_type):
    cache = cache_type(max_bytes=128)
    events = []

    def validate():
        events.append("validate")

    def build():
        events.append("build")
        return canvases()

    cache.fetch(binding(), validate=validate, build=build, validate_hits=False)
    cache.fetch(binding(), validate=validate, build=build, validate_hits=False)
    assert events == ["validate", "build"]
    with pytest.raises(TypeError):
        cache.fetch(binding(), validate=validate, build=build, validate_hits=0)
    disabled = cache_type(max_bytes=0)
    events.clear()
    disabled.fetch(binding(), validate=validate, build=build, validate_hits=False)
    disabled.fetch(binding(), validate=validate, build=build, validate_hits=False)
    assert events == ["validate", "build", "validate", "build"]


def test_builder_and_augmentation_writes_cannot_poison_cache(cache_type):
    cache = cache_type(max_bytes=128)
    original = canvases()
    expected = {name: array.tobytes() for name, array in original.items()}
    first = cache.fetch(binding(), validate=lambda: None, build=lambda: original)
    for array in original.values():
        array.fill(99)
    for array in first.values():
        array.fill(255)
        assert array.flags.writeable
    first.clear()
    second = cache.fetch(binding(), validate=lambda: None, build=lambda: pytest.fail("hit"))
    for name, array in second.items():
        assert array.tobytes() == expected[name]
        assert array.flags.owndata


def test_changed_file_rejected_on_warm_hit(cache_type, tmp_path):
    product = tmp_path / "synthetic-product.bin"
    product.write_bytes(b"authorized synthetic bytes")
    authority = binding()
    authority["product_sha256"] = hashlib.sha256(product.read_bytes()).hexdigest()
    cache = cache_type(max_bytes=128)

    def validate():
        if hashlib.sha256(product.read_bytes()).hexdigest() != authority["product_sha256"]:
            raise ValueError("product changed")

    cache.fetch(authority, validate=validate, build=canvases)
    product.write_bytes(b"mutated synthetic bytes")
    with pytest.raises(ValueError, match="product changed"):
        cache.fetch(authority, validate=validate, build=lambda: pytest.fail("must not build"))


@pytest.mark.parametrize("field", ["raw_sha256", "product_sha256", "registration", "preprocess"])
def test_each_authority_component_changes_key(cache_type, field):
    cache = cache_type(max_bytes=512)
    first = binding()
    changed = binding()
    changed[field] = "different authorized binding"
    calls = []

    def build():
        calls.append(1)
        return canvases()

    cache.fetch(first, validate=lambda: None, build=build)
    cache.fetch(changed, validate=lambda: None, build=build)
    cache.fetch(dict(reversed(list(first.items()))), validate=lambda: None, build=build)
    assert len(calls) == 2


def test_lru_eviction_and_memory_bound(cache_type):
    cache = cache_type(max_bytes=8)
    calls = []

    def fetch(identifier):
        def build():
            calls.append(identifier)
            return {"plane": np.full(4, identifier, dtype=np.uint8)}

        return cache.fetch({"id": identifier}, validate=lambda: None, build=build)

    for identifier in [1, 2, 1, 3, 1, 2]:
        fetch(identifier)
        assert cache.current_bytes <= 8
        assert len(cache) <= 2
    assert calls == [1, 2, 3, 2]
    assert cache.current_bytes == 8


@pytest.mark.parametrize("budget", [0, 127])
def test_disabled_or_oversized_cache_still_validates_and_copies(cache_type, budget):
    cache = cache_type(max_bytes=budget)
    events = []
    original = canvases()
    for unused in range(2):
        result = cache.fetch(
            binding(),
            validate=lambda: events.append("validate"),
            build=lambda: events.append("build") or original,
        )
        result["query"].fill(0)
    assert events == ["validate", "build", "validate", "build"]
    assert original["query"].tobytes() == canvases()["query"].tobytes()
    assert cache.current_bytes == len(cache) == 0


def test_oversized_entry_does_not_evict_retained_entry(cache_type):
    cache = cache_type(max_bytes=4)
    cache.fetch({"id": 1}, validate=lambda: None,
                build=lambda: {"plane": np.ones(4, dtype=np.uint8)})
    cache.fetch({"id": 2}, validate=lambda: None,
                build=lambda: {"plane": np.ones(5, dtype=np.uint8)})
    cache.fetch({"id": 1}, validate=lambda: None,
                build=lambda: pytest.fail("oversized entry must not evict"))
    assert cache.current_bytes == 4
    assert len(cache) == 1


def test_nested_readonly_binding_is_supported(cache_type):
    authority = binding()
    authority["registration"] = MappingProxyType(authority["registration"])
    cache = cache_type(max_bytes=128)
    cache.fetch(MappingProxyType(authority), validate=lambda: None, build=canvases)
    cache.fetch(binding(), validate=lambda: None, build=lambda: pytest.fail("same binding"))


def test_validation_failure_before_cold_build_does_not_retain(cache_type):
    cache = cache_type(max_bytes=128)

    def validate():
        raise RuntimeError("authentication failed")

    with pytest.raises(RuntimeError, match="authentication failed"):
        cache.fetch(binding(), validate=validate, build=lambda: pytest.fail("not authorized"))
    assert cache.current_bytes == len(cache) == 0


def test_failed_builder_does_not_evict_prior_valid_entry(cache_type):
    cache = cache_type(max_bytes=128)
    cache.fetch(binding(), validate=lambda: None, build=canvases)

    def build():
        raise RuntimeError("construction failed")

    with pytest.raises(RuntimeError, match="construction failed"):
        cache.fetch({"other": "binding"}, validate=lambda: None, build=build)
    cache.fetch(binding(), validate=lambda: None, build=lambda: pytest.fail("prior entry"))
    assert cache.current_bytes == 128
    assert len(cache) == 1


def test_storage_contains_only_immutable_payloads(cache_type):
    cache = cache_type(max_bytes=128)
    cache.fetch(binding(), validate=lambda: None, build=canvases)
    for entry in cache._entries.values():
        assert isinstance(entry, tuple)
        for name, shape, payload in entry:
            assert isinstance(shape, tuple)
            assert isinstance(payload, bytes)
            view = np.frombuffer(payload, dtype=np.uint8)
            assert not view.flags.writeable
            with pytest.raises(ValueError):
                view.setflags(write=True)


def test_noncontiguous_arrays_roundtrip(cache_type):
    original = np.arange(24, dtype=np.uint8).reshape(4, 6)[:, ::2]
    cache = cache_type(max_bytes=12)
    for unused in range(2):
        result = cache.fetch({"id": 1}, validate=lambda: None, build=lambda: {"plane": original})
        assert result["plane"].shape == original.shape
        assert result["plane"].tobytes() == original.tobytes()


def test_cache_does_not_consume_global_or_private_rng(cache_type):
    private = random.Random(42)
    numpy_private = np.random.default_rng(42)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    private_before = private.getstate()
    generator_before = repr(numpy_private.bit_generator.state)
    cache = cache_type(max_bytes=128)
    for unused in range(2):
        cache.fetch(binding(), validate=lambda: None, build=canvases)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert private.getstate() == private_before
    assert repr(numpy_private.bit_generator.state) == generator_before


@pytest.mark.parametrize("budget", [-1, 1.5, True, "128"])
def test_invalid_budget_rejected(cache_type, budget):
    with pytest.raises((TypeError, ValueError)):
        cache_type(max_bytes=budget)


@pytest.mark.parametrize("authority", [{}, {1: "not string"}, {"id": float("nan")}, {"id": object()}])
def test_invalid_binding_rejected_before_callbacks(cache_type, authority):
    cache = cache_type(max_bytes=128)
    with pytest.raises((TypeError, ValueError)):
        cache.fetch(authority, validate=lambda: pytest.fail("invalid binding"), build=canvases)


@pytest.mark.parametrize("callback", ["validate", "build"])
def test_binding_mutation_during_callback_is_rejected(cache_type, callback):
    cache = cache_type(max_bytes=128)
    authority = binding()

    def mutate():
        authority["registration"]["inverse"][2] = 5
        return canvases() if callback == "build" else None

    with pytest.raises(ValueError, match="binding changed"):
        cache.fetch(authority, validate=mutate if callback == "validate" else lambda: None,
                    build=mutate if callback == "build" else canvases)
    assert cache.current_bytes == len(cache) == 0


def test_false_validation_cannot_serve_cached_arrays(cache_type):
    cache = cache_type(max_bytes=128)
    cache.fetch(binding(), validate=lambda: None, build=canvases)
    with pytest.raises(ValueError, match="validation"):
        cache.fetch(binding(), validate=lambda: False, build=lambda: pytest.fail("validation failed"))


@pytest.mark.parametrize("arrays", [{}, {"plane": np.zeros(4)}, {"plane": np.zeros(0, dtype=np.uint8)},
                                     {"plane": [1, 2]}, {1: np.ones(4, dtype=np.uint8)}])
def test_invalid_builder_output_is_not_cached(cache_type, arrays):
    cache = cache_type(max_bytes=128)
    with pytest.raises((TypeError, ValueError)):
        cache.fetch(binding(), validate=lambda: None, build=lambda: arrays)
    assert cache.current_bytes == len(cache) == 0
