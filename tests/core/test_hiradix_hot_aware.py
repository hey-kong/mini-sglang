from __future__ import annotations

from typing import Any, cast

import pytest
import torch

import minisgl.core as core
import minisgl.hicache.controller as hicache_controller_module
from minisgl.hicache.controller import Ack, HiCacheController
from minisgl.kvcache.hiradix_cache import HiRadixCacheHandle, HiRadixPrefixCache, HiRadixTreeNode


class _FakeEvent:
    def query(self) -> bool:
        return True

    def elapsed_time(self, _other) -> float:
        return 1.0


@pytest.fixture(autouse=True)
def reset_global_ctx():
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = old_ctx


def _build_cache_with_one_cuda_host_node(length: int = 4) -> tuple[HiRadixPrefixCache, HiRadixCacheHandle]:
    core.set_global_ctx(core.Context(page_size=1))
    cache = HiRadixPrefixCache(device=torch.device("cpu"))

    key = torch.arange(length, dtype=torch.int32)
    cuda_indices = torch.arange(length, dtype=torch.int32)
    node = HiRadixTreeNode(cache.key_fn)
    node.set_key_value(key=key, cuda_value=cuda_indices.clone())
    node.set_parent(cache.root_node)
    cache.evictable_size += length

    handle = HiRadixCacheHandle(cached_len=length, node=node)
    host_indices = torch.arange(100, 100 + length, dtype=torch.int32)
    cache.set_host(handle, host_indices)
    return cache, handle


def _build_stub_controller(
    cache: HiRadixPrefixCache,
    ack: Ack,
    released_holder: list[torch.Tensor],
) -> HiCacheController:
    controller = object.__new__(HiCacheController)
    controller.hiradix_cache = cache
    controller.ack_load_queue = []
    controller.ack_write_queue = [ack]
    controller.hot_aware_enabled = False
    controller.released_cuda_tokens = 0
    controller.released_cuda_acks = 0
    controller.write_ack_processed = 0
    controller.free_cuda_indices = lambda indices: released_holder.append(indices.clone())
    controller._log_transaction = cast(Any, lambda ack, stage: None)
    return controller


def _patch_all_reduce_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hicache_controller_module.dist,
        "all_reduce",
        lambda _tensor, op=None, group=None: None,
    )


def test_refresh_drop_cuda_suffix_after_write_ack(monkeypatch: pytest.MonkeyPatch):
    cache, handle = _build_cache_with_one_cuda_host_node(length=4)

    # Mimic prepare_write behavior: lock once, then refresh() unlocks and drops.
    cache.lock_handle(handle, unlock=False)

    released: list[torch.Tensor] = []
    ack = Ack(
        ack_id=1,
        handles=[handle],
        drop_lens=[4],
        drop_flags=[True],
        num_tokens=4,
        start_event=cast(torch.Event, _FakeEvent()),
        finish_event=cast(torch.Event, _FakeEvent()),
    )
    controller = _build_stub_controller(cache, ack, released)
    _patch_all_reduce_noop(monkeypatch)

    controller.refresh(tp_cpu_group=cast(Any, None))

    assert len(controller.ack_write_queue) == 0
    assert len(released) == 1
    assert torch.equal(released[0], torch.arange(4, dtype=torch.int32))
    assert handle.node.on_host_only()
    assert cache.evictable_size == 0


def test_refresh_skip_drop_when_ref_count_still_positive(monkeypatch: pytest.MonkeyPatch):
    cache, handle = _build_cache_with_one_cuda_host_node(length=4)

    # One lock for write transaction + one extra lock for in-flight user.
    cache.lock_handle(handle, unlock=False)
    cache.lock_handle(handle, unlock=False)

    released: list[torch.Tensor] = []
    ack = Ack(
        ack_id=2,
        handles=[handle],
        drop_lens=[4],
        drop_flags=[True],
        num_tokens=4,
        start_event=cast(torch.Event, _FakeEvent()),
        finish_event=cast(torch.Event, _FakeEvent()),
    )
    controller = _build_stub_controller(cache, ack, released)
    _patch_all_reduce_noop(monkeypatch)

    controller.refresh(tp_cpu_group=cast(Any, None))

    assert len(released) == 0
    assert handle.node._cuda_value is not None
    assert handle.node._host_value is not None
    assert handle.node.ref_count == 1
    assert cache.evictable_size == 0

    # Cleanup the extra lock and verify size info can be restored.
    cache.lock_handle(handle, unlock=True)
    assert cache.evictable_size == 4


def test_refresh_skip_drop_when_drop_flag_is_false(monkeypatch: pytest.MonkeyPatch):
    cache, handle = _build_cache_with_one_cuda_host_node(length=4)

    # Mimic prepare_write behavior: refresh() will unlock this lock.
    cache.lock_handle(handle, unlock=False)

    released: list[torch.Tensor] = []
    ack = Ack(
        ack_id=3,
        handles=[handle],
        drop_lens=[4],
        drop_flags=[False],
        num_tokens=4,
        start_event=cast(torch.Event, _FakeEvent()),
        finish_event=cast(torch.Event, _FakeEvent()),
    )
    controller = _build_stub_controller(cache, ack, released)
    _patch_all_reduce_noop(monkeypatch)

    controller.refresh(tp_cpu_group=cast(Any, None))

    assert len(controller.ack_write_queue) == 0
    assert len(released) == 0
    assert handle.node._cuda_value is not None
    assert handle.node._host_value is not None
    assert handle.node.ref_count == 0
    assert cache.evictable_size == 4
