from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Literal

import torch

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool
    from minisgl.moe import BaseMoeBackend


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # cpu tensor
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        self.input_ids = torch.cat([self.input_ids, next_token])

    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0

    def __repr__(self) -> str:
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


class PrefillLayerTimer:
    def __init__(self) -> None:
        self._segments: Dict[int, List[tuple[torch.Event, torch.Event]]] = {}
        self._active_starts: Dict[int, torch.Event] = {}

    def start_layer(self, layer_id: int) -> None:
        start = _create_timing_event()
        start.record(torch.cuda.current_stream())
        self._active_starts[layer_id] = start

    def pause_layer(self, layer_id: int) -> None:
        start = self._active_starts.pop(layer_id, None)
        if start is None:
            return
        end = _create_timing_event()
        end.record(torch.cuda.current_stream())
        self._segments.setdefault(layer_id, []).append((start, end))

    def resume_layer(self, layer_id: int) -> None:
        self.start_layer(layer_id)

    def end_layer(self, layer_id: int) -> None:
        self.pause_layer(layer_id)

    def layer_durations(self) -> Dict[int, float]:
        return {
            layer_id: sum(start.elapsed_time(end) for start, end in segments)
            for layer_id, segments in sorted(self._segments.items())
        }

    def log(self, logger) -> None:
        durations = self.layer_durations()
        if not durations:
            return
        total = sum(durations.values())
        per_layer = ", ".join(
            f"layer_{layer_id}={duration:.2f} ms" for layer_id, duration in durations.items()
        )
        logger.info_rank0(f"Prefill layer pure durations: total={total:.2f} ms, {per_layer}")


def _create_timing_event() -> torch.Event:
    return torch.cuda.Event(enable_timing=True)  # type: ignore


@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    # these fields should be set by scheduler
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    out_loc: torch.Tensor = field(init=False)
    padded_reqs: List[Req] = field(init=False)
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)
    # this field should be set by Context.forward_batch for prefill batches
    prefill_layer_timer: PrefillLayerTimer | None = field(default=None, init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)


@dataclass
class Context:
    page_size: int
    # NOTE: this table always treat page_size = 1
    page_table: torch.Tensor = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)
    moe_backend: BaseMoeBackend = field(init=False)
    kv_cache: BaseKVCachePool = field(init=False)
    _batch: Batch | None = field(default=None, init=False)
    _prefill_layer_timer: PrefillLayerTimer | None = field(default=None, init=False)

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @property
    def prefill_layer_timer(self) -> PrefillLayerTimer | None:
        return self._prefill_layer_timer

    @contextmanager
    def forward_batch(self, batch: Batch):
        assert self._batch is None, "Nested forward_batch is not allowed"
        timer = PrefillLayerTimer() if batch.is_prefill else None
        try:
            self._batch = batch
            self._prefill_layer_timer = timer
            batch.prefill_layer_timer = timer
            yield
        finally:
            self._batch = None
            self._prefill_layer_timer = None


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX


def get_global_ctx_optional() -> Context | None:
    return _GLOBAL_CTX
