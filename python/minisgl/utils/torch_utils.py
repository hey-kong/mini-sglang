from __future__ import annotations

import functools
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@contextmanager
def torch_dtype(dtype: torch.dtype):
    import torch  # real import when used

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


_NVTX_STATE = threading.local()


def _nvtx_stack() -> list[str]:
    stack = getattr(_NVTX_STATE, "stack", None)
    if stack is None:
        stack = []
        _NVTX_STATE.stack = stack
    return stack


@contextmanager
def nvtx_pause_current_ranges():
    """Temporarily suspend ranges opened by :func:`nvtx_annotate`.

    This is useful for stream waits that must be enqueued in the middle of a
    layer.  The wait operation is emitted outside the layer NVTX range, then the
    range is reopened with the same name so profiler aggregations report pure
    layer compute time instead of compute plus the stream-wait gap.
    """
    import torch.cuda.nvtx as nvtx

    stack = _nvtx_stack()
    paused = stack.copy()
    for _ in reversed(paused):
        nvtx.range_pop()
    stack.clear()
    try:
        yield
    finally:
        for display_name in paused:
            nvtx.range_push(display_name)
            stack.append(display_name)


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    import torch.cuda.nvtx as nvtx

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))
            stack = _nvtx_stack()
            nvtx.range_push(display_name)
            stack.append(display_name)
            try:
                return fn(self, *args, **kwargs)
            finally:
                stack.pop()
                nvtx.range_pop()

        return wrapper

    return decorator
